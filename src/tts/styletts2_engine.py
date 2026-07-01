"""
StyleTTS2 GPU TTS engine (optional, CPU-safe).

load() returns False silently when:
  - No CUDA GPU detected
  - StyleTTS2 repo not cloned at docreader/StyleTTS2/
  - LibriTTS checkpoint not present

When load() returns False the caller falls back to XTTS/Piper.
"""

import logging
import sys
import numpy as np
from pathlib import Path

from src.config import STYLETTS2_DIR, EMOTION_CLIPS_DIR

logger = logging.getLogger(__name__)

MODEL_DIR       = STYLETTS2_DIR / "Models" / "LibriTTS"
CHECKPOINT_NAME = "epochs_2nd_00020.pth"
CONFIG_NAME     = "config.yml"
SAMPLE_RATE     = 24000

# Emotion label → reference wav filename
_EMOTION_CLIP_MAP = {
    "NEUTRAL":    "neutral.wav",
    "INTENSE":    "intense.wav",
    "COLD":       "cold.wav",
    "EXPRESSIVE": "expressive.wav",
}


class StyleTTS2Engine:
    """
    Wraps StyleTTS2 for sentence-level emotion-aware synthesis.
    All heavy imports happen inside load() so CPU users pay nothing.
    """

    def __init__(self):
        self._model = None
        self._sampler = None
        self._loaded = False
        self._style_cache: dict = {}

    def load(self) -> bool:
        """
        Try to initialise StyleTTS2 on GPU.
        Returns True on success, False on any failure (no exception raised).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return False

            if not STYLETTS2_DIR.exists():
                return False

            checkpoint = MODEL_DIR / CHECKPOINT_NAME
            config_file = MODEL_DIR / CONFIG_NAME
            if not checkpoint.exists() or not config_file.exists():
                return False

            sts2_str = str(STYLETTS2_DIR)
            if sts2_str not in sys.path:
                sys.path.insert(0, sts2_str)

            import yaml  # type: ignore
            from models import build_model  # type: ignore
            from utils import recursive_munch  # type: ignore
            from Munch import Munch  # type: ignore

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            model_params = recursive_munch(config["model_params"])
            model = build_model(model_params, text_aligner=None, pitch_extractor=None, bert=None)
            params = torch.load(str(checkpoint), map_location="cpu")
            model_state = params.get("net", params)

            for key, module in model.items():
                if key in model_state:
                    try:
                        module.load_state_dict(model_state[key])
                    except Exception:
                        pass
                module.eval().cuda()

            from diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule  # type: ignore
            sampler = DiffusionSampler(
                model.diffusion.diffusion,
                sampler=ADPM2Sampler(),
                sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                clamp=False,
            )

            self._model = model
            self._sampler = sampler
            self._loaded = True
            logger.info("StyleTTS2 loaded on GPU successfully")
            return True

        except Exception as exc:
            logger.debug("StyleTTS2 load() failed (%s: %s) — skipping", type(exc).__name__, exc)
            return False

    def _compute_style(self, emotion: str) -> "torch.Tensor":  # noqa: F821
        """Compute a style vector from the matching reference clip. Cached per emotion."""
        if emotion in self._style_cache:
            return self._style_cache[emotion]

        import torch
        import torchaudio  # type: ignore

        clip_name = _EMOTION_CLIP_MAP.get(emotion, _EMOTION_CLIP_MAP["NEUTRAL"])
        clip_path = EMOTION_CLIPS_DIR / clip_name

        if not clip_path.exists():
            clip_path = EMOTION_CLIPS_DIR / _EMOTION_CLIP_MAP["NEUTRAL"]

        if clip_path.exists():
            wav, sr = torchaudio.load(str(clip_path))
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            wav = wav.mean(0, keepdim=True).cuda()
        else:
            wav = torch.zeros(1, SAMPLE_RATE * 5).cuda()

        with torch.no_grad():
            ref_s = self._model.style_encoder(wav.unsqueeze(1))

        self._style_cache[emotion] = ref_s
        return ref_s

    def synthesize_chunk(self, text: str, emotion: str = "NEUTRAL") -> bytes:
        """
        Synthesise text with StyleTTS2 and return raw PCM int16 bytes at SAMPLE_RATE Hz.

        emotion: one of NEUTRAL / INTENSE / COLD / EXPRESSIVE
        On any error returns 200 ms of silence rather than raising.
        """
        if not self._loaded or self._model is None:
            return _silence_bytes(200)

        try:
            import torch
            from text_utils import TextCleaner  # type: ignore
            import phonemizer  # type: ignore

            text = text.strip()
            if not text:
                return _silence_bytes(100)

            style_vec = self._compute_style(emotion.upper())

            cleaner = TextCleaner()
            tokens = cleaner.text_to_sequence(text)
            tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()
            input_lengths = torch.LongTensor([tokens.shape[-1]]).cuda()

            with torch.no_grad():
                pred_dur = self._model.length_regulator(tokens, input_lengths)
                wav = self._sampler(
                    style_vec,
                    pred_dur,
                    input_lengths,
                    diffusion_steps=5,
                    embedding_scale=1.0,
                )

            wav_np = wav.squeeze().cpu().numpy()
            max_val = np.abs(wav_np).max()
            if max_val > 0:
                wav_np = wav_np / max_val
            pcm = (wav_np * 32767).astype(np.int16)

            if len(pcm) < 100:
                return _silence_bytes(200)

            return pcm.tobytes()

        except Exception as exc:
            logger.error("synthesize_chunk error: %s", exc)
            return _silence_bytes(200)


def _silence_bytes(ms: int) -> bytes:
    """Return ms milliseconds of silence as int16 PCM bytes."""
    samples = int(SAMPLE_RATE * ms / 1000)
    return np.zeros(samples, dtype=np.int16).tobytes()
