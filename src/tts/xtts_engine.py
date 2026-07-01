"""
XTTS v2 wrapper — CPU voice cloning engine.

XTTS v2 is thread-safe for CPU inference, so no Lock() is needed.
Outputs float32 list at 24000 Hz; converted to int16 PCM bytes.
Returns b"" on any synthesis error — caller skips empty segments.
"""

import logging
import re
from pathlib import Path

import numpy as np

from src.config import VOICES_REF_DIR
from src.extraction.extractor import DocReaderError

logger = logging.getLogger(__name__)

_AUDIO_EXTS = (".ogg", ".mp3", ".flac", ".m4a")


def _ensure_wav(path: Path) -> Path:
    """
    If path is not a .wav, convert it to .wav using soundfile and return the
    new path. Result is cached alongside the original file.
    Returns the original path unchanged if it is already .wav or conversion fails.
    """
    if path.suffix.lower() == ".wav":
        return path
    wav_path = path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    try:
        import soundfile as sf
        data, samplerate = sf.read(str(path))
        sf.write(str(wav_path), data, samplerate)
        logger.info("Converted %s → %s", path.name, wav_path.name)
        return wav_path
    except Exception as exc:
        logger.warning("Could not convert %s to WAV: %s", path.name, exc)
        return path


def _find_clip_any_format(stem: str) -> Path | None:
    """
    Look for voices/reference/<stem>.wav first, then any supported audio format.
    Returns a .wav Path (converting if needed) or None.
    """
    wav = VOICES_REF_DIR / f"{stem}.wav"
    if wav.exists():
        return wav
    for ext in _AUDIO_EXTS:
        alt = VOICES_REF_DIR / f"{stem}{ext}"
        if alt.exists():
            return _ensure_wav(alt)
    return None


class XTTSEngine:
    def __init__(self):
        self.model = None
        self.reference_clip = None

    def load(self) -> None:
        """Load XTTS v2 to CPU and run a warm-up pass."""
        import os
        import torch
        import torchaudio

        os.environ["COQUI_TOS_AGREED"] = "1"
        espeak_dir = r"C:\Program Files\eSpeak NG"
        if os.path.isdir(espeak_dir):
            os.add_dll_directory(espeak_dir)

        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

        import warnings
        warnings.filterwarnings("ignore", message=".*set_audio_backend.*")
        try:
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            pass

        from TTS.api import TTS
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.model.to("cpu")
        self.reference_clip = self._find_reference_clip()
        clip = self.reference_clip
        if clip:
            self.model.tts(text="warmup", language="en", speaker_wav=clip)
        else:
            self.model.tts(text="warmup", language="en", speaker="Ana Florence")
        logger.info("XTTS v2 loaded — voice cloning active")

    def _split_for_xtts(self, text: str, max_chars: int = 200) -> list:
        """
        Split text into segments under max_chars.
        XTTS crashes on segments over ~250 chars.
        Splits on sentence boundaries first, then commas, then hard-cuts.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result = []
        for sent in sentences:
            if len(sent) <= max_chars:
                result.append(sent)
            else:
                parts = re.split(r',\s*', sent)
                buf = ""
                for part in parts:
                    if len(buf) + len(part) + 2 <= max_chars:
                        buf = (buf + ", " + part).strip(", ")
                    else:
                        if buf:
                            result.append(buf)
                        while len(part) > max_chars:
                            result.append(part[:max_chars])
                            part = part[max_chars:]
                        buf = part
                if buf:
                    result.append(buf)
        return [s.strip() for s in result if s.strip()]

    def synthesize_chunk(self, text: str, reference_clip: str = None) -> bytes:
        """
        Synthesize text and return raw int16 PCM bytes at 24000 Hz.
        Splits every chunk into ≤200-char units to prevent XTTS v2's internal
        index-out-of-range crash on long sentences.
        Returns b"" on any error.
        """
        if self.model is None:
            raise DocReaderError("XTTS v2 not loaded.")
        clip = reference_clip or self.reference_clip
        use_clip = clip and Path(clip).exists()
        kw = {"language": "en"}
        if use_clip:
            kw["speaker_wav"] = clip
        else:
            kw["speaker"] = "Ana Florence"

        # 200ms of silence at 24000 Hz mono int16
        _silence_200ms = b"\x00" * (24000 * 2 // 5)

        units = self._split_for_xtts(text, max_chars=200)
        raw = b""
        for unit in units:
            try:
                wav = self.model.tts(text=unit, **kw)
                arr = np.array(wav, dtype=np.float32)
                piece = (np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes()
                if len(piece) < 1000:
                    logger.warning(
                        "Short output (%d bytes) for unit '%s...' — inserting silence",
                        len(piece), unit[:40]
                    )
                    raw += _silence_200ms
                else:
                    raw += piece
            except Exception as exc:
                logger.error("Synthesis error on unit (%d chars): %s", len(unit), exc)
                raw += _silence_200ms
        if len(raw) < 1000:
            logger.warning("Empty output for chunk (%d chars) — inserting silence", len(text))
            import numpy as _np
            return _np.zeros(int(24000 * 0.2), dtype=_np.int16).tobytes()
        return raw

    def _find_reference_clip(self) -> str | None:
        """Return path to neutral voice clip, falling back to any clip in reference/."""
        clip = _find_clip_any_format("neutral")
        if clip:
            logger.info("Default voice: %s (your recorded voice)", clip.name)
            return str(clip)
        for ext in (".wav",) + _AUDIO_EXTS:
            found = list(VOICES_REF_DIR.glob(f"*{ext}"))
            if found:
                converted = _ensure_wav(found[0])
                logger.info("Default voice: %s", converted.name)
                return str(converted)
        logger.info("No reference clips found — using Ana Florence built-in voice")
        return None


# ---------------------------------------------------------------------------
# Module-level singleton — mirrors tts_engine.py public API
# ---------------------------------------------------------------------------

_engine = XTTSEngine()


def load_engine() -> None:
    _engine.load()


def synthesize_chunk(text: str, reference_clip: str = None) -> bytes:
    return _engine.synthesize_chunk(text, reference_clip)
