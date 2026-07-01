"""
TTS engine router with Piper as CPU fallback.

Engine priority at startup:
  1. StyleTTS2  (GPU, emotion-aware — requires CUDA + cloned repo)
  2. XTTS v2    (CPU, voice cloning — requires TTS==0.22.0)
  3. Piper      (CPU fallback — always available)

Windows note: select.select() does not work on Windows pipes, so a daemon
thread continuously drains stdout into a queue.Queue instead.
"""

import logging
import queue
import subprocess
import threading

from src.config import PIPER_DIR, PIPER_EXE, VOICE_MODEL_PATH
from src.extraction.extractor import DocReaderError

logger = logging.getLogger(__name__)


class TTSEngine:
    """Piper TTS subprocess wrapper with background PCM reader thread."""

    def __init__(self):
        self._process   = None
        self._pcm_queue = queue.Queue()
        self._lock      = threading.Lock()
        self._reader    = None

    def load(self) -> None:
        """Spawn one persistent piper.exe process. The model loads once here."""
        if not PIPER_EXE.exists():
            raise DocReaderError(
                "Piper executable not found. Please run scripts/setup.py first."
            )
        if not VOICE_MODEL_PATH.exists():
            raise DocReaderError(
                "Voice model not found. Please run scripts/setup.py first."
            )

        cmd = [
            str(PIPER_EXE),
            "--model", str(VOICE_MODEL_PATH),
            "--output-raw",
            "--quiet",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(PIPER_DIR),
        )

        self._pcm_queue = queue.Queue()
        self._reader = threading.Thread(
            target=self._reader_thread, daemon=True, name="piper-reader"
        )
        self._reader.start()

    def _reader_thread(self) -> None:
        """Drain piper stdout into _pcm_queue continuously."""
        try:
            while True:
                chunk = self._process.stdout.read1(65536)
                if not chunk:
                    break
                self._pcm_queue.put(chunk)
        except Exception:
            pass

    def synthesize_chunk(self, text: str) -> bytes:
        """
        Write one text chunk to the running piper process and return the
        raw 16-bit mono PCM bytes at 22050 Hz.
        """
        if self._process is None or self._process.poll() is not None:
            raise DocReaderError("Piper process crashed. Restart the app.")

        with self._lock:
            self._process.stdin.write(text.encode("utf-8") + b"\n")
            self._process.stdin.flush()

            parts = []
            empty_streak = 0
            while empty_streak < 2:
                timeout = 10.0 if not parts else 0.5
                try:
                    parts.append(self._pcm_queue.get(timeout=timeout))
                    empty_streak = 0
                except queue.Empty:
                    if not parts:
                        break
                    empty_streak += 1

        return b"".join(parts)

    def shutdown(self) -> None:
        """Gracefully stop the piper process."""
        if self._process is None:
            return
        try:
            self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None


# ---------------------------------------------------------------------------
# Module-level singletons — router selects engine at startup
# ---------------------------------------------------------------------------

_piper         = TTSEngine()
_engine        = _piper
_xtts          = None
_styletts2     = None
_use_xtts      = False
_use_styletts2 = False

_engine_name: str = "Piper"


def get_engine_name() -> str:
    """Return the name of the currently active TTS engine."""
    return _engine_name


def shutdown_engine() -> None:
    """Shut down the Piper subprocess on application exit."""
    _piper.shutdown()


def load_engine() -> None:
    """Try StyleTTS2 (GPU) → XTTS v2 → Piper in priority order."""
    global _styletts2, _use_styletts2, _xtts, _use_xtts, _engine_name

    # --- Attempt 1: StyleTTS2 (GPU) ---
    try:
        from src.tts.styletts2_engine import StyleTTS2Engine
        eng = StyleTTS2Engine()
        if eng.load():
            _styletts2 = eng
            _use_styletts2 = True
            _engine_name = "StyleTTS2"
            logger.info("StyleTTS2 active — GPU emotion-aware synthesis enabled")
            return
    except Exception as exc:
        logger.warning("StyleTTS2 not available (%s): %s", type(exc).__name__, exc)

    # --- Attempt 2: XTTS v2 ---
    try:
        from src.tts.xtts_engine import XTTSEngine
        eng = XTTSEngine()
        eng.load()
        _xtts = eng
        _use_xtts = True
        _engine_name = "XTTS"
        logger.info("XTTS v2 active — voice cloning enabled")
        return
    except Exception as exc:
        import traceback
        logger.warning("XTTS v2 unavailable (%s): %s", type(exc).__name__, exc)
        logger.debug(traceback.format_exc())

    # --- Fallback: Piper ---
    _piper.load()
    _use_xtts = False
    _use_styletts2 = False
    _engine_name = "Piper"
    logger.info("Piper active (fallback) — install TTS==0.22.0 for voice cloning")


def synthesize_chunk(text: str, reference_clip: str = None, emotion: str = "NEUTRAL") -> bytes:
    """Route synthesis to the active engine."""
    if _use_styletts2 and _styletts2 is not None:
        return _styletts2.synthesize_chunk(text, emotion=emotion)
    if _use_xtts and _xtts is not None:
        return _xtts.synthesize_chunk(text, reference_clip)
    return _piper.synthesize_chunk(text)
