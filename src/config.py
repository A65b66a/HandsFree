"""
Centralized configuration: runtime paths, Ollama settings, and application limits.

Runtime assets (Piper binary, voice models, book profiles, library data) live in
data/ at the project root. This directory is gitignored — populate it via
scripts/setup.py on first run.
"""

from pathlib import Path

# ── Project layout ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent          # d:\book_reader\
DATA_ROOT    = PROJECT_ROOT / "data"                 # runtime assets live here

# ── TTS paths ─────────────────────────────────────────────────────────────────

PIPER_DIR        = DATA_ROOT / "piper"
PIPER_EXE        = PIPER_DIR / "piper.exe"
VOICE_MODEL_PATH = DATA_ROOT / "voices" / "en_US-lessac-medium.onnx"
VOICES_REF_DIR   = DATA_ROOT / "voices" / "reference"
EMOTION_CLIPS_DIR = DATA_ROOT / "voices" / "emotion_clips"
STYLETTS2_DIR    = DATA_ROOT / "StyleTTS2"

# ── Storage paths ─────────────────────────────────────────────────────────────

PROFILES_DIR = DATA_ROOT / "book_profiles"
LIBRARY_DIR  = DATA_ROOT / "library_data"

# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3.2:3b"
OLLAMA_TIMEOUT = 120  # seconds — chapters can be long

# ── Application limits ────────────────────────────────────────────────────────

FILE_SIZE_LIMIT = 50 * 1024 * 1024  # 50 MB — covers large EPUBs
MAX_CHARS       = 500               # max chars per TTS chunk
MAX_TOTAL_CHARS = 50_000            # max chars for single-file conversion
