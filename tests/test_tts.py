"""
Tests for src/tts — engine routing and Piper wrapper.
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Engine name
# ---------------------------------------------------------------------------

def test_get_engine_name_returns_string():
    from src.tts.tts_engine import get_engine_name
    name = get_engine_name()
    assert isinstance(name, str)
    assert name in ("Piper", "XTTS", "StyleTTS2")


# ---------------------------------------------------------------------------
# TTSEngine (Piper wrapper)
# ---------------------------------------------------------------------------

class TestTTSEngine:
    def test_shutdown_noop_when_not_started(self):
        """shutdown() must not raise when no process was started."""
        from src.tts.tts_engine import TTSEngine
        engine = TTSEngine()
        engine.shutdown()  # should not raise

    def test_synthesize_chunk_raises_when_not_loaded(self):
        from src.tts.tts_engine import TTSEngine, DocReaderError
        engine = TTSEngine()
        with pytest.raises(DocReaderError):
            engine.synthesize_chunk("hello")

    def test_load_raises_docreader_error_when_piper_missing(self):
        """load() must raise DocReaderError (not a raw FileNotFoundError)."""
        from src.tts.tts_engine import TTSEngine, DocReaderError
        engine = TTSEngine()
        with patch("src.tts.tts_engine.PIPER_EXE") as mock_exe:
            mock_exe.exists.return_value = False
            with pytest.raises(DocReaderError, match="Piper executable"):
                engine.load()


# ---------------------------------------------------------------------------
# XTTS split_for_xtts
# ---------------------------------------------------------------------------

class TestXTTSEngine:
    def test_split_short_text_unchanged(self):
        from src.tts.xtts_engine import XTTSEngine
        engine = XTTSEngine()
        text = "Hello world."
        parts = engine._split_for_xtts(text, max_chars=200)
        assert parts == ["Hello world."]

    def test_split_long_text_into_multiple_parts(self):
        from src.tts.xtts_engine import XTTSEngine
        engine = XTTSEngine()
        long_text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100 + "."
        parts = engine._split_for_xtts(long_text, max_chars=120)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 120

    def test_no_empty_parts(self):
        from src.tts.xtts_engine import XTTSEngine
        engine = XTTSEngine()
        parts = engine._split_for_xtts("Hello. World.", max_chars=200)
        assert all(p.strip() != "" for p in parts)


# ---------------------------------------------------------------------------
# synthesize_chunk routing
# ---------------------------------------------------------------------------

def test_synthesize_chunk_routes_to_piper_by_default():
    """When no GPU engine is loaded, synthesize_chunk calls Piper."""
    from src.tts import tts_engine

    mock_bytes = b"\x00" * 100
    with patch.object(tts_engine._piper, "synthesize_chunk", return_value=mock_bytes) as mock_synth:
        # Force Piper path
        original_xtts = tts_engine._use_xtts
        original_stt  = tts_engine._use_styletts2
        tts_engine._use_xtts = False
        tts_engine._use_styletts2 = False
        try:
            result = tts_engine.synthesize_chunk("test text")
            mock_synth.assert_called_once_with("test text")
            assert result == mock_bytes
        finally:
            tts_engine._use_xtts = original_xtts
            tts_engine._use_styletts2 = original_stt
