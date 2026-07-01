"""
Tests for src/extraction — extractor and preprocessor.
"""

import pytest
from src.extraction.extractor import split_chapters, DocReaderError, SUPPORTED_EXTENSIONS
from src.extraction.preprocessor import clean_text, chunk_text, MAX_CHARS, MAX_TOTAL_CHARS


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_removes_em_dash(self):
        result = clean_text("word—another")
        assert "—" not in result
        assert ", " in result

    def test_removes_control_chars(self):
        result = clean_text("hello\x00world\x1f!")
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "hello" in result

    def test_normalizes_unicode(self):
        # NFKC should normalize ligatures and compatibility chars
        result = clean_text("ﬁle")   # ﬁ is a ligature (U+FB01)
        assert result == "file"

    def test_collapses_multiple_blank_lines(self):
        result = clean_text("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_collapses_spaces(self):
        result = clean_text("a     b")
        assert "  " not in result


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_returns_one_chunk(self):
        chunks = chunk_text("Hello world.", max_chars=500)
        assert len(chunks) == 1

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", max_chars=500)
        assert chunks == []

    def test_chunks_respect_max_chars(self):
        long_sentence = "A" * 50 + ". "
        text = long_sentence * 30
        chunks = chunk_text(text, max_chars=200)
        for chunk in chunks:
            assert len(chunk) <= 210  # small buffer for joining

    def test_no_empty_chunks(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_hard_split_very_long_word(self):
        # A word longer than max_chars should still be split
        text = "A" * 600
        chunks = chunk_text(text, max_chars=200)
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# split_chapters
# ---------------------------------------------------------------------------

class TestSplitChapters:
    def test_detects_two_chapters(self):
        body1 = "In the beginning there was darkness and silence and nothing moved at all."
        body2 = "More body text here that is long enough to count as a chapter body indeed."
        text = (
            f"Chapter 1: The Beginning\n{body1}\n\n"
            f"Chapter 2: The Middle\n{body2}"
        )
        chapters = split_chapters(text)
        assert len(chapters) == 2
        assert chapters[0][0] == 1
        assert chapters[0][1] == "The Beginning"
        assert chapters[1][0] == 2

    def test_single_chapter_returns_empty(self):
        text = "Chapter 1: Only One\nSome body text."
        chapters = split_chapters(text)
        assert chapters == []

    def test_strips_short_bodies(self):
        text = (
            "Chapter 1: Short\nToo short.\n\n"
            "Chapter 2: Long\n" + "Real content. " * 20
        )
        chapters = split_chapters(text)
        # Chapter 1 body is too short (<10 words) — should be excluded
        assert all(len(ch[2].split()) >= 10 for ch in chapters)

    def test_case_insensitive(self):
        text = (
            "CHAPTER 1: Upper\n" + "Content here. " * 15 + "\n\n"
            "chapter 2: lower\n" + "Content here. " * 15
        )
        chapters = split_chapters(text)
        assert len(chapters) == 2


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS
# ---------------------------------------------------------------------------

def test_supported_extensions_set():
    assert ".epub" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".xyz" not in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# MAX_CHARS / MAX_TOTAL_CHARS sanity
# ---------------------------------------------------------------------------

def test_constants_reasonable():
    assert MAX_CHARS == 500
    assert MAX_TOTAL_CHARS == 50_000
    assert MAX_TOTAL_CHARS > MAX_CHARS
