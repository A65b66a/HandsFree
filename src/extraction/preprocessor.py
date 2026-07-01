"""
Text cleaning and sentence-boundary chunking before TTS synthesis.
"""

import re
import unicodedata

from src.config import MAX_CHARS, MAX_TOTAL_CHARS

__all__ = ["clean_text", "chunk_text", "MAX_CHARS", "MAX_TOTAL_CHARS"]


def clean_text(text: str) -> str:
    """Normalize and clean text for TTS input."""
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove non-printable / control characters (keep \t, \n, \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Replace TTS-hostile special characters
    replacements = [
        ("—", ", "),   # em dash → comma
        ("–", "-"),    # en dash → hyphen
        ("“", '"'),    # left double quote
        ("”", '"'),    # right double quote
        ("‘", "'"),    # left single quote
        ("’", "'"),    # right single quote
        ("…", "..."),  # ellipsis
        (" ", " "),    # non-breaking space
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    # 4. Collapse 3+ blank lines into a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Collapse runs of spaces/tabs within a line
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list:
    """
    Split text into chunks of at most max_chars, breaking only at sentence
    boundaries. Returns a list of non-empty chunk strings.
    """
    paragraphs = text.split("\n\n")
    sentences = []
    sentence_pattern = re.compile(
        r"(?<=[.!?。！？])\s+(?=[A-Z一-鿿Ѐ-ӿ])|"
        r"(?<=[.!?])\s*\n"
    )
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = sentence_pattern.split(para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    chunks = []
    current_parts = []
    current_len = 0

    for sentence in sentences:
        sentence_fragments = _hard_split(sentence, max_chars)

        for frag in sentence_fragments:
            frag_len = len(frag)
            separator = " " if current_parts else ""
            would_be = current_len + len(separator) + frag_len

            if current_parts and would_be > max_chars:
                chunks.append(" ".join(current_parts))
                current_parts = [frag]
                current_len = frag_len
            else:
                current_parts.append(frag)
                current_len += len(separator) + frag_len

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [c for c in chunks if c.strip()]


def _hard_split(text: str, max_chars: int) -> list:
    """Split a long sentence at word boundaries to fit within max_chars."""
    if len(text) <= max_chars:
        return [text]

    fragments = []
    while len(text) > max_chars:
        split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        fragments.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        fragments.append(text)

    return fragments
