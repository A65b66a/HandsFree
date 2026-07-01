"""
Emotion tagger: Groq sentence-level emotion labels + Ollama chunk rewriter (legacy).

Legacy API:  tag_chunks(chunks)                    → Ollama / Pre-generated path
New API:     tag_sentence_emotions(sentences, key) → StyleTTS2 path

tag_sentence_emotions returns one label per sentence:
  NEUTRAL | INTENSE | COLD | EXPRESSIVE
Falls back to all-NEUTRAL on any Groq error.
"""

import json
import logging
import re

import requests

from src.narration.script_writer import write_chapter_script

logger = logging.getLogger(__name__)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL   = "llama-3.3-70b-versatile"
_VALID_LABELS = {"NEUTRAL", "INTENSE", "COLD", "EXPRESSIVE"}
_BATCH_SIZE   = 40  # sentences per Groq call (stays well within token limits)


def _split_into_n_chunks(text: str, n: int) -> list:
    """
    Split text into exactly n chunks by sentence boundaries.
    Falls back to character-count distribution if fewer sentence breaks exist.
    """
    if n <= 1:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= n:
        return sentences + [""] * (n - len(sentences))

    per_bucket = len(sentences) / n
    chunks     = []
    buf        = []
    threshold  = per_bucket

    for i, sent in enumerate(sentences):
        buf.append(sent)
        if i + 1 >= threshold and len(chunks) < n - 1:
            chunks.append(" ".join(buf))
            buf       = []
            threshold += per_bucket

    if buf:
        chunks.append(" ".join(buf))

    while len(chunks) < n:
        chunks.append("")
    return chunks[:n]


def tag_sentence_emotions(sentences: list, groq_api_key: str) -> list:
    """
    Call Groq llama-3.3-70b-versatile to label each sentence with one of:
      NEUTRAL | INTENSE | COLD | EXPRESSIVE

    Returns a list of labels with the same length as `sentences`.
    Falls back to all-NEUTRAL on any error.
    """
    if not sentences:
        return []
    if not groq_api_key:
        return ["NEUTRAL"] * len(sentences)

    labels: list = []

    for batch_start in range(0, len(sentences), _BATCH_SIZE):
        batch = sentences[batch_start : batch_start + _BATCH_SIZE]

        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
        prompt = (
            "Label each numbered sentence below with exactly one emotion tag.\n"
            "Valid tags: NEUTRAL, INTENSE, COLD, EXPRESSIVE\n\n"
            "NEUTRAL    — calm narration, descriptions, ordinary dialogue\n"
            "INTENSE    — combat, urgency, high stakes, threat, rage\n"
            "COLD       — scheming, calculation, detached internal thought\n"
            "EXPRESSIVE — warmth, humour, surprise, grief, wonder\n\n"
            "Return ONLY a JSON array of strings, one per sentence, in order.\n"
            "Example for 3 sentences: [\"NEUTRAL\", \"INTENSE\", \"COLD\"]\n\n"
            f"{numbered}"
        )

        try:
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": _GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 256,
            }
            resp = requests.post(_GROQ_API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()

            content = resp.json()["choices"][0]["message"]["content"].strip()

            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array in response")

            raw_labels = json.loads(json_match.group(0))

            batch_labels = []
            for lbl in raw_labels:
                upper = str(lbl).upper().strip()
                batch_labels.append(upper if upper in _VALID_LABELS else "NEUTRAL")

            while len(batch_labels) < len(batch):
                batch_labels.append("NEUTRAL")
            labels.extend(batch_labels[: len(batch)])

        except Exception as exc:
            logger.warning("Groq call failed: %s — defaulting to NEUTRAL", exc)
            labels.extend(["NEUTRAL"] * len(batch))

    return labels


def tag_chunks(chunks: list) -> list:
    """
    Legacy interface: join chunks, rewrite via Ollama, split back to same count.

    Falls back to original chunks unchanged on any Ollama failure.
    """
    if not chunks:
        return chunks

    full_text = " ".join(c.strip() for c in chunks if c.strip())
    if not full_text:
        return chunks

    script = write_chapter_script(0, "chapter", full_text)

    if script == full_text or not script.strip():
        return chunks

    return _split_into_n_chunks(script, len(chunks))
