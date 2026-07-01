"""
Full-chapter Ollama performance script generator.

Pass 1 of the two-pass pipeline:
  All chapters → Ollama (llama3.2:3b) → .txt scripts cached to disk

Scripts are cached so Pass 2 (TTS synthesis) can run independently.
If Pass 2 crashes, restart and Pass 1 output is fully preserved.
"""

import logging
import os

import requests

from src.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from src.narration.book_profiler import load_or_build_profile, generate_narrator_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional audiobook narrator director.
Your job is to rewrite the provided chapter as a complete
narrator performance script.

Rules:
1. Preserve every plot event, every character name, every
   piece of dialogue — do NOT summarize or skip content.
2. Write in narrator voice. Add natural transitions between
   scenes ("Meanwhile...", "Hours later...", "In the shadows...").
3. Insert performance cues inline using ONLY these markers:
     [breath]  — natural breath pause (after commas, between thoughts)
     [pause]   — dramatic silence (before reveals, after shocks)
     *word*    — stress this word (key names, important concepts)
     [slow]    — slow delivery (gravitas, tragedy, weight)
     [fast]    — speed up (action, urgency, chase)
4. Maximum 10 cue insertions per paragraph.
5. For dialogue: add speaker context if not clear.
   e.g. "he said coldly," becomes "[slow] Fang Yuan's voice was ice."
6. Genre context: this is a dark cultivation fantasy (Reverend Insanity).
   The tone is calculating, cold, ambitious. The protagonist is ruthless.
   Reflect that in the narration style.
7. Return ONLY the performance script. No explanations. No preamble.\
"""


def _call_ollama(text: str, system_prompt: str = None) -> str:
    """Single Ollama call with optional custom system prompt. Returns response text."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    payload = {
        "model":  OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": text,
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def write_chapter_script(chapter_num: int, title: str, body: str,
                         narrator_prompt: str = None) -> str:
    """
    Send full chapter body to Ollama and return the performance script.

    If body > 6000 chars: splits into two halves, calls Ollama twice, joins results.
    narrator_prompt: optional custom system prompt (from book_profiler).
    Returns original body unchanged on any error. Never raises.
    """
    if not body.strip():
        return body

    try:
        if len(body) <= 6000:
            result = _call_ollama(body, system_prompt=narrator_prompt)
        else:
            mid   = len(body) // 2
            split = body.rfind(". ", mid - 200, mid + 200)
            if split == -1:
                split = mid
            else:
                split += 2
            part_a = _call_ollama(body[:split], system_prompt=narrator_prompt)
            part_b = _call_ollama(body[split:], system_prompt=narrator_prompt)
            result = part_a.rstrip() + "\n\n" + part_b.lstrip()

        if len(result) < len(body) * 0.6:
            logger.warning(
                "Ch%d: response too short (%d vs %d) — using original",
                chapter_num, len(result), len(body)
            )
            return body

        return result

    except Exception as exc:
        logger.warning("Ch%d: Ollama error (%s: %s) — using original", chapter_num, type(exc).__name__, exc)
        return body


def write_all_scripts(chapters: list, scripts_dir: str,
                      progress_callback=None) -> dict:
    """
    Pass 1: rewrite all chapters as performance scripts and cache to disk.

    chapters:          list of (num, title, body) tuples
    scripts_dir:       directory to save Chapter_NNNN_script.txt files
    progress_callback: optional callback(chapter_num, total)

    Skips chapters whose script file already exists (resume support).
    Returns {chapter_num: script_text}.
    """
    os.makedirs(scripts_dir, exist_ok=True)
    total   = len(chapters)
    scripts = {}

    for idx, (num, title, body) in enumerate(chapters, 1):
        script_path = os.path.join(scripts_dir, f"Chapter_{num:04d}_script.txt")

        if os.path.isfile(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                scripts[num] = f.read()
            logger.debug("Ch%d: loaded from cache (skipping Ollama)", num)
        else:
            logger.info("Ch%d/%d: writing script for '%s'...", num, total, title)
            script = write_chapter_script(num, title, body)
            scripts[num] = script
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

        if progress_callback:
            progress_callback(idx, total)

    return scripts


def load_cached_scripts(chapters: list, scripts_dir: str) -> dict:
    """
    Load all cached scripts from disk without calling Ollama.
    Falls back to original chapter body if script file is missing.
    Returns {chapter_num: script_text}.
    """
    scripts = {}
    for num, title, body in chapters:
        script_path = os.path.join(scripts_dir, f"Chapter_{num:04d}_script.txt")
        if os.path.isfile(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                scripts[num] = f.read()
        else:
            logger.debug("Ch%d: no cached script — using original body", num)
            scripts[num] = body
    return scripts


def write_all_scripts_with_profile(chapters: list, scripts_dir: str,
                                   book_title: str, first_chapter_body: str,
                                   progress_callback=None) -> dict:
    """
    Write all scripts using a book-specific custom narrator prompt.

    1. Load or build the book profile via book_profiler.
    2. Generate a custom narrator prompt from the profile.
    3. Call write_all_scripts() with that prompt for each chapter.

    Returns {chapter_num: script_text}.
    """
    logger.info("Building/loading profile for '%s'...", book_title)

    try:
        profile = load_or_build_profile(book_title, first_chapter_body)
        narrator_prompt = generate_narrator_prompt(profile)

        genres   = ", ".join(profile.get("genres", [])[:3]) or "unknown"
        tone     = ", ".join(profile.get("tone", [])[:2]) or "neutral"
        register = profile.get("emotional_register", "neutral")
        logger.info("Profile: %s — %s — narrator: %s", genres, tone, register)

    except Exception as exc:
        logger.warning("Error building profile: %s — using generic prompt", exc)
        narrator_prompt = None

    os.makedirs(scripts_dir, exist_ok=True)
    total   = len(chapters)
    scripts = {}

    for idx, (num, title, body) in enumerate(chapters, 1):
        script_path = os.path.join(scripts_dir, f"Chapter_{num:04d}_script.txt")

        if os.path.isfile(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                scripts[num] = f.read()
            logger.debug("Ch%d: loaded from cache (skipping Ollama)", num)
        else:
            logger.info("Ch%d/%d: writing script for '%s'...", num, total, title)
            script = write_chapter_script(num, title, body, narrator_prompt=narrator_prompt)
            scripts[num] = script
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

        if progress_callback:
            progress_callback(idx, total)

    return scripts
