"""
Maps emotion labels to reference WAV clip paths for XTTS v2 voice cloning.

Clips live in docreader/voices/reference/.  Only neutral.wav and epic.wav are
currently recorded.  All other emotions fall back to neutral.wav until their
clips are added.
"""

import logging
from pathlib import Path

from src.config import VOICES_REF_DIR
from src.tts.xtts_engine import _find_clip_any_format

logger = logging.getLogger(__name__)

EMOTION_TO_CLIP = {
    "cold"    : "neutral.wav",   # fallback until cold.wav recorded
    "tense"   : "neutral.wav",   # fallback until tense.wav recorded
    "grief"   : "neutral.wav",   # fallback until grief.wav recorded
    "contempt": "neutral.wav",   # fallback until contempt.wav recorded
    "epic"    : "epic.wav",      # available
    "rage"    : "neutral.wav",   # fallback until rage.wav recorded
    "tender"  : "neutral.wav",   # fallback until tender.wav recorded
    "neutral" : "neutral.wav",   # primary default
}


def get_reference_clip(emotion: str) -> str | None:
    """
    Return the full WAV path for the given emotion.
    Accepts any audio format — non-WAV files are auto-converted.
    Falls back to neutral clip if the mapped file is missing.
    Unknown emotion labels resolve to neutral.
    """
    stem = Path(EMOTION_TO_CLIP.get(emotion, "neutral.wav")).stem
    clip = _find_clip_any_format(stem)
    if clip:
        return str(clip)
    neutral = _find_clip_any_format("neutral")
    return str(neutral) if neutral else None


def validate_clips() -> dict:
    """
    Check all 8 emotions against their mapped clip files.
    Logs a startup summary and returns {emotion: True/False}.
    """
    results = {}
    for emotion, filename in EMOTION_TO_CLIP.items():
        stem = Path(filename).stem
        clip = _find_clip_any_format(stem)
        exists = clip is not None
        results[emotion] = exists
        is_own_clip = stem == emotion
        if is_own_clip and exists:
            logger.info("%-20s YOUR VOICE ACTIVE", clip.name)
        elif exists:
            logger.info("%-10s using %s fallback", f"{emotion}.wav", clip.name)
        else:
            logger.warning("%-10s clip missing, no fallback found", f"{emotion}.wav")
    return results
