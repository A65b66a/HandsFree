"""
pydub-based audio stitching: combine raw PCM chunks into a single WAV file.
"""

import os
import re
import tempfile

from pydub import AudioSegment

SAMPLE_RATE  = 24000   # XTTS v2 / StyleTTS2 native output rate
SAMPLE_WIDTH = 2       # bytes — 16-bit PCM
CHANNELS     = 1       # mono


def raw_pcm_to_segment(raw_bytes: bytes) -> AudioSegment:
    """Wrap raw PCM bytes in a pydub AudioSegment."""
    return AudioSegment(
        data=raw_bytes,
        sample_width=SAMPLE_WIDTH,
        frame_rate=SAMPLE_RATE,
        channels=CHANNELS,
    )


def normalize_audio(segment: AudioSegment,
                    target_rms_db: float = -15.0) -> AudioSegment:
    """
    Normalize segment to target RMS dB, then apply a -1.0 dB peak ceiling
    to prevent clipping. Skips silent segments (dBFS == -inf).
    """
    if segment.dBFS == float("-inf"):
        return segment
    gain       = target_rms_db - segment.dBFS
    normalized = segment.apply_gain(gain)
    peak       = normalized.max_dBFS
    if peak > -1.0:
        normalized = normalized.apply_gain(-1.0 - peak)
    return normalized


def concat_with_silence(
    segments: list,
    silence_ms: int = 80,   # 80ms matches natural narrator breath rhythm
) -> AudioSegment:
    """Concatenate AudioSegments with a fixed silence gap between each."""
    silence = AudioSegment.silent(duration=silence_ms, frame_rate=SAMPLE_RATE)
    result  = segments[0]
    for seg in segments[1:]:
        result = result + silence + seg
    return result


def concat_with_variable_silence(
    segments: list,
) -> AudioSegment:
    """
    Concatenate (AudioSegment, trailing_silence_ms) tuples.
    Silence is capped at 400ms to prevent dead air over 1 second.
    The last segment's trailing silence is ignored.
    """
    result = segments[0][0]
    for seg, silence_ms in segments[1:]:
        silence_ms = min(silence_ms, 400)
        if silence_ms > 0:
            sil    = AudioSegment.silent(duration=silence_ms, frame_rate=SAMPLE_RATE)
            result = result + sil
        result = result + seg
    return result


def export_wav(segment: AudioSegment) -> str:
    """Export an AudioSegment to a temporary WAV file. Returns the path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    segment.export(path, format="wav")
    return path


def export_wav_named(segment: AudioSegment, directory: str, filename: str) -> str:
    """Export an AudioSegment to a named WAV file in directory. Returns the path."""
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", filename)
    path      = os.path.join(directory, safe_name)
    segment.export(path, format="wav")
    return path
