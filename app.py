"""
ai-book-narrator — Document to Audiobook
Gradio desktop app: upload a document, extract text, convert to narrated WAV.

Setup (run once):
    pip install -e .                       # installs package + dependencies
    python scripts/setup.py               # downloads piper.exe + voice model

Launch:
    python app.py
    Then open http://localhost:7860
"""

import atexit
import logging
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before any module import so GROQ_API_KEY is visible everywhere.
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    candidates = [
        Path(__file__).parent / ".env",
    ]
    for _env_path in candidates:
        if _env_path.exists():
            for _line in _env_path.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
            break

_load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Monkey-patch gradio-client 1.3.0 bug BEFORE anything else uses it.
#
# gradio/routes.py calls api_info() which calls
# gradio_client.utils._json_schema_to_python_type(). When a component schema
# has  additionalProperties: false  (a Python bool), the function crashes with
# TypeError: argument of type 'bool' is not iterable.
# Fix: guard the inner function so boolean schemas return "Any".
# ---------------------------------------------------------------------------
def _apply_gradio_client_patch() -> None:
    try:
        import gradio_client.utils as _gcu
        _original = _gcu._json_schema_to_python_type

        def _safe(schema, defs=None):
            if not isinstance(schema, dict):
                return "Any"
            return _original(schema, defs)

        _gcu._json_schema_to_python_type = _safe
    except Exception:
        pass


_apply_gradio_client_patch()

# ---------------------------------------------------------------------------
# Logging setup — configure before importing src modules so their loggers
# are captured from the start.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Application imports
# ---------------------------------------------------------------------------
import gradio as gr

from src.config import FILE_SIZE_LIMIT, MAX_TOTAL_CHARS
from src.extraction.extractor import (
    DocReaderError,
    extract_text,
    split_chapters,
    extract_epub_chapters,
)
from src.extraction.preprocessor import clean_text, chunk_text
from src.tts.tts_engine import (
    load_engine,
    synthesize_chunk,
    get_engine_name,
    shutdown_engine,
)
from src.audio_utils import (
    raw_pcm_to_segment,
    concat_with_silence,
    export_wav,
    export_wav_named,
)
from src.library import get_library_manager
from src.narration.emotion_tagger import tag_chunks, tag_sentence_emotions
from src.narration.style_director import get_reference_clip, validate_clips
from src.narration.book_profiler import load_or_build_profile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(chapter_num: int, title: str) -> str:
    """Build a zero-padded filename like 'Chapter_03_The_Dark_Forest.wav'."""
    slug = title.replace(" ", "_")[:60]
    return f"Chapter_{chapter_num:02d}_{slug}.wav"


def _chapter_label(num: int, title: str) -> str:
    return f"Chapter {num}: {title}"


def _synthesize_text(text: str, emotion_mode: bool = False, groq_key: str = "") -> list:
    """
    Clean → chunk → (optionally tag emotions) → synthesize sequentially.
    Returns list of AudioSegments.

    When StyleTTS2 is active and groq_key is provided, uses Groq for
    sentence-level emotion tagging and synthesises per sentence.
    Otherwise falls back to Ollama chunk-level rewriting.
    """
    cleaned = clean_text(text)

    # StyleTTS2 + Groq: sentence-level emotion synthesis
    if emotion_mode and groq_key and get_engine_name() == "StyleTTS2":
        sentences = re.split(r'(?<=[.!?])\s+', cleaned.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        emotion_labels = tag_sentence_emotions(sentences, groq_key)
        segments = []
        for sentence, label in zip(sentences, emotion_labels):
            raw = synthesize_chunk(sentence, reference_clip=None, emotion=label)
            if raw:
                segments.append(raw_pcm_to_segment(raw))
        return segments

    # Standard / Ollama chunk path
    chunks = chunk_text(cleaned, max_chars=1000)
    if not chunks:
        return []

    if emotion_mode:
        emotion_labels = tag_chunks(chunks)
    else:
        emotion_labels = ["neutral"] * len(chunks)

    segments = []
    for chunk, emotion in zip(chunks, emotion_labels):
        ref_clip = get_reference_clip(emotion) if emotion_mode else None
        raw = synthesize_chunk(chunk, reference_clip=ref_clip)
        if raw:
            segments.append(raw_pcm_to_segment(raw))
    return segments


# ---------------------------------------------------------------------------
# Library UI handlers
# ---------------------------------------------------------------------------

def refresh_library_view():
    """Generate a formatted list of all books in the library."""
    lib_mgr = get_library_manager()
    books = lib_mgr.get_all_books()

    if not books:
        return gr.update(
            value="**No books in library yet.** Convert a document to see it here."
        )

    library_text = "## Converted Books\n\n"

    for i, book in enumerate(books, 1):
        book_id  = book["book_id"]
        title    = book["title"]
        chapters_count = book["chapters"]
        created  = book["created"][:10]

        library_text += f"### {i}. {title}\n"
        library_text += f"**Created:** {created} | **Chapters:** {chapters_count}\n\n"

        book_data = lib_mgr.get_book(book_id)
        if book_data:
            for chapter in book_data["chapters"]:
                ch_num   = chapter["num"]
                ch_title = chapter["title"]
                duration = chapter["duration"]
                library_text += f"  - **Chapter {ch_num}:** {ch_title} ({duration})\n"

        library_text += "\n"

    return gr.update(value=library_text)


# ---------------------------------------------------------------------------
# Chapter navigation handlers
# ---------------------------------------------------------------------------

def on_chapter_select(chapter_label, chapter_list):
    """Load the selected chapter into the audio player."""
    if not chapter_list or not chapter_label:
        return gr.update(), 0
    choices = [_chapter_label(n, t) for n, t, _ in chapter_list]
    try:
        idx = choices.index(chapter_label)
        _, _, wav_path = chapter_list[idx]
        return gr.update(value=wav_path), idx
    except (ValueError, IndexError):
        return gr.update(), 0


def on_prev_chapter(chapter_list, chapter_idx):
    """Move to the previous chapter."""
    if not chapter_list or chapter_idx <= 0:
        return gr.update(), gr.update(), chapter_idx
    new_idx = chapter_idx - 1
    num, title, wav_path = chapter_list[new_idx]
    return gr.update(value=wav_path), gr.update(value=_chapter_label(num, title)), new_idx


def on_next_chapter(chapter_list, chapter_idx):
    """Move to the next chapter."""
    if not chapter_list or chapter_idx >= len(chapter_list) - 1:
        return gr.update(), gr.update(), chapter_idx
    new_idx = chapter_idx + 1
    num, title, wav_path = chapter_list[new_idx]
    return gr.update(value=wav_path), gr.update(value=_chapter_label(num, title)), new_idx


# ---------------------------------------------------------------------------
# Pre-generation pipeline — generate scripts with book profiler
# ---------------------------------------------------------------------------

def pregenerate_scripts(file, progress=gr.Progress()):
    """
    Pre-generate Ollama performance scripts for ALL chapters using
    the book profiler to create a custom narrator prompt.

    Returns (status_message, profile_summary, book_profile_display).
    """
    if file is None:
        return "No file uploaded.", "", gr.update(visible=False, value="")

    try:
        size = os.path.getsize(file.name)
    except OSError:
        size = 0

    if size > FILE_SIZE_LIMIT:
        return (
            "File exceeds 50 MB limit.",
            "",
            gr.update(visible=False, value=""),
        )

    ext = os.path.splitext(file.name)[1].lower()
    progress(0, desc="Extracting text...")

    if ext == ".epub":
        try:
            chapters = extract_epub_chapters(file.name)
        except DocReaderError as exc:
            return f"EPUB error: {exc}", "", gr.update(visible=False, value="")
    else:
        try:
            raw_text = extract_text(file.name)
        except DocReaderError as exc:
            return f"Text extraction error: {exc}", "", gr.update(visible=False, value="")

        chapters = split_chapters(raw_text)
        if not chapters:
            chapters = [(1, "Full Book", raw_text)]

    if not chapters:
        return "No chapters found.", "", gr.update(visible=False, value="")

    progress(0.1, desc="Building book profile...")

    book_title = Path(file.name).stem
    first_chapter_text = chapters[0][2]

    try:
        profile = load_or_build_profile(book_title, first_chapter_text)
    except Exception as exc:
        logger.warning("Profile error: %s", exc)
        profile = {}

    genres   = ", ".join(profile.get("genres", [])[:3]) or "Unknown"
    tone     = ", ".join(profile.get("tone", [])[:2]) or "Neutral"
    register = profile.get("emotional_register", "neutral")
    country  = profile.get("country", "unknown")

    profile_md = f"""## Book Profile

**Title:** {profile.get('title', book_title)}
**Genres:** {genres}
**Tone:** {tone}
**Narrator Register:** {register}
**Country:** {country}
**Pacing:** {profile.get('pacing', 'medium')}
**Dialogue Density:** {profile.get('dialogue_density', 'medium')}
**Action Density:** {profile.get('action_density', 'medium')}

**Narrative Style:** {profile.get('narration_style', 'Standard narration')}

**Protagonist:** {profile.get('protagonist_nature', 'Unknown')}
"""

    scripts_dir = Path(tempfile.gettempdir()) / f"scripts_{book_title}"
    scripts_dir.mkdir(exist_ok=True)

    from src.narration.script_writer import write_all_scripts_with_profile

    def progress_callback(idx, total):
        progress((0.1 + idx / total * 0.8), desc=f"Generating scripts: {idx}/{total}...")

    try:
        scripts = write_all_scripts_with_profile(
            chapters,
            str(scripts_dir),
            book_title,
            first_chapter_text,
            progress_callback=progress_callback,
        )
        num_scripts = len(scripts)
        status = f"✓ Pre-generated {num_scripts} performance scripts with custom narrator prompt!"
    except Exception as exc:
        status = f"Script generation error: {exc}"
        num_scripts = 0

    progress(1.0, desc="Done!")

    return (
        status,
        f"**{num_scripts} scripts cached**",
        gr.update(visible=True, value=profile_md),
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def on_file_upload(file):
    """
    Extract text on upload; return:
      (preview, word/chapter count, convert_btn state, error_box, to_chapter update)
    """
    _no_change = gr.update()

    if file is None:
        return (
            "",
            "Extracted: 0 words",
            gr.update(interactive=False),
            gr.update(visible=False, value=""),
            _no_change,
        )

    try:
        size = os.path.getsize(file.name)
    except OSError:
        size = 0

    if size > FILE_SIZE_LIMIT:
        return (
            "",
            "Extracted: 0 words",
            gr.update(interactive=False),
            gr.update(
                visible=True,
                value="**Error:** File exceeds 50 MB limit. Please use a smaller document.",
            ),
            _no_change,
        )

    ext = os.path.splitext(file.name)[1].lower()

    if ext == ".epub":
        try:
            chapters = extract_epub_chapters(file.name)
        except DocReaderError as exc:
            return (
                "",
                "Extracted: 0 words",
                gr.update(interactive=False),
                gr.update(visible=True, value=f"**Error:** {exc}"),
                _no_change,
            )
        if not chapters:
            return (
                "",
                "Extracted: 0 words",
                gr.update(interactive=False),
                gr.update(visible=True, value="**Error:** No readable chapters found in this EPUB."),
                _no_change,
            )
        word_count = sum(len(body.split()) for _, _, body in chapters)
        preview    = chapters[0][2][:500]
        total_ch   = len(chapters)
        secs       = total_ch * 210
        h          = secs // 3600
        m          = (secs % 3600) // 60
        time_str   = f"{h}h {m}m" if h > 0 else f"{m}m"
        count_label = (
            f"Extracted: {word_count:,} words — "
            f"**{total_ch} chapters** — "
            f"⏱ Full book ~{time_str} on CPU"
        )
        default_to = 5 if total_ch > 500 else min(50, total_ch)
        return (
            preview,
            count_label,
            gr.update(interactive=True),
            gr.update(visible=False, value=""),
            gr.update(value=default_to),
        )

    try:
        raw_text = extract_text(file.name)
    except DocReaderError as exc:
        return (
            "",
            "Extracted: 0 words",
            gr.update(interactive=False),
            gr.update(visible=True, value=f"**Error:** {exc}"),
            _no_change,
        )

    word_count = len(raw_text.split())
    preview = raw_text[:500]

    chapters = split_chapters(raw_text)
    if chapters:
        total_ch     = len(chapters)
        secs         = total_ch * 210
        h            = secs // 3600
        m            = (secs % 3600) // 60
        time_str     = f"{h}h {m}m" if h > 0 else f"{m}m"
        count_label  = (
            f"Extracted: {word_count:,} words — "
            f"**{total_ch} chapters detected** — "
            f"⏱ Full book ~{time_str} on CPU"
        )
        default_to   = 5 if total_ch > 500 else min(50, total_ch)
        to_ch_update = gr.update(value=default_to)
    else:
        count_label  = f"Extracted: {word_count:,} words"
        to_ch_update = _no_change

    warning = ""
    if not chapters and len(raw_text) > MAX_TOTAL_CHARS:
        warning = (
            f"**Warning:** Document exceeds {MAX_TOTAL_CHARS:,} characters. "
            "Only the first 50,000 characters will be converted."
        )

    return (
        preview,
        count_label,
        gr.update(interactive=True),
        gr.update(visible=bool(warning), value=warning),
        to_ch_update,
    )


def convert_to_audio(file, from_ch, to_ch, narration_mode, groq_key, progress=gr.Progress()):
    """Full pipeline: extract → slice chapter range → TTS → WAV/ZIP export."""
    emotion_mode = narration_mode.startswith("Emotional")
    _no_audio    = gr.update(visible=False, value=None)
    _no_file     = gr.update(visible=False, value=None)
    _no_err      = gr.update(visible=False, value="")
    _no_dropdown = gr.update(visible=False, choices=[], value=None)
    _no_prev     = gr.update(visible=False)
    _no_next     = gr.update(visible=False)
    _lib_refresh = refresh_library_view()

    def _err(msg):
        return (
            _no_audio, _no_file, "",
            gr.update(visible=True, value=f"**Error:** {msg}"),
            _lib_refresh,
            _no_dropdown, _no_prev, _no_next,
            [], 0,
        )

    if file is None:
        return _err("No file uploaded.")

    try:
        size = os.path.getsize(file.name)
    except OSError:
        size = 0

    if size > FILE_SIZE_LIMIT:
        return _err("File exceeds 50 MB limit. Please use a smaller document.")

    ext = os.path.splitext(file.name)[1].lower()
    progress(0, desc="Extracting text...")

    if ext == ".epub":
        try:
            chapters = extract_epub_chapters(file.name)
        except DocReaderError as exc:
            return _err(str(exc))
        if not chapters:
            return _err("No readable chapters found in this EPUB.")

        start = max(0, int(from_ch) - 1)
        end   = min(len(chapters), int(to_ch))
        chapters = chapters[start:end]
        if not chapters:
            return _err("No chapters in that range.")

        return _convert_chapters(chapters, progress, _err, _no_err, emotion_mode, groq_key)

    try:
        raw_text = extract_text(file.name)
    except DocReaderError as exc:
        return _err(str(exc))

    chapters = split_chapters(raw_text)
    if chapters:
        start = max(0, int(from_ch) - 1)
        end   = min(len(chapters), int(to_ch))
        chapters = chapters[start:end]
        if not chapters:
            return _err("No chapters in that range.")
        return _convert_chapters(chapters, progress, _err, _no_err, emotion_mode, groq_key)
    else:
        return _convert_single(raw_text, progress, _err, _no_err, emotion_mode, groq_key)


def _convert_chapters(chapters, progress, _err, _no_err, emotion_mode=False, groq_key=""):
    """Convert each chapter; return per-chapter WAVs packaged as a ZIP."""
    total_chapters = len(chapters)
    tmp_dir = tempfile.mkdtemp()
    chapter_wavs = []
    first_wav_path = None

    for ch_idx, (num, title, body) in enumerate(chapters):
        chapter_label = f"Chapter {num}: {title}"
        progress(
            (ch_idx, total_chapters),
            desc=f"Processing {chapter_label} ({ch_idx + 1}/{total_chapters})...",
        )

        try:
            segments = _synthesize_text(body, emotion_mode, groq_key)
        except MemoryError:
            return _err("File too large for available memory. Try a shorter document.")
        except DocReaderError as exc:
            if not chapter_wavs:
                return _err(str(exc))
            break

        if not segments:
            continue

        try:
            combined  = concat_with_silence(segments, silence_ms=80)
            filename  = _safe_filename(num, title)
            wav_path  = export_wav_named(combined, tmp_dir, filename)
            chapter_wavs.append((num, title, wav_path))
            if first_wav_path is None:
                first_wav_path = wav_path
        except Exception as exc:
            return _err(f"Audio export failed for {chapter_label}: {exc}")

    if not chapter_wavs:
        return _err("No audio was generated.")

    progress(0.98, desc="Packaging ZIP...")

    try:
        zip_fd, zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(zip_fd)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for _, _, wav in chapter_wavs:
                zf.write(wav, arcname=os.path.basename(wav))
    except Exception as exc:
        return _err(f"ZIP creation failed: {exc}")

    try:
        lib_mgr = get_library_manager()
        lib_mgr.add_book(f"Book ({len(chapter_wavs)} chapters)", chapter_wavs)
    except Exception:
        pass

    completed = len(chapter_wavs)
    status = (
        f"Conversion complete — {completed} of {len(chapters)} chapters converted. "
        "Download the ZIP for all chapters."
    )
    if completed < len(chapters):
        status += " (Some chapters failed and were skipped.)"

    chapter_labels = [_chapter_label(n, t) for n, t, _ in chapter_wavs]
    library_update = refresh_library_view()

    return (
        gr.update(visible=True, value=first_wav_path),
        gr.update(visible=True, value=zip_path, label=f"Download All Chapters ZIP ({completed} files)"),
        status,
        _no_err,
        library_update,
        gr.update(visible=True, choices=chapter_labels, value=chapter_labels[0]),
        gr.update(visible=True),
        gr.update(visible=True),
        chapter_wavs,
        0,
    )


def _convert_single(raw_text, progress, _err, _no_err, emotion_mode=False, groq_key=""):
    """Convert text as one file (no chapters detected)."""
    truncated = False
    if len(raw_text) > MAX_TOTAL_CHARS:
        raw_text  = raw_text[:MAX_TOTAL_CHARS]
        truncated = True

    progress(0.05, desc="Cleaning text...")
    progress(0.10, desc="Splitting into chunks...")
    if emotion_mode:
        progress(0.20, desc="Tagging emotions...")
    progress(0.30, desc="Synthesizing audio...")

    try:
        audio_segments = _synthesize_text(raw_text, emotion_mode, groq_key)
    except MemoryError:
        return _err("Out of memory during synthesis. Try a shorter document.")
    except DocReaderError as exc:
        return _err(str(exc))

    if not audio_segments:
        return _err("No audio was generated.")

    progress(0.90, desc="Stitching audio...")
    try:
        combined = concat_with_silence(audio_segments, silence_ms=80)
        wav_path = export_wav(combined)
    except Exception as exc:
        return _err(f"Audio export failed: {exc}")

    progress(1.0, desc="Done.")

    try:
        lib_mgr = get_library_manager()
        lib_mgr.add_single_file("Converted Book", wav_path)
    except Exception:
        pass

    duration_sec = len(combined) / 1000.0
    status = f"Conversion complete — {duration_sec:.1f}s of audio."
    if truncated:
        status += " (Document was truncated to 50,000 characters.)"

    library_update = refresh_library_view()

    return (
        gr.update(visible=True, value=wav_path),
        gr.update(visible=True, value=wav_path, label="Download WAV"),
        status,
        _no_err,
        library_update,
        gr.update(visible=False, choices=[], value=None),
        gr.update(visible=False),
        gr.update(visible=False),
        [],
        0,
    )


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DocReader — Text to Audio", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# DocReader — Text to Audio")
        gr.Markdown(
            "Upload a document and convert it to speech using Piper TTS — "
            "fully offline, no GPU required. "
            "**Chapter-aware:** documents with chapter headings produce one WAV per chapter, packaged as a ZIP."
        )

        chapter_list_state = gr.State([])
        chapter_idx_state  = gr.State(0)

        file_input = gr.File(
            label="Upload Document (.pdf .epub .docx .txt .md)",
            file_count="single",
        )

        text_preview = gr.Textbox(
            label="Text Preview (first 500 characters)",
            interactive=False,
            lines=6,
            max_lines=6,
            placeholder="Upload a file to see a preview here...",
        )

        word_count_label = gr.Markdown("Extracted: 0 words")

        narration_mode = gr.Dropdown(
            choices=[
                "Standard — XTTS direct (fastest)",
                "Emotional — live Ollama per chapter",
                "Pre-generated — use cached scripts",
            ],
            value="Standard — XTTS direct (fastest)",
            label="Narration mode",
        )

        groq_key_input = gr.Textbox(
            label="Groq API Key (required for Emotional mode with StyleTTS2)",
            placeholder="gsk_...",
            type="password",
            value=GROQ_API_KEY,
        )

        with gr.Accordion("Advanced: Pre-generate Scripts with AI Profiler", open=False):
            gr.Markdown(
                "Generate Ollama performance scripts with auto-detected book genre, tone, "
                "and style. Scripts are cached for fast replays."
            )
            pregenerate_btn    = gr.Button("🧠 Analyze Book & Pre-generate Scripts", variant="secondary")
            pregenerate_status = gr.Markdown("")
            book_profile_display = gr.Markdown("", visible=False)

        with gr.Row():
            from_chapter = gr.Number(label="From chapter", value=1,  minimum=1, precision=0)
            to_chapter   = gr.Number(label="To chapter",   value=50, minimum=1, precision=0)

        convert_btn  = gr.Button("Convert to Audio", variant="primary", interactive=False)
        status_text  = gr.Markdown("")

        audio_output = gr.Audio(
            label="Audio Preview (first chapter / full file)",
            type="filepath",
            visible=False,
            interactive=False,
        )

        chapter_dropdown = gr.Dropdown(
            choices=[],
            label="Select Chapter",
            visible=False,
            interactive=True,
        )

        with gr.Row():
            prev_btn = gr.Button("◀ Previous Chapter", visible=False, variant="secondary")
            next_btn = gr.Button("Next Chapter ▶",     visible=False, variant="secondary")

        file_output = gr.File(label="Download WAV", visible=False)
        error_box   = gr.Markdown("", visible=False)

        gr.Markdown("---")
        gr.Markdown("## Your Library")
        gr.Markdown("*Your converted books appear below*")
        library_view = gr.Markdown(
            value="**No books in library yet.** Convert a document to see it here."
        )

        # ---------------------------------------------------------------------------
        # Wiring
        # ---------------------------------------------------------------------------

        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[text_preview, word_count_label, convert_btn, error_box, to_chapter],
        )

        pregenerate_btn.click(
            fn=pregenerate_scripts,
            inputs=[file_input],
            outputs=[pregenerate_status, status_text, book_profile_display],
        )

        convert_btn.click(
            fn=convert_to_audio,
            inputs=[file_input, from_chapter, to_chapter, narration_mode, groq_key_input],
            outputs=[
                audio_output, file_output, status_text, error_box, library_view,
                chapter_dropdown, prev_btn, next_btn,
                chapter_list_state, chapter_idx_state,
            ],
        )

        chapter_dropdown.change(
            fn=on_chapter_select,
            inputs=[chapter_dropdown, chapter_list_state],
            outputs=[audio_output, chapter_idx_state],
        )

        prev_btn.click(
            fn=on_prev_chapter,
            inputs=[chapter_list_state, chapter_idx_state],
            outputs=[audio_output, chapter_dropdown, chapter_idx_state],
        )

        next_btn.click(
            fn=on_next_chapter,
            inputs=[chapter_list_state, chapter_idx_state],
            outputs=[audio_output, chapter_dropdown, chapter_idx_state],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ai-book-narrator — Document to Audiobook")
    print("=" * 40)
    print("Loading TTS engine...")

    try:
        load_engine()
        print(f"Engine ready: {get_engine_name()}\n")
    except DocReaderError as exc:
        print(f"\nStartup error: {exc}")
        print("Run  python scripts/setup.py  first, then restart.")
        sys.exit(1)

    atexit.register(shutdown_engine)

    clips = validate_clips()
    missing = [e for e, ok in clips.items() if not ok]
    if missing:
        print(f"[StyleTTS] Missing reference clips: {missing}")
        print("[StyleTTS] Using neutral voice for those emotions.")

    demo = build_ui()

    print("Starting Gradio — open http://localhost:7860 in your browser.")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        show_api=False,
    )
