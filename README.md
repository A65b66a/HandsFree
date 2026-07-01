# ai-book-narrator

Convert any document into a narrated audiobook — fully offline, no GPU required.

Supports PDF, EPUB, DOCX, TXT, and Markdown. Chapter-aware documents produce one WAV file per chapter, packaged as a ZIP. An optional AI narration layer adds emotion-aware synthesis and book-specific narrator personality.

---

## Features

| Feature | Description |
|---|---|
| **Multi-format extraction** | PDF, EPUB (chapter-aware), DOCX, TXT, Markdown |
| **Three-engine TTS** | Piper (CPU) → XTTS v2 (CPU + voice cloning) → StyleTTS2 (GPU + emotions) |
| **Emotion-aware narration** | Groq tags each sentence NEUTRAL / INTENSE / COLD / EXPRESSIVE |
| **Book profiling** | Scrapes NovelUpdates + Goodreads, analyzes first chapter with Ollama |
| **Script caching** | Ollama scripts cached to disk; synthesis reruns without re-calling AI |
| **Audio library** | JSON-backed metadata store with in-app chapter navigation |
| **Fully offline** | Core Piper synthesis requires no internet or cloud API |

---

## Architecture

```
app.py                         ← Gradio UI + event handlers
│
└── src/
    ├── config.py              ← All paths, constants, Ollama settings
    │
    ├── extraction/
    │   ├── extractor.py       ← PDF/EPUB/DOCX/TXT/MD parsing
    │   └── preprocessor.py   ← Unicode normalization, sentence chunking
    │
    ├── tts/
    │   ├── tts_engine.py      ← Engine router + Piper subprocess wrapper
    │   ├── xtts_engine.py     ← XTTS v2 voice cloning (CPU)
    │   └── styletts2_engine.py ← StyleTTS2 emotion-aware synthesis (GPU)
    │
    ├── narration/
    │   ├── script_writer.py   ← Ollama performance script generator (cached)
    │   ├── emotion_tagger.py  ← Groq sentence-level emotion labels
    │   ├── style_director.py  ← Emotion → reference WAV clip mapping
    │   └── book_profiler.py   ← Web scrape + Ollama book analysis
    │
    ├── audio_utils.py         ← pydub: PCM wrapping, concat, WAV export
    └── library.py             ← JSON-backed converted-book store
```

### TTS Engine Priority

```
load_engine() tries in order:
  1. StyleTTS2  ── GPU available? CUDA + StyleTTS2 repo cloned? → GPU synthesis
  2. XTTS v2   ── TTS==0.22.0 installed? → CPU voice cloning
  3. Piper      ── always available (fallback) → fast CPU synthesis
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | [Gradio](https://gradio.app/) 4.44 |
| Document parsing | PyMuPDF, ebooklib, python-docx, BeautifulSoup4 |
| TTS engines | [Piper](https://github.com/rhasspy/piper), [XTTS v2](https://github.com/coqui-ai/TTS), [StyleTTS2](https://github.com/yl4579/StyleTTS2) |
| Emotion tagging | [Groq API](https://console.groq.com) (llama-3.3-70b-versatile) |
| Book profiling | [Ollama](https://ollama.ai) (llama3.2:3b) |
| Audio processing | pydub, NumPy |
| Runtime | Python 3.10+ · Windows (primary) · Linux (untested) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-book-narrator.git
cd ai-book-narrator
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies

**Piper only (fastest, no GPU required):**
```bash
pip install -e .
```

**With XTTS v2 voice cloning (CPU):**
```bash
pip install -e ".[xtts]"
```

**With StyleTTS2 (GPU, requires CUDA):**
```bash
pip install -e ".[styletts2]"
# Also clone the StyleTTS2 repo into docreader/StyleTTS2/
# and download the LibriTTS checkpoint — see StyleTTS2 README
```

### 4. Download Piper and voice model

```bash
python scripts/setup.py
```

This downloads `piper.exe` (~50 MB) and `en_US-lessac-medium.onnx` (~60 MB) into `docreader/`.

### 5. Configure (optional)

```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY if you want Emotional mode
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required for Emotional narration mode with StyleTTS2. Free tier available. |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model for script rewriting and book analysis. |

All paths (Piper executable, voice models, book profiles, library) are resolved automatically from `src/config.py`.

---

## Usage

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

### Basic workflow

1. Upload a PDF, EPUB, DOCX, TXT, or MD file.
2. See the word count and detected chapter structure in the preview.
3. Select a chapter range (default: chapters 1–50).
4. Choose a narration mode:
   - **Standard** — direct TTS synthesis (fastest)
   - **Emotional** — Ollama rewrites each chunk as a performance script
   - **Pre-generated** — uses cached scripts from a previous Analyze run
5. Click **Convert to Audio**.
6. Download the WAV (single file) or ZIP (chapter mode).

### Voice cloning (XTTS v2)

Record your voice in `docreader/voices/reference/neutral.wav` (10–30 seconds of clean speech). XTTS v2 will clone this voice for all synthesis.

### Emotion clips (StyleTTS2)

Place reference WAVs in `docreader/voices/emotion_clips/`:
- `neutral.wav`, `intense.wav`, `cold.wav`, `expressive.wav`

StyleTTS2 uses these to compute style vectors per emotion label.

### Book profiler + AI scripts

Click **"Analyze Book & Pre-generate Scripts"** in the Advanced accordion. The profiler:
1. Searches NovelUpdates and Goodreads for genre/tone metadata.
2. Sends the first 2,000 characters to Ollama for style analysis.
3. Generates a custom narrator prompt and rewrites all chapters as performance scripts.
4. Caches scripts to disk — subsequent runs skip the Ollama step.

---

## Screenshots

> _Add screenshots of the Gradio UI here._

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `scripts/setup.py` | Download Piper executable and voice model |
| `scripts/narrator_debug.py` | Audio quality analyser (silence, volume, rhythm, pitch) |
| `scripts/convert_audio.py` | Convert any audio format to WAV |

---

## Future Improvements

- [ ] Linux/Mac support (Piper binary selection by platform)
- [ ] Multi-language voice models (Piper supports 40+ languages)
- [ ] Streaming synthesis (play while converting)
- [ ] Custom chapter regex patterns via config
- [ ] EPUB metadata extraction (cover art, author, publisher)
- [ ] Export to MP3 alongside WAV
- [ ] Audiobook M4B container output

---

## License

[MIT](LICENSE)
