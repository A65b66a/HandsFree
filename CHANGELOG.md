# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2024

### Added
- Multi-format document parsing: PDF, EPUB, DOCX, TXT, Markdown
- Three-engine TTS router with automatic hardware detection:
  - **Piper** (CPU, always available, ~1× real-time)
  - **XTTS v2** (CPU, voice cloning from reference WAV clips, ~0.5× real-time)
  - **StyleTTS2** (GPU, emotion-aware synthesis, ~2× real-time on RTX 3090)
- Chapter-aware EPUB extraction with dual parser (ebooklib + zipfile fallback)
- Sentence-level emotion tagging via Groq API (llama-3.3-70b-versatile)
- Four emotion labels: NEUTRAL, INTENSE, COLD, EXPRESSIVE
- Book profiling: dual web scrape (NovelUpdates, Goodreads) + Ollama first-chapter analysis
- Dynamic narrator prompt generation from book profile
- Performance script caching (Pass 1 Ollama → disk; Pass 2 TTS reads cache)
- JSON-backed audio library with chapter navigation UI
- WAV export: single file or per-chapter ZIP
- Gradio UI with chapter range selector and audio player
- Fully offline operation (no cloud dependency for core Piper synthesis)
