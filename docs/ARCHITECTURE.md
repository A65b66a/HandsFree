# Architecture Notes

See the Architecture section in [README.md](../README.md) for the folder structure diagram.

## TTS Engine Decision Tree

```
load_engine() on startup
│
├── torch.cuda.is_available() AND StyleTTS2/ repo cloned?
│   └── YES → StyleTTS2Engine.load() → returns True?
│       └── YES → _engine_name = "StyleTTS2"  (emotion-aware GPU synthesis)
│
├── TTS==0.22.0 installed?
│   └── YES → XTTSEngine.load() succeeds?
│       └── YES → _engine_name = "XTTS"  (voice cloning, CPU)
│
└── ALWAYS → TTSEngine (Piper).load() → _engine_name = "Piper"
```

## Data Directory Layout

```
docreader/                       ← runtime asset root (not versioned)
├── piper/
│   └── piper.exe                ← Piper TTS binary (Windows AMD64)
├── voices/
│   ├── en_US-lessac-medium.onnx ← Piper voice model
│   ├── en_US-lessac-medium.onnx.json
│   ├── reference/               ← voice cloning clips (XTTS v2)
│   │   ├── neutral.wav          ← your recorded voice (10–30s)
│   │   └── epic.wav             ← optional alternative
│   └── emotion_clips/           ← StyleTTS2 style reference clips
│       ├── neutral.wav
│       ├── intense.wav
│       ├── cold.wav
│       └── expressive.wav
├── StyleTTS2/                   ← cloned repo (GPU users only)
├── book_profiles/               ← cached JSON profiles (auto-generated)
└── library_data/
    └── library.json             ← converted book metadata
```

## Known Design Trade-offs

**Coupling: `style_director.py` → `xtts_engine._find_clip_any_format`**
`style_director` imports a private helper from `xtts_engine` to resolve voice clips in any audio format. This cross-domain coupling is intentional for now — the helper is specific to the audio-clip resolution logic. A future refactor would expose a `find_audio_clip()` utility in `config.py`.

**Mixed responsibilities in `app.py`**
The Gradio event handlers and the synthesis pipeline (`_synthesize_text`, `_convert_chapters`, `_convert_single`) both live in `app.py`. Extracting the pipeline into `src/narration/pipeline.py` would be the next refactoring step, but requires careful decoupling from Gradio's `progress()` callback type.
