# Changelog

All notable changes to **sonixvoiceagent** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] - 2026-04-06

### Changed — TTS (Breaking improvement)

- **Replaced** edge-tts (`te-IN-ShrutiNeural`) as primary TTS with **Meta MMS-TTS** (`facebook/mms-tts-tel`)
  - VITS architecture trained specifically on Telugu speech corpus
  - Sounds natural and native — not robotic or synthesised-sounding
  - Runs entirely on local GPU (CUDA), ~460 MB model, no internet required
  - Model downloads automatically from HuggingFace on first server start
  - Output: 16 000 Hz PCM → polyphase-resampled to 24 000 Hz for the browser
  - Resampling priority: `torchaudio` → `scipy` → numpy linear interpolation
- **edge-tts** retained as Fallback 1 (internet required, Microsoft Azure neural)
- **gTTS** retained as Fallback 2 (internet required, Google TTS)
- Added GPU warmup at startup — no cold-start lag on first utterance

### Changed — LLM

- Documented `gemma4:31b` (Google Gemma 4, April 2026) as the recommended upgrade option
  - ~62.5 GB, fits in 80 GB A100
  - Excellent Telugu comprehension and natural phrasing
- Default remains `qwen2.5:72b` for stability (already pulled)
- Added model comparison table to README and `.env.example`

### Added

- `scipy>=1.11.0` to `requirements.txt` for high-quality polyphase resampling
- `_resample()` helper with torchaudio → scipy → numpy fallback chain
- `_apply_fade()` helper for click-free chunk boundaries in TTS output
- `_pcm_to_int16()` helper for consistent audio normalisation
- `schedule_tts_warmup()` pre-loads MMS-TTS model at server startup
- `CHANGELOG.md` (this file)

### Fixed

- RAG docs path resolves correctly when server runs from `backend/` directory
  (`RAG_DOCS_DIR=../docs` in `.env`)

---

## [1.0.0] - 2026-04-06

### Initial release

- Full-duplex Telugu voice agent over WebSocket + FastAPI
- **ASR**: Soniox streaming (`soniox_multilingual_2`, `te`) → Whisper large-v3 fallback
- **LLM**: Ollama `qwen2.5:72b` — open-source, local GPU, no API key required
- **TTS**: edge-tts `te-IN-ShrutiNeural` (24 kHz) → gTTS fallback
- Browser client: Web Audio API, AudioWorklet PCM capture, VAD interrupt
- FAISS RAG with Qobox knowledge base (Telugu + English)
- Sliding-window conversation memory (8 turns)
- Based on architecture of [voiceagentcloud](https://github.com/Raghavendraqbox/voiceagentcloud)
