# Changelog

All notable changes to **sonixvoiceagent** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.1.0] - 2026-04-09

### Fixed — ASR (Soniox v2 API migration)

- **Migrated** Soniox ASR from deprecated v1 API (`soniox.transcribe_live`, `soniox.speech_service`) to v2 API (`SonioxClient`, `RealtimeSTTSession`, `RealtimeSTTConfig`)
- **Fixed** `_SONIOX_AVAILABLE = False` caused by broken v1 imports — Soniox is now correctly detected and used when API key is present
- **Added** `enable_language_identification=False` to `RealtimeSTTConfig` — prevents Soniox auto-switching transcription to English mid-speech
- **Added** `_SonioxFatalError` exception class for non-retryable errors (e.g. 402 balance exhausted)
- **Fixed** tight retry loop on Soniox 402 error — now immediately falls back to Whisper large-v3 permanently for the session instead of hammering the API every second

### Fixed — TTS voice consistency

- **Fixed** mid-response voice change: MMS-TTS failing on a single sentence would silently fall back to edge-tts, causing a jarring voice switch mid-conversation
- **Added** `_mms_available` session flag to `VoiceTTSHandler` — once MMS-TTS fails, edge-tts is used for all subsequent sentences in that session, keeping the voice consistent

### Fixed — LLM / neutral stubs

- **Removed** English string `"Sorry, please give me a moment while I check on that."` from `neutral_stubs` for both Dari and Pashto — MMS-TTS synthesising English text caused voice distortion and quality degradation
- **Replaced** with native Dari (`بسیار ممنون، یک لحظه صبر کنید.`) and Pashto (`مننه، یو شیبه صبر وکړئ.`) equivalents
- **Updated** system prompt for both Dari and Pashto: explicitly instructs the LLM to always respond in the target language even when ASR transcription appears in English

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
