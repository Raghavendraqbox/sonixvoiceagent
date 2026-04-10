# Changelog

All notable changes to **sonixvoiceagent** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.3.0] - 2026-04-10

### Fixed — ElevenLabs voice selection and echo-triggered interrupts

#### ElevenLabs now actually called on every turn
- **Fixed** `backend/tts.py`: ElevenLabs was short-circuiting before making any HTTP request because `ELEVENLABS_VOICE_ID_PASHTO_MALE` / `_FEMALE` were empty. Added default stock voice IDs (`pNInz6obpgDQGcFmaJgB` Adam / `EXAVITQu4vr4xnSDxMaL` Bella) as fallback so the API is always called when a key is present.
- **Added** distinct HTTP error log lines for 401 (bad key) and 402 (free-plan limit) to make account issues immediately obvious in logs.
- **Improved** synthesis log: now prints `selected_voice=male/female → voice_id=<id>` and success line includes voice name + byte count for easy tracing.

#### Male / Female voice selection respected by ElevenLabs
- **Fixed** `backend/tts.py`: when `voice=female` is selected in the UI, `ELEVENLABS_VOICE_ID_PASHTO_FEMALE` (Bella) is used; `voice=male` uses `ELEVENLABS_VOICE_ID_PASHTO_MALE` (Adam). Previously both resolved to empty string and short-circuited.
- **Updated** `.env`: `ELEVENLABS_VOICE_ID_PASHTO_MALE` and `ELEVENLABS_VOICE_ID_PASHTO_FEMALE` now have explicit default values with override instructions.

#### Girl-voice (MMS-TTS) eliminated from ElevenLabs sessions
- **Fixed** `backend/tts.py` engine priority: when a cloud TTS engine (ElevenLabs, Narakeet, etc.) is explicitly selected from the UI, MMS-TTS (`facebook/mms-tts-pps`) is no longer inserted as fallback #2. MMS produces a robotic female-sounding voice that was heard mid-conversation when ElevenLabs failed. Cloud sessions now fall back only to `edge → gtts`.
- **Result**: consistent single voice throughout a session — no more sudden voice switches mid-conversation.

#### Echo-triggered barge-in / mid-sentence interrupts suppressed
- **Fixed** `backend/session_manager.py`: ASR kept running while TTS was playing through speakers. The bot's own audio was picked up by the microphone, re-transcribed, and treated as a new user utterance — cancelling the bot's current speech mid-sentence (especially visible on the greeting).
- **Added** `SessionManager._drain_echo_transcripts()`: after every TTS turn (greeting + LLM responses), any transcripts that queued while the bot was speaking are discarded before waiting for the next real user utterance.

---

## [2.2.0] - 2026-04-10

### Fixed — ElevenLabs in-app noise diagnostics and decode integrity

- **Fixed** MP3 decode path in `backend/tts.py`: replaced raw PyAV plane-byte extraction with `to_ndarray()` frame extraction to avoid padding/stride artefacts that can sound like hiss/static in app playback.
- **Added** optional debug A/B audio dump capability for ElevenLabs:
  - `DEBUG_TTS_DUMP_AUDIO` (default `false`)
  - `DEBUG_TTS_DUMP_DIR` (default `./debug_audio`)
- **Added** paired debug outputs per synthesis:
  - source API audio (`.mp3`)
  - decoded playback audio (`.decoded.wav`, 24kHz mono int16)
- **Added** `ELEVENLABS_NOISE_DEBUG_TEST_PLAN.md` with a detailed validation workflow, result matrix, and cleanup guidance.

### Changed — Strict Dari enforcement

- **Enforced** Dari TTS as **Dari-only** runtime behavior:
  - Dari now always uses `facebook/mms-tts-fas`
  - Dari no longer falls back to edge-tts/gTTS Persian voices
  - On MMS failure, strict fallback is silence (to prevent language drift)
- **Updated** TTS startup/health descriptions in `backend/main.py` to reflect strict Dari + configurable Pashto behavior.

### Docs / Config cleanup

- **Updated** `.env.example` comments to clearly state strict Dari runtime behavior and mark `fa-IR` values as compatibility/testing-only.
- **Updated** `backend/config.py` comments for `fa`/`fa-IR` fields to avoid implying non-Dari fallback behavior in strict mode.

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
