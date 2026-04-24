# Changelog

All notable changes to **sonixvoiceagent** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [3.0.0] - 2026-04-24

### Added — Multi-provider STT/TTS support + Azure integration

#### STT — Four new cloud providers (`backend/asr.py`, `backend/config.py`)
- **Added** Microsoft Azure STT (`_run_azure_stt_session`, `_azure_transcribe`) — supports `te-IN` and `kn-IN`; uses `AZURE_STT_KEY` + `AZURE_STT_REGION`
- **Added** Google Cloud STT (`_run_google_stt_session`, `_google_transcribe`) — REST batch, supports `te-IN` and `kn-IN`; uses `GOOGLE_STT_API_KEY`
- **Added** Amazon Transcribe STT (`_run_amazon_transcribe_session`, `_amazon_transcribe_utterance`) — streaming SDK, supports `te-IN` and `kn-IN`; uses shared AWS credentials
- **Extended** `set_engine()` to accept `google`, `azure`, `amazon` in addition to existing `auto`, `sarvam`, `soniox`, `whisper`

#### STT — `STT_ENGINE` env var (`backend/config.py`, `backend/main.py`)
- **Added** `default_stt_engine` field to `AppConfig` — reads `STT_ENGINE` from `.env` (default: `auto`)
- **Fixed** WebSocket endpoint to use `config.default_stt_engine` as the server-wide default instead of hardcoded `"auto"`, so you can set `STT_ENGINE=azure` in `.env` without touching any URL

#### TTS — Azure TTS priority updated (`.env`, `.env.example`)
- **Added** `azure_tts` to TTS engine priority chains for both Telugu and Kannada
- **Default priorities** changed to `azure_tts,sarvam,edge,gtts` — Azure goes first when key is present

#### Config — All providers now have complete `.env` and `.env.example` entries
- **Added** `AZURE_STT_KEY`, `AZURE_STT_REGION`, `AZURE_TTS_KEY`, `AZURE_TTS_REGION`
- **Added** `GOOGLE_STT_API_KEY`
- **Added** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`
- **Added** `SARVAM_STT_MODEL`, `SARVAM_SPEAKER_KANNADA`, `SARVAM_SPEAKER_KANNADA_MALE`
- **Added** `GOOGLE_TTS_VOICE_KANNADA`, `GOOGLE_TTS_VOICE_KANNADA_MALE`
- **Added** `ELEVENLABS_VOICE_ID_TELUGU_FEMALE`, `ELEVENLABS_VOICE_ID_TELUGU_MALE` with working default voice IDs
- **Added** all Amazon Polly, Gnani, TTSMaker, Narakeet, MicMonster, Speakatoo vars
- All empty keys are safely skipped — no errors if a key is not set

---

## [2.9.0] - 2026-04-16

### Fixed — LLM timeout, TTS cascade latency, Sarvam speech rate

#### LLM — 72b cold-start timeout (`backend/llm.py`)
- **Raised** Ollama HTTP read timeout from `60s` → `300s` to accommodate `qwen2.5:72b` first-inference VRAM load time
- Previously the second user turn would always hit the 60-second hard limit while the model paged layers into GPU memory, falling back to a neutral stub (`"అర్థమైంది, చూడనివ్వండి."`) with no real LLM response

#### TTS — cascade latency eliminated (`backend/tts.py`, `.env`)
- **Root cause**: `TELUGU_TTS_ENGINE_PRIORITY` was trying 4 engines with placeholder/invalid API keys before reaching `edge-tts`, burning ~1.3s of wasted HTTP calls on every single response
- **Fixed** default priority to skip engines without valid keys; now goes `sarvam → edge → gtts`

#### Sarvam TTS — speech rate reduced (`backend/tts.py`, `backend/config.py`)
- **Fixed** hardcoded `pace: 1.65` (65% above normal) → now driven by `SARVAM_PACE` env var (default `1.0` = natural speed)
- Added `sarvam_pace: float` field to `TTSConfig` in `config.py`
- Added `SARVAM_PACE` to `.env.example` with tuning guide (`0.5` slow → `1.0` normal → `2.0` fast)

#### Infrastructure
- Installed `qwen2.5:72b` (47 GB) on NVIDIA A100-SXM4-80GB — full model served locally via Ollama
- Verified Sarvam API key and `bulbul:v2` model working for Telugu TTS

---

## [2.8.0] - 2026-04-11

### Changed — Two-column UI layout + conversation display fix

#### Side-by-side layout
- **Redesigned** page layout from single vertical column to a two-panel side-by-side design:
  - **Left panel**: Conversation transcript (live during session, persists after stopping) + Download Transcript button
  - **Right panel**: Language selector (Dari / Pashto), Voice selector (Male / Female), status orb, Start/Stop button, mic level meter, status bar
- Conversation log now uses `max-height: calc(100vh - 260px)` — fills available screen height naturally without overflowing
- Download Transcript button is full-width inside the left panel, only appears after stopping
- Responsive: stacks to single column (controls above, conversation below) on screens narrower than 700px

#### Conversation visibility fix
- **Fixed** double-bubble bug: `tts_start` was always calling `startBotBubble()`, creating a second empty bot bubble even when `bot_text_fragment` had already opened one. Now guarded with `if (!currentBotMsgEl)` so only one bubble per turn is created.
- **Added** `— Session ended —` divider at the bottom of the conversation log when a session stops, making it clear the chat history is complete
- Conversation header dot indicator turns green while session is active

---

## [2.7.0] - 2026-04-10

### Fixed — Dari LLM response quality (natural Afghan Dari, no Iranian Persian mixing)

- **Rewrote** Afghan Dari system prompt in `backend/config.py` with explicit vocabulary enforcement:
  - Banned Iranian Persian words with direct Afghan Dari replacements: `استم` not `هستم`, `بلی` not `بله`, `می‌خواهید` not `در نظر دارید`, `اتومیشن تست` not `خودکارسازی`, `چه قسم` not `چه نوع`
  - Banned literary/formal connectors unnatural in spoken Dari: `چنین`, `مذکور`, `لذا`, `بنابراین`
  - Added explicit grammar examples: `می‌توانم کمک کنم` (not `کمک کرده می‌توانم`)
  - Added instruction to ask for repetition on garbled ASR: `'بخشش می‌خواهم، دوباره بگویید لطفاً.'`
  - Replaced English service names with Afghan Dari-natural forms: `سافت‌ویر تیستنگ`, `اتومیشن تست`, `پرفارمنس تست`
- **Fixed** off-topic template sentence the LLM was copying verbatim — previously it used `کمک کرده می‌توانم` which LLM echoed in all responses; now uses `می‌توانم کمک کنم`
- **Result**: LLM now produces natural-sounding Afghan phone-call Dari without Iranian Persian vocabulary or grammatical constructions

---

## [2.6.0] - 2026-04-10

### Fixed — Dari female voice, transcript UI, and timestamp accuracy

#### Dari female voice now uses ElevenLabs Bella
- **Fixed** Dari female was silently getting male voice because the old backend (MMS strict mode, single-voice) was still running. New `_synthesize_dari()` now correctly maps `voice=female` → `ELEVENLABS_VOICE_ID_DARI_FEMALE` (Bella) and `voice=male` → `ELEVENLABS_VOICE_ID_DARI_MALE` (Adam).
- **Fixed** Backend startup message in `main.py`: was `Dari=MMS-TTS strict` — now reads `Dari=ElevenLabs→MMS(prs) fallback` to accurately reflect the active code path.

#### Scrollable conversation transcript
- **Fixed** Conversation log height: `min-height: 260px; max-height: 460px` with `scroll-behavior: smooth` — conversation now scrolls naturally without overflowing the page layout.

#### Bot response timestamp accuracy
- **Fixed** Bot bubble timestamp was stamped at `tts_start` (fired immediately after user turn) so User and Assistant showed the same time. Now stamped inside `finalizeBotBubble()` when the LLM + TTS cycle actually completes — gives accurate per-turn timestamps.

#### Download transcript format
- **Improved** Transcript file format: language native name header (`دری` / `پښتو`), turn count, `─────` dividers between turns, per-turn timestamps matching the UI display time.

---

## [2.5.0] - 2026-04-10

### Added — ElevenLabs TTS for Dari (male & female)

- **Added** ElevenLabs as primary TTS engine for Dari language (previously Dari used MMS-TTS strict mode only).
- **Added** `ELEVENLABS_VOICE_ID_DARI_MALE` and `ELEVENLABS_VOICE_ID_DARI_FEMALE` env vars in `.env` and `config.py` — defaults to Adam / Bella multilingual; override with a cloned Afghan Dari voice for best accent quality.
- **Changed** `backend/tts.py`: Dari TTS path is now `_synthesize_dari()` — ElevenLabs first, falls back to MMS Afghan Dari (`facebook/mms-tts-prs`) if ElevenLabs fails, then silence.
- **Result**: Dari and Pashto now both use ElevenLabs for male/female voice selection with the same fallback safety net.

---

## [2.4.0] - 2026-04-10

### Fixed — Afghan Dari TTS (not Iranian Persian)

- **Fixed** `backend/config.py`: Dari MMS-TTS model changed from `facebook/mms-tts-fas` (general/Iranian Persian) to `facebook/mms-tts-prs` (ISO 639-3 `prs` = Dari, Afghan Persian specifically). This ensures the synthesised voice uses the correct Afghan Dari accent rather than Iranian Farsi.
- **Fixed** Dari greeting text — now uses Afghan Dari vocabulary: `مرستیار` (assistant), `استم` (not Iranian `هستم`), Afghan phrasing throughout.
- **Fixed** Dari neutral stubs — `بخشش می‌خواهم` (Afghan) replaces `ببخشید` (Iranian).
- **Rewritten** Dari LLM system prompt to explicitly enforce Afghan Dari grammar and vocabulary over Iranian Persian: instructs model to use `استم/استید`, `بلی` not `بله`, `بخشش می‌خواهم` not `ببخشید`, `مرستیار` for assistant, and Afghan sentence patterns.

### Added — Conversation History UI + Download Transcript

- **Added** scrollable conversation history panel replacing the simple single-turn "You said / Assistant" cards. Every user utterance and bot response stacks as labelled chat bubbles (user = orange, right-aligned; assistant = purple, left-aligned).
- **Added** live bot response streaming — LLM fragments appear word by word in the bubble as they arrive from the server, with a subtle streaming opacity until the turn completes.
- **Added** partial ASR preview below the log — italic live text shows while the user is still speaking, then disappears when the final transcript arrives.
- **Added** **Download Transcript** button — appears automatically when a session is stopped. Exports the full conversation as a timestamped plain-text `.txt` file (`qobox-transcript-YYYY-MM-DD-HH-MM-SS.txt`) with language, date, turn-by-turn user/assistant lines.
- **Changed** conversation log and history are cleared automatically when a new session starts.

### Changed — UI cleanup

- **Removed** TTS Engine dropdown from the UI (ElevenLabs runs automatically via server-side priority; no engine selection needed).
- **Simplified** Voice dropdown label: `Male (default)` → `Male`, `Female` unchanged.

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
