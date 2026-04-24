# Sonix Voice Agent — Real-Time Telugu & Kannada Voice AI

A production-ready, full-duplex voice AI agent for **Telugu** (తెలుగు) and **Kannada** (ಕನ್ನಡ). Speak in Telugu or Kannada and get an intelligent voice response in real time. Runs entirely on your own GPU — no OpenAI or cloud AI API needed.

```
Browser Mic
    │
    ▼  PCM 16kHz (WebSocket binary)
STT Engine  ── Speech-to-Text ─────────────────────────────────────┐
    │  Sarvam saarika:v2.5 (primary, best for Indian languages)     │
    │  Azure STT → Google STT → Amazon Transcribe (cloud options)   │
    │  Soniox cloud streaming → Whisper large-v3 local (fallbacks)  │
    ▼  Final transcript                                             │
Qwen2.5:72b via Ollama  ── Telugu / Kannada LLM response ──────────┤
    │  (open-source, runs on your GPU)                              │
    ▼  Text (streamed sentence by sentence)                         │
TTS Engine  ── Neural Speech ───────────────────────────────────────┘
    │  Azure TTS → Sarvam bulbul:v2 → edge-tts → gTTS
    │  Telugu: te-IN-ShrutiNeural / te-IN-MohanNeural (Azure)
    │  Kannada: kn-IN-SapnaNeural / kn-IN-GaganNeural (Azure)
    ▼  PCM 24kHz (WebSocket binary)
Browser Speaker
```

---

## Table of Contents

1. [What You Need](#1-what-you-need)
2. [Step-by-Step Installation](#2-step-by-step-installation)
3. [Start the Agent](#3-start-the-agent)
4. [Open in Browser](#4-open-in-browser)
5. [Health Check](#5-health-check)
6. [Configuration](#6-configuration)
7. [Choosing a GPU & Model](#7-choosing-a-gpu--model)
8. [ASR Options](#8-asr-options)
9. [TTS Voices](#9-tts-voices)
10. [Adding Your Own Knowledge Base](#10-adding-your-own-knowledge-base)
11. [Troubleshooting](#11-troubleshooting)
12. [Architecture Deep Dive](#12-architecture-deep-dive)
13. [Project Structure](#13-project-structure)

---

## 1. What You Need

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Ubuntu 20.04+ / Debian 11+ | Ubuntu 22.04 |
| **Python** | 3.10 | 3.11 |
| **GPU** | 8 GB VRAM (for 7B model) | 80 GB A100 (for 72B model) |
| **CUDA** | 12.x | 12.4 |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 60 GB free | 100 GB free |
| **Internet** | Required (first-time downloads) | — |
| **Browser** | Chrome or Edge | Chrome |

> **No experience needed.** Follow every step below exactly and you will have a working Telugu/Kannada voice agent.

---

## 2. Step-by-Step Installation

### Step 1 — Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-pip zstd
```

### Step 2 — Clone the repository

```bash
git clone -b generic-telugu-kannada https://github.com/Raghavendraqbox/sonixvoiceagent.git
cd sonixvoiceagent
```

### Step 3 — Install PyTorch with CUDA

> Skip this if PyTorch is already installed. Run `python3 -c "import torch; print(torch.__version__)"` to check.

```bash
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

This downloads ~2.5 GB. Wait for it to finish.

### Step 4 — Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### Step 5 — Install Ollama (local LLM server)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify installation:
```bash
ollama --version
```

### Step 6 — Start Ollama

```bash
ollama serve &
```

You should see: `Ollama is running`

### Step 7 — Download a multilingual LLM

Choose the model that fits your GPU VRAM (see [Section 7](#7-choosing-a-gpu--model)):

```bash
# If you have 6 GB VRAM
ollama pull qwen2.5:7b

# If you have 10 GB VRAM
ollama pull qwen2.5:14b

# If you have 20 GB VRAM
ollama pull qwen2.5:32b

# If you have 80 GB VRAM (recommended)
ollama pull qwen2.5:72b
```

> The download is large (7B ≈ 4 GB, 72B ≈ 47 GB). This only happens once.

### Step 8 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
LANGUAGE=telugu          # or: kannada
OLLAMA_MODEL=qwen2.5:72b # match what you pulled above
```

Paste any API keys you have for STT/TTS — everything else falls back gracefully (see Sections 8 and 9).

---

## 3. Start the Agent

```bash
cd backend
python3 main.py
```

You should see:

```
Telugu & Kannada Voice AI Agent starting…
Default language : telugu
Supported        : telugu, kannada
LLM  : Ollama qwen2.5:72b @ http://localhost:11434
LLM warm-up starting — loading qwen2.5:72b into VRAM…
LLM warm-up complete — qwen2.5:72b ready
Server ready.
```

> **LLM warm-up:** On the very first start, warm-up takes 60–120 seconds while the 72B model loads from disk into VRAM. The server says `Server ready.` only after warm-up completes.

---

## 4. Open in Browser

Open **Chrome** or **Edge** and go to:

```
http://localhost:8000
```

1. Select your language: **Telugu (తెలుగు)** or **Kannada (ಕನ್ನಡ)**
2. Select voice: **Male** or **Female**
3. Click **Start Conversation**
4. Allow microphone access when the browser asks
5. Speak in Telugu or Kannada — the agent will respond in the same language

> **Must use Chrome or Edge.** Firefox has limited Web Audio API support.

> **Must be on `localhost` or HTTPS.** Browsers block microphone on plain HTTP for remote IPs. If accessing from another machine, use an SSH tunnel.

---

## 5. Health Check

```bash
curl http://localhost:8000/health
```

```bash
curl http://localhost:8000/languages
```

---

## 6. Configuration

All settings live in `.env` (copied from `.env.example`). Keys you leave empty are safely skipped.

### Quick-start minimal config

```env
LANGUAGE=telugu
OLLAMA_MODEL=qwen2.5:72b
```

### STT engine selection

```env
# Set once in .env — applies to all sessions
# Options: auto | sarvam | soniox | google | azure | amazon | whisper
STT_ENGINE=auto

# Override per-session via WebSocket URL:
# ws://host/ws?language=telugu&stt_engine=azure
```

### TTS engine priority

```env
# Comma-separated — first engine that succeeds wins
# Options: azure_tts | sarvam | google_tts | amazon_polly | elevenlabs |
#          gnani | ttsmaker | edge | gtts
TELUGU_TTS_ENGINE_PRIORITY=azure_tts,sarvam,edge,gtts
KANNADA_TTS_ENGINE_PRIORITY=azure_tts,sarvam,edge,gtts
```

### API keys summary

| Provider | Variables | Used for |
|---|---|---|
| Sarvam AI | `SARVAM_API_KEY` | STT + TTS (best for Indian languages) |
| Azure | `AZURE_STT_KEY` + `AZURE_STT_REGION` | STT |
| Azure | `AZURE_TTS_KEY` + `AZURE_TTS_REGION` | TTS |
| Google | `GOOGLE_STT_API_KEY` | STT |
| Google | `GOOGLE_TTS_API_KEY` | TTS |
| Amazon | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | STT (Transcribe) + TTS (Polly) |
| Soniox | `SONIOX_API_KEY` | STT (real-time streaming) |
| ElevenLabs | `ELEVENLABS_API_KEY` | TTS |
| Gnani | `GNANI_API_KEY` + `GNANI_CLIENT_ID` | TTS |
| edge-tts | — | TTS (free, no key) |
| gTTS | — | TTS (free, no key) |

---

## 7. Choosing a GPU & Model

| Model | VRAM | Download | Telugu Quality | Kannada Quality | Speed |
|-------|------|----------|----------------|-----------------|-------|
| `qwen2.5:7b` | ~6 GB | ~4 GB | ★★★☆☆ Good | ★★★☆☆ Good | Fastest |
| `qwen2.5:14b` | ~10 GB | ~9 GB | ★★★★☆ Very Good | ★★★★☆ Good | Very Fast |
| `qwen2.5:32b` | ~20 GB | ~19 GB | ★★★★★ Excellent | ★★★★☆ Very Good | Fast |
| `qwen2.5:72b` | ~48 GB | ~47 GB | ★★★★★ Best | ★★★★★ Best | Medium |

- **A100 80GB** → `qwen2.5:72b` (recommended)
- **RTX 4090 (24 GB)** → `qwen2.5:32b`
- **RTX 3080 (10 GB)** → `qwen2.5:14b`
- **RTX 3060 / 6–8 GB GPU** → `qwen2.5:7b`

---

## 8. ASR Options

### Priority chain (STT_ENGINE=auto)

```
Sarvam AI saarika:v2.5  (SARVAM_API_KEY set)   ← primary
    ↓ if key missing
Soniox stt-rt-v4        (SONIOX_API_KEY set)   ← real-time partials
    ↓ if key missing
faster-whisper large-v3 (local GPU)             ← always available
```

### All STT options

| Engine | Key needed | Latency | Partial results | Best for |
|--------|-----------|---------|-----------------|---------|
| `sarvam` | `SARVAM_API_KEY` | ~300ms | No | Telugu/Kannada accuracy |
| `azure` | `AZURE_STT_KEY` | ~200ms | No | General purpose |
| `google` | `GOOGLE_STT_API_KEY` | ~300ms | No | General purpose |
| `amazon` | AWS credentials | ~400ms | No | AWS ecosystem |
| `soniox` | `SONIOX_API_KEY` | ~100ms | Yes | Real-time streaming |
| `whisper` | None (local GPU) | ~800ms | No | No internet needed |

To switch engine for all sessions:
```env
STT_ENGINE=azure
```

To switch for a single session (WebSocket URL):
```
ws://host/ws?language=telugu&stt_engine=azure
```

---

## 9. TTS Voices

### Telugu

| Engine | Voice | Key needed |
|--------|-------|-----------|
| Azure TTS | te-IN-ShrutiNeural (F) / te-IN-MohanNeural (M) | `AZURE_TTS_KEY` |
| Sarvam AI | anushka (F) / abhilash (M) — `bulbul:v2` | `SARVAM_API_KEY` |
| Google TTS | te-IN-Standard-A (F) / te-IN-Standard-B (M) | `GOOGLE_TTS_API_KEY` |
| ElevenLabs | multilingual v2 | `ELEVENLABS_API_KEY` |
| edge-tts | te-IN-ShrutiNeural (F) / te-IN-MohanNeural (M) | Free |
| gTTS | Telugu `te` | Free |

### Kannada

| Engine | Voice | Key needed |
|--------|-------|-----------|
| Azure TTS | kn-IN-SapnaNeural (F) / kn-IN-GaganNeural (M) | `AZURE_TTS_KEY` |
| Sarvam AI | anushka (F) / abhilash (M) — `bulbul:v2` | `SARVAM_API_KEY` |
| Google TTS | kn-IN-Standard-A (F) / kn-IN-Standard-B (M) | `GOOGLE_TTS_API_KEY` |
| ElevenLabs | multilingual v2 | `ELEVENLABS_API_KEY` |
| edge-tts | kn-IN-SapnaNeural (F) / kn-IN-GaganNeural (M) | Free |
| gTTS | Kannada `kn` | Free |

### Sarvam AI speaker list

**Female:** anushka, manisha, vidya, arya, ritu, priya, neha, pooja, simra, kavya, ishita, shreya, roopa, tanya, suhani, kavitha, rupal

**Male:** abhilash, karun, hitesh, aditya, rahul, rohan, amit, dev, ratan, varun, manan, sumit, kabir, aayan, shubh, ashutosh, advait, anand, tarun, mani, gokul, vijay, mohit, rehan, soham

---

## 10. Adding Your Own Knowledge Base

Place `.txt` files in the `docs/` folder to give the agent domain knowledge.

```
# company_info.txt
Our company offers:
- Customer support: 9am–6pm
- Technical helpdesk: 24/7, dial 1800-XXX-XXXX
```

Steps:
1. Copy your `.txt` file to `docs/`
2. Delete the cached index: `rm -rf backend/faiss_index/`
3. Restart the server — it rebuilds automatically.

---

## 11. Troubleshooting

### Bot doesn't respond

```bash
curl http://localhost:11434/   # Check Ollama is running
ollama serve &                 # Start if not running
ollama list                    # Verify model is pulled
```

### No transcription / ASR not working

- With Whisper (default): first run downloads ~3 GB — watch logs for `Loading Whisper large-v3…`
- Check your `STT_ENGINE` in `.env` and verify the matching API key is set
- Run with debug: `cd backend && python3 main.py` (log level info shows which engine starts)

### No audio / TTS silent

- Check which engine is first in `TELUGU_TTS_ENGINE_PRIORITY` / `KANNADA_TTS_ENGINE_PRIORITY`
- Verify the matching API key is set in `.env`
- edge-tts and gTTS always work as final fallback (require internet)

### Azure key not working (401)

- Verify `AZURE_STT_KEY` / `AZURE_TTS_KEY` are set correctly in `.env`
- Verify region matches your endpoint: `https://<region>.api.cognitive.microsoft.com/`
- Key 1 and Key 2 from Azure Portal are interchangeable — use either

### Microphone not working in browser

- Use **Chrome** or **Edge** only
- Access via `http://localhost:8000` (not via IP address)
- Remote access: use SSH tunnel:

```bash
ssh -L 8000:localhost:8000 user@your-server-ip
# Then open http://localhost:8000
```

### CUDA out of memory

```bash
ollama pull qwen2.5:7b
# Update .env: OLLAMA_MODEL=qwen2.5:7b
```

### Port already in use

```bash
fuser -k 8000/tcp
```

---

## 12. Architecture Deep Dive

### How a conversation turn works

```
1. Browser (UI)      → User selects Telugu or Kannada from dropdown
2. WebSocket connect → ws://host/ws?language=telugu&stt_engine=auto
3. Browser mic       → AudioWorklet captures PCM at 16 kHz, mono
4. WebSocket binary  → 100ms chunks (3200 bytes) sent to server
5. ASR               → engine selected by STT_ENGINE (default: auto)
                       buffers frames, fires final transcript on silence
6. VAD interrupt     → if user speaks mid-response, stops TTS instantly
7. LLM (Qwen2.5)    → receives transcript + history + RAG context
                       system prompt: respond in Telugu/Kannada only
                       streams tokens; each sentence dispatches to TTS
8. TTS               → engine priority: azure_tts → sarvam → edge → gtts
                       streams 60ms PCM chunks (24 kHz) back over WebSocket
9. Browser speaker  → Web Audio API schedules chunks back-to-back; gapless
```

### LLM warm-up

At startup, a small dummy request forces the 72B model into GPU VRAM before any user connects. Without this, the first query would hit a 60–120 second cold-start delay. After warm-up, first-token latency is under 1 second.

### Full-duplex interrupt

While the bot is speaking, the browser continuously listens. The moment you speak:
1. Browser sends `{"type":"interrupt"}` to server
2. Server cancels TTS mid-stream
3. Browser stops audio playback immediately
4. ASR processes your new speech

### Language selection

```
Frontend dropdown  →  WebSocket URL: /ws?language=telugu
                   →  ASRHandler(language="telugu")      → STT lang="te-IN"
                   →  VoiceTTSHandler(language="telugu") → Azure/Sarvam te-IN
                   →  VoiceLLMClient(language="telugu")  → Telugu system prompt
```

---

## 13. Project Structure

```
sonixvoiceagent/
│
├── backend/
│   ├── main.py             FastAPI app + WebSocket (?language=, ?stt_engine=, ?tts_engine=)
│   ├── asr.py              ASRHandler: Sarvam → Azure → Google → Amazon → Soniox → Whisper
│   ├── llm.py              VoiceLLMClient: Ollama streaming, language-aware prompts
│   ├── tts.py              VoiceTTSHandler: Azure → Sarvam → edge-tts → gTTS priority chain
│   ├── session_manager.py  Per-session state, LLM warm-up, conversation loop
│   ├── config.py           LANGUAGE_CONFIGS for Telugu/Kannada + all settings
│   ├── memory.py           Sliding-window conversation history (20 turns)
│   ├── rag.py              FAISS RAG retriever + embeddings
│   └── faiss_index/        Auto-generated FAISS index (do not commit)
│
├── frontend/
│   └── index.html          UI: language + voice selector + full-duplex voice
│
├── docs/                   Place your .txt knowledge base files here
│
├── requirements.txt        Python dependencies
├── .env.example            All environment variables with docs — copy to .env
├── .env                    Your local config (do not commit)
├── CHANGELOG.md            Version history
└── README.md               This file
```

---

## Features

- **Telugu & Kannada** — native voice via Azure TTS neural voices (te-IN / kn-IN) and Sarvam AI bulbul:v2
- **Multi-provider STT** — Sarvam, Azure, Google, Amazon Transcribe, Soniox, Whisper — set via `STT_ENGINE` in `.env`
- **Multi-provider TTS** — Azure, Sarvam, Google, ElevenLabs, Amazon Polly, Gnani, edge-tts, gTTS — priority chain in `.env`
- **Per-session language** — switch between Telugu and Kannada from the browser dropdown
- **Full-duplex** — interrupt the bot mid-sentence; it stops and listens instantly
- **LLM warm-up** — 72B model pre-loaded into VRAM at startup; first-token latency <1s
- **Streaming pipeline** — first audio response within ~200ms of speech ending (warm model)
- **Open-source LLM** — Qwen2.5 runs locally on GPU via Ollama; zero cloud AI cost
- **RAG** — knowledge base embedded in FAISS for domain-specific answers
- **Conversation memory** — remembers the last 20 turns per session

---

## Credits

- **ASR (primary)**: [Sarvam AI](https://sarvam.ai) — `saarika:v2.5` best-in-class Indian language STT
- **ASR (cloud)**: [Azure Speech](https://azure.microsoft.com/en-us/products/ai-services/speech-service), [Google STT](https://cloud.google.com/speech-to-text), [Amazon Transcribe](https://aws.amazon.com/transcribe/), [Soniox](https://soniox.com)
- **ASR (local)**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local GPU Telugu/Kannada recognition
- **LLM**: [Ollama](https://ollama.com) + [Qwen2.5](https://qwenlm.github.io/) — open-source multilingual LLM
- **TTS (primary)**: [Azure TTS](https://azure.microsoft.com/en-us/products/ai-services/text-to-speech) — `te-IN-ShrutiNeural`, `kn-IN-SapnaNeural`
- **TTS (Indian)**: [Sarvam AI](https://sarvam.ai) — `bulbul:v2` high-quality Telugu/Kannada TTS
- **TTS (fallback)**: [edge-tts](https://github.com/rany2/edge-tts) — Microsoft Azure neural voices (free)
- **RAG**: [FAISS](https://github.com/facebookresearch/faiss) + [sentence-transformers](https://www.sbert.net/)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
