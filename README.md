# Sonix Voice Agent — Real-Time Full-Duplex Dari & Pashto Voice AI

A production-ready, full-duplex voice AI agent for **Dari** (دری) and **Pashto** (پښتو). Speak in Dari or Pashto → get an intelligent voice response in real time. Runs entirely on your own GPU — no OpenAI or cloud AI API needed.

```
Browser Mic
    │
    ▼  PCM 16kHz (WebSocket binary)
Soniox ASR  ──── Dari / Pashto Speech-to-Text ─────────────┐
    │  (fallback: Whisper large-v3)                         │
    ▼  Final transcript                                     │
Qwen2.5 via Ollama  ── Dari / Pashto LLM response ─────────┤
    │  (open-source, runs on your GPU)                      │
    ▼  Text (streamed sentence by sentence)                 │
Meta MMS-TTS  ── Dari / Pashto Neural Speech ───────────────┘
    │  facebook/mms-tts-prs (Dari)
    │  facebook/mms-tts-pbt (Pashto)
    │  (fallback: edge-tts → gTTS)
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

> **No experience needed.** Follow every step below exactly and you will have a working Dari/Pashto voice agent.

---

## 2. Step-by-Step Installation

### Step 1 — Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-pip zstd
```

### Step 2 — Clone the repository

```bash
git clone -b daripastho https://github.com/Raghavendraqbox/sonixvoiceagent.git
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

This installs FastAPI, Whisper, Meta MMS-TTS, uroman, edge-tts, FAISS, sentence-transformers, and all other required packages.

> **Note:** The `uroman` package is required for Dari/Pashto MMS-TTS models. It is included in `requirements.txt` and installed automatically.

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
# If you have 6 GB VRAM (default — good Dari, reasonable Pashto)
ollama pull qwen2.5:7b

# If you have 10 GB VRAM (better quality)
ollama pull qwen2.5:14b

# If you have 20 GB VRAM (best Dari quality)
ollama pull qwen2.5:32b

# If you have 80 GB VRAM (best overall)
ollama pull qwen2.5:72b
```

> The download is large (7B = ~4 GB, 72B = ~47 GB). This only happens once.

Verify the model downloaded:
```bash
ollama list
```

### Step 8 — Set the default language (optional)

```bash
cp .env.example .env
```

Open `.env` and set your preferred default language:

```env
LANGUAGE=dari    # or: pashto
OLLAMA_MODEL=qwen2.5:7b
```

The language can also be changed per-session from the browser UI dropdown. See [Section 6](#6-configuration) for all options.

---

## 3. Start the Agent

```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

You should see:
```
Dari & Pashto Voice AI Agent starting…
Default language : dari
Supported        : dari, pashto
ASR  : Soniox (stt-rt-v4) → Whisper large-v3 fallback
LLM  : Ollama qwen2.5:7b @ http://localhost:11434
TTS  : MMS-TTS (local GPU) → edge-tts → gTTS
Server ready.
```

> **First run note:** The MMS-TTS model for your language (~460 MB) downloads automatically from HuggingFace on first use. Whisper large-v3 (~3 GB) also downloads automatically on first speech input.

---

## 4. Open in Browser

Open **Chrome** or **Edge** and go to:

```
http://localhost:8000
```

1. Select your language from the **Language** dropdown: **Dari (دری)** or **Pashto (پښتو)**
2. Click **Start Conversation**
3. Allow microphone access when the browser asks
4. Speak in Dari or Pashto — the agent will respond in the same language

> **Must use Chrome or Edge.** Firefox has limited Web Audio API support.

> **Must be on `localhost` or HTTPS.** Browsers block microphone on plain HTTP for remote IPs. If accessing from another machine, set up HTTPS or use an SSH tunnel.

---

## 5. Health Check

At any time, check that all components are running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "supported_languages": ["dari", "pashto"],
  "default_language": "dari",
  "asr": "whisper-large-v3 (local GPU)",
  "llm": "ollama/qwen2.5:7b @ http://localhost:11434",
  "tts": "dari: mms-tts strict | pashto: configurable chain (24kHz output)"
}
```

List available languages:
```bash
curl http://localhost:8000/languages
```

---

## 6. Configuration

All settings are controlled via environment variables. Copy `.env.example` to `.env` and edit:

```env
# ── Language ──────────────────────────────────────────────────────────────
# Server-side default language (also selectable per-session in the UI)
LANGUAGE=dari                   # Options: dari | pashto

# ── ASR (Speech-to-Text) ──────────────────────────────────────────────────
# Optional: Soniox API key for best accuracy
# Leave empty to use Whisper large-v3 locally (no API key needed)
# Get a free key at: https://soniox.com/dashboard
SONIOX_API_KEY=
SONIOX_MODEL=stt-rt-v4          # Current recommended Soniox model

# ── LLM (Language Model) ──────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b         # Change to match what you pulled in Step 7
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=150

# ── TTS voices (edge-tts fallback — optional overrides) ───────────────────
# Dari voices (fa-IR locale):
#   fa-IR-DilaraNeural  ← female (default)
#   fa-IR-FaridNeural   ← male
TTS_VOICE_DARI=fa-IR-DilaraNeural

# Pashto voices (ps-AF locale):
#   ps-AF-LatifaNeural    ← female (default)
#   ps-AF-GulNawazNeural  ← male
TTS_VOICE_PASHTO=ps-AF-LatifaNeural

# ── RAG (Knowledge Base) ──────────────────────────────────────────────────
RAG_DOCS_DIR=../docs            # Folder with your .txt knowledge base files
RAG_TOP_K=3                     # Number of knowledge chunks per query

# ── Server ────────────────────────────────────────────────────────────────
SERVER_PORT=8000
LOG_LEVEL=info                  # debug | info | warning | error
```

---

## 7. Choosing a GPU & Model

| Model | VRAM | Download | Dari Quality | Pashto Quality | Speed |
|-------|------|----------|-------------|----------------|-------|
| `qwen2.5:7b` | ~6 GB | ~4 GB | ★★★☆☆ Good | ★★★☆☆ Reasonable | Fastest |
| `qwen2.5:14b` | ~10 GB | ~9 GB | ★★★★☆ Very Good | ★★★☆☆ Good | Very Fast |
| `qwen2.5:32b` | ~20 GB | ~19 GB | ★★★★★ Excellent | ★★★★☆ Good | Fast |
| `qwen2.5:72b` | ~48 GB | ~47 GB | ★★★★★ Best | ★★★★☆ Very Good | Medium |
| `aya-expanse` | varies | varies | ★★★★☆ Good | ★★★☆☆ Reasonable | Medium |

**Which one to pick:**
- **A100 80GB** → `qwen2.5:72b` (best quality)
- **RTX 3090 / 4090 (24 GB)** → `qwen2.5:32b`
- **RTX 3080 (10 GB)** → `qwen2.5:14b`
- **RTX 3060 / any 6-8 GB GPU** → `qwen2.5:7b` (default)

After pulling a different model, update `.env`:
```env
OLLAMA_MODEL=qwen2.5:32b
```
Then restart the server.

> **Pashto tip:** For the best Pashto responses, the community model `junaid008/qehwa-pashto-llm` (Qwen2.5-7B fine-tuned for Pashto, Apache 2.0) can be converted to a GGUF Ollama Modelfile for improved Pashto generation.

---

## 8. ASR Options

The agent supports two ASR backends per language:

### Option A: Whisper large-v3 (default, no setup needed)
- Runs on your GPU locally
- No API key required
- ~3 GB model, downloads automatically on first use
- Dari → `language="fa"` | Pashto → `language="ps"`

### Option B: Soniox (best accuracy)
- Cloud-based streaming ASR
- **Superior** accuracy and real-time partial transcripts
- Dari → `language_code="fa"` | Pashto → `language_code="ps"`
- Requires internet + free API key

To enable Soniox:
1. Get a free API key at [soniox.com/dashboard](https://soniox.com/dashboard)
2. Add to `.env`: `SONIOX_API_KEY=your-key-here`
3. Restart the server

| | Soniox | Whisper large-v3 |
|-|--------|-----------------|
| Accuracy | ★★★★★ | ★★★★☆ |
| Response latency | ~100ms | ~800ms |
| Partial results | Yes (real-time) | No |
| Requires internet | Yes | No |
| API key | Yes (free tier) | No |

---

## 9. TTS Voices

### Primary: Meta MMS-TTS (local GPU — no internet required)

| Language | Model | Type |
|----------|-------|------|
| Dari | `facebook/mms-tts-prs` | Afghan Persian VITS |
| Pashto | `facebook/mms-tts-pbt` | Southern Pashto VITS |

Models download automatically from HuggingFace (~460 MB each) on first use. Both models require `uroman` for romanisation — installed automatically via `requirements.txt`.

### Fallback 1: edge-tts (Microsoft Azure — free, internet required)

| Language | Female Voice | Male Voice |
|----------|-------------|------------|
| Dari | `fa-IR-DilaraNeural` | `fa-IR-FaridNeural` |
| Pashto | `ps-AF-LatifaNeural` | `ps-AF-GulNawazNeural` |

Override in `.env`:
```env
TTS_VOICE_DARI=fa-IR-FaridNeural      # Switch to male Dari voice
TTS_VOICE_PASHTO=ps-AF-GulNawazNeural # Switch to male Pashto voice
```

### Fallback 2: gTTS (Google TTS — free, internet required)
Uses Persian (`fa`) for both languages as the closest available Google voice.

---

## 10. Adding Your Own Knowledge Base

The agent uses RAG (Retrieval-Augmented Generation) to answer questions from a custom knowledge base. By default it contains Qobox company information in English, Dari, and Pashto.

To add your own content:

1. Create a `.txt` file in English (the LLM will respond in the user's language):

```
# my_company.txt
Our company is XYZ. We provide software testing services.
Contact us at info@xyz.com or call +1-555-000-0000.
```

2. Place the file in the `docs/` folder:

```bash
cp my_company.txt docs/
```

3. Delete the cached FAISS index so it rebuilds:

```bash
rm -rf backend/faiss_index/
```

4. Restart the server — it will automatically embed and index your new content.

---

## 11. Troubleshooting

### "Bot doesn't respond" / LLM not working

```bash
# Check if Ollama is running
curl http://localhost:11434/

# If not running, start it
ollama serve &

# Check which models are available
ollama list

# If your model isn't listed, pull it
ollama pull qwen2.5:7b
```

Make sure `OLLAMA_MODEL` in `.env` matches exactly what `ollama list` shows.

---

### "No transcription" / ASR not working

- **With Soniox:** Check that `SONIOX_API_KEY` is set correctly in `.env`. Verify your key at [soniox.com/dashboard](https://soniox.com/dashboard).
- **With Whisper:** First run downloads ~3 GB. Watch server logs — you'll see `Loading Whisper large-v3 for Dari…`. Wait for it to finish.

```bash
cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

---

### "No audio / TTS is silent"

- **MMS-TTS:** Model downloads ~460 MB from HuggingFace on first use. Check server logs for `Loading facebook/mms-tts-prs…`.
- **uroman missing:** Run `pip3 install uroman` — required for Dari/Pashto MMS-TTS models.
- **edge-tts:** Requires internet. Check your connection.
- If all else fails, gTTS (Google TTS) is used automatically.

---

### "Microphone not working in browser"

- Use **Chrome** or **Edge** only
- Access via `http://localhost:8000` (not via IP address)
- If you must access from another machine, use an SSH tunnel:

```bash
# On your local machine
ssh -L 8000:localhost:8000 user@your-server-ip
# Then open http://localhost:8000
```

---

### "CUDA out of memory"

Switch to a smaller model:
```bash
ollama pull qwen2.5:7b
```
Then update `.env`:
```env
OLLAMA_MODEL=qwen2.5:7b
```

---

### "uroman error on startup"

```bash
pip3 install uroman
```

The `uroman` package is required for Dari (`facebook/mms-tts-prs`) and Pashto (`facebook/mms-tts-pbt`) TTS models. It should be installed by `pip install -r requirements.txt`, but can be installed manually.

---

### Port already in use

```bash
lsof -i :8000
kill -9 <PID>
```

---

## 12. Architecture Deep Dive

### How a conversation turn works

```
1. Browser (UI)      → User selects Dari or Pashto from dropdown
2. WebSocket connect → ws://host/ws?language=dari (or pashto)
3. Browser mic       → AudioWorklet captures PCM at 16 kHz, mono
4. WebSocket binary  → 100ms chunks (3200 bytes) sent to server
5. ASR               → Soniox (fa/ps) or Whisper (fa/ps)
                       streams partial transcripts; fires final on silence
6. VAD interrupt     → if user speaks mid-response, stops TTS instantly
7. LLM (Qwen2.5)    → receives final transcript + history + RAG context
                       system prompt instructs: respond in Dari/Pashto script
                       streams tokens; each sentence dispatches to TTS
8. TTS (MMS-TTS)    → facebook/mms-tts-prs (Dari) or mms-tts-pbt (Pashto)
                       uroman romanises input text automatically
                       streams 60ms PCM chunks (24 kHz) back over WebSocket
9. Browser speaker  → Web Audio API schedules chunks back-to-back; gapless
```

### Full-duplex interrupt flow

While the bot is speaking, the browser continuously listens for your voice. The moment you speak:
1. Browser detects audio energy above threshold
2. Sends `{"type":"interrupt"}` to server
3. Server cancels TTS generation mid-stream
4. Browser stops audio playback immediately
5. ASR starts processing your new speech

### Language selection flow

```
Frontend dropdown  →  WebSocket URL: /ws?language=dari
                   →  server validates language
                   →  ASRHandler(language="dari")  → soniox_lang="fa", whisper="fa"
                   →  VoiceTTSHandler(language="dari") → model=mms-tts-prs
                   →  VoiceLLMClient(language="dari")  → system_prompt in Dari
```

### RAG pipeline

At startup, all `.txt` files in `docs/` are:
1. Split into 300-token chunks with 50-token overlap
2. Embedded using `sentence-transformers/all-MiniLM-L6-v2`
3. Stored in a FAISS index on disk

On each user query, the top-3 most relevant chunks are retrieved and injected into the LLM system prompt. The LLM then responds in Dari or Pashto using that context.

---

## 13. Project Structure

```
sonixvoiceagent/
│
├── backend/
│   ├── main.py             FastAPI app + WebSocket (?language=dari/pashto)
│   ├── asr.py              ASRHandler: Soniox → Whisper large-v3 fallback
│   ├── llm.py              VoiceLLMClient: Ollama streaming, language-aware prompts
│   ├── tts.py              VoiceTTSHandler: MMS-TTS → edge-tts → gTTS
│   ├── session_manager.py  Per-session state, language wiring, conversation loop
│   ├── config.py           LANGUAGE_CONFIGS for Dari/Pashto + all settings
│   ├── memory.py           Sliding-window conversation history (8 turns)
│   ├── rag.py              FAISS RAG retriever + embeddings
│   └── faiss_index/        Auto-generated FAISS index (do not commit)
│
├── frontend/
│   └── index.html          English UI: language selector + full-duplex voice
│
├── docs/
│   ├── qobox_company_info.txt   Qobox KB in English + Dari + Pashto
│   └── qobox_telugu.txt         Legacy Telugu KB (still indexed by RAG)
│
├── .claude/
│   └── settings.local.json      Claude Code project permissions
│
├── requirements.txt        Python dependencies (includes uroman)
├── .env.example            Environment variable template
├── CHANGELOG.md            Version history
└── README.md               This file
```

---

## Features

- **Dari & Pashto** — native voice synthesis via Meta MMS-TTS (local GPU, no internet needed for TTS)
- **English UI** — all interface text in English; transcript cards render RTL Arabic/Pashto script
- **Per-session language** — switch between Dari and Pashto from the browser dropdown before connecting
- **Full-duplex** — interrupt the bot mid-sentence; it stops and listens instantly
- **Streaming pipeline** — first audio response within ~200ms of speech ending
- **Open-source LLM** — Qwen2.5 runs locally on GPU via Ollama; zero cloud AI cost
- **RAG** — bilingual knowledge base (English + Dari + Pashto) embedded in FAISS
- **Conversation memory** — remembers the last 8 turns per session
- **Robust fallbacks** — MMS-TTS → edge-tts → gTTS | Soniox → Whisper | every component has a fallback

---

## Credits

- **ASR**: [Soniox](https://soniox.com) — streaming Dari/Pashto speech recognition
- **LLM**: [Ollama](https://ollama.com) + [Qwen2.5](https://qwenlm.github.io/) — open-source multilingual LLM
- **TTS**: [Meta MMS-TTS](https://huggingface.co/facebook/mms-tts) — VITS models for Dari & Pashto
- **TTS fallback**: [edge-tts](https://github.com/rany2/edge-tts) — Microsoft neural voices
- **RAG**: [FAISS](https://github.com/facebookresearch/faiss) + [sentence-transformers](https://www.sbert.net/)
- **uroman**: [USC ISI uroman](https://github.com/isi-nlp/uroman) — script romanisation for MMS-TTS

---

## License

MIT License — see [LICENSE](LICENSE) for details.
