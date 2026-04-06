# Sonix Voice Agent — Real-Time Full-Duplex Telugu Voice AI

A production-ready, full-duplex Telugu voice AI agent. Speak in Telugu → get an intelligent Telugu voice response in real time. Runs entirely on your own GPU — no OpenAI or cloud AI API needed.

```
Browser Mic
    │
    ▼  PCM 16kHz (WebSocket binary)
Soniox ASR  ──── Telugu Speech-to-Text ──────────────┐
    │  (fallback: Whisper large-v3)                   │
    ▼  Final transcript                               │
Qwen2.5:72b via Ollama  ── Telugu LLM response ──────┤
    │  (open-source, runs on your GPU)                │
    ▼  Text (streamed sentence by sentence)           │
edge-tts Telugu Neural Voice  ── Telugu Speech ───────┘
    │  (fallback: gTTS)
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

> **No experience needed.** Follow every step below exactly and you will have a working Telugu voice agent.

---

## 2. Step-by-Step Installation

### Step 1 — Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-pip zstd
```

### Step 2 — Clone the repository

```bash
git clone https://github.com/Raghavendraqbox/sonixvoiceagent.git
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

This installs FastAPI, Whisper, edge-tts, FAISS, sentence-transformers, and all other required packages.

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

### Step 7 — Download the Telugu LLM

Choose the model that fits your GPU VRAM (see [Section 7](#7-choosing-a-gpu--model)):

```bash
# If you have 80 GB VRAM (best quality — recommended for production)
ollama pull qwen2.5:72b

# If you have 20 GB VRAM
ollama pull qwen2.5:32b

# If you have 10 GB VRAM
ollama pull qwen2.5:14b

# If you have 6 GB VRAM (minimum)
ollama pull qwen2.5:7b
```

> The download is large (7B = ~4 GB, 72B = ~47 GB). This only happens once.

Verify the model downloaded:
```bash
ollama list
```

### Step 8 — (Optional) Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and set your model if you chose something other than the default:

```env
OLLAMA_MODEL=qwen2.5:32b
```

Everything else works without changes. See [Section 6](#6-configuration) for all options.

---

## 3. Start the Agent

```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

You should see:
```
Telugu Voice AI Agent starting…
ASR  : Soniox (te) → Whisper large-v3 fallback
LLM  : Ollama qwen2.5:72b @ http://localhost:11434
TTS  : edge-tts te-IN-ShrutiNeural → gTTS fallback
Server ready.
```

The server is now running on port `8000`.

> **First run note:** Whisper large-v3 (~3 GB) downloads automatically on the first speech input. This is a one-time download.

---

## 4. Open in Browser

Open **Chrome** or **Edge** and go to:

```
http://localhost:8000
```

1. Click **మాట్లాడడం ప్రారంభించండి** (Start Talking)
2. Allow microphone access when the browser asks
3. Speak in Telugu — the agent will respond in Telugu voice

> **Must use Chrome or Edge.** Firefox has limited Web Audio API support and may not work correctly.

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
  "asr": "whisper-large-v3 (local GPU)",
  "llm": "ollama/qwen2.5:72b @ http://localhost:11434",
  "tts": "edge-tts/te-IN-ShrutiNeural (24kHz)"
}
```

If `status` is `ok`, everything is working.

---

## 6. Configuration

All settings are controlled via environment variables. Copy `.env.example` to `.env` and edit:

```env
# ── ASR (Speech-to-Text) ──────────────────────────────────────────────────
# Optional: Soniox API key for best Telugu accuracy
# Leave empty to use Whisper large-v3 locally (no API key needed)
# Get a free key at: https://soniox.com/dashboard
SONIOX_API_KEY=

SONIOX_LANGUAGE=te
SONIOX_MODEL=soniox_multilingual_2

# ── LLM (Language Model) ──────────────────────────────────────────────────
# Ollama runs locally — no API key needed
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:72b      # Change to match what you pulled in Step 7
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=150

# ── TTS (Text-to-Speech) ──────────────────────────────────────────────────
# Telugu neural voices (free, requires internet)
# Female voice: te-IN-ShrutiNeural  ← default
# Male voice:   te-IN-MohanNeural
TTS_VOICE=te-IN-ShrutiNeural

# ── RAG (Knowledge Base) ──────────────────────────────────────────────────
RAG_DOCS_DIR=../docs           # Folder with your .txt knowledge base files
RAG_TOP_K=3                    # Number of knowledge chunks to retrieve per query

# ── Server ────────────────────────────────────────────────────────────────
SERVER_PORT=8000
LOG_LEVEL=info                 # debug | info | warning | error
```

---

## 7. Choosing a GPU & Model

| Model | VRAM Needed | Download Size | Telugu Quality | Speed |
|-------|-------------|---------------|----------------|-------|
| `qwen2.5:72b` | ~80 GB | ~47 GB | ★★★★★ Best | Medium |
| `qwen2.5:32b` | ~20 GB | ~19 GB | ★★★★★ Excellent | Fast |
| `qwen2.5:14b` | ~10 GB | ~9 GB | ★★★★☆ Very Good | Very Fast |
| `qwen2.5:7b` | ~6 GB | ~4 GB | ★★★☆☆ Good | Fastest |
| `gemma4:31b` | ~63 GB | ~21 GB | ★★★★★ Excellent | Medium |

**Which one to pick:**
- **A100 80GB** → `qwen2.5:72b` (best quality, fits perfectly)
- **RTX 3090 / 4090 (24 GB)** → `qwen2.5:32b`
- **RTX 3080 (10 GB)** → `qwen2.5:14b`
- **RTX 3060 / any 6-8 GB GPU** → `qwen2.5:7b`

After pulling a different model, update `.env`:
```env
OLLAMA_MODEL=qwen2.5:32b
```

Then restart the server.

---

## 8. ASR Options

The agent supports two ASR backends:

### Option A: Whisper large-v3 (default, no setup needed)
- Runs on your GPU locally
- No API key required
- ~3 GB model, downloads automatically on first use
- Good Telugu accuracy

### Option B: Soniox (best Telugu accuracy)
- Cloud-based streaming ASR
- **Superior** Telugu accuracy vs Whisper
- Real-time partial transcripts (words appear as you speak)
- Requires internet + free API key

To use Soniox:
1. Get a free API key at [soniox.com/dashboard](https://soniox.com/dashboard)
2. Install the package: `pip3 install soniox`
3. Add to `.env`: `SONIOX_API_KEY=your-key-here`
4. Restart the server

| | Soniox | Whisper large-v3 |
|-|--------|-----------------|
| Telugu accuracy | ★★★★★ | ★★★★☆ |
| Response latency | ~100ms | ~800ms |
| Partial results | Yes (real-time) | No |
| Requires internet | Yes | No |
| API key | Yes (free tier) | No |

---

## 9. TTS Voices

The agent uses **edge-tts** for Telugu neural voice synthesis (free, powered by Microsoft Azure).

| Voice ID | Gender | Description |
|----------|--------|-------------|
| `te-IN-ShrutiNeural` | Female | Default — natural, clear |
| `te-IN-MohanNeural` | Male | Natural male Telugu voice |

Change the voice in `.env`:
```env
TTS_VOICE=te-IN-MohanNeural
```

> edge-tts requires internet access. If the network is unavailable, gTTS (Google TTS) is used as a fallback.

---

## 10. Adding Your Own Knowledge Base

The agent uses RAG (Retrieval-Augmented Generation) to answer questions from a custom knowledge base. By default it contains Qobox company information.

To add your own content:

1. Create a `.txt` file with your knowledge in plain text (Telugu or English, or both):

```
# my_company.txt
Our company is XYZ. We provide software testing services.
మా కంపెనీ XYZ. మేము సాఫ్ట్‌వేర్ టెస్టింగ్ సేవలు అందిస్తాము.
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
ollama pull qwen2.5:72b
```

Make sure `OLLAMA_MODEL` in `.env` matches exactly what `ollama list` shows.

---

### "No transcription" / ASR not working

- **With Soniox:** Check that `SONIOX_API_KEY` is set correctly in `.env`. Try visiting [soniox.com/dashboard](https://soniox.com/dashboard) to verify your key.
- **With Whisper:** First run downloads ~3 GB. Watch server logs — you'll see `Loading model...`. Wait for it to finish.

```bash
# Check server logs for errors
cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

---

### "No audio / TTS is silent"

- edge-tts requires internet. Check your connection.
- Try a ping: `ping tts.microsoft.com`
- If no internet, gTTS (Google) is used automatically.

---

### "Microphone not working in browser"

- Use **Chrome** or **Edge** only
- Access via `http://localhost:8000` (not via IP address like `http://192.168.x.x:8000`)
- If you must access from another machine: set up HTTPS, or use an SSH tunnel:

```bash
# On your local machine — tunnel remote port 8000 to localhost
ssh -L 8000:localhost:8000 user@your-server-ip
# Then open http://localhost:8000 in your browser
```

---

### "CUDA out of memory"

Switch to a smaller model:
```bash
ollama pull qwen2.5:14b
```

Then update `.env`:
```env
OLLAMA_MODEL=qwen2.5:14b
```

---

### Port already in use

```bash
# Find what's using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>
```

---

### Server crashes on startup

Run with debug logging to see the exact error:
```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

---

## 12. Architecture Deep Dive

### How a conversation turn works

```
1. Browser mic       → AudioWorklet captures PCM at 16 kHz, mono
2. WebSocket binary  → 100ms chunks (3200 bytes) sent to server
3. ASR (Soniox/Whisper) → streams partial transcripts back; fires final on silence
4. VAD interrupt     → if user speaks mid-response, {"type":"interrupt"} stops TTS instantly
5. LLM (Qwen2.5)    → receives final transcript + conversation history + RAG context
                       streams tokens; each complete sentence dispatches to TTS
6. TTS (edge-tts)   → synthesises each sentence fragment in real time
                       streams 60ms PCM chunks (24 kHz) back over WebSocket
7. Browser speaker  → Web Audio API schedules chunks back-to-back; gapless playback
```

### Full-duplex interrupt flow

While the bot is speaking, the browser continuously listens for your voice. The moment you speak:
1. Browser detects audio energy above threshold
2. Sends `{"type":"interrupt"}` to server
3. Server cancels TTS generation mid-stream
4. Browser stops audio playback immediately
5. ASR starts processing your new speech

This gives a natural conversational feel — you never have to wait for the bot to finish.

### RAG pipeline

At startup, all `.txt` files in `docs/` are:
1. Split into 300-token chunks with 50-token overlap
2. Embedded using `sentence-transformers/all-MiniLM-L6-v2`
3. Stored in a FAISS index on disk

On each user query, the top-3 most relevant chunks are retrieved and injected into the LLM system prompt.

---

## 13. Project Structure

```
sonixvoiceagent/
│
├── backend/
│   ├── main.py             FastAPI app + WebSocket endpoint
│   ├── asr.py              Soniox streaming ASR → Whisper large-v3 fallback
│   ├── llm.py              Ollama Qwen2.5 client (streaming)
│   ├── tts.py              edge-tts Telugu TTS → gTTS fallback
│   ├── session_manager.py  Per-session state, interrupt logic, pipeline wiring
│   ├── config.py           All settings (env-var overridable)
│   ├── memory.py           Sliding-window conversation history (8 turns)
│   ├── rag.py              FAISS RAG retriever + embeddings
│   └── faiss_index/        Auto-generated FAISS index (do not commit)
│
├── frontend/
│   └── index.html          Single-file browser client (Web Audio API + WebSocket)
│
├── docs/
│   └── qobox_telugu.txt    Default Qobox knowledge base (Telugu + English)
│
├── requirements.txt        Python dependencies
├── .env.example            Environment variable template
├── CHANGELOG.md            Version history
└── README.md               This file
```

---

## Features

- **Full-duplex** — interrupt the bot mid-sentence; it stops and listens instantly
- **Streaming pipeline** — first audio response within ~200ms of speech ending
- **Telugu-first** — native Telugu voice via edge-tts neural models
- **Bilingual** — responds in Telugu when spoken to in Telugu, English in English
- **Open-source LLM** — Qwen2.5 runs locally on GPU via Ollama; zero cloud AI cost
- **RAG** — Qobox knowledge base embedded in FAISS for accurate company-specific answers
- **Conversation memory** — remembers the last 8 turns per session
- **Robust fallbacks** — Soniox → Whisper | edge-tts → gTTS | every component has a fallback

---

## Credits

- **ASR**: [Soniox](https://soniox.com) — streaming Telugu speech recognition
- **LLM**: [Ollama](https://ollama.com) + [Qwen2.5](https://qwenlm.github.io/) — open-source multilingual LLM
- **TTS**: [edge-tts](https://github.com/rany2/edge-tts) — Microsoft neural Telugu voices
- **RAG**: [FAISS](https://github.com/facebookresearch/faiss) + [sentence-transformers](https://www.sbert.net/)
- Built on architecture from [voiceagentcloud](https://github.com/Raghavendraqbox/voiceagentcloud)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
