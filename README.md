# Sonix Voice Agent — Real-Time Telugu & Kannada Voice AI

A production-ready, full-duplex voice AI agent for **Telugu** (తెలుగు) and **Kannada** (ಕನ್ನಡ). Speak in Telugu or Kannada and get an intelligent voice response in real time. Runs entirely on your own GPU — no OpenAI or cloud AI API needed.

```
Browser Mic
    │
    ▼  PCM 16kHz (WebSocket binary)
Whisper large-v3  ── Telugu / Kannada Speech-to-Text ────────┐
    │  (GPU, local — no API key required)                     │
    ▼  Final transcript                                       │
Qwen2.5:72b via Ollama  ── Telugu / Kannada LLM response ────┤
    │  (open-source, runs on your GPU)                        │
    ▼  Text (streamed sentence by sentence)                   │
Sarvam AI / edge-tts / gTTS  ── Neural Speech ───────────────┘
    │  Telugu  : Sarvam bulbul:v2 → edge-tts → gTTS
    │  Kannada : edge-tts → gTTS (configurable priority)
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

This installs FastAPI, faster-whisper, Meta MMS-TTS, edge-tts, FAISS, sentence-transformers, and all other required packages.

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
# If you have 6 GB VRAM (good multilingual quality)
ollama pull qwen2.5:7b

# If you have 10 GB VRAM (better quality)
ollama pull qwen2.5:14b

# If you have 20 GB VRAM (best quality)
ollama pull qwen2.5:32b

# If you have 80 GB VRAM (best overall — recommended)
ollama pull qwen2.5:72b
```

> The download is large (7B ≈ 4 GB, 72B ≈ 47 GB). This only happens once.

Verify the model downloaded:
```bash
ollama list
```

### Step 8 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your language and model:

```env
LANGUAGE=telugu          # or: kannada
OLLAMA_MODEL=qwen2.5:72b # match what you pulled above
```

The language can also be changed per-session from the browser UI dropdown.

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
ASR  : Soniox (stt-rt-v4) → Whisper large-v3 fallback
LLM  : Ollama qwen2.5:72b @ http://localhost:11434
LLM warm-up starting — loading qwen2.5:72b into VRAM…
LLM warm-up complete — qwen2.5:72b ready (0.3s, first-token latency will now be <1s)
Server ready.
```

> **LLM warm-up:** On the very first start after pulling the model, warm-up takes 60–120 seconds while the 72B model loads from disk into VRAM. Subsequent starts are near-instant because the model stays cached by Ollama. The server reports "Server ready." only after the warm-up completes — users will never hit the cold-start delay.

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
  "supported_languages": ["telugu", "kannada"],
  "default_language": "telugu",
  "asr": "whisper-large-v3 (local GPU)",
  "llm": "ollama/qwen2.5:72b @ http://localhost:11434",
  "tts": "telugu: mms-tts strict | kannada: configurable chain (24kHz output)"
}
```

List available languages:
```bash
curl http://localhost:8000/languages
```

---

## 6. Configuration

All settings are in `.env` (copy from `.env.example`):

```env
# ── Language ──────────────────────────────────────────────────────────────
LANGUAGE=telugu                 # Options: telugu | kannada

# ── ASR (Speech-to-Text) ──────────────────────────────────────────────────
# Leave SONIOX_API_KEY empty to use Whisper large-v3 locally (recommended)
# Set a valid key from https://soniox.com/dashboard to use Soniox cloud ASR
SONIOX_API_KEY=
SONIOX_MODEL=stt-rt-v4

# ── LLM (Language Model) ──────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:72b        # Change to match what you pulled
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=400           # Higher = more complete sentences; lower = faster

# ── TTS — Sarvam AI (primary for Telugu) ──────────────────────────────────
# Get a key at: https://sarvam.ai
SARVAM_API_KEY=your-sarvam-key-here
SARVAM_SPEAKER_TELUGU=anushka           # Female Telugu voice
SARVAM_SPEAKER_TELUGU_MALE=abhilash     # Male Telugu voice
SARVAM_MODEL=bulbul:v2
SARVAM_PACE=1.0                         # 0.5 (slow) → 1.0 (normal) → 2.0 (fast)

# ── Telugu TTS engine priority ────────────────────────────────────────────
# Comma-separated list tried in order until one succeeds.
# Options: sarvam | edge | gtts
TELUGU_TTS_ENGINE_PRIORITY=sarvam,edge,gtts

# ── Kannada TTS engine priority ───────────────────────────────────────────
# Options: mms | edge | gtts
KANNADA_TTS_ENGINE_PRIORITY=mms,edge,gtts

# ── TTS voices (edge-tts fallback) ────────────────────────────────────────
TTS_VOICE_TELUGU=te-IN-ShrutiNeural     # Female Telugu
TTS_VOICE_KANNADA=kn-IN-SapnaNeural    # Female Kannada

# ── RAG (Knowledge Base) ──────────────────────────────────────────────────
RAG_DOCS_DIR=./docs
RAG_TOP_K=3

# ── Server ────────────────────────────────────────────────────────────────
SERVER_PORT=8000
LOG_LEVEL=info
```

---

## 7. Choosing a GPU & Model

| Model | VRAM | Download | Telugu Quality | Kannada Quality | Speed |
|-------|------|----------|----------------|-----------------|-------|
| `qwen2.5:7b` | ~6 GB | ~4 GB | ★★★☆☆ Good | ★★★☆☆ Good | Fastest |
| `qwen2.5:14b` | ~10 GB | ~9 GB | ★★★★☆ Very Good | ★★★★☆ Good | Very Fast |
| `qwen2.5:32b` | ~20 GB | ~19 GB | ★★★★★ Excellent | ★★★★☆ Very Good | Fast |
| `qwen2.5:72b` | ~48 GB | ~47 GB | ★★★★★ Best | ★★★★★ Best | Medium |

**Which one to pick:**
- **A100 80GB** → `qwen2.5:72b` (best quality, recommended)
- **RTX 3090 / 4090 (24 GB)** → `qwen2.5:32b`
- **RTX 3080 (10 GB)** → `qwen2.5:14b`
- **RTX 3060 / any 6–8 GB GPU** → `qwen2.5:7b`

After pulling a different model, update `.env`:
```env
OLLAMA_MODEL=qwen2.5:32b
```
Then restart the server.

---

## 8. ASR Options

### Option A: Whisper large-v3 (default, recommended — no setup needed)

- Runs on your GPU locally
- No API key required
- ~3 GB model, downloads automatically on first use
- Telugu → `language="te"` | Kannada → `language="kn"`
- Starts in ~2 seconds on sessions after first download

### Option B: Soniox (optional — better real-time accuracy)

- Cloud-based streaming ASR
- Real-time partial transcripts while you speak
- Requires internet + free API key from [soniox.com/dashboard](https://soniox.com/dashboard)
- Telugu → `language_code="te"` | Kannada → `language_code="kn"`

To enable Soniox:
1. Get a free API key at [soniox.com/dashboard](https://soniox.com/dashboard)
2. Add to `.env`: `SONIOX_API_KEY=your-key-here`
3. Restart the server

| | Whisper large-v3 | Soniox |
|-|-----------------|--------|
| Accuracy | ★★★★☆ | ★★★★★ |
| Response latency | ~800ms | ~100ms |
| Partial results | No | Yes (real-time) |
| Requires internet | No | Yes |
| API key | No | Yes (free tier) |

> **Default is Whisper.** If `SONIOX_API_KEY` is empty or not set, the server skips Soniox entirely and goes straight to Whisper — no delay, no failed connection attempt.

---

## 9. TTS Voices

### Telugu — Primary: Sarvam AI (`bulbul:v2`)

Sarvam AI provides the highest-quality Telugu neural TTS. Requires a free API key from [sarvam.ai](https://sarvam.ai).

**Female speakers:** anushka, manisha, vidya, arya, ritu, priya, neha, pooja, simra, kavya, ishita, shreya, roopa, tanya, sunny, suhani, kavitha, rupal

**Male speakers:** abhilash, karun, hitesh, aditya, rahul, rohan, amit, dev, ratan, varun, manan, sumit, kabir, aayan, shubh, ashutosh, advait, anand, tarun, mani, gokul, vijay, mohit, rehan, soham

Set in `.env`:
```env
SARVAM_API_KEY=your-key-here
SARVAM_SPEAKER_TELUGU=anushka       # default female
SARVAM_SPEAKER_TELUGU_MALE=abhilash # default male
```

If Sarvam is not configured, Telugu falls back to **edge-tts** (Microsoft Azure free):
- Female: `te-IN-ShrutiNeural`
- Male: `te-IN-MohanNeural`

### Kannada — Primary: Meta MMS-TTS (local GPU)

Kannada uses `facebook/mms-tts-kan` — a local VITS model that runs on your GPU. Downloads ~460 MB from HuggingFace automatically on first use. No internet required after download.

Falls back to **edge-tts** if MMS fails:
- Female: `kn-IN-SapnaNeural`
- Male: `kn-IN-GaganNeural`

### TTS engine priority

Control the fallback chain via `.env`:
```env
TELUGU_TTS_ENGINE_PRIORITY=sarvam,edge,gtts
KANNADA_TTS_ENGINE_PRIORITY=mms,edge,gtts
```

---

## 10. Adding Your Own Knowledge Base

The agent uses RAG (Retrieval-Augmented Generation) to answer questions from a custom knowledge base. Place `.txt` files in the `docs/` folder to give the agent domain knowledge.

Example:
```
# company_info.txt
Our company offers the following services:
- Customer support: available 9am–6pm
- Technical helpdesk: available 24/7, dial 1800-XXX-XXXX
- Billing queries: email billing@example.com
```

Steps:
1. Copy your `.txt` file to `docs/`
2. Delete the cached FAISS index so it rebuilds:
   ```bash
   rm -rf backend/faiss_index/
   ```
3. Restart the server — it will automatically embed and index your content.

The LLM reads retrieved context and responds in Telugu or Kannada regardless of the language the knowledge base is written in.

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

> **First start can take 60–120 seconds** — the 72B model loads from disk into VRAM. The server logs `LLM warm-up starting…` and only says `Server ready.` after the model is fully loaded. This wait only happens once per cold start.

---

### "No transcription" / ASR not working

With Whisper (default): first run downloads ~3 GB. Watch server logs for `Loading Whisper large-v3 for Telugu…`. Wait for it to finish.

```bash
cd backend && python3 main.py --log-level debug
```

---

### "No audio / TTS is silent"

- **Sarvam (Telugu):** Check that `SARVAM_API_KEY` is set in `.env`. Get a key at [sarvam.ai](https://sarvam.ai).
- **MMS-TTS (Kannada):** Model downloads ~460 MB from HuggingFace on first use. Check server logs.
- **edge-tts:** Requires internet. Check your connection.
- gTTS is used automatically as a final fallback.

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

### Port already in use

```bash
fuser -k 8000/tcp
```

---

## 12. Architecture Deep Dive

### How a conversation turn works

```
1. Browser (UI)      → User selects Telugu or Kannada from dropdown
2. WebSocket connect → ws://host/ws?language=telugu (or kannada)
3. Browser mic       → AudioWorklet captures PCM at 16 kHz, mono
4. WebSocket binary  → 100ms chunks (3200 bytes) sent to server
5. ASR               → Whisper large-v3 (te/kn)
                       buffers frames, fires final transcript on silence
6. VAD interrupt     → if user speaks mid-response, stops TTS instantly
7. LLM (Qwen2.5)    → receives final transcript + history + RAG context
                       system prompt instructs: respond in Telugu/Kannada
                       streams tokens; each sentence dispatches to TTS
8. TTS               → Telugu: Sarvam → edge-tts → gTTS (priority chain)
                       Kannada: MMS-TTS → edge-tts → gTTS
                       streams 60ms PCM chunks (24 kHz) back over WebSocket
9. Browser speaker  → Web Audio API schedules chunks back-to-back; gapless
```

### LLM warm-up (startup)

On server startup, `session_manager.warmup_llm()` fires a small dummy request to Ollama. This forces the 72B model to load its 44 GB of layers into GPU VRAM before any user connects. Without this, the first user query would hit a 60–120 second cold-start delay. After warm-up, first-token latency is under 1 second.

### Full-duplex interrupt flow

While the bot is speaking, the browser continuously listens for your voice. The moment you speak:
1. Browser detects audio energy above threshold
2. Sends `{"type":"interrupt"}` to server
3. Server cancels TTS generation mid-stream
4. Browser stops audio playback immediately
5. ASR starts processing your new speech

### Barge-in transcript drain

If the user speaks multiple times while the LLM is still generating (e.g., "hello?" "can you hear me?"), those utterances queue up. When the LLM finishes, only the **most recent** utterance is processed — all intermediate ones are discarded. This prevents a flood of stale questions being answered sequentially.

### Language selection flow

```
Frontend dropdown  →  WebSocket URL: /ws?language=telugu
                   →  server validates language
                   →  ASRHandler(language="telugu")  → whisper lang="te"
                   →  VoiceTTSHandler(language="telugu") → Sarvam bulbul:v2
                   →  VoiceLLMClient(language="telugu")  → system prompt in Telugu
```

### RAG pipeline

At startup, all `.txt` files in `docs/` are:
1. Split into 300-token chunks with 50-token overlap
2. Embedded using `sentence-transformers/all-MiniLM-L6-v2`
3. Stored in a FAISS index on disk

On each user query, the top-3 most relevant chunks are retrieved and injected into the LLM prompt. The LLM responds in Telugu or Kannada using that context.

---

## 13. Project Structure

```
sonixvoiceagent/
│
├── backend/
│   ├── main.py             FastAPI app + WebSocket (?language=telugu/kannada)
│   ├── asr.py              ASRHandler: Whisper large-v3 (primary) → Soniox (optional)
│   ├── llm.py              VoiceLLMClient: Ollama streaming, language-aware prompts
│   ├── tts.py              VoiceTTSHandler: Sarvam / MMS-TTS / edge-tts / gTTS
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
│                           (empty by default — add your own domain content)
│
├── requirements.txt        Python dependencies
├── .env.example            Environment variable template
├── .env                    Your local config (do not commit)
├── CHANGELOG.md            Version history
└── README.md               This file
```

---

## Features

- **Telugu & Kannada** — native voice synthesis via Sarvam AI (Telugu) and Meta MMS-TTS (Kannada)
- **English UI** — all interface text in English; transcript cards render Telugu/Kannada script
- **Per-session language** — switch between Telugu and Kannada from the browser dropdown
- **Full-duplex** — interrupt the bot mid-sentence; it stops and listens instantly
- **Barge-in drain** — only the most recent utterance is processed when user speaks during LLM generation
- **LLM warm-up** — 72B model pre-loaded into VRAM at startup; first-token latency <1s for all users
- **Streaming pipeline** — first audio response within ~200ms of speech ending (warm model)
- **Open-source LLM** — Qwen2.5 runs locally on GPU via Ollama; zero cloud AI cost
- **RAG** — knowledge base embedded in FAISS for domain-specific answers
- **Conversation memory** — remembers the last 20 turns per session
- **Robust fallbacks** — Sarvam → edge-tts → gTTS (Telugu) | MMS-TTS → edge-tts → gTTS (Kannada) | Whisper (default ASR) → Soniox (optional)

---

## Credits

- **ASR**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local GPU Telugu/Kannada speech recognition
- **ASR (optional)**: [Soniox](https://soniox.com) — streaming cloud ASR
- **LLM**: [Ollama](https://ollama.com) + [Qwen2.5](https://qwenlm.github.io/) — open-source multilingual LLM
- **TTS (Telugu)**: [Sarvam AI](https://sarvam.ai) — `bulbul:v2` high-quality Telugu neural TTS
- **TTS (Kannada)**: [Meta MMS-TTS](https://huggingface.co/facebook/mms-tts) — `facebook/mms-tts-kan` local GPU model
- **TTS (fallback)**: [edge-tts](https://github.com/rany2/edge-tts) — Microsoft Azure neural voices
- **RAG**: [FAISS](https://github.com/facebookresearch/faiss) + [sentence-transformers](https://www.sbert.net/)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
