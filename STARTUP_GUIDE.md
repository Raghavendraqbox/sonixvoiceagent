# Dari & Pashto Voice Agent — Complete Startup Guide

This guide walks you through starting every component of the voice agent from scratch.  
No prior technical knowledge required.

---

## What Does Each Part Do?

| Component | What It Is | What It Does |
|-----------|-----------|--------------|
| **LLM** (Ollama + qwen2.5:7b) | The AI brain | Understands what the user said and generates a reply |
| **ASR** (Soniox or Whisper) | Speech-to-Text | Converts the user's voice into text |
| **TTS** (MMS-TTS / edge-tts) | Text-to-Speech | Converts the AI's text reply back into voice |
| **FastAPI Server** | The glue | Connects everything and serves the web interface |

---

## Step 0 — Requirements

Make sure you have these installed before starting:

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **NVIDIA GPU with CUDA** (recommended) — CPU works but will be slow
- **Internet connection** — needed for Soniox ASR and to download models on first run

Check your Python version:
```bash
python --version
```

---

## Step 1 — Install Python Dependencies

Open a terminal, go to the project folder, and run:

```bash
cd /path/to/sonixvoiceagent

# If you have a GPU, install PyTorch with CUDA first:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Then install all other dependencies:
pip install -r requirements.txt
```

> **No GPU?** Skip the torch line above and just run `pip install -r requirements.txt`.  
> Whisper will run on CPU (slower, but works).

---

## Step 2 — Start the LLM (Ollama)

Ollama runs the AI language model locally on your machine.

### 2a. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> **Windows?** Download the installer from [ollama.com](https://ollama.com)

### 2b. Start the Ollama server

Open a **new terminal window** and run:

```bash
ollama serve
```

Leave this terminal open. Ollama must keep running in the background.

### 2c. Download the AI model

Open **another terminal** and run:

```bash
ollama pull qwen2.5:7b
```

This downloads ~4.7 GB. Only needed once — it won't download again next time.

**Verify it worked:**
```bash
ollama list
```
You should see `qwen2.5:7b` in the list.

---

## Step 3 — Set Up ASR (Speech Recognition)

You have two options. Option A is better quality; Option B requires no account.

### Option A — Soniox (Recommended, Cloud-based)

Soniox gives accurate real-time Dari and Pashto transcription.

1. Sign up for a free API key at **[soniox.com/dashboard](https://soniox.com/dashboard)**
2. Copy your API key — it looks like: `1f9a7bb0dc3534ffde3192ff825d4959...`
3. You will pass it when starting the server in Step 4.

### Option B — Whisper (No Account Needed, Local GPU)

If you skip the Soniox key, the server automatically uses Whisper large-v3.

- Downloads ~3 GB on first use (automatic, no action needed)
- Requires a CUDA GPU for reasonable speed
- Slightly less accurate for Dari/Pashto than Soniox

---

## Step 4 — Start the Voice Agent Server

Navigate to the `backend` folder and start the server:

### With Soniox API key (recommended):

**Linux / Mac:**
```bash
cd /path/to/sonixvoiceagent/backend
SONIOX_API_KEY=your_key_here python main.py
```

**Windows (Command Prompt):**
```cmd
cd C:\path\to\sonixvoiceagent\backend
set SONIOX_API_KEY=your_key_here
python main.py
```

**Windows (PowerShell):**
```powershell
cd C:\path\to\sonixvoiceagent\backend
$env:SONIOX_API_KEY="your_key_here"
python main.py
```

### Without Soniox key (Whisper fallback):

```bash
cd /path/to/sonixvoiceagent/backend
python main.py
```

### What you should see on startup:

```
Dari & Pashto Voice AI Agent starting…
Default language : dari
ASR  : Soniox (stt-rt-v4) → Whisper large-v3 fallback
LLM  : Ollama qwen2.5:7b @ http://localhost:11434
TTS  : MMS-TTS (local GPU) → edge-tts → gTTS
Server ready.
Uvicorn running on http://0.0.0.0:8000
```

---

## Step 5 — Open the Web Interface

Open your browser and go to:

```
http://localhost:8000
```

Select **Dari** or **Pashto**, allow microphone access, and start speaking.

---

## How TTS (Voice Output) Works

TTS starts automatically when the server starts — no extra steps needed.  
It tries each option in order and uses the first one that works:

| Priority | Engine | Requires |
|----------|--------|----------|
| 1st | **MMS-TTS** `facebook/mms-tts-fas` (Dari) | Local GPU — downloads ~460 MB on first run |
| 2nd | **edge-tts** Microsoft neural voices | Internet connection |
| 3rd | **gTTS** Google TTS | Internet connection |

> Pashto uses edge-tts (`ps-AF-LatifaNeural`) since no MMS-TTS model exists for Pashto.

---

## Quick Reference — All Commands Together

Open **3 terminal windows** and run one command in each:

**Terminal 1 — Ollama:**
```bash
ollama serve
```

**Terminal 2 — Pull model (first time only):**
```bash
ollama pull qwen2.5:7b
```

**Terminal 3 — Voice Agent:**
```bash
cd /path/to/sonixvoiceagent/backend
SONIOX_API_KEY=your_key_here python main.py
```

Then open **http://localhost:8000** in your browser.

---

## Check Everything Is Working

Visit the health check URL:
```
http://localhost:8000/health
```

You should see something like:
```json
{
  "status": "ok",
  "asr": "soniox/stt-rt-v4",
  "llm": "ollama/qwen2.5:7b @ http://localhost:11434",
  "tts": "mms-tts (local GPU) → edge-tts → gTTS (24kHz output)"
}
```

---

## Environment Variables (Optional Customization)

You can change defaults by setting these before starting the server:

| Variable | Default | Description |
|----------|---------|-------------|
| `SONIOX_API_KEY` | _(none)_ | Soniox API key — enables cloud ASR |
| `LANGUAGE` | `dari` | Default language: `dari` or `pashto` |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Any model you have pulled via Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `SERVER_PORT` | `8000` | Port the web server listens on |
| `LOG_LEVEL` | `info` | Log verbosity: `debug`, `info`, `warning` |

Example — run on a different port with Pashto as default:
```bash
SONIOX_API_KEY=your_key LANGUAGE=pashto SERVER_PORT=9000 python main.py
```

---

## Troubleshooting

**"Connection refused" on http://localhost:8000**
- The server is not running. Go back to Step 4.

**"ollama: command not found"**
- Ollama is not installed. Go back to Step 2a.

**No voice output / TTS silent**
- Check your browser has not muted the tab.
- MMS-TTS model may still be downloading — wait a moment.
- edge-tts requires internet; check your connection.

**ASR shows Whisper instead of Soniox**
- `SONIOX_API_KEY` was not set. Check the variable name and value.

**Whisper is very slow**
- You are on CPU. A CUDA GPU is strongly recommended.
- Consider using Soniox (Option A in Step 3) which runs in the cloud.

**Ollama model not found**
- Run `ollama pull qwen2.5:7b` and wait for it to finish.

**GPU out of memory**
- Use a smaller model: `ollama pull qwen2.5:3b` and set `OLLAMA_MODEL=qwen2.5:3b`.
