# First Time Setup — Dari & Pashto Voice Agent

Follow this guide once on any new machine. After this, starting the agent takes ~15 seconds.

**Total time: ~15–25 minutes** (mostly downloading models — depends on internet speed)

---

## What You Need Before Starting

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS | Ubuntu 20.04+ / Windows 10+ / macOS 12+ | Linux recommended |
| Python | 3.11 or higher | [python.org/downloads](https://www.python.org/downloads/) |
| RAM | 16 GB | 32 GB recommended |
| GPU | NVIDIA with 8 GB VRAM | CPU works but is slow |
| Disk space | 15 GB free | For models and packages |
| Internet | Required | For first-time downloads |

Check your Python version:
```bash
python --version
# Must show Python 3.11.x or higher
```

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Raghavendraqbox/sonixvoiceagent.git
cd sonixvoiceagent
git checkout daripastho
```

---

## Step 2 — Install Python Packages

**~5–10 minutes**

If you have an NVIDIA GPU, install PyTorch with CUDA first:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> Skip the above line if you have no GPU (CPU-only mode).

Then install all project dependencies:
```bash
pip install -r requirements.txt
```

---

## Step 3 — Install Ollama (the AI brain)

**~2 minutes**

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download and run the installer from [ollama.com](https://ollama.com)

Verify it installed:
```bash
ollama --version
```

---

## Step 4 — Download the AI Language Model

**~5–10 minutes** (downloads 4.7 GB)

First, start the Ollama server in a separate terminal:
```bash
ollama serve
```

Then in another terminal, download the model:
```bash
ollama pull qwen2.5:7b
```

Wait for it to finish. You will see `success` when done.

Verify:
```bash
ollama list
# Should show: qwen2.5:7b
```

> Keep the `ollama serve` terminal open — it must stay running.

---

## Step 5 — Get a Soniox API Key (Speech Recognition)

**~2 minutes**

Soniox provides accurate real-time Dari and Pashto speech recognition.

1. Go to **[soniox.com/dashboard](https://soniox.com/dashboard)**
2. Sign up for a free account
3. Copy your API key (looks like: `1f9a7bb0dc3534ff...`)

> **No key?** The agent will automatically fall back to Whisper (local GPU). It downloads ~3 GB on first use and is slightly less accurate.

---

## Step 6 — Start the Voice Agent

**~10 seconds** (first run downloads the TTS voice model ~460 MB automatically)

Navigate to the backend folder:
```bash
cd sonixvoiceagent/backend
```

Start with your Soniox key:

**Linux / macOS:**
```bash
SONIOX_API_KEY=your_key_here python main.py
```

**Windows (Command Prompt):**
```cmd
set SONIOX_API_KEY=your_key_here
python main.py
```

**Windows (PowerShell):**
```powershell
$env:SONIOX_API_KEY="your_key_here"
python main.py
```

**Without Soniox key (Whisper fallback):**
```bash
python main.py
```

---

## Step 7 — Open the Web Interface

Wait until you see this in the terminal:
```
Server ready.
Uvicorn running on http://0.0.0.0:8000
```

Then open your browser and go to:
```
http://localhost:8000
```

Select **Dari** or **Pashto**, allow microphone access, and click **Start Conversation**.

---

## Verify Everything Is Working

Open this URL in your browser:
```
http://localhost:8000/health
```

You should see:
```json
{
  "status": "ok",
  "asr": "soniox/stt-rt-v4",
  "llm": "ollama/qwen2.5:7b @ http://localhost:11434",
  "tts": "mms-tts (local GPU) → edge-tts → gTTS (24kHz output)"
}
```

---

## What Gets Downloaded Automatically

These are downloaded once and cached — never downloaded again:

| Item | Size | When |
|------|------|------|
| `qwen2.5:7b` LLM model | 4.7 GB | Step 4 (manual pull) |
| `facebook/mms-tts-fas` Dari TTS | ~460 MB | First server start |
| `all-MiniLM-L6-v2` embeddings | ~90 MB | First server start |
| Whisper `large-v3` *(no Soniox key only)* | ~3 GB | First call |

---

## Time Summary

| Step | Time |
|------|------|
| Clone repo | < 1 min |
| Install Python packages | 5–10 min |
| Install Ollama | 1–2 min |
| Download `qwen2.5:7b` | 5–10 min |
| First server start (TTS model download) | 1–2 min |
| **Total** | **~15–25 min** |

After first-time setup, starting the agent every time takes **~15 seconds**.

---

## Troubleshooting

**`python: command not found`**
Use `python3` instead of `python`, or install Python 3.11+ from [python.org](https://www.python.org/downloads/).

**`ollama: command not found`**
Ollama is not installed. Repeat Step 3.

**`connection refused` on http://localhost:8000**
The server is not running. Go back to Step 6.

**`qwen2.5:7b` not found error**
Run `ollama pull qwen2.5:7b` and make sure `ollama serve` is running.

**TTS model download fails**
Check your internet connection. The model downloads from HuggingFace on first run.

**Out of GPU memory**
Use a smaller model: `ollama pull qwen2.5:3b` then start with `OLLAMA_MODEL=qwen2.5:3b python main.py`.

**Microphone not working**
Make sure your browser has microphone permission. Click the lock icon in the browser address bar and allow microphone access.
