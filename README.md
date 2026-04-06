# Sonix Voice Agent — Real-Time Full-Duplex Telugu Voice AI

A production-ready, full-duplex Telugu voice AI agent built for cloud GPU environments.  
Speak in Telugu → get an intelligent Telugu voice response in real time.

```
Browser Mic
    │
    ▼  PCM 16kHz (WebSocket binary)
Soniox ASR  ─── Telugu STT ──────────────────────┐
    │                                             │
    ▼  Final transcript                           │
Qwen2.5:72b (Ollama)  ── Telugu LLM response ───┤
    │                                             │
    ▼  Text fragments (streamed)                  │
MMS-TTS facebook/mms-tts-tel  ── Telugu VITS ────┘
    │  (resampled 16kHz → 24kHz)
    ▼  PCM 24kHz (WebSocket binary)
Browser Speaker
```

**Fully open-source stack** — LLM and TTS both run on your GPU via Ollama/HuggingFace, no API keys needed for inference.

---

## Features

- **Full-duplex** — user can interrupt the bot mid-sentence; it stops and listens instantly
- **Streaming pipeline** — first audio response arrives within ~200ms of speech ending
- **Telugu-first** — Soniox ASR gives state-of-the-art Telugu accuracy; Meta MMS-TTS provides native, natural Telugu voice (not robotic)
- **Bilingual** — responds in Telugu when the user speaks Telugu, English when spoken to in English
- **Open-source LLM** — Qwen2.5-32B runs locally on GPU via Ollama; no OpenAI / Claude API required
- **RAG** — Qobox knowledge base embedded in FAISS for accurate company-specific answers
- **Conversation memory** — remembers the last 8 turns per session
- **Robust fallbacks** — Soniox → Whisper large-v3 | MMS-TTS → edge-tts → gTTS | Ollama → neutral stub

---

## Architecture

```
backend/
├── main.py            FastAPI app + WebSocket endpoint
├── asr.py             Soniox streaming ASR (Whisper fallback)
├── llm.py             Ollama LLM client (Qwen2.5 streaming)
├── tts.py             edge-tts Telugu TTS (gTTS fallback)
├── session_manager.py Per-session state, interrupt logic, pipeline wiring
├── config.py          All settings (env-var overridable)
├── memory.py          Sliding-window conversation history
└── rag.py             FAISS RAG with sentence-transformer embeddings

frontend/
└── index.html         Single-file browser client (Web Audio API + WebSocket)

docs/
└── qobox_telugu.txt   Qobox knowledge base (Telugu + English)
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.11+ |
| CUDA GPU | 16+ GB VRAM for Qwen2.5-32B; 8+ GB for Qwen2.5-14B |
| Ollama | Local LLM server |
| Soniox API key | For Telugu ASR (optional — falls back to Whisper) |

---

## Quick Start

### 1. Install PyTorch (CUDA)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Telugu LLM (choose based on your VRAM)
ollama pull qwen2.5:32b    # ~20 GB VRAM — best quality
ollama pull qwen2.5:14b    # ~10 GB VRAM — good balance
ollama pull qwen2.5:7b     #  ~6 GB VRAM — lighter option

# Start the Ollama server
ollama serve
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — minimum required settings:

```env
# Soniox API key for Telugu ASR (get free key at https://soniox.com/dashboard)
# Leave empty to use Whisper large-v3 locally instead
SONIOX_API_KEY=your-soniox-api-key

# Which Ollama model to use
OLLAMA_MODEL=qwen2.5:32b
```

### 5. Start the server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Open the browser

```
http://localhost:8000
```

Click **మాట్లాడడం ప్రారంభించండి** (Start Talking), allow microphone, and speak in Telugu.

---

## Recommended LLM Models for Telugu

All models run locally via Ollama — no internet connection or API key needed for inference.

| Model | VRAM | Telugu Quality | Notes |
|-------|------|---------------|-------|
| `qwen2.5:32b` | ~20 GB | ★★★★★ | Best overall Telugu quality |
| `qwen2.5:14b` | ~10 GB | ★★★★☆ | Good balance of quality/speed |
| `qwen2.5:7b`  |  ~6 GB | ★★★☆☆ | For smaller GPUs |
| `gemma3:27b`  | ~18 GB | ★★★★☆ | Google Gemma 3, good multilingual |
| `aya:35b`     | ~22 GB | ★★★★☆ | Cohere multilingual model |

Set the model in `.env`:
```env
OLLAMA_MODEL=qwen2.5:32b
```

---

## ASR: Soniox vs Whisper

| | Soniox | Whisper large-v3 |
|-|--------|-----------------|
| Telugu accuracy | ★★★★★ | ★★★★☆ |
| Latency | ~100ms (streaming) | ~800ms (batch) |
| Interim results | Yes (real-time) | No |
| Requires internet | Yes (API) | No (local GPU) |
| API key needed | Yes (free tier available) | No |

**Recommendation**: Use Soniox for best Telugu accuracy. Fall back to Whisper if you need a fully offline setup.

Get your free Soniox key at [soniox.com/dashboard](https://soniox.com/dashboard).

---

## TTS: Native Telugu Voice

### Primary: Meta MMS-TTS (`facebook/mms-tts-tel`)

- VITS model trained specifically on Telugu speech
- Natural, native-sounding pronunciation — not robotic
- Runs on local GPU, ~460 MB model, no internet required
- Downloaded automatically from HuggingFace on first run

### Fallback: edge-tts Telugu neural voices

Used automatically when MMS-TTS fails. Change the fallback voice via `.env`:

| Voice | Gender | Quality |
|-------|--------|---------|
| `te-IN-ShrutiNeural` | Female | ★★★★☆ (default fallback) |
| `te-IN-MohanNeural`  | Male   | ★★★★☆ |

```env
TTS_VOICE=te-IN-MohanNeural
```

---

## Environment Variables

```env
# ── ASR ──────────────────────────────────────────────────────────────────
SONIOX_API_KEY=           # Soniox API key (leave empty for Whisper fallback)
SONIOX_LANGUAGE=te        # ASR language code
SONIOX_MODEL=soniox_multilingual_2

# ── LLM (local Ollama — no API key needed) ───────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:32b  # Change to qwen2.5:14b or qwen2.5:7b if VRAM limited
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=150

# ── TTS ──────────────────────────────────────────────────────────────────
TTS_VOICE=te-IN-ShrutiNeural   # or te-IN-MohanNeural

# ── RAG ──────────────────────────────────────────────────────────────────
RAG_DOCS_DIR=./docs
RAG_TOP_K=3

# ── Server ───────────────────────────────────────────────────────────────
SERVER_PORT=8000
LOG_LEVEL=info
```

---

## Project Structure

```
sonixvoiceagent/
├── backend/
│   ├── main.py            # FastAPI + WebSocket server
│   ├── asr.py             # Soniox ASR → Whisper fallback
│   ├── llm.py             # Ollama Qwen2.5 (open-source, local)
│   ├── tts.py             # edge-tts Telugu → gTTS fallback
│   ├── session_manager.py # Session lifecycle + conversation loop
│   ├── config.py          # Centralised settings
│   ├── memory.py          # Conversation history (sliding window)
│   └── rag.py             # FAISS RAG retriever
├── frontend/
│   └── index.html         # Browser client (Web Audio API, WebSocket)
├── docs/
│   └── qobox_telugu.txt   # Qobox knowledge base
├── requirements.txt
├── .env.example
└── README.md
```

---

## How Full-Duplex Works

1. **Browser** captures microphone PCM (16kHz) via AudioWorklet and sends 100ms chunks over WebSocket
2. **Soniox ASR** streams partial + final transcripts back to the server in real time
3. **VAD interrupt** — if the user speaks while the bot is talking, a `{"type":"interrupt"}` message is sent immediately; the backend cancels TTS and the browser stops playback
4. **LLM streaming** — Qwen2.5 streams tokens via Ollama; each complete sentence is dispatched to TTS immediately
5. **TTS streaming** — edge-tts synthesizes each sentence fragment and streams 60ms PCM chunks back over WebSocket
6. **Browser playback** — Web Audio API schedules PCM chunks back-to-back on a 24kHz AudioContext for gapless playback

---

## Adding Your Own Knowledge Base

Place `.txt` files in the `docs/` directory. They are automatically chunked, embedded, and indexed into FAISS at startup.

```bash
# Example: add a product catalog
cp my_products.txt docs/

# Delete the cached index to rebuild on next start
rm -rf faiss_index/
```

---

## Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "asr": "soniox/soniox_multilingual_2 (language=te)",
  "llm": "ollama/qwen2.5:32b @ http://localhost:11434",
  "tts": "edge-tts/te-IN-ShrutiNeural (24kHz)"
}
```

---

## Troubleshooting

**Bot doesn't respond / LLM error in logs**
```bash
# Check Ollama is running
ollama serve

# Check model is pulled
ollama list

# Pull if missing
ollama pull qwen2.5:32b
```

**ASR not transcribing**
- Check `SONIOX_API_KEY` in `.env`
- If key is absent, Whisper large-v3 is used — first run downloads ~3 GB

**No audio / TTS silent**
- edge-tts requires internet access (Microsoft Azure service, free)
- Check your network; gTTS fallback also requires internet

**Microphone not working**
- Use Chrome or Edge (best Web Audio API support)
- Serve over HTTPS or `localhost` (microphone requires a secure context)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Credits

- **ASR**: [Soniox](https://soniox.com) — streaming Telugu speech recognition
- **LLM**: [Ollama](https://ollama.com) + [Qwen2.5](https://qwenlm.github.io/) — open-source multilingual LLM
- **TTS**: [edge-tts](https://github.com/rany2/edge-tts) — Microsoft neural Telugu voices
- **RAG**: [FAISS](https://github.com/facebookresearch/faiss) + [sentence-transformers](https://www.sbert.net/)
- Based on the architecture of [voiceagentcloud](https://github.com/Raghavendraqbox/voiceagentcloud)
