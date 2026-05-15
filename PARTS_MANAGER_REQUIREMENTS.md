# Parts Manager Platform — UI & Voice Requirements Specification

This document captures everything requested for the **Parts Manager Platform** voice UI, so you can rebuild or port it (e.g. from an existing Telugu STT/TTS stack) without re-reading the full conversation.

**Source persona content:** `parts_manager_professional.pdf` (project root)

**Current reference implementation:** PersonaPlex / Moshi (`moshi.server` + React client in `client/`)

---

## 1. Product overview

| Item | Requirement |
|------|-------------|
| Product name | **Parts Manager Platform** |
| Former name | PersonaPlex (must **not** appear in UI title, favicon, or user-facing copy) |
| Domain | UAE automotive spare-parts marketplace (workshops, retailers, platform ops, car owners) |
| Interaction | Real-time **voice conversation** (user speaks, agent speaks back) |
| Currency / locale | **AED**, UAE cities (Dubai, Sharjah, Abu Dhabi, Al Quoz, etc.) |

### Tagline

> Voice-powered spare parts marketplace for workshops, retailers, and customers in the UAE.

---

## 2. Branding & UI (no NVIDIA)

| Element | Value |
|---------|--------|
| Page title | `Parts Manager Platform` |
| Favicon | Custom SVG (`/favicon.svg`) — **no** NVIDIA logo or PersonaPlex assets |
| Primary color | `#1e4d8c` (blue) |
| Light accent | `#2d6cb5` |
| Theme | Light only |
| Background | `bg-neutral-50`, body text `text-zinc-700` |

### Homepage layout

1. **Header:** App title + tagline  
2. **Persona selector:** 4 cards in a 2-column grid (mobile: 1 column)  
   - Selected card: blue border `border-[#1e4d8c]`, light blue background  
3. **Role prompt:** Read-only/editable textarea showing full system prompt for selected persona  
4. **Sample questions:** Bullet list (hints only — not buttons that reset prompt)  
5. **Voice file label:** e.g. `Voice file: NATF2.pt (female)`  
6. **Connect & talk** button → requests microphone → opens conversation view  

### Conversation view

- Shows: app title, persona name, persona role  
- Connect / Disconnect control + connection status dot (blue = connected)  
- Server audio visualizer + user audio  
- Text transcript display  
- Download recording link: `parts_manager_audio.*`

---

## 3. Server & deployment requirements

| Requirement | Detail |
|-------------|--------|
| Protocol | **HTTP only** (no HTTPS / no `--ssl`) |
| Port | `8998` (default) |
| Host | `0.0.0.0` for remote access |
| Static UI | Serve built client from `client/dist` via `--static /path/to/client/dist` |
| Model | NVIDIA PersonaPlex-7B (`nvidia/personaplex-7b-v1` on Hugging Face) |
| HF token | Required (`HF_TOKEN` env var) |
| GPU | CUDA; on this environment **disable cuDNN** before start (see §7) |

### Start command (reference)

```bash
cd moshi
python -c "
import torch
torch.backends.cudnn.enabled = False
import runpy, sys
sys.argv = ['moshi.server', '--host', '0.0.0.0', '--static', '/workspace/personaplex/client/dist']
runpy.run_module('moshi.server', run_name='__main__')
"
```

### Build client

```bash
cd client && npm install && npm run build
```

---

## 4. Voice / TTS requirements (PersonaPlex stack)

> **Important for STT/TTS migration:** The current app uses **full-duplex speech-to-speech** (one model handles listen + speak). A separate STT → LLM → TTS pipeline is a **different architecture**; use this section for **behavioral** requirements, not exact APIs.

### 4.1 Female voice (required)

| Setting | Value |
|---------|--------|
| Voice embedding file | **`NATF2.pt`** for all personas |
| Voice family | `NATF*` = Natural **Female** (NVIDIA PersonaPlex) |
| Do **not** use | `NATM*` (male), `VARM*` (male variety) unless explicitly testing male |

PersonaPlex voice naming (from upstream README):

```
Natural(female): NATF0, NATF1, NATF2, NATF3
Natural(male):   NATM0, NATM1, NATM2, NATM3
Variety(female): VARF0–VARF4
Variety(male):   VARM0–VARM4
```

### 4.2 Audio playback (current = default)

| Setting | Value |
|---------|--------|
| Playback speed | **1.0** (normal — no slowdown) |
| Output gain | **1.0** (default loudness — no extra boost) |
| Decode gain | **1.0** |
| Master gain | **1.0** |

*(Previously requested louder + ~78% speed; reverted to default.)*

### 4.3 Speech style (append to every persona text prompt)

```
Always speak with a clear female voice. Speak at a calm, moderate pace with neutral international English. Avoid strong regional accents. Use short, clear sentences.
```

### 4.4 Critical lesson: text prompt vs voice file

PersonaPlex uses **two layers**:

1. **`voice_prompt`** — `.pt` file (timbre / pitch bias)  
2. **`text_prompt`** — role instructions (strong influence on how speech sounds)

**Do not** use prompts like “your name is Ahmed” + “respond as Ahmed” while expecting a female voice — the model will often sound **male** even with `NATF2.pt`.

**Do** use a **female speaker name** (Layla, Fatima, Priya, Amira) and phrases like “You are a woman”, “female voice”, “Do not use a male voice”.

### 4.5 WebSocket parameters (PersonaPlex client)

Sent on connect to `/api/chat`:

| Parameter | Typical value |
|-----------|----------------|
| `voice_prompt` | `NATF2.pt` |
| `text_prompt` | Full persona system prompt |
| `text_temperature` | `0.7` |
| `text_topk` | `25` |
| `audio_temperature` | `0.8` |
| `audio_topk` | `250` |
| `pad_mult` | `0` |
| `text_seed` / `audio_seed` | Random per session |

Server logs to verify:

```
voice prompt: .../voices/NATF2.pt
text prompt: ...your name is Layla... female voice...
```

---

## 5. Four personas (from PDF)

Each persona has:

- **UI label** — who the scenario is about (from PDF)  
- **Speaker** — who the **voice agent** should be (female)  
- **Voice file** — `NATF2.pt`  
- **Sample questions** — example user utterances  

---

### Persona 1 — Workshop owner / buyer

| Field | Value |
|-------|--------|
| ID | `workshop-owner` |
| UI name | Ahmed Al Hashimi |
| Role | Workshop Owner / Parts Manager (Buyer) |
| Subtitle | Al Ghurair Auto Repairs, Al Quoz, Dubai |
| **Speaker (voice)** | **Layla** (female assistant) — not Ahmed |
| Voice file | `NATF2.pt` |

**System prompt (body):**

```
You work for Parts Manager which is the UAE automotive spare parts marketplace and your name is Layla. You are a woman and you speak with a female voice. You assist workshop owners such as Ahmed Al Hashimi at Al Ghurair Auto Repairs in Al Quoz, Dubai. Information: Help find parts by make, model, year, and VIN; create RFQs; compare quotes in AED; place orders; track deliveries; handle returns. When the user speaks, answer as Layla the female assistant in first person. Do not role-play as Ahmed or use a male voice.
```

**Sample questions (from PDF):**

- Find me a front left headlight for a 2020 Toyota Camry V6.
- Show me all pending RFQs I submitted this week.
- Where is my delivery from Pro Parts?

**PDF context (Ahmed):** Buyer at medium workshop; needs fast parts lookup, RFQs, quotes, delivery tracking, returns.

---

### Persona 2 — Retailer / supplier

| Field | Value |
|-------|--------|
| ID | `retailer-supplier` |
| UI name | Fatima Al Marri |
| Role | Retailer / Parts Supplier (Seller) |
| Subtitle | Gulf Star Auto Parts, Sharjah |
| **Speaker (voice)** | **Fatima Al Marri** (female) |
| Voice file | `NATF2.pt` |

**System prompt (body):**

```
You work for Gulf Star Auto Parts which is a large retailer in Sharjah and your name is Fatima Al Marri. You are a woman and you speak with a female voice. You sell on the Parts Manager platform. Information: Receive RFQs, send quotes in AED with OEM and aftermarket options, update delivery status, handle returns, review team performance. Response-rate goal is 75 percent. When the user speaks, answer as Fatima in first person with a female voice. Be professional and concise.
```

**Sample questions:**

- Show me new RFQs for Toyota and Lexus only.
- Dashboard of my team's response rates this week.
- Mark Order 2080 as ready for dispatch.

**PDF context (Fatima):** Seller; RFQ filtering, quoting OEM/aftermarket, dispatch, returns, team metrics, commission 5%.

---

### Persona 3 — Platform admin / operator

| Field | Value |
|-------|--------|
| ID | `platform-admin` |
| UI name | Priya Sharma |
| Role | Platform Admin (Operator) |
| Subtitle | Parts Manager HQ — Operations |
| **Speaker (voice)** | **Priya Sharma** (female) |
| Voice file | `NATF2.pt` |

**System prompt (body):**

```
You work for Parts Manager HQ which operates the UAE marketplace and your name is Priya Sharma. You are a woman and you speak with a female voice. You are the platform operations manager. Information: Monitor system health, retailer onboarding, disputes, and revenue in AED. Target quote-to-order time is 30 minutes. When the user speaks, answer as Priya in first person with a female voice. Use clear operational language.
```

**Sample questions:**

- Run a health check on the system.
- Show retailers with zero RFQ response in 48 hours.
- Show today's platform revenue.

**PDF context (Priya):** HQ ops; health checks, inactive retailers, disputes/POD, matching rules, revenue metrics (Dubai/Sharjah/Abu Dhabi).

---

### Persona 4 — Car owner (end customer)

| Field | Value |
|-------|--------|
| ID | `car-owner` |
| UI name | Carlos Fernandez |
| Role | End Customer (Car Owner) |
| Subtitle | 2016 Jeep Wrangler — City Garage, Dubai |
| **Speaker (voice)** | **Amira** (female assistant) — not Carlos |
| Voice file | `NATF2.pt` |

**System prompt (body):**

```
You work for Parts Manager on the City Garage customer portal and your name is Amira. You are a woman and you speak with a female voice. You help car owners such as Carlos Fernandez who has a 2016 Jeep Wrangler in Dubai. Information: Explain repairs, OEM versus aftermarket parts in AED, delivery timing, and invoices that separate parts from labor. When the user speaks, answer as Amira the female assistant in first person. Do not role-play as Carlos or use a male voice. Use simple friendly language.
```

**Sample questions:**

- My brakes are grinding — what's wrong?
- OEM vs aftermarket brake pads for my Jeep?
- Is the part here yet for my repair?

**PDF context (Carlos):** Non-mechanic; wants transparency on parts vs labor, repair status, simple language.

---

## 6. Persona selection UX requirements

- User must **select exactly one** persona before connecting  
- Selection sets both `text_prompt` and `voice_prompt`  
- Persist selected persona ID in `localStorage` (`partsManagerPersonaId`)  
- Prompt schema version key: `partsManagerPromptVersion` (for migrations when prompts/voice change)  
- Default persona: `workshop-owner` (Ahmed scenario / Layla speaker)  
- On persona change: refresh prompt text + voice file to match persona config  

---

## 7. Environment notes (this deployment)

| Issue | Workaround |
|-------|------------|
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` on warmup | Set `torch.backends.cudnn.enabled = False` before starting server |
| Stale UI | Hard refresh (Ctrl+Shift+R) after `npm run build` |
| Stale voice | Disconnect WebSocket and reconnect; check logs for `NATF2.pt` |

---

## 8. Porting to STT + TTS (e.g. Telugu stack)

If you replace PersonaPlex with **STT → LLM → TTS**:

| PersonaPlex concept | Your stack equivalent |
|---------------------|------------------------|
| `voice_prompt` / NATF2.pt | TTS voice ID / speaker profile (pick one **female** Telugu or English voice per persona) |
| `text_prompt` | LLM system prompt (use §5 prompts + §4.3 suffix) |
| Full-duplex audio stream | STT on mic input + TTS on LLM output (higher latency; no overlap unless you add barge-in) |
| WebSocket `/api/chat` | Your own API: audio in → text → LLM → audio out |
| Opus codec / worklet | Your Telugu pipeline’s audio format & playback |

### Minimum behavioral parity checklist

- [ ] Title: **Parts Manager Platform**  
- [ ] No NVIDIA / PersonaPlex branding  
- [ ] HTTP (or your chosen protocol)  
- [ ] 4 persona cards with correct labels  
- [ ] Female TTS voice for all personas  
- [ ] Female speaker in system prompt (Layla / Fatima / Priya / Amira — not Ahmed/Carlos as speaker)  
- [ ] Neutral accent / clear pace in prompt  
- [ ] Default playback speed and volume (no artificial slow/loud unless you want it)  
- [ ] UAE / AED / automotive domain in prompts  
- [ ] Sample questions visible per persona  

---

## 9. Key files in this repo (reference)

| Path | Purpose |
|------|---------|
| `client/src/config/personas.ts` | All 4 personas, prompts, `NATF2.pt` |
| `client/src/config/branding.ts` | Title, colors, tagline |
| `client/src/config/audioPlayback.ts` | Gain/speed + speech style suffix |
| `client/src/pages/Queue/Queue.tsx` | Homepage + persona selector |
| `client/src/pages/Conversation/` | Live call UI |
| `moshi/moshi/server.py` | Backend WebSocket server |
| `parts_manager_professional.pdf` | Original persona Q&A source |
| `scripts/verify_personas.py` | Voice file + WebSocket smoke test |

---

## 10. Change log (conversation summary)

| Request | Status |
|---------|--------|
| Run `python -m moshi.server` | Done — HTTP, cuDNN workaround |
| Rebrand to Parts Manager Platform | Done |
| Remove NVIDIA favicon / title | Done |
| 4 personas from PDF | Done |
| Selectable persona + talk | Done |
| Female voice only | Done — `NATF2.pt` |
| Louder + slower speech | Tried — **reverted to default** |
| Neutral accent in prompt | Done |
| Restart server | Multiple times — use §3 command |

---

*Generated for handoff to a new STT/TTS implementation. Update voice file names and API details to match your Telugu stack while keeping §2, §5, and §6 behavior.*
