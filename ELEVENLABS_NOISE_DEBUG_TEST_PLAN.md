# ElevenLabs Noise Debug Test Plan

This guide helps you verify where audio noise is introduced:

- ElevenLabs generation layer
- Backend decode/transcode layer
- Frontend playback/scheduling layer

It is designed for your current app flow and does not change production behavior unless you enable debug flags.

---

## 1) Goal

Confirm this hypothesis:

- Audio in ElevenLabs History is clean.
- Noise appears only when listening from the app.
- Therefore, the issue is likely after TTS generation (decode/stream/playback path).

---

## 2) What was added for debugging

When enabled, the backend can dump both:

- Source ElevenLabs response (`.mp3`)
- Decoded PCM output saved as `.decoded.wav` (24kHz, mono, int16)

This lets you do A/B listening:

- If source MP3 is clean but decoded WAV is noisy -> backend decode/transcode issue.
- If both are clean but app playback is noisy -> frontend playback issue.
- If both are noisy -> source/provider/input issue.

---

## 3) Prerequisites

- Backend and frontend run normally.
- ElevenLabs API key and voice IDs are configured.
- You can run with Pashto + `tts_engine=elevenlabs`.

Recommended tools for listening:

- VLC
- `ffplay` (if FFmpeg installed)

---

## 4) Environment setup

In your `.env` file, add/update:

```env
DEBUG_TTS_DUMP_AUDIO=true
DEBUG_TTS_DUMP_DIR=./debug_audio
```

Keep these as-is for testing:

```env
PASHTO_TTS_ENGINE_PRIORITY=elevenlabs,edge,gtts
```

Or choose ElevenLabs from UI dropdown if available.

---

## 5) Run test scenario

1. Restart backend after `.env` changes.
2. Open app UI.
3. Select:
   - Language: `Pashto`
   - TTS engine: `ElevenLabs`
4. Start conversation.
5. Speak 3-5 short utterances with different content:
   - Short greeting
   - Longer sentence
   - Numbers or names
6. Stop conversation.

---

## 6) Locate output files

Check folder:

`./debug_audio`

For each TTS response you should see pairs like:

- `elevenlabs-<session>-<timestamp>.mp3`
- `elevenlabs-<session>-<timestamp>.decoded.wav`

If no files are created:

- Confirm `DEBUG_TTS_DUMP_AUDIO=true`
- Confirm backend restarted
- Confirm ElevenLabs path was actually used

---

## 7) A/B listening procedure

For each pair:

1. Play the `.mp3`
2. Play the matching `.decoded.wav`
3. Compare:
   - Hiss/static
   - Crackle/clicks
   - Distortion
   - Volume pumping

Optional waveform/spectrum check:

- Open both in Audacity and compare noise floor and peaks.

---

## 8) Result interpretation matrix

### Case A: MP3 clean, WAV noisy

Likely backend decode/transcode issue.

Check:

- PyAV frame extraction path
- Resampler output format assumptions
- PCM conversion scaling/clipping

### Case B: MP3 clean, WAV clean, app noisy

Likely frontend scheduling/playback issue.

Check:

- Queue/scheduling overlap in Web Audio
- Buffer start times and drift
- Repeated/late chunks
- Any double playback from reconnect paths

### Case C: MP3 noisy, WAV noisy

Likely upstream/source issue.

Check:

- ElevenLabs request payload (voice/model/settings)
- Input text characteristics
- Voice quality for selected language

### Case D: Noisy only on some utterances

Likely timing/buffer edge condition.

Check:

- Chunk boundary behavior
- Rapid interrupt/resume sequences
- Reconnect during playback

---

## 9) Minimal reproducible test script (manual)

Use the same short text each run for consistency:

- "سلام، زه څنګه مرسته کولی شم؟"
- "مهرباني وکړئ د خپلې ستونزې په اړه لږ معلومات راکړئ."
- "ستاسو مننه، زه به دا تایید کړم."

Run each phrase at least 3 times and compare output pairs.

---

## 10) Logs to capture

Collect backend logs around:

- ElevenLabs request success line
- Debug dump saved path line
- Any MP3 decode/resample warnings/errors

Also record:

- Browser/OS
- Whether headphones or speakers were used
- Whether noise suppression/echo cancellation was enabled

---

## 11) Fast commands (optional)

If using FFmpeg tools:

```bash
ffprobe "./debug_audio/<file>.mp3"
ffprobe "./debug_audio/<file>.decoded.wav"
ffplay "./debug_audio/<file>.mp3"
ffplay "./debug_audio/<file>.decoded.wav"
```

---

## 12) After testing (important)

Disable debug dumping to avoid disk growth:

```env
DEBUG_TTS_DUMP_AUDIO=false
```

Then restart backend.

---

## 13) Next fix path (based on outcome)

- If decode layer is culprit: we harden backend decode and add integrity assertions.
- If playback layer is culprit: we add frontend playback diagnostics (queue depth, schedule offsets, underrun counters) and optional "direct WAV playback mode" for isolation.

---

## 14) Quick checklist

- [ ] Debug flags enabled
- [ ] Backend restarted
- [ ] ElevenLabs path used
- [ ] MP3/WAV file pairs generated
- [ ] A/B listening completed
- [ ] Outcome mapped to matrix
- [ ] Debug flags disabled after test

