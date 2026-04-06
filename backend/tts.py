"""
tts.py — Telugu TTS handler using edge-tts (te-IN-ShrutiNeural / te-IN-MohanNeural).

Primary:    Microsoft edge-tts with Telugu neural voices
Fallback 1: gTTS (Google Text-to-Speech, Telugu)
Fallback 2: Silence padding

Audio output: PCM 16-bit signed LE, 24000 Hz, mono
  → matches PLAYBACK_SAMPLE_RATE in frontend/index.html

Cancel semantics:
  Setting cancel_event aborts synthesis within one 60ms chunk.
"""

import asyncio
import io
import logging
import queue as _queue
import threading
from typing import Callable, Awaitable

import numpy as np

from config import config

logger = logging.getLogger(__name__)

# TTS native output sample rate (edge-tts → 24 kHz PCM after resampling)
TTS_RATE = config.tts.sample_rate   # 24000 Hz

# Type alias
AudioSendCallback = Callable[[bytes], Awaitable[None]]

# Sentinel for empty queue poll
_QUEUE_EMPTY = object()


# ---------------------------------------------------------------------------
# TeluguTTSHandler
# ---------------------------------------------------------------------------

class TeluguTTSHandler:
    """
    Primary: edge-tts with Telugu neural voice (te-IN-ShrutiNeural).
    Fallback: gTTS (Google TTS, language=te).

    Each synthesis call streams PCM chunks to the client as soon as they are
    ready — first audio arrives within ~100ms of the call.
    """

    def __init__(
        self,
        session_id: str,
        send_audio_cb: AudioSendCallback,
        cancel_event: asyncio.Event,
    ) -> None:
        self.session_id = session_id
        self._send_audio = send_audio_cb
        self._cancel_event = cancel_event

    async def synthesize_and_stream(self, text: str) -> bool:
        """
        Synthesize `text` to Telugu speech and stream PCM to the client.

        Returns True if synthesis completed without interruption, False if
        cancel_event was set mid-stream.
        """
        if not text.strip():
            return True
        logger.info(
            "TTS synthesize: %.60s", text,
            extra={"session_id": self.session_id},
        )
        return await self._synthesize_edge_tts(text)

    # ------------------------------------------------------------------
    # edge-tts synthesis
    # ------------------------------------------------------------------

    async def _synthesize_edge_tts(self, text: str) -> bool:
        """
        Stream Telugu speech via edge-tts (te-IN-ShrutiNeural).

        edge-tts returns MP3 chunks; we decode with PyAV and resample to 24kHz
        PCM for the browser AudioContext.
        """
        if self._cancel_event.is_set():
            return False

        try:
            import edge_tts
            import av
        except ImportError:
            logger.warning(
                "edge-tts or PyAV not installed — falling back to gTTS",
                extra={"session_id": self.session_id},
            )
            return await self._synthesize_gtts(text)

        try:
            communicate = edge_tts.Communicate(text, voice=config.tts.edge_tts_voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
                if self._cancel_event.is_set():
                    return False

            if not audio_bytes:
                logger.warning(
                    "edge-tts returned empty audio",
                    extra={"session_id": self.session_id},
                )
                return await self._synthesize_gtts(text)

            if self._cancel_event.is_set():
                return False

            # Decode MP3 → PCM via PyAV, resample to TTS_RATE
            pcm_bytes = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self._decode_mp3_to_pcm(audio_bytes)
            )
            if pcm_bytes is None:
                return await self._synthesize_gtts(text)

            # Stream in small chunks for low cancel latency
            bpc = int(TTS_RATE * config.tts.chunk_ms / 1000) * 2  # bytes per chunk
            for i in range(0, len(pcm_bytes), bpc):
                if self._cancel_event.is_set():
                    return False
                await self._send_audio(pcm_bytes[i: i + bpc])
                await asyncio.sleep(0)  # yield to event loop

            return not self._cancel_event.is_set()

        except Exception as exc:
            logger.error(
                "edge-tts error: %s", exc,
                extra={"session_id": self.session_id},
            )
            return await self._synthesize_gtts(text)

    def _decode_mp3_to_pcm(self, audio_bytes: bytes) -> bytes | None:
        """
        Decode MP3 bytes to int16 PCM at TTS_RATE using PyAV.

        Returns raw PCM bytes (int16 LE), or None on error.
        """
        try:
            import av
            buf = io.BytesIO(audio_bytes)
            container = av.open(buf)
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=TTS_RATE,
            )
            frames: list = []
            for frame in container.decode(audio=0):
                for r in resampler.resample(frame):
                    frames.append(np.frombuffer(bytes(r.planes[0]), dtype=np.int16))
            for r in resampler.resample(None):
                frames.append(np.frombuffer(bytes(r.planes[0]), dtype=np.int16))
            container.close()

            if not frames:
                return None

            pcm = np.concatenate(frames)

            # Trim leading/trailing silence
            nz = np.where(np.abs(pcm) > 160)[0]
            if len(nz):
                pcm = pcm[nz[0]: nz[-1] + 1]

            # Fade in/out to avoid clicks at boundaries
            fade = int(TTS_RATE * 0.015)
            if len(pcm) > fade * 2:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                f = pcm.astype(np.float32)
                f[:fade] *= ramp
                f[-fade:] *= ramp[::-1]
                pcm = np.clip(f, -32768, 32767).astype(np.int16)

            return pcm.tobytes()

        except Exception as exc:
            logger.error("PyAV decode error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # gTTS fallback
    # ------------------------------------------------------------------

    async def _synthesize_gtts(self, text: str) -> bool:
        """
        Fallback TTS using Google Text-to-Speech (Telugu).

        gTTS returns MP3; decoded to PCM same as edge-tts path.
        """
        if self._cancel_event.is_set():
            return False

        try:
            from gtts import gTTS
            import av
        except ImportError:
            logger.warning(
                "gTTS or PyAV not installed — using silence",
                extra={"session_id": self.session_id},
            )
            return await self._synthesize_silence(text)

        try:
            loop = asyncio.get_running_loop()

            def _run_gtts() -> bytes:
                buf = io.BytesIO()
                tts = gTTS(text=text, lang=config.tts.gtts_language, slow=False)
                tts.write_to_fp(buf)
                return buf.getvalue()

            audio_bytes = await loop.run_in_executor(None, _run_gtts)

            if not audio_bytes or self._cancel_event.is_set():
                return not self._cancel_event.is_set()

            pcm_bytes = await loop.run_in_executor(
                None, lambda: self._decode_mp3_to_pcm(audio_bytes)
            )
            if pcm_bytes is None:
                return await self._synthesize_silence(text)

            bpc = int(TTS_RATE * config.tts.chunk_ms / 1000) * 2
            for i in range(0, len(pcm_bytes), bpc):
                if self._cancel_event.is_set():
                    return False
                await self._send_audio(pcm_bytes[i: i + bpc])
                await asyncio.sleep(0)

            return not self._cancel_event.is_set()

        except Exception as exc:
            logger.error(
                "gTTS error: %s", exc,
                extra={"session_id": self.session_id},
            )
            return await self._synthesize_silence(text)

    # ------------------------------------------------------------------
    # Silence fallback
    # ------------------------------------------------------------------

    async def _synthesize_silence(self, text: str) -> bool:
        """Last resort — send silence proportional to text length."""
        import struct
        words = text.split()
        spc = TTS_RATE // 5   # 200ms of silence per word
        silence = struct.pack(f"<{spc}h", *([0] * spc))
        for _ in words:
            if self._cancel_event.is_set():
                return False
            await self._send_audio(silence)
            await asyncio.sleep(0.2)
        return True


# ---------------------------------------------------------------------------
# TTSOrchestrator — drains a queue of text fragments, synthesizes in order
# ---------------------------------------------------------------------------

class TTSOrchestrator:
    """
    Drains a queue of text fragments and synthesizes them sequentially.
    Checks cancel_event between every fragment.
    """

    def __init__(
        self,
        session_id: str,
        tts_handler: TeluguTTSHandler,
        cancel_event: asyncio.Event,
    ) -> None:
        self.session_id = session_id
        self._tts = tts_handler
        self._cancel_event = cancel_event
        self._fragment_queue: asyncio.Queue = asyncio.Queue()
        self._active = False

    @property
    def fragment_queue(self) -> asyncio.Queue:
        return self._fragment_queue

    async def run(self) -> None:
        self._active = True
        logger.debug("TTSOrchestrator started", extra={"session_id": self.session_id})

        while True:
            if self._cancel_event.is_set():
                # Drain stale fragments
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

            try:
                fragment = await asyncio.wait_for(self._fragment_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if fragment is None:
                break

            if self._cancel_event.is_set():
                break

            completed = await self._tts.synthesize_and_stream(fragment)
            if not completed:
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

        self._active = False
        logger.debug("TTSOrchestrator stopped", extra={"session_id": self.session_id})

    def is_active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# Warmup stub (no-op for edge-tts; kept for API compatibility with session_manager)
# ---------------------------------------------------------------------------

def schedule_tts_warmup():
    """
    edge-tts has no heavy model to warm up — this is a no-op.
    Kept for API compatibility with session_manager.initialize_rag().
    """
    logger.info("edge-tts Telugu TTS ready (no warmup needed)")
