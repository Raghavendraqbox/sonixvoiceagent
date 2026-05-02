"""
vad.py — Voice activity detection for the audio ingest pipeline.

Provides a small, thread-safe wrapper around the `silero-vad` ONNX model so the
server can distinguish real speech from background TV / chatter / fans / noise
before the audio reaches a cloud STT API.

Audio contract: 16-bit signed little-endian PCM, 16 kHz mono. Any chunk length
is accepted; we internally accumulate into the 512-sample windows that Silero
VAD requires at 16 kHz.

Graceful fallback: if `silero-vad` (or its onnxruntime/torch dependency) is not
importable, every VAD created by `make_session_vad()` degrades to the existing
RMS-energy gate using `config.audio.vad_rms_threshold`. The rest of the pipeline
keeps working unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Silero VAD requires exactly 512 samples per inference at 16 kHz (32 ms).
_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 512


def _pcm_rms_int16(pcm_bytes: bytes) -> float:
    """Return normalized RMS for 16-bit little-endian mono PCM."""
    n = len(pcm_bytes) // 2
    if n <= 0:
        return 0.0
    arr = np.frombuffer(pcm_bytes[: n * 2], dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(arr * arr))) / 32768.0


# ---------------------------------------------------------------------------
# Fallback: pure-RMS gate
# ---------------------------------------------------------------------------

class _RMSFallback:
    """Pure RMS-based gate. Used when Silero VAD cannot be loaded."""

    kind: str = "rms"

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def is_speech(self, pcm_bytes: bytes) -> bool:
        return _pcm_rms_int16(pcm_bytes) > self._threshold

    def reset(self) -> None:  # parity with _SileroVAD
        return


# ---------------------------------------------------------------------------
# Silero VAD (preferred)
# ---------------------------------------------------------------------------

# Module-level cache for the ONNX model so we only load weights once even when
# many sessions ask for VAD instances. Each session still receives its own
# wrapper (and therefore its own LSTM state) — the underlying ONNX runtime
# session is what's expensive to construct, not its hidden state.
_silero_load_lock = threading.Lock()
_silero_load_attempted = False
_silero_model = None      # type: ignore[var-annotated]
_silero_torch = None      # type: ignore[var-annotated]


def _try_load_silero():
    """Lazy-load Silero VAD once. Returns (model, torch) or (None, None)."""
    global _silero_load_attempted, _silero_model, _silero_torch
    if _silero_load_attempted:
        return _silero_model, _silero_torch
    with _silero_load_lock:
        if _silero_load_attempted:
            return _silero_model, _silero_torch
        _silero_load_attempted = True
        try:
            import torch
            from silero_vad import load_silero_vad

            model = load_silero_vad(onnx=True)
            try:
                model.reset_states()
            except Exception:
                pass
            _silero_model = model
            _silero_torch = torch
            logger.info(
                "Silero VAD loaded (onnx=True, window=%d samples @ %d Hz)",
                _WINDOW_SAMPLES,
                _SAMPLE_RATE,
            )
        except Exception as exc:
            logger.warning(
                "Silero VAD unavailable (%s) — using RMS fallback", exc
            )
            _silero_model = None
            _silero_torch = None
    return _silero_model, _silero_torch


class _SileroVAD:
    """Per-session Silero VAD wrapper.

    State (LSTM hidden + tail buffer of partial windows) is per-instance so
    concurrent sessions never interfere with each other's classifications.
    The underlying ONNX model object is shared, but `model.reset_states()` is
    called when this instance constructs and any time `reset()` is invoked.
    """

    kind: str = "silero"

    def __init__(self, threshold: float, rms_floor: float) -> None:
        self._threshold = float(threshold)
        self._rms_floor = float(rms_floor)
        self._tail = np.empty((0,), dtype=np.float32)
        self._lock = threading.Lock()

        model, torch = _try_load_silero()
        self._model = model
        self._torch = torch
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        """Reset RNN state and pending tail buffer for a new utterance."""
        with self._lock:
            self._tail = np.empty((0,), dtype=np.float32)
            if self._model is not None:
                try:
                    self._model.reset_states()
                except Exception:
                    pass

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Return True if the chunk contains speech.

        Behaviour:
          - If RMS energy is below `rms_floor`, short-circuit to False without
            running ONNX inference (saves CPU on near-silent chunks).
          - Otherwise, the chunk is concatenated with any leftover samples from
            the previous call, sliced into 512-sample windows, and the highest
            speech probability across windows is compared against `threshold`.
          - Any tail under 512 samples is held for the next call so no audio is
            lost across the boundary.
        """
        if self._model is None:
            return _pcm_rms_int16(pcm_bytes) > self._rms_floor

        # Cheap energy pre-filter to skip ONNX entirely on silent frames.
        if _pcm_rms_int16(pcm_bytes) < self._rms_floor:
            return False

        sample_count = len(pcm_bytes) // 2
        if sample_count == 0:
            return False
        samples = (
            np.frombuffer(pcm_bytes[: sample_count * 2], dtype="<i2").astype(
                np.float32
            )
            / 32768.0
        )

        torch = self._torch
        with self._lock:
            samples = np.concatenate([self._tail, samples])
            n_windows = len(samples) // _WINDOW_SAMPLES
            if n_windows == 0:
                self._tail = samples
                return False

            speech_detected = False
            for i in range(n_windows):
                window = samples[
                    i * _WINDOW_SAMPLES : (i + 1) * _WINDOW_SAMPLES
                ]
                tensor = torch.from_numpy(window)
                with torch.no_grad():
                    conf = float(self._model(tensor, _SAMPLE_RATE).item())
                if conf >= self._threshold:
                    speech_detected = True
                # keep iterating so the model's internal state stays in sync
                # with the audio timeline even after the answer is decided.

            consumed = n_windows * _WINDOW_SAMPLES
            self._tail = samples[consumed:]

        return speech_detected


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_session_vad():
    """Return a stateful, per-session VAD instance.

    Resolves config at call time so test/runtime overrides take effect:
      - `config.audio.use_silero_vad` toggles the engine.
      - `config.audio.silero_vad_threshold` sets the speech probability cutoff.
      - `config.audio.vad_rms_threshold` is used as both the RMS pre-filter
        floor for Silero and the threshold for the RMS fallback.
    """
    from config import config  # local import to avoid circulars at module load

    rms_threshold = float(config.audio.vad_rms_threshold)

    if not config.audio.use_silero_vad:
        return _RMSFallback(threshold=rms_threshold)

    silero_threshold = float(config.audio.silero_vad_threshold)
    # Floor the energy pre-filter at half the RMS threshold so very low audio
    # is rejected without running Silero, but real speech with quiet plosives
    # still reaches the model.
    rms_floor = max(rms_threshold * 0.5, 0.003)

    vad = _SileroVAD(threshold=silero_threshold, rms_floor=rms_floor)
    if vad.loaded:
        return vad

    return _RMSFallback(threshold=rms_threshold)


__all__ = ["make_session_vad"]
