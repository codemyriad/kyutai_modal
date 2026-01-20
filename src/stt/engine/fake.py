"""Fake engine for CPU-based testing.

Returns deterministic output based on audio characteristics,
allowing reliable unit tests without GPU dependencies.
"""

import hashlib

import numpy as np

from stt.constants import SAMPLE_RATE


class FakeEngine:
    """Deterministic CPU engine for testing.

    Generates predictable transcriptions based on audio length and content hash.
    This allows testing the full server pipeline without GPU dependencies.
    """

    def __init__(self, latency_ms: float = 0.0):
        """Initialize the fake engine.

        Args:
            latency_ms: Simulated inference latency in milliseconds.
        """
        self._latency_ms = latency_ms
        self._call_count = 0

    def transcribe_batch(self, audio_list: list[np.ndarray]) -> list[str]:
        """Generate deterministic transcriptions based on audio properties.

        Args:
            audio_list: List of float32 audio arrays.

        Returns:
            List of fake transcription strings.
        """
        import time

        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)

        self._call_count += 1
        results = []

        for audio in audio_list:
            duration_s = len(audio) / SAMPLE_RATE
            audio_hash = self._hash_audio(audio)
            text = f"[fake:{audio_hash[:8]}|{duration_s:.2f}s]"
            results.append(text)

        return results

    def warmup(self) -> None:
        """No-op warmup for fake engine."""
        pass

    @property
    def call_count(self) -> int:
        """Number of transcribe_batch calls made."""
        return self._call_count

    @staticmethod
    def _hash_audio(audio: np.ndarray) -> str:
        """Generate a short hash of audio content for deterministic output."""
        # Use first 100 samples (or all if shorter) for hash
        samples = audio[: min(100, len(audio))]
        data = samples.tobytes()
        return hashlib.sha256(data).hexdigest()
