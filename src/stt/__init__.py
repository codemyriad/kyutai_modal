"""Kyutai STT service package."""

from stt.constants import (
    CHUNK_BYTES,
    CHUNK_SAMPLES,
    FRAME_SAMPLES,
    MODEL_ID,
    SAMPLE_RATE,
)

__all__ = [
    "SAMPLE_RATE",
    "FRAME_SAMPLES",
    "CHUNK_SAMPLES",
    "CHUNK_BYTES",
    "MODEL_ID",
]
