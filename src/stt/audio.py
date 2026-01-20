"""Audio conversion and chunking utilities.

All functions work with 24kHz mono PCM16 audio as bytes or float32 numpy arrays.
"""

from collections.abc import Iterator

import numpy as np

from stt.constants import BYTES_PER_SAMPLE, SAMPLE_RATE


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float32 array normalized to [-1, 1].

    Args:
        data: Raw PCM16 little-endian audio bytes.

    Returns:
        Float32 numpy array with values in [-1, 1].
    """
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 array [-1, 1] to PCM16 bytes.

    Args:
        audio: Float32 numpy array with values in [-1, 1].

    Returns:
        Raw PCM16 little-endian audio bytes.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def chunk_audio(data: bytes, chunk_size: int) -> Iterator[bytes]:
    """Split audio bytes into fixed-size chunks.

    Args:
        data: Raw PCM16 audio bytes.
        chunk_size: Size of each chunk in bytes.

    Yields:
        Chunks of the specified size. The last chunk may be smaller.
    """
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def validate_audio_format(data: bytes) -> bool:
    """Check if audio data has valid PCM16 format.

    Args:
        data: Raw audio bytes to validate.

    Returns:
        True if data length is even (valid PCM16), False otherwise.
    """
    return len(data) % BYTES_PER_SAMPLE == 0


def samples_to_bytes(num_samples: int) -> int:
    """Convert sample count to byte count for PCM16."""
    return num_samples * BYTES_PER_SAMPLE


def bytes_to_samples(num_bytes: int) -> int:
    """Convert byte count to sample count for PCM16."""
    return num_bytes // BYTES_PER_SAMPLE


def duration_samples(duration_ms: int) -> int:
    """Calculate number of samples for a given duration in milliseconds."""
    return SAMPLE_RATE * duration_ms // 1000


def duration_bytes(duration_ms: int) -> int:
    """Calculate number of bytes for a given duration in milliseconds."""
    return samples_to_bytes(duration_samples(duration_ms))
