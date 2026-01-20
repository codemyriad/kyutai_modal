"""Unit tests for audio conversion utilities."""

import numpy as np
import pytest

from stt.audio import (
    bytes_to_samples,
    chunk_audio,
    duration_bytes,
    duration_samples,
    float32_to_pcm16,
    pcm16_to_float32,
    samples_to_bytes,
    validate_audio_format,
)
from stt.constants import CHUNK_BYTES, CHUNK_MS, CHUNK_SAMPLES, FRAME_SAMPLES, SAMPLE_RATE


class TestPCM16Conversion:
    """Tests for PCM16 <-> float32 conversion."""

    def test_pcm16_to_float32_zeros(self):
        """Zero bytes should produce zero array."""
        data = bytes(100)  # 50 samples of zeros
        result = pcm16_to_float32(data)
        assert result.dtype == np.float32
        assert len(result) == 50
        np.testing.assert_array_equal(result, np.zeros(50, dtype=np.float32))

    def test_pcm16_to_float32_max_values(self):
        """Max int16 should map to ~1.0."""
        # Max positive: 32767
        data = np.array([32767], dtype=np.int16).tobytes()
        result = pcm16_to_float32(data)
        assert abs(result[0] - 1.0) < 0.0001

        # Max negative: -32768
        data = np.array([-32768], dtype=np.int16).tobytes()
        result = pcm16_to_float32(data)
        assert abs(result[0] - (-1.0)) < 0.0001

    def test_float32_to_pcm16_roundtrip(self):
        """Conversion should be reversible within precision limits."""
        original = np.array([0.0, 0.5, -0.5, 0.99, -0.99], dtype=np.float32)
        pcm_bytes = float32_to_pcm16(original)
        recovered = pcm16_to_float32(pcm_bytes)
        np.testing.assert_allclose(recovered, original, atol=0.0001)

    def test_float32_to_pcm16_clipping(self):
        """Values outside [-1, 1] should be clipped."""
        original = np.array([2.0, -2.0], dtype=np.float32)
        pcm_bytes = float32_to_pcm16(original)
        recovered = pcm16_to_float32(pcm_bytes)
        np.testing.assert_allclose(recovered, [1.0, -1.0], atol=0.0001)


class TestChunking:
    """Tests for audio chunking."""

    def test_chunk_audio_exact_division(self):
        """Data that divides evenly into chunks."""
        data = bytes(100)
        chunks = list(chunk_audio(data, 25))
        assert len(chunks) == 4
        assert all(len(c) == 25 for c in chunks)

    def test_chunk_audio_with_remainder(self):
        """Data that doesn't divide evenly."""
        data = bytes(100)
        chunks = list(chunk_audio(data, 30))
        assert len(chunks) == 4
        assert [len(c) for c in chunks] == [30, 30, 30, 10]

    def test_chunk_audio_empty(self):
        """Empty data should produce no chunks."""
        chunks = list(chunk_audio(b"", 10))
        assert chunks == []

    def test_chunk_audio_smaller_than_chunk_size(self):
        """Data smaller than chunk size."""
        data = bytes(10)
        chunks = list(chunk_audio(data, 100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 10


class TestValidation:
    """Tests for audio format validation."""

    def test_validate_audio_format_valid(self):
        """Even byte count is valid PCM16."""
        assert validate_audio_format(bytes(100)) is True
        assert validate_audio_format(bytes(0)) is True
        assert validate_audio_format(bytes(2)) is True

    def test_validate_audio_format_invalid(self):
        """Odd byte count is invalid PCM16."""
        assert validate_audio_format(bytes(1)) is False
        assert validate_audio_format(bytes(101)) is False


class TestHelperFunctions:
    """Tests for conversion helper functions."""

    def test_samples_to_bytes(self):
        """Sample count to byte count."""
        assert samples_to_bytes(1) == 2
        assert samples_to_bytes(100) == 200
        assert samples_to_bytes(FRAME_SAMPLES) == FRAME_SAMPLES * 2

    def test_bytes_to_samples(self):
        """Byte count to sample count."""
        assert bytes_to_samples(2) == 1
        assert bytes_to_samples(200) == 100
        assert bytes_to_samples(CHUNK_BYTES) == CHUNK_SAMPLES

    def test_duration_samples(self):
        """Duration in ms to sample count."""
        assert duration_samples(1000) == SAMPLE_RATE  # 1 second
        assert duration_samples(80) == FRAME_SAMPLES  # 80ms
        assert duration_samples(CHUNK_MS) == CHUNK_SAMPLES

    def test_duration_bytes(self):
        """Duration in ms to byte count."""
        assert duration_bytes(1000) == SAMPLE_RATE * 2  # 1 second
        assert duration_bytes(CHUNK_MS) == CHUNK_BYTES


class TestConstants:
    """Tests for audio constants consistency."""

    def test_frame_samples(self):
        """80ms at 24kHz = 1920 samples."""
        assert FRAME_SAMPLES == 24000 * 80 // 1000

    def test_chunk_samples(self):
        """CHUNK_MS at 24kHz = CHUNK_SAMPLES."""
        assert CHUNK_SAMPLES == 24000 * CHUNK_MS // 1000

    def test_chunk_bytes(self):
        """CHUNK_SAMPLES * 2 bytes = CHUNK_BYTES."""
        assert CHUNK_BYTES == CHUNK_SAMPLES * 2
