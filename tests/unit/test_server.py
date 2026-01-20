"""Unit tests for the FastAPI server with FakeEngine."""

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from stt.audio import float32_to_pcm16
from stt.constants import MIN_AUDIO_BYTES
from stt.engine.fake import FakeEngine
from stt.server import StreamSession, create_app


class TestStreamSession:
    """Tests for StreamSession buffer management."""

    def test_initial_state(self):
        """Session should start with empty buffer."""
        session = StreamSession("test-id")
        assert session.buffer_bytes == 0
        assert session.buffer_samples == 0
        assert not session.has_enough_data()

    def test_append_data(self):
        """Appending data should increase buffer size."""
        session = StreamSession("test-id", chunk_threshold=100)
        session.append(bytes(50))
        assert session.buffer_bytes == 50
        assert session.buffer_samples == 25

    def test_has_enough_data(self):
        """has_enough_data should respect threshold."""
        session = StreamSession("test-id", chunk_threshold=100)

        session.append(bytes(50))
        assert not session.has_enough_data()

        session.append(bytes(50))
        assert session.has_enough_data()

    def test_has_minimum_audio(self):
        """has_minimum_audio should respect minimum threshold."""
        session = StreamSession("test-id")

        # Very short audio (100 bytes = 50 samples = ~2ms)
        session.append(bytes(100))
        assert not session.has_minimum_audio()

        # Still not enough (less than 1 second)
        session.append(bytes(10000))
        assert not session.has_minimum_audio()

        # Add enough to reach minimum (1 second = 48000 bytes)
        session.append(bytes(MIN_AUDIO_BYTES))
        assert session.has_minimum_audio()

    def test_flush_clears_buffer(self):
        """Flush should return data and clear buffer."""
        session = StreamSession("test-id")
        audio = np.array([0.5, -0.5], dtype=np.float32)
        pcm = float32_to_pcm16(audio)

        session.append(pcm)
        result = session.flush()

        assert session.buffer_bytes == 0
        np.testing.assert_allclose(result, audio, atol=0.0001)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self):
        """Health endpoint should return status ok."""
        engine = FakeEngine()
        app = create_app(engine, use_batching=False)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["sample_rate"] == 24000
        assert "model" in data


class TestWebSocketEndpoint:
    """Tests for the WebSocket /v1/stream endpoint."""

    @pytest.fixture
    def engine(self):
        return FakeEngine()

    @pytest.fixture
    def app(self, engine):
        return create_app(engine, use_batching=False)

    @pytest.mark.asyncio
    async def test_websocket_accept(self, app):
        """WebSocket should accept connection."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Use TestClient for WebSocket
            with TestClient(app) as tc:
                with tc.websocket_connect("/v1/stream") as ws:
                    ws.send_bytes(b"EOS")
                    response = ws.receive_json()
                    assert response["status"] == "complete"

    @pytest.mark.asyncio
    async def test_websocket_single_chunk(self, app):
        """Single chunk transcription."""
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/stream") as ws:
                # Send 1 second of audio (enough to trigger transcription)
                audio = np.zeros(24000, dtype=np.float32)
                pcm = float32_to_pcm16(audio)
                ws.send_bytes(pcm)
                ws.send_bytes(b"EOS")

                # Should get transcription then complete
                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("status") == "complete":
                        break

                # Should have at least the complete message
                assert any(m.get("status") == "complete" for m in messages)

    @pytest.mark.asyncio
    async def test_websocket_multiple_chunks(self, app):
        """Multiple chunks should accumulate and trigger transcription."""
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/stream") as ws:
                # Send in 4 chunks (each 6000 samples = 0.25s)
                for _ in range(4):
                    audio = np.zeros(6000, dtype=np.float32)
                    pcm = float32_to_pcm16(audio)
                    ws.send_bytes(pcm)

                ws.send_bytes(b"EOS")

                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("status") == "complete":
                        break

                assert any(m.get("status") == "complete" for m in messages)

    @pytest.mark.asyncio
    async def test_websocket_invalid_audio(self, app):
        """Odd byte count should return error."""
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/stream") as ws:
                # Send odd number of bytes (invalid PCM16)
                ws.send_bytes(bytes(101))

                msg = ws.receive_json()
                assert "error" in msg

                # Still should be able to send EOS
                ws.send_bytes(b"EOS")
                complete = ws.receive_json()
                assert complete["status"] == "complete"

    @pytest.mark.asyncio
    async def test_websocket_eos_with_remaining_buffer(self, app):
        """EOS should process remaining buffer if it meets minimum length."""
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/stream") as ws:
                # Send 2 seconds of audio (meets minimum)
                audio = np.zeros(48000, dtype=np.float32)
                pcm = float32_to_pcm16(audio)
                ws.send_bytes(pcm)
                ws.send_bytes(b"EOS")

                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("status") == "complete":
                        break

                # Should have processed the remaining buffer
                assert any(m.get("status") == "complete" for m in messages)

    @pytest.mark.asyncio
    async def test_websocket_short_audio_no_crash(self, app):
        """Very short audio should not crash, just skip transcription.

        This test verifies the fix for the IndexError bug that occurred
        when audio was too short (<1 second) causing model.generate() to fail
        with 'index -1 is out of bounds for dimension 0 with size 0'.
        """
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/stream") as ws:
                # Send very short audio (only 100ms - way below minimum)
                audio = np.zeros(2400, dtype=np.float32)  # 100ms at 24kHz
                pcm = float32_to_pcm16(audio)
                ws.send_bytes(pcm)
                ws.send_bytes(b"EOS")

                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("status") == "complete":
                        break

                # Should complete without crash, no transcription for short audio
                assert any(m.get("status") == "complete" for m in messages)
                # Should NOT have any text messages (audio too short)
                text_msgs = [m for m in messages if "text" in m]
                assert len(text_msgs) == 0


class TestServerWithBatching:
    """Tests for server with batching enabled."""

    @pytest.mark.asyncio
    async def test_batching_enabled(self):
        """Server should work with batching enabled."""
        engine = FakeEngine()
        app = create_app(engine, use_batching=True)

        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/stream") as ws:
                audio = np.zeros(24000, dtype=np.float32)
                pcm = float32_to_pcm16(audio)
                ws.send_bytes(pcm)
                ws.send_bytes(b"EOS")

                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("status") == "complete":
                        break

                assert any(m.get("status") == "complete" for m in messages)
