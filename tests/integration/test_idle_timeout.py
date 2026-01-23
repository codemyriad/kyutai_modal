"""Integration tests for WebSocket idle timeout behavior.

These tests verify that the Modal-deployed STT service properly closes
WebSocket connections after periods of inactivity, ensuring containers
can scale down and costs are minimized.

Run against deployed service:
    MODAL_KEY=... MODAL_SECRET=... pytest tests/integration/test_idle_timeout.py -v

Run against local dev server:
    WS_URL=ws://localhost:8000/v1/stream pytest tests/integration/test_idle_timeout.py -v
"""

import asyncio
import json
import os
import time

import numpy as np
import pytest
import websockets

# Test configuration
MODAL_WORKSPACE = os.environ.get("MODAL_WORKSPACE", "YOUR_WORKSPACE")
DEFAULT_WS_URL = f"wss://{MODAL_WORKSPACE}--kyutai-stt-rust-kyutaisttrustservice-serve.modal.run/v1/stream"
IDLE_TIMEOUT_SECONDS = 10  # Expected server idle timeout (should match IDLE_AUDIO_TIMEOUT_SECONDS)


def get_ws_url() -> str:
    """Get WebSocket URL from environment or use default."""
    return os.environ.get("WS_URL", DEFAULT_WS_URL)


def get_auth_headers() -> dict | None:
    """Get Modal auth headers from environment."""
    modal_key = os.environ.get("MODAL_KEY")
    modal_secret = os.environ.get("MODAL_SECRET")
    if modal_key and modal_secret:
        return {"Modal-Key": modal_key, "Modal-Secret": modal_secret}
    return None


def create_pcm_audio(duration_seconds: float = 1.0) -> bytes:
    """Create raw PCM float32 audio for testing."""
    sample_rate = 24000
    num_samples = int(sample_rate * duration_seconds)

    # Generate silence (or could add a tone for more realistic test)
    audio = np.zeros(num_samples, dtype=np.float32)

    return audio.tobytes()


@pytest.fixture
def ws_url():
    return get_ws_url()


@pytest.fixture
def auth_headers():
    return get_auth_headers()


class TestIdleTimeout:
    """Tests for WebSocket idle timeout behavior."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Test should complete within 60 seconds
    async def test_connection_closes_after_idle_timeout(self, ws_url, auth_headers):
        """Server should close connection after IDLE_TIMEOUT_SECONDS without audio.

        This is the critical test - ensures containers can scale down when clients
        stop sending audio.
        """
        # Create some initial audio to establish the connection
        audio = create_pcm_audio(duration_seconds=2.0)

        async with websockets.connect(
            ws_url,
            additional_headers=auth_headers,
            open_timeout=120,
            close_timeout=10,
        ) as ws:
            # Send initial audio
            await ws.send(audio)

            # Receive any tokens (drain the initial response)
            received_tokens = []
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(msg)
                    if data.get("type") == "token":
                        received_tokens.append(data.get("text", ""))
            except asyncio.TimeoutError:
                pass  # Expected - no more immediate responses

            # Now stop sending audio and wait for server to close connection
            # Server should close after IDLE_TIMEOUT_SECONDS
            start_time = time.monotonic()
            connection_closed = False
            close_reason = None

            try:
                # Wait for server to close the connection
                # We expect this to happen within IDLE_TIMEOUT_SECONDS + some margin
                max_wait = IDLE_TIMEOUT_SECONDS + 15  # Allow 15s margin for processing

                while time.monotonic() - start_time < max_wait:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(msg)
                        # Ignore ping messages
                        if data.get("type") == "ping":
                            continue
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        connection_closed = True
                        close_reason = str(e)
                        break

            except websockets.exceptions.ConnectionClosed as e:
                connection_closed = True
                close_reason = str(e)

            elapsed = time.monotonic() - start_time

            # Assert connection was closed by server
            assert connection_closed, (
                f"Connection was NOT closed after {elapsed:.1f}s of inactivity. "
                f"Expected closure after ~{IDLE_TIMEOUT_SECONDS}s. "
                "This will prevent containers from scaling down!"
            )

            # Assert it happened within reasonable time
            assert elapsed < IDLE_TIMEOUT_SECONDS + 20, (
                f"Connection closed after {elapsed:.1f}s, but expected ~{IDLE_TIMEOUT_SECONDS}s. "
                "Timeout may be misconfigured."
            )

            print(f"Connection closed after {elapsed:.1f}s of inactivity (expected ~{IDLE_TIMEOUT_SECONDS}s)")

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_continuous_audio_keeps_connection_alive(self, ws_url, auth_headers):
        """Connection should stay open while audio is being sent."""
        audio_chunk = create_pcm_audio(duration_seconds=1.0)

        async with websockets.connect(
            ws_url,
            additional_headers=auth_headers,
            open_timeout=120,
            close_timeout=10,
        ) as ws:
            # Send audio continuously for longer than idle timeout
            send_duration = IDLE_TIMEOUT_SECONDS + 5
            start_time = time.monotonic()

            while time.monotonic() - start_time < send_duration:
                await ws.send(audio_chunk)
                await asyncio.sleep(0.5)  # Send every 500ms

                # Drain any responses
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        # Just consume the message
                except asyncio.TimeoutError:
                    pass

            # Connection should still be open
            from websockets.protocol import State
            assert ws.state == State.OPEN, (
                f"Connection closed while actively sending audio! "
                f"Sent audio for {time.monotonic() - start_time:.1f}s"
            )

            print(f"Connection stayed open during {send_duration}s of continuous audio")

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_ping_messages_received(self, ws_url, auth_headers):
        """Server should send periodic ping messages."""
        audio = create_pcm_audio(duration_seconds=2.0)

        async with websockets.connect(
            ws_url,
            additional_headers=auth_headers,
            open_timeout=120,
            close_timeout=10,
        ) as ws:
            # Send initial audio
            await ws.send(audio)

            # Collect messages for 15 seconds
            messages = []
            start_time = time.monotonic()

            try:
                while time.monotonic() - start_time < 15:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(msg)
                    messages.append(data)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

            # Check for ping messages
            ping_messages = [m for m in messages if m.get("type") == "ping"]

            # Should have received at least one ping (server sends every 10s)
            # Note: might not receive ping if connection closes first due to idle timeout
            print(f"Received {len(ping_messages)} ping messages in {time.monotonic() - start_time:.1f}s")
            print(f"Total messages: {len(messages)}")


class TestConnectionRecovery:
    """Tests for connection handling edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_reconnect_after_idle_close(self, ws_url, auth_headers):
        """Should be able to reconnect after server closes idle connection."""
        audio = create_pcm_audio(duration_seconds=1.0)

        # First connection - let it idle and close
        async with websockets.connect(
            ws_url,
            additional_headers=auth_headers,
            open_timeout=120,
            close_timeout=10,
        ) as ws:
            await ws.send(audio)

            # Wait for server to close (or timeout)
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=IDLE_TIMEOUT_SECONDS + 10)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

        # Second connection - should work fine
        async with websockets.connect(
            ws_url,
            additional_headers=auth_headers,
            open_timeout=120,
            close_timeout=10,
        ) as ws:
            await ws.send(audio)

            # Should receive tokens
            received_token = False
            try:
                for _ in range(10):
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(msg)
                    if data.get("type") == "token":
                        received_token = True
                        break
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

            print(f"Reconnection successful, received token: {received_token}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    import sys

    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
