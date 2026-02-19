#!/usr/bin/env -S uv run
# /// script
# dependencies = ["websockets", "numpy", "httpx"]
# ///
"""Quick smoke test for the local Kyutai STT service.

Checks /health, opens a WebSocket to /v1/stream, sends 3 seconds of silence,
and prints any token responses.
"""

import asyncio
import json
import sys

import httpx
import numpy as np
import websockets

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.101:8000"
WS_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://") + "/v1/stream"
HEALTH_URL = BASE_URL + "/health"


async def main():
    # 1. Health check
    print(f"Checking health at {HEALTH_URL}...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(HEALTH_URL, timeout=10)
        resp.raise_for_status()
        health = resp.json()
        print(f"  Health: {health}")
        assert health["status"] == "ok", f"Unexpected health status: {health}"

    # 2. WebSocket test
    print(f"\nConnecting to {WS_URL}...")
    async with websockets.connect(WS_URL, open_timeout=30) as ws:
        print("  Connected, sending 3s of PCM silence...")

        # Generate 3 seconds of silence at 24kHz float32
        silence = np.zeros(24000 * 3, dtype=np.float32)
        await ws.send(silence.tobytes())

        # Collect responses for up to 10 seconds
        tokens = []
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                if data.get("type") == "token":
                    tokens.append(data.get("text", ""))
                    print(f"  Token: {data.get('text', '')!r}")
        except asyncio.TimeoutError:
            pass

        print(f"\nReceived {len(tokens)} tokens")

    print("\nSmoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
