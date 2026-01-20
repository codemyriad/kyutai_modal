#!/usr/bin/env python3
"""Measure transcription latency for the STT endpoint."""

import asyncio
import json
import os
import time

import numpy as np
import websockets


async def measure_latency(uri: str, wav_path: str = "samples/wav24k/chunk_0.wav", auth_headers: dict | None = None):
    """Send audio and measure time to first response."""
    import soundfile as sf

    # Load real audio file
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 24000:
        raise ValueError(f"Expected 24kHz, got {sr}")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    audio_seconds = len(audio) / 24000

    print(f"Audio: {audio_seconds:.1f}s ({len(pcm)} bytes)")
    print(f"Connecting to {uri}...")
    connect_start = time.perf_counter()

    async with websockets.connect(
        uri,
        additional_headers=auth_headers,
        open_timeout=180,
        close_timeout=10,
    ) as ws:
        connect_time = time.perf_counter() - connect_start
        print(f"Connected in {connect_time:.2f}s")

        # Send audio
        send_start = time.perf_counter()
        await ws.send(pcm)
        await ws.send(b"EOS")
        send_time = time.perf_counter() - send_start
        print(f"Sent {len(pcm)} bytes in {send_time:.3f}s")

        # Wait for response
        recv_start = time.perf_counter()
        first_response = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=60)
            data = json.loads(msg)
            if first_response is None and "text" in data:
                first_response = time.perf_counter() - recv_start
                print(f"First transcription in {first_response:.3f}s: {data.get('text', '')[:50]}")
            if data.get("status") == "complete":
                break

        total_time = time.perf_counter() - send_start
        print(f"\nTotal round-trip: {total_time:.3f}s")
        print(f"  Connect: {connect_time:.2f}s")
        print(f"  Send: {send_time:.3f}s")
        print(f"  First response: {first_response:.3f}s" if first_response else "  No transcription returned")

        return {
            "connect_time": connect_time,
            "send_time": send_time,
            "first_response": first_response,
            "total_time": total_time,
        }


async def main():
    uri = "wss://silviot--kyutai-stt-kyutaisttservice-serve.modal.run/v1/stream"

    # Auth from environment
    modal_key = os.environ.get("MODAL_KEY")
    modal_secret = os.environ.get("MODAL_SECRET")
    auth_headers = None
    if modal_key and modal_secret:
        auth_headers = {"Modal-Key": modal_key, "Modal-Secret": modal_secret}
        print("Using Modal proxy authentication\n")

    # Run 3 tests to see cold vs warm latency
    for i in range(3):
        print(f"\n{'='*50}")
        print(f"Test {i+1}/3")
        print('='*50)
        await measure_latency(uri, auth_headers=auth_headers)
        if i < 2:
            print("\nWaiting 5s before next test...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
