#!/usr/bin/env python3
"""Measure transcription latency for the STT endpoint."""

import asyncio
import json
import os
import time

import numpy as np
import sphn
import websockets


async def measure_latency(uri: str, wav_path: str = "samples/wav24k/chunk_0.wav", auth_headers: dict | None = None):
    """Send audio and measure time to first response."""
    import soundfile as sf

    # Load real audio file
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 24000:
        raise ValueError(f"Expected 24kHz, got {sr}")

    audio_seconds = len(audio) / 24000
    print(f"Audio: {audio_seconds:.1f}s ({len(audio)} samples)")

    # Encode to Opus in streaming chunks
    encoder = sphn.OpusStreamWriter(24000)
    frame_size = 960  # 40ms at 24kHz (valid Opus frame size)
    opus_chunks = []

    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        # append_pcm expects a 1D array
        encoder.append_pcm(frame)
        chunk = encoder.read_bytes()
        if chunk:
            opus_chunks.append(chunk)

    # Final read to get any remaining data
    final_chunk = encoder.read_bytes()
    if final_chunk:
        opus_chunks.append(final_chunk)

    opus_bytes = b"".join(opus_chunks)
    print(f"Opus encoded: {len(opus_bytes)} bytes")

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

        # Send all audio at once (Opus packets must be complete)
        send_start = time.perf_counter()
        await ws.send(opus_bytes)
        send_time = time.perf_counter() - send_start
        print(f"Sent {len(opus_bytes)} bytes in {send_time:.3f}s")

        # Receive tokens
        recv_start = time.perf_counter()
        first_token_time = None
        all_tokens = []

        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(msg)

                if data.get("type") == "token":
                    text = data.get("text", "")
                    all_tokens.append(text)
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - recv_start
                        print(f"First token in {first_token_time:.3f}s: '{text}'")
                elif data.get("type") == "vad_end":
                    print(f"VAD end detected")
                elif "text" in data:  # Legacy format
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - recv_start
                        print(f"First response in {first_token_time:.3f}s: {data.get('text', '')[:50]}")
                elif data.get("status") == "complete":
                    break
        except asyncio.TimeoutError:
            print("Receive timeout (expected for streaming)")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed (session ended)")

        total_time = time.perf_counter() - send_start
        full_text = "".join(all_tokens)

        print(f"\nTranscription: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
        print(f"\nTotal round-trip: {total_time:.3f}s")
        print(f"  Connect: {connect_time:.2f}s")
        print(f"  Send: {send_time:.3f}s")
        print(f"  First token: {first_token_time:.3f}s" if first_token_time else "  No tokens received")
        print(f"  Total tokens: {len(all_tokens)}")

        return {
            "connect_time": connect_time,
            "send_time": send_time,
            "first_token_time": first_token_time,
            "total_time": total_time,
            "token_count": len(all_tokens),
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
