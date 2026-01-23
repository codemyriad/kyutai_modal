#!/usr/bin/env python3
"""Debug script to see exactly what the STT server returns."""

import asyncio
import json
import os
import sys
import time

import numpy as np
import websockets


async def test_stt_server(url: str, wav_path: str):
    """Connect to STT server and print all responses."""
    import soundfile as sf

    # Load audio
    audio, sr = sf.read(wav_path, dtype="float32")
    print(f"Loaded {wav_path}: {len(audio)/sr:.2f}s @ {sr}Hz")

    # WebSocket URL - keep as-is for custom URLs
    ws_url = url.replace("https://", "wss://").replace("http://", "ws://")

    print(f"Connecting to {ws_url}...")

    # Modal proxy auth
    headers = {}
    modal_key = os.environ.get("MODAL_KEY")
    modal_secret = os.environ.get("MODAL_SECRET")
    if modal_key and modal_secret:
        headers["Modal-Key"] = modal_key
        headers["Modal-Secret"] = modal_secret
        print("Using Modal proxy authentication")

    try:
        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            open_timeout=120,
            close_timeout=10,
        ) as ws:
            print("Connected!")

            # Send audio in chunks (raw PCM float32)
            chunk_size = int(sr * 0.08)  # 80ms chunks
            chunks_sent = 0
            recv_count = 0
            tokens = []

            async def send_audio():
                nonlocal chunks_sent
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i : i + chunk_size]
                    if chunk.size == 0:
                        continue
                    # Send raw PCM float32 bytes
                    data = np.asarray(chunk, dtype=np.float32).tobytes()
                    await ws.send(data)
                    chunks_sent += 1
                    if chunks_sent <= 5 or chunks_sent % 50 == 0:
                        print(f"  Sent chunk {chunks_sent} ({len(data)} bytes)")
                    await asyncio.sleep(0.02)  # Send faster than realtime

                print(f"Audio send complete: {chunks_sent} chunks")

            async def recv_messages():
                nonlocal recv_count
                start = time.perf_counter()
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        recv_count += 1
                        elapsed = time.perf_counter() - start

                        if isinstance(msg, str):
                            # JSON message from our proxy
                            try:
                                data = json.loads(msg)
                                msg_type = data.get("type", "unknown")
                                if msg_type == "token":
                                    text = data.get("text", "")
                                    tokens.append(text)
                                    print(f"  [{elapsed:.2f}s] TOKEN: {repr(text)}")
                                elif msg_type == "ping":
                                    print(f"  [{elapsed:.2f}s] PING")
                                elif msg_type == "vad_end":
                                    print(f"  [{elapsed:.2f}s] VAD_END")
                                else:
                                    print(f"  [{elapsed:.2f}s] JSON: {data}")
                            except json.JSONDecodeError:
                                print(f"  [{elapsed:.2f}s] STRING: {repr(msg)}")
                        else:
                            print(f"  [{elapsed:.2f}s] BYTES ({len(msg)}): {msg[:50]}...")
                    except asyncio.TimeoutError:
                        print(f"  Receive timeout (no message for 10s)")
                        break
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"  Connection closed: {e}")
                        break

            # Run send and recv concurrently
            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(recv_messages())

            await send_task
            print("Waiting for responses (max 15s after send)...")
            try:
                await asyncio.wait_for(recv_task, timeout=15.0)
            except asyncio.TimeoutError:
                pass

            print(f"\nSummary: sent {chunks_sent} chunks, received {recv_count} messages")
            if tokens:
                print(f"Transcription: {''.join(tokens)}")
            else:
                print("No transcription tokens received")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "wss://silviot--kyutai-stt-rust-kyutaisttrustservice-serve.modal.run/v1/stream"
    wav = sys.argv[2] if len(sys.argv) > 2 else "samples/wav24k/chunk_0.wav"
    asyncio.run(test_stt_server(url, wav))
