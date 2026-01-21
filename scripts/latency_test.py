#!/usr/bin/env python3
"""Measure transcription latency for the STT endpoint."""

import asyncio
import json
import os
import time

import numpy as np
import websockets

# Modal workspace name (set via MODAL_WORKSPACE env var or change default)
MODAL_WORKSPACE = os.environ.get("MODAL_WORKSPACE", "YOUR_WORKSPACE")


async def measure_latency(uri: str, wav_path: str = "samples/wav24k/chunk_0.wav", auth_headers: dict | None = None):
    """Send audio and measure time to first response."""
    import soundfile as sf

    # Load real audio file
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 24000:
        raise ValueError(f"Expected 24kHz, got {sr}")

    audio_seconds = len(audio) / 24000
    print(f"Audio: {audio_seconds:.1f}s ({len(audio)} samples)")

    pcm_bytes = np.asarray(audio, dtype=np.float32).tobytes()
    print(f"PCM payload: {len(pcm_bytes)} bytes")

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

        # Send all audio at once (raw PCM float32 LE, 24kHz mono)
        send_start = time.perf_counter()
        await ws.send(pcm_bytes)
        send_time = time.perf_counter() - send_start
        print(f"Sent {len(pcm_bytes)} bytes in {send_time:.3f}s")

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
                    print("VAD end detected")
                elif data.get("type") == "ping":
                    pass  # Server keepalive, ignore
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


async def run_parallel_test(uri: str, num_streams: int, wav_path: str, auth_headers: dict | None, warmup: bool = True):
    """Run multiple streams in parallel to test concurrent handling."""

    if warmup:
        print("Warmup request (excludes cold boot from stats)...")
        try:
            await measure_latency(uri, wav_path=wav_path, auth_headers=auth_headers)
            print("Warmup complete.\n")
        except Exception as e:
            print(f"Warmup failed: {e}\n")

    print(f"Running {num_streams} parallel streams...")

    async def labeled_test(idx: int):
        result = await measure_latency(uri, wav_path=wav_path, auth_headers=auth_headers)
        return idx, result

    tasks = [labeled_test(i) for i in range(num_streams)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n{'='*50}")
    print("Parallel Test Summary")
    print('='*50)

    successful = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Stream {i+1}: FAILED - {result}")
        else:
            idx, metrics = result
            if metrics.get("first_token_time"):
                successful.append(metrics["first_token_time"])
                print(f"  Stream {i+1}: first_token={metrics['first_token_time']:.3f}s, tokens={metrics['token_count']}")
            else:
                print(f"  Stream {i+1}: No tokens received")

    if successful:
        print(f"\n  Avg first token: {sum(successful)/len(successful):.3f}s")
        print(f"  Min first token: {min(successful):.3f}s")
        print(f"  Max first token: {max(successful):.3f}s")

    return successful


def get_app_url(app_name: str) -> str:
    """Get WebSocket URL for a Modal app."""
    # Modal URL pattern: wss://{workspace}--{app-name}-{class}-{method}.modal.run
    return f"wss://{MODAL_WORKSPACE}--{app_name}-kyutaisttservice-serve.modal.run/v1/stream"


async def deploy_gpu_variant(gpu: str, auth_headers: dict | None) -> str | None:
    """Deploy a GPU variant and return its URL."""
    import subprocess

    app_name = f"kyutai-stt-{gpu.lower()}"
    print(f"Deploying {gpu} to app '{app_name}'...")

    env = os.environ.copy()
    env["KYUTAI_GPU"] = gpu
    env["KYUTAI_APP_NAME"] = app_name

    proc = subprocess.run(
        ["uvx", "modal", "deploy", "src/stt/modal_app.py"],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"Deploy failed for {gpu}:")
        print(proc.stderr)
        return None

    return get_app_url(app_name)


async def compare_gpus(gpus: list[str], num_streams: int, wav_path: str, auth_headers: dict | None):
    """Deploy each GPU to separate apps, then test all in parallel."""
    import subprocess

    # Step 1: Deploy all GPU variants (can be done in parallel)
    print("=" * 60)
    print("DEPLOYING GPU VARIANTS")
    print("=" * 60)

    gpu_urls = {}
    for gpu in gpus:
        url = await deploy_gpu_variant(gpu, auth_headers)
        if url:
            gpu_urls[gpu] = url
            print(f"  {gpu}: {url}")

    if not gpu_urls:
        print("No deployments succeeded!")
        return

    # Step 2: Warmup all variants in parallel
    print(f"\n{'='*60}")
    print("WARMING UP ALL VARIANTS")
    print('='*60)

    async def warmup_one(gpu: str, uri: str):
        print(f"  Warming up {gpu}...")
        try:
            await measure_latency(uri, wav_path=wav_path, auth_headers=auth_headers)
            print(f"  {gpu} warm")
            return True
        except Exception as e:
            print(f"  {gpu} warmup failed: {e}")
            return False

    warmup_tasks = [warmup_one(gpu, url) for gpu, url in gpu_urls.items()]
    await asyncio.gather(*warmup_tasks)

    # Step 3: Test each GPU variant
    print(f"\n{'='*60}")
    print("RUNNING LATENCY TESTS")
    print('='*60)

    results = {}
    for gpu, uri in gpu_urls.items():
        print(f"\n--- {gpu} ---")
        latencies = await run_parallel_test(uri, num_streams, wav_path, auth_headers, warmup=False)
        if latencies:
            results[gpu] = {
                "avg": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "samples": latencies,
            }

    # Print comparison
    print(f"\n{'='*60}")
    print("GPU COMPARISON RESULTS")
    print('='*60)
    print(f"{'GPU':<12} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10}")
    print('-'*42)
    for gpu, stats in sorted(results.items(), key=lambda x: x[1]["avg"]):
        print(f"{gpu:<12} {stats['avg']:<10.3f} {stats['min']:<10.3f} {stats['max']:<10.3f}")

    if results:
        best = min(results.items(), key=lambda x: x[1]["avg"])
        print(f"\nBest GPU: {best[0]} (avg {best[1]['avg']:.3f}s)")

    # Show cleanup command
    print(f"\nTo clean up test apps:")
    for gpu in gpu_urls:
        print(f"  modal app stop kyutai-stt-{gpu.lower()}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Measure STT transcription latency")
    parser.add_argument("--parallel", "-p", type=int, default=0,
                        help="Number of parallel streams (0 = sequential mode)")
    parser.add_argument("--runs", "-n", type=int, default=3,
                        help="Number of test runs (sequential mode)")
    parser.add_argument("--wav", type=str, default="samples/wav24k/chunk_0.wav",
                        help="Path to test audio file")
    parser.add_argument("--compare-gpus", type=str, default="",
                        help="Compare GPUs (comma-separated, e.g., 'T4,L4,A10G,A100')")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup request")
    args = parser.parse_args()

    uri = f"wss://{MODAL_WORKSPACE}--kyutai-stt-kyutaisttservice-serve.modal.run/v1/stream"

    # Auth from environment
    modal_key = os.environ.get("MODAL_KEY")
    modal_secret = os.environ.get("MODAL_SECRET")
    auth_headers = None
    if modal_key and modal_secret:
        auth_headers = {"Modal-Key": modal_key, "Modal-Secret": modal_secret}
        print("Using Modal proxy authentication\n")

    if args.compare_gpus:
        gpus = [g.strip() for g in args.compare_gpus.split(",")]
        num_streams = args.parallel if args.parallel > 0 else 3
        await compare_gpus(gpus, num_streams, args.wav, auth_headers)
    elif args.parallel > 0:
        await run_parallel_test(uri, args.parallel, args.wav, auth_headers, warmup=not args.no_warmup)
    else:
        # Sequential mode
        for i in range(args.runs):
            print(f"\n{'='*50}")
            print(f"Test {i+1}/{args.runs}")
            print('='*50)
            await measure_latency(uri, wav_path=args.wav, auth_headers=auth_headers)


if __name__ == "__main__":
    asyncio.run(main())
