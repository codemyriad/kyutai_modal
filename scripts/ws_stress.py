#!/usr/bin/env python3
"""Concurrent WebSocket stress test for STT service.

Sends multiple audio streams in parallel and validates transcription quality.

Usage:
    uv run scripts/ws_stress.py --uri wss://your-app.modal.run/v1/stream --concurrency 10

Dependencies:
    uv pip install websockets soundfile numpy
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


def normalize_text(s: str) -> str:
    """Normalize text for WER comparison."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer(ref: str, hyp: str) -> float:
    """Calculate Word Error Rate using edit distance."""
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()

    if not r:
        return 0.0 if not h else 1.0

    # Classic DP edit distance
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, dp[0] = dp[0], i
        for j, hw in enumerate(h, 1):
            cur = dp[j]
            cost = 0 if rw == hw else 1
            dp[j] = min(
                dp[j] + 1,  # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,  # substitution
            )
            prev = cur

    return dp[-1] / len(r)


def wav24k_to_pcm16_chunks(wav_path: Path, chunk_ms: int):
    """Load WAV and yield PCM16 chunks."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

    if sr != 24000:
        raise ValueError(f"{wav_path} sample rate is {sr}, expected 24000")
    if audio.ndim != 1:
        raise ValueError(f"{wav_path} must be mono")

    # float32 [-1,1] -> int16
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    b = pcm_i16.tobytes()

    # bytes per ms at 24kHz, 16-bit
    bytes_per_ms = 24000 * 2 // 1000
    step = chunk_ms * bytes_per_ms

    for i in range(0, len(b), step):
        yield b[i : i + step]


async def run_one(
    uri: str,
    sample: dict,
    chunk_ms: int,
    wer_max: float,
    realtime: bool = False,
) -> tuple[str, bool, float, str, str]:
    """Run a single WebSocket stream and return results."""
    wav_path = Path(sample["wav24k_path"])
    ref = sample.get("ref_text", "")

    hyp_parts = []

    async with websockets.connect(uri, max_size=2**24) as ws:

        async def sender():
            for chunk in wav24k_to_pcm16_chunks(wav_path, chunk_ms):
                await ws.send(chunk)
                if realtime:
                    await asyncio.sleep(chunk_ms / 1000.0)
            await ws.send(b"EOS")

        async def receiver():
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    continue
                data = json.loads(msg)
                if data.get("text"):
                    hyp_parts.append(data["text"])
                if data.get("final") is True or data.get("status") == "complete":
                    break

        await asyncio.gather(sender(), receiver())

    hyp = " ".join(hyp_parts).strip()
    w = wer(ref, hyp) if ref else 0.0
    ok = w <= wer_max or not ref  # Pass if no reference text
    return sample["id"], ok, w, ref, hyp


async def main() -> None:
    ap = argparse.ArgumentParser(description="WebSocket STT stress test")
    ap.add_argument(
        "--uri",
        required=True,
        help="WebSocket URI (wss://your-app.modal.run/v1/stream)",
    )
    ap.add_argument(
        "--manifest",
        default="samples/manifest.jsonl",
        help="Path to samples manifest",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent streams",
    )
    ap.add_argument(
        "--chunk-ms",
        type=int,
        default=480,
        help="Audio chunk size in milliseconds",
    )
    ap.add_argument(
        "--wer-max",
        type=float,
        default=0.25,
        help="Maximum acceptable WER",
    )
    ap.add_argument(
        "--realtime",
        action="store_true",
        help="Send audio at real-time pace (for latency testing)",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run scripts/prepare_samples.py first to create test samples")
        sys.exit(1)

    samples = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not samples:
        print("Error: No samples in manifest")
        sys.exit(1)

    # Cycle samples if fewer than concurrency
    jobs = [samples[i % len(samples)] for i in range(args.concurrency)]

    print(f"Running {args.concurrency} concurrent streams...")
    print(f"URI: {args.uri}")
    print(f"Chunk size: {args.chunk_ms}ms")
    print(f"WER threshold: {args.wer_max}")
    print()

    results = await asyncio.gather(
        *(run_one(args.uri, s, args.chunk_ms, args.wer_max, args.realtime) for s in jobs)
    )

    # Print results
    passed = 0
    failed = 0
    bad_results = []

    for sid, ok, w, ref, hyp in results:
        status = "OK" if ok else "FAIL"
        print(f"{sid}: {status}  WER={w:.3f}")
        if ok:
            passed += 1
        else:
            failed += 1
            bad_results.append((sid, w, ref, hyp))

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(results)}")

    if bad_results:
        print("\nFailures (showing first 2):")
        for sid, w, ref, hyp in bad_results[:2]:
            print(f"\n--- {sid} WER={w:.3f}")
            print(f"REF: {ref[:200]}..." if len(ref) > 200 else f"REF: {ref}")
            print(f"HYP: {hyp[:200]}..." if len(hyp) > 200 else f"HYP: {hyp}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
