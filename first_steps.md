## Development strategy: “sealed GPU core” + CPU-first iteration

### 0) Fix the invariants (so everything else can move fast)

From your deployment notes, the hard constraints that should become *testable constants* are: **24kHz mono**, **80ms frames = 1920 samples**, and a *recommended* streaming chunk around **480ms = 11520 samples = 23040 bytes (PCM16)**. 
On Modal, concurrency should be expressed with `@modal.concurrent(max_inputs=..., target_inputs=...)` so you can absorb spikes while autoscaling toward the target. ([Modal][1])

---

## 1) Code organization: isolate the GPU-dependent “engine”

Create a tiny, stable surface area that the rest of the system calls:

**`Engine` protocol (sealed boundary)**

* `transcribe_batch(audio_list: list[np.ndarray]) -> list[str]`  *(pure function-ish; easiest to test)*
* optional: `warmup()` *(captured in snapshot)*

Everything else (WebSocket framing, buffering, batching, metrics, auth, routing) depends only on this protocol and can be developed/tested on CPU.

**Suggested repo layout**

```
src/
  stt/
    constants.py        # SR=24000, FRAME=1920, CHUNK=11520, etc.
    audio.py            # pcm16<->float32, chunking, resampling helpers
    engine/
      protocol.py       # Engine Protocol
      kyutai.py         # REAL engine (transformers + cuda)
      fake.py           # CPU fake engine (deterministic for tests)
    server.py           # FastAPI WS: depends only on Engine
    batching.py         # cross-stream batcher: depends only on Engine
tests/
  unit/                 # CPU-only
  integration_gpu/      # runs on Modal GPU
samples/
  manifest.jsonl
  wav24k/...
scripts/
  prepare_samples.py
  ws_stress.py
```

---

## 2) Modal dev loop: “hot reload” + model cached outside your code

### 2.1 Use `modal serve` for rapid iteration on the GPU node

`modal serve` runs your web endpoints and hot-reloads code. ([Modal][2])
So your development loop becomes: edit locally → Modal reloads → hit `-dev` URL.

### 2.2 Put model weights in a Modal Volume (not in your image) for dev

Bake weights into the image for prod if you want, but for dev you don’t want a 3GB rebuild on every tweak. Use a **Volume** for `/models` so code reloads don’t imply re-downloading or rebuilding. (Volumes require explicit commit/reload semantics.) ([Modal][3])

### 2.3 Keep cold starts low with snapshots once the engine is stable

Enable memory snapshot + GPU snapshot so containers restart fast after reloads / scale-from-zero. ([Modal][4])
(Your notes already use `enable_memory_snapshot=True` + `experimental_options={"enable_gpu_snapshot": True}`.) 

---

## 3) Testing strategy: CPU-default, GPU only for the sealed engine + a few end-to-end checks

### 3.1 CPU unit tests (run everywhere, fast)

Test:

* audio conversions: PCM16 bytes ↔ float32 [-1,1]
* chunking: exactly `FRAME_SAMPLES=1920`, `CHUNK_SAMPLES=11520` boundaries
* websocket session buffering logic (append/flush/EOS)
* batching scheduler (fairness, max batch size, timeout behavior)
* server routing and message formats using a `FakeEngine`

### 3.2 “Sealed” GPU integration tests (only a small suite)

On Modal GPU, run:

* **engine smoke**: load model, warmup, one short clip
* **engine regression**: run N known samples; compare to reference transcript with WER tolerance
* **concurrency**: 10 parallel streams (or whatever your target) and verify no cross-talk + acceptable latency

Modal supports WebSockets on `@modal.asgi_app()` / web endpoints. ([Modal][5])

### 3.3 Golden references: generated once, used everywhere

Your idea (generate transcripts with Whisper and store alongside samples) is the right shape, but treat them as *reference*, not strict equality:

* normalize text (lowercase, strip punctuation, collapse whitespace)
* assert **WER <= threshold** (e.g., 0.25) rather than exact match

---

## 4) Audio samples pipeline (parallel-ready): resample → whisper transcript → manifest

You want short clips (3–10s) in both English and French, and enough variety to detect concurrency issues (silence, numbers, punctuation, accents). Your server expects **24kHz** audio. 

### `scripts/prepare_samples.py` (Whisper + 24kHz assets + manifest)

This:

* converts any input audio to `samples/wav24k/*.wav` (PCM16, mono, 24k)
* runs Whisper (via `faster-whisper`) to produce a reference transcript
* writes `samples/manifest.jsonl` with `{id, wav24k_path, ref_text, sha256, duration_s}`

```python
#!/usr/bin/env python3
import argparse, hashlib, json, subprocess
from pathlib import Path

def sh(*cmd):
    subprocess.check_call(cmd)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ffmpeg_to_wav24k(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh(
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "24000",
        "-c:a", "pcm_s16le",
        str(dst),
    )

def duration_seconds(p: Path) -> float:
    # ffprobe JSON
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(p),
    ])
    return float(json.loads(out)["format"]["duration"])

def whisper_transcribe(wav_path: Path, model: str) -> str:
    # faster-whisper (pip install faster-whisper)
    from faster_whisper import WhisperModel
    wm = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _info = wm.transcribe(str(wav_path), vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory with raw audio files")
    ap.add_argument("--out-dir", default="samples", help="Output samples directory")
    ap.add_argument("--whisper-model", default="large-v3", help="faster-whisper model name")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav24k"
    manifest_path = out_dir / "manifest.jsonl"
    wav_dir.mkdir(parents=True, exist_ok=True)

    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
    raw_files = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in exts]

    with manifest_path.open("w", encoding="utf-8") as mf:
        for src in raw_files:
            sample_id = src.stem
            wav24k = wav_dir / f"{sample_id}.wav"
            ffmpeg_to_wav24k(src, wav24k)

            ref = whisper_transcribe(wav24k, args.whisper_model)
            rec = {
                "id": sample_id,
                "wav24k_path": str(wav24k),
                "ref_text": ref,
                "sha256": sha256_file(wav24k),
                "duration_s": duration_seconds(wav24k),
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {manifest_path} with {len(raw_files)} samples")

if __name__ == "__main__":
    main()
```

Install notes (CLI-first):

* `sudo apt-get install -y ffmpeg`
* `pip install faster-whisper`

---

## 5) Concurrent inference validation plan (10 streams) + a stress harness

### 5.1 What “correct” means under concurrency

For each client stream:

* the server returns only that stream’s text (no cross-talk)
* final transcript meets WER threshold against the sample’s `ref_text`
* timing is sane (e.g., completes within `audio_duration + budget`)

Use the known streaming chunk size guidance: **send 480ms chunks** (or smaller) derived from the 24kHz/80ms frame structure. 

### `scripts/ws_stress.py` (parallel WebSocket clients)

This feeds N samples concurrently and validates final transcripts (you can promote this into `pytest` later).

```python
#!/usr/bin/env python3
import argparse, asyncio, json, re
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # classic DP edit distance
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, dp[0] = dp[0], i
        for j, hw in enumerate(h, 1):
            cur = dp[j]
            cost = 0 if rw == hw else 1
            dp[j] = min(
                dp[j] + 1,      # del
                dp[j - 1] + 1,  # ins
                prev + cost,    # sub
            )
            prev = cur
    return dp[-1] / len(r)

def wav24k_to_pcm16_chunks(wav_path: Path, chunk_ms: int):
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != 24000:
        raise ValueError(f"{wav_path} sample rate is {sr}, expected 24000")
    if audio.ndim != 1:
        raise ValueError(f"{wav_path} must be mono")
    # float32 [-1,1] -> int16 little endian
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    b = pcm_i16.tobytes()
    bytes_per_ms = 24000 * 2 // 1000
    step = chunk_ms * bytes_per_ms
    for i in range(0, len(b), step):
        yield b[i:i+step]

async def run_one(uri: str, sample: dict, chunk_ms: int, wer_max: float):
    wav_path = Path(sample["wav24k_path"])
    ref = sample["ref_text"]

    hyp_parts = []
    async with websockets.connect(uri, max_size=2**24) as ws:
        async def sender():
            for chunk in wav24k_to_pcm16_chunks(wav_path, chunk_ms):
                await ws.send(chunk)
                # optional pacing: uncomment to mimic real-time streaming
                # await asyncio.sleep(chunk_ms / 1000.0)
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
    w = wer(ref, hyp)
    ok = w <= wer_max
    return sample["id"], ok, w, ref, hyp

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help="wss://.../v1/transcribe (or /v1/stream)")
    ap.add_argument("--manifest", default="samples/manifest.jsonl")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--chunk-ms", type=int, default=480)
    ap.add_argument("--wer-max", type=float, default=0.25)
    args = ap.parse_args()

    samples = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    if not samples:
        raise SystemExit("No samples in manifest")

    # cycle samples if fewer than concurrency
    jobs = [samples[i % len(samples)] for i in range(args.concurrency)]
    results = await asyncio.gather(*(run_one(args.uri, s, args.chunk_ms, args.wer_max) for s in jobs))

    bad = [r for r in results if not r[1]]
    for sid, ok, w, _ref, _hyp in results:
        print(f"{sid}: {'OK' if ok else 'FAIL'}  WER={w:.3f}")
    if bad:
        print("\nFailures (showing first 2):")
        for sid, _ok, w, ref, hyp in bad[:2]:
            print(f"\n--- {sid} WER={w:.3f}\nREF: {ref}\nHYP: {hyp}")
        raise SystemExit(2)

if __name__ == "__main__":
    asyncio.run(main())
```

Dependencies:

* `pip install websockets soundfile numpy`

---

## 6) How this fits your Modal deployment shape

* Keep your production knobs aligned with your notes: L40S, 10 target streams, burst above target, and snapshots for cold-start control. 
* In dev, use `modal serve` + Volume-cached model to iterate on everything *outside* `engine/kyutai.py`. ([Modal][2])
* When `engine/kyutai.py` changes, run the GPU integration suite on Modal; otherwise, run CPU tests locally/CI.

This gives you a stable, well-tested GPU core, while letting nearly all server/concurrency/product work proceed on non-GPU machines.

[1]: https://modal.com/docs/reference/modal.concurrent?utm_source=chatgpt.com "modal.concurrent | Modal Docs"
[2]: https://modal.com/docs/reference/cli/serve?utm_source=chatgpt.com "modal serve | Modal Docs"
[3]: https://modal.com/docs/guide/volumes?utm_source=chatgpt.com "Volumes | Modal Docs"
[4]: https://modal.com/docs/guide/memory-snapshot?utm_source=chatgpt.com "Memory Snapshot | Modal Docs"
[5]: https://modal.com/docs/guide/webhooks?utm_source=chatgpt.com "Web endpoints | Modal Docs"
