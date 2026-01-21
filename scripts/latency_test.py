#!/usr/bin/env python3
"""Measure transcription latency for the STT endpoint."""

import asyncio
import contextlib
import queue
import json
import os
import signal
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import websockets

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:
    Console = Live = Panel = Progress = SpinnerColumn = TextColumn = BarColumn = TaskProgressColumn = TimeElapsedColumn = Table = Text = None
    RICH_AVAILABLE = False

# Modal workspace name (set via MODAL_WORKSPACE env var or change default)
MODAL_WORKSPACE = os.environ.get("MODAL_WORKSPACE", "YOUR_WORKSPACE")
CHUNK_DURATION_S = 0.08  # 80ms chunks to mirror realtime streaming


async def _stream_audio(
    ws,
    audio: np.ndarray,
    sample_rate: int,
    real_time_factor: float,
    progress_update=None,
    playback_buffer=None,
) -> float:
    """Stream audio over WebSocket at (1x, 2x, 4x, ...) realtime speed."""
    chunk_size = int(sample_rate * CHUNK_DURATION_S)
    rtf = max(0.25, real_time_factor)  # avoid zero/negative/too-slow
    send_start = time.perf_counter()
    bytes_sent = 0

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if chunk.size == 0:
            continue
        await ws.send(np.asarray(chunk, dtype=np.float32).tobytes())
        if playback_buffer is not None:
            playback_buffer.write(chunk)
        bytes_sent += chunk.nbytes
        if progress_update:
            progress_update(bytes_sent)
        # Sleep to simulate live streaming (faster if rtf > 1.0)
        await asyncio.sleep(len(chunk) / sample_rate / rtf)

    return time.perf_counter() - send_start


async def _quit_watcher(stop_event: asyncio.Event):
    """Watch stdin for 'q' to request stop."""
    if not sys.stdin.isatty():
        return
    loop = asyncio.get_running_loop()
    while not stop_event.is_set():
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        if line.strip().lower() == "q":
            print("\nQuit requested (q)")
            stop_event.set()
            break


async def measure_latency(
    uri: str,
    wav_path: str = "samples/wav24k/chunk_0.wav",
    auth_headers: dict | None = None,
    real_time_factor: float = 1.0,
    expected_text: str | None = None,
    stop_event: asyncio.Event | None = None,
    playback: bool = False,
    playback_device: int | None = None,
):
    """Send audio in streaming chunks and measure token latencies."""
    import soundfile as sf

    # Load real audio file
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 24000:
        raise ValueError(f"Expected 24kHz, got {sr}")

    # Optional word-level timeline (produced by annotate_samples.py)
    words_path = f"{wav_path}.words.json"
    word_timeline = None
    if os.path.exists(words_path):
        try:
            import json as _json

            word_timeline = _json.loads(Path(words_path).read_text(encoding="utf-8"))
        except Exception:
            word_timeline = None

    playback_stream = None
    playback_buffer = None
    if playback:
        try:
            import sounddevice as sd

            class PlaybackBuffer:
                def __init__(self):
                    self._q = queue.Queue()
                    self._buf = np.array([], dtype=np.float32)

                def write(self, chunk: np.ndarray):
                    self._q.put(chunk.astype(np.float32))

                def read(self, frames: int) -> np.ndarray:
                    out = []
                    needed = frames
                    while needed > 0:
                        if self._buf.size == 0:
                            try:
                                self._buf = self._q.get_nowait()
                            except queue.Empty:
                                out.append(np.zeros(needed, dtype=np.float32))
                                break
                        take = min(needed, self._buf.size)
                        out.append(self._buf[:take])
                        self._buf = self._buf[take:]
                        needed -= take
                    return np.concatenate(out) if out else np.zeros(frames, dtype=np.float32)

            playback_buffer = PlaybackBuffer()

            def callback(outdata, frames, time_info, status):
                data = playback_buffer.read(frames)
                if data.size < frames:
                    data = np.pad(data, (0, frames - data.size))
                outdata[:, 0] = data

            playback_stream = sd.OutputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                device=playback_device,
                callback=callback,
                blocksize=int(sr * CHUNK_DURATION_S),
            )
            playback_stream.start()
            print(f"Playback enabled (device: {playback_device or 'default'})")
        except Exception as exc:
            print(f"Playback disabled (failed to init): {exc}")
            playback_stream = None
            playback_buffer = None

    audio_seconds = len(audio) / 24000
    print(f"Audio: {audio_seconds:.1f}s ({len(audio)} samples)")

    pcm_bytes = np.asarray(audio, dtype=np.float32).tobytes()
    print(f"PCM payload: {len(pcm_bytes)} bytes")

    print(f"Connecting to {uri}...")
    connect_start = time.perf_counter()

    if stop_event and stop_event.is_set():
        return {}

    console = Console() if RICH_AVAILABLE else None
    live = None
    state = {}
    bear_frames = ["ʕ•ᴥ•ʔ", "ʕᵔᴥᵔʔ", "ʕ•̀ᴥ•́ʔ✧", "ʕ•ᴥ•ʔ♪"]

    def render_dashboard():
        if not RICH_AVAILABLE:
            return None
        pct = (
            state["sent_bytes"] / state["total_bytes"]
            if state.get("total_bytes")
            else 0.0
        )
        bar_len = 24
        filled = int(bar_len * pct)
        bar = f"[cyan]{'█'*filled}[/cyan][dim]{'·'*(bar_len-filled)}[/dim] {pct*100:5.1f}%"

        def fmt(v, suffix="s"):
            return f"{v:.3f}{suffix}" if v is not None else "--"

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", ratio=2, style="white")
        table.add_column(justify="right", ratio=1, style="cyan")

        table.add_row("Status", state.get("status", "--"))
        table.add_row("RT factor", f"{state.get('rtf', 1.0)}x")
        table.add_row("Audio", f"{state.get('audio_len', 0):.1f}s")
        table.add_row("Connect", fmt(state.get("connect_time")))
        table.add_row("Send", bar if state.get("total_bytes") else "--")
        table.add_row("Send time", fmt(state.get("send_time")))
        table.add_row("Tokens", str(state.get("tokens", 0)))
        table.add_row("First token", fmt(state.get("first_token")))
        table.add_row("10th token", fmt(state.get("tenth_token")))
        table.add_row("Last token", fmt(state.get("last_token")))
        table.add_row("Total time", fmt(state.get("total_time")))
        if state.get("words_total"):
            table.add_row(
                "Words",
                f"{state.get('words_matched', 0)}/{state.get('words_total')} "
                f"({(state.get('words_matched', 0)/state.get('words_total'))*100:4.1f}%)",
            )
        if state.get("similarity") is not None:
            table.add_row("Similarity", f"{state['similarity']*100:5.1f}%")

        bear = bear_frames[state.get("bear_idx", 0) % len(bear_frames)]
        content = table

        if state.get("expected_text") and Text:
            exp_text = state.get("expected_render") or Text(state["expected_text"])
            grid = Table.grid(expand=True)
            grid.add_column(ratio=1)
            grid.add_column(ratio=2)
            grid.add_row(table, Panel(exp_text, title="Expected vs recognized", border_style="green"))
            content = grid

        return Panel(content, title=f"[magenta]{bear} Kyutai STT Latency[/magenta]", border_style="magenta")

    async with websockets.connect(
        uri,
        additional_headers=auth_headers,
        open_timeout=180,
        close_timeout=10,
    ) as ws:
        connect_time = time.perf_counter() - connect_start
        print(f"Connected in {connect_time:.2f}s")

        # Init UI state
        state = {
            "status": "connected",
            "rtf": real_time_factor,
            "audio_len": audio_seconds,
            "sent_bytes": 0,
            "total_bytes": audio.nbytes,
            "tokens": 0,
            "connect_time": connect_time,
            "first_token": None,
            "tenth_token": None,
            "last_token": None,
            "send_time": None,
            "total_time": None,
            "bear_idx": 0,
            "expected_text": expected_text,
            "expected_render": None,
            "similarity": None,
            "word_timeline": word_timeline,
            "words_matched": 0,
            "words_total": len(word_timeline) if word_timeline else 0,
        }

        if RICH_AVAILABLE:
            live = Live(render_dashboard(), console=console, refresh_per_second=12, screen=True)
            live.start()

        # Stream audio in chunks to mirror realtime usage
        print(f"Streaming audio at {real_time_factor}x realtime...")
        start_time = time.perf_counter()

        def progress_update(sent_bytes: int):
            state["sent_bytes"] = sent_bytes
            state["status"] = "streaming"
            if live:
                live.update(render_dashboard())

        send_task = asyncio.create_task(
            _stream_audio(
                ws,
                audio,
                sr,
                real_time_factor,
                progress_update=progress_update,
                playback_buffer=playback_buffer,
            )
        )

        # Receive tokens
        tokens = []
        token_times = []
        last_msg_ts = time.perf_counter()
        recognized_text = ""

        def update_expected_render():
            if not expected_text:
                return
            matcher = SequenceMatcher(None, expected_text, recognized_text)
            state["similarity"] = matcher.ratio()
            if not Text:
                return
            text_out = Text()
            last_idx = 0
            for i, j, n in matcher.get_matching_blocks():
                if i > last_idx:
                    text_out.append(expected_text[last_idx:i], style="dim")
                if n:
                    text_out.append(expected_text[i : i + n], style="bold green")
                last_idx = i + n
            if last_idx < len(expected_text):
                text_out.append(expected_text[last_idx:], style="dim")
            state["expected_render"] = text_out
            if state.get("word_timeline"):
                recognized_words = [w.strip(" ,.!?;:").lower() for w in recognized_text.split() if w.strip()]
                expected_words = [w["word"].strip(" ,.!?;:").lower() for w in state["word_timeline"]]
                matched = 0
                for rw, ew in zip(recognized_words, expected_words):
                    if rw == ew:
                        matched += 1
                    else:
                        break
                state["words_matched"] = matched

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                # Stop after inactivity once audio is fully sent
                timeout = 2.0
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if send_task.done() and (time.perf_counter() - last_msg_ts) > 5.0:
                        break
                    if stop_event and stop_event.is_set():
                        break
                    continue

                data = json.loads(msg)
                last_msg_ts = time.perf_counter()

                if data.get("type") == "token":
                    text = data.get("text", "")
                    tokens.append(text)
                    recognized_text = "".join(tokens)
                    token_times.append(last_msg_ts - start_time)
                    if len(token_times) == 1:
                        print(f"First token in {token_times[0]:.3f}s: '{text}'")
                        state["first_token"] = token_times[0]
                    if len(token_times) == 10:
                        state["tenth_token"] = token_times[9]
                    state["last_token"] = token_times[-1]
                    state["tokens"] = len(tokens)
                    state["bear_idx"] = len(tokens)
                    update_expected_render()
                    if live:
                        live.update(render_dashboard())
                elif data.get("type") == "vad_end":
                    print("VAD end detected")
                elif data.get("type") == "ping":
                    pass  # Server keepalive, ignore
                elif "text" in data:  # Legacy format
                    tokens.append(data.get("text", ""))
                    recognized_text = "".join(tokens)
                    token_times.append(last_msg_ts - start_time)
                    if len(token_times) == 1:
                        print(f"First response in {token_times[0]:.3f}s: {data.get('text', '')[:50]}")
                        state["first_token"] = token_times[0]
                    if len(token_times) == 10:
                        state["tenth_token"] = token_times[9]
                    state["last_token"] = token_times[-1]
                    state["tokens"] = len(tokens)
                    state["bear_idx"] = len(tokens)
                    update_expected_render()
                    if live:
                        live.update(render_dashboard())
                elif data.get("status") == "complete":
                    break
        except asyncio.TimeoutError:
            print("Receive timeout (expected for streaming)")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed (session ended)")

        if stop_event and stop_event.is_set() and not send_task.done():
            send_task.cancel()
            with contextlib.suppress(Exception):
                await send_task
        send_time = await send_task
        total_time = time.perf_counter() - start_time
        full_text = "".join(tokens)
        state["send_time"] = send_time
        state["total_time"] = total_time
        state["status"] = "complete"
    if live:
        live.update(render_dashboard())
        live.stop()

    if playback_stream:
        with contextlib.suppress(Exception):
            playback_stream.stop()
            playback_stream.close()

    print(f"\nTranscription: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
    print(f"\nTotal round-trip: {total_time:.3f}s")
    print(f"  Connect: {connect_time:.2f}s")
    print(f"  Send: {send_time:.3f}s")
    if token_times:
        late = token_times[min(len(token_times) - 1, 9)]
        print(f"  First token: {token_times[0]:.3f}s")
        print(f"  10th token: {late:.3f}s" if len(token_times) >= 10 else f"  Last token: {token_times[-1]:.3f}s")
        print(f"  Final token: {token_times[-1]:.3f}s")
        if expected_text and state.get("similarity") is not None:
            print(f"  Similarity vs expected: {state['similarity']*100:5.1f}%")
    else:
        print("  No tokens received")
    print(f"  Total tokens: {len(tokens)}")

    return {
        "connect_time": connect_time,
        "send_time": send_time,
        "first_token_time": token_times[0] if token_times else None,
        "last_token_time": token_times[-1] if token_times else None,
        "total_time": total_time,
        "token_count": len(tokens),
        "similarity": state.get("similarity"),
    }


async def run_parallel_test(
    uri: str,
    num_streams: int,
    wav_path: str,
    auth_headers: dict | None,
    real_time_factor: float,
    expected_text: str | None,
    warmup: bool = True,
    stop_event: asyncio.Event | None = None,
    playback: bool = False,
    playback_device: int | None = None,
):
    """Run multiple streams in parallel to test concurrent handling."""

    if warmup:
        print("Warmup request (excludes cold boot from stats)...")
        try:
            await measure_latency(
                uri,
                wav_path=wav_path,
                auth_headers=auth_headers,
                real_time_factor=real_time_factor,
                expected_text=expected_text,
                stop_event=stop_event,
            )
            print("Warmup complete.\n")
        except Exception as e:
            print(f"Warmup failed: {e}\n")

    print(f"Running {num_streams} parallel streams...")

    async def labeled_test(idx: int):
        result = await measure_latency(
            uri,
            wav_path=wav_path,
            auth_headers=auth_headers,
            real_time_factor=real_time_factor,
            expected_text=expected_text,
            stop_event=stop_event,
            playback=playback,
            playback_device=playback_device,
        )
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
                last = metrics.get("last_token_time")
                print(
                    f"  Stream {i+1}: "
                    f"first={metrics['first_token_time']:.3f}s, "
                    f"last={(last if last is not None else float('nan')):.3f}s, "
                    f"tokens={metrics['token_count']}"
                )
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


async def compare_gpus(
    gpus: list[str],
    num_streams: int,
    wav_path: str,
    auth_headers: dict | None,
    real_time_factor: float,
    expected_text: str | None,
    stop_event: asyncio.Event | None = None,
    playback: bool = False,
    playback_device: int | None = None,
):
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
            await measure_latency(
                uri,
                wav_path=wav_path,
                auth_headers=auth_headers,
                real_time_factor=real_time_factor,
                expected_text=expected_text,
                stop_event=stop_event,
                playback=playback,
                playback_device=playback_device,
            )
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
        latencies = await run_parallel_test(
            uri,
            num_streams,
            wav_path,
            auth_headers,
            real_time_factor,
            expected_text,
            warmup=False,
            stop_event=stop_event,
            playback=playback,
            playback_device=playback_device,
        )
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
    parser.add_argument("--rtf", type=float, default=1.0,
                        help="Realtime factor for streaming (1.0=real time, 2.0=2x faster, 4.0=4x)")
    parser.add_argument("--compare-gpus", type=str, default="",
                        help="Compare GPUs (comma-separated, e.g., 'T4,L4,A10G,A100')")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup request")
    parser.add_argument("--expected", type=str, default=None,
                        help="Expected transcript text (for visualization/quality)")
    parser.add_argument("--expected-file", type=str, default=None,
                        help="Path to expected transcript text file (falls back to <wav>.txt if present)")
    parser.add_argument("--playback", action="store_true",
                        help="Play audio locally while streaming")
    parser.add_argument("--playback-device", type=int, default=None,
                        help="Output device index for playback (sounddevice)")
    args = parser.parse_args()

    uri = f"wss://{MODAL_WORKSPACE}--kyutai-stt-kyutaisttservice-serve.modal.run/v1/stream"
    expected_text = args.expected
    if expected_text is None:
        candidate = args.expected_file
        if candidate is None:
            default_txt = f"{args.wav}.txt"
            candidate = default_txt if os.path.exists(default_txt) else None
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    expected_text = f.read().strip()
            except Exception:
                expected_text = None

    # Auth from environment
    modal_key = os.environ.get("MODAL_KEY")
    modal_secret = os.environ.get("MODAL_SECRET")
    auth_headers = None
    if modal_key and modal_secret:
        auth_headers = {"Modal-Key": modal_key, "Modal-Secret": modal_secret}
        print("Using Modal proxy authentication\n")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
    except NotImplementedError:
        pass
    quit_task = asyncio.create_task(_quit_watcher(stop_event))

    try:
        if args.compare_gpus:
            gpus = [g.strip() for g in args.compare_gpus.split(",")]
            num_streams = args.parallel if args.parallel > 0 else 3
            await compare_gpus(
                gpus,
                num_streams,
                args.wav,
                auth_headers,
                args.rtf,
                expected_text,
                stop_event=stop_event,
                playback=args.playback,
                playback_device=args.playback_device,
            )
        elif args.parallel > 0:
            await run_parallel_test(
                uri,
                args.parallel,
                args.wav,
                auth_headers,
                args.rtf,
                expected_text,
                warmup=not args.no_warmup,
                stop_event=stop_event,
                playback=args.playback,
                playback_device=args.playback_device,
            )
        else:
            # Sequential mode
            for i in range(args.runs):
                print(f"\n{'='*50}")
                print(f"Test {i+1}/{args.runs}")
                print('='*50)
                await measure_latency(
                    uri,
                    wav_path=args.wav,
                    auth_headers=auth_headers,
                    real_time_factor=args.rtf,
                    expected_text=expected_text,
                    stop_event=stop_event,
                    playback=args.playback,
                    playback_device=args.playback_device,
                )
                if stop_event.is_set():
                    break
    finally:
        stop_event.set()
        quit_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await quit_task
        # Restore terminal if rich left it in alt screen
        if RICH_AVAILABLE and console:
            with contextlib.suppress(Exception):
                console.print("\033[?1049l")  # exit alt screen


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted, exiting cleanly.")
