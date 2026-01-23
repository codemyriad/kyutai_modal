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
from pathlib import Path

import numpy as np
import websockets

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:
    Console = Live = Panel = Text = Group = None
    RICH_AVAILABLE = False


class PlaybackBuffer:
    """Thread-safe buffer for streaming audio playback."""

    def __init__(self):
        self._q: queue.Queue[np.ndarray] = queue.Queue()
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

# Modal workspace name (set via MODAL_WORKSPACE env var or change default)
MODAL_WORKSPACE = os.environ.get("MODAL_WORKSPACE", "YOUR_WORKSPACE")
CHUNK_DURATION_S = 0.08  # 80ms chunks to mirror realtime streaming


def _normalize_word(w: str) -> str:
    """Normalize a word for comparison."""
    return w.lower().strip(",.!?;:'\"")


def _words_match(w1: str, w2: str) -> bool:
    """Check if two words match, allowing for minor spelling differences."""
    n1 = _normalize_word(w1)
    n2 = _normalize_word(w2)
    if n1 == n2:
        return True
    # Allow edit distance of 1 for words of 5+ chars (handles memorising/memorizing)
    if len(n1) >= 5 and len(n2) >= 5:
        if abs(len(n1) - len(n2)) <= 1:
            # Simple check: same start and end, differ by at most 1 char
            diffs = sum(1 for a, b in zip(n1, n2) if a != b)
            diffs += abs(len(n1) - len(n2))
            return diffs <= 1
    return False


def _color_diff(expected: str, actual: str) -> str:
    """
    Color the actual text using greedy prefix-aligned matching.

    - Green: matching words
    - Red: extra words in actual (not in expected)
    - Dim + strikethrough: missing words (skipped in expected)
    - Dim: pending words (not yet reached in expected)

    Uses greedy matching optimized for streaming transcription:
    each actual word matches with the earliest available expected word.
    """
    if not actual:
        return "[dim]...[/dim]"

    exp_words = expected.split()
    act_words = actual.split()

    # Greedy prefix-aligned matching
    parts = []
    exp_idx = 0

    for act_word in act_words:
        # Find the first matching expected word from current position
        found_idx = None
        for i in range(exp_idx, len(exp_words)):
            if _words_match(exp_words[i], act_word):
                found_idx = i
                break

        if found_idx is not None:
            # Mark all skipped expected words as missing (strikethrough)
            for j in range(exp_idx, found_idx):
                parts.append(f"[dim strike]{exp_words[j]}[/dim strike]")
            # Mark this as a match
            parts.append(f"[green]{act_word}[/green]")
            exp_idx = found_idx + 1
        else:
            # No match found - this is an extra word
            parts.append(f"[red]{act_word}[/red]")

    # Remaining expected words are pending (future text, just dim)
    for i in range(exp_idx, len(exp_words)):
        parts.append(f"[dim]{exp_words[i]}[/dim]")

    return " ".join(parts)


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

    try:
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
    except websockets.exceptions.ConnectionClosed:
        # Server closed connection (e.g., finished processing) - this is normal
        pass

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
    live: "Live | None" = None,
    render_fn: "callable | None" = None,
    state: dict | None = None,
):
    """Send audio in streaming chunks and measure token latencies.

    If `live` and `render_fn` are provided, uses an external Rich Live display
    instead of creating its own (for persistent TUI across multiple tests).
    If `state` is provided, updates it in place (for external state tracking).
    """
    import soundfile as sf

    # Load real audio file
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 24000:
        raise ValueError(f"Expected 24kHz, got {sr}")

    # Add 2 seconds of silence at the end to let the model finish
    silence = np.zeros(int(sr * 2), dtype=np.float32)
    audio = np.concatenate([audio, silence])

    playback_stream = None
    playback_buffer = None
    if playback:
        try:
            import sounddevice as sd

            # Disable input device to prevent BT headphones switching to handsfree profile
            sd.default.device = (None, playback_device)

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
        except Exception as exc:
            print(f"Playback disabled (failed to init): {exc}")
            playback_stream = None
            playback_buffer = None

    audio_seconds = len(audio) / sr
    if not RICH_AVAILABLE:
        print(f"Audio: {audio_seconds:.1f}s ({len(audio)} samples)")
        print(f"Connecting to {uri}...")

    if stop_event and stop_event.is_set():
        return {}

    console = Console() if RICH_AVAILABLE else None
    external_live = live is not None  # Track if we're using an external display
    if state is None:
        state = {}
    bear_frames = ["ʕ•ᴥ•ʔ", "ʕᵔᴥᵔʔ", "ʕ•̀ᴥ•́ʔ✧", "ʕ•ᴥ•ʔ♪"]

    # Initialize state for "connecting" phase
    connect_start = time.perf_counter()
    state.update({
        "status": "connecting",
        "rtf": real_time_factor,
        "audio_len": audio_seconds,
        "sent_bytes": 0,
        "total_bytes": audio.nbytes,
        "tokens": 0,
        "connect_start": connect_start,
        "connect_time": None,
        "first_token": None,
        "send_time": None,
        "total_time": None,
        "bear_idx": 0,
        "expected_text": expected_text,
        "recognized_text": "",
    })

    def render_dashboard():
        if not RICH_AVAILABLE:
            return None

        def fmt(v):
            return f"[cyan]{v:.2f}s[/cyan]" if v is not None else "[dim]--[/dim]"

        # Status with color
        status = state.get("status", "--")
        if status == "connecting" and state.get("connect_start"):
            elapsed = time.perf_counter() - state["connect_start"]
            status = f"[yellow]connecting ({elapsed:.1f}s)[/yellow]"
        elif status == "streaming":
            status = f"[cyan]{status}[/cyan]"
        elif status == "complete":
            status = f"[green]{status}[/green]"

        # Compact progress bar
        pct = state["sent_bytes"] / state["total_bytes"] if state.get("total_bytes") else 0.0
        bar_len = 20
        filled = int(bar_len * pct)
        bar = f"[cyan]{'█'*filled}[/cyan][dim]{'·'*(bar_len-filled)}[/dim]"

        # Build compact display
        lines = []
        lines.append(f"[dim]{state.get('audio_len', 0):.1f}s @ {state.get('rtf', 1.0)}x[/dim]  {status}  [dim]connect:[/dim]{fmt(state.get('connect_time'))}  [dim]first:[/dim]{fmt(state.get('first_token'))}")
        lines.append(f"{bar} {pct*100:5.1f}%")

        if state.get("expected_text"):
            expected = state["expected_text"]
            actual = state.get("recognized_text", "")
            lines.append(f"[blue]expect:[/blue] {expected}")
            lines.append(f"[blue]actual:[/blue] {_color_diff(expected, actual)}")

        bear = bear_frames[state.get("bear_idx", 0) % len(bear_frames)]
        content = "\n".join(lines)
        return Panel(content, title=f"[magenta]{bear} STT[/magenta]", border_style="magenta", padding=(0, 1))

    # Use render_fn if provided (for external TUI), otherwise local render_dashboard
    render = render_fn if render_fn else render_dashboard

    # Start Live display BEFORE connecting so we can show connection progress
    if RICH_AVAILABLE and not external_live:
        live = Live(render(), console=console, refresh_per_second=12, screen=True)
        live.start()

    # Background task to update display while connecting
    async def update_while_connecting():
        while state.get("status") == "connecting":
            if live:
                live.update(render())
            await asyncio.sleep(0.25)

    update_task = asyncio.create_task(update_while_connecting()) if live else None

    # Initialize variables that may be used after the try block
    connect_time = 0.0
    send_time = 0.0
    total_time = 0.0
    token_times: list[float] = []
    tokens: list[str] = []
    full_text = ""

    try:
        async with websockets.connect(
            uri,
            additional_headers=auth_headers,
            open_timeout=180,
            close_timeout=10,
        ) as ws:
            connect_time = time.perf_counter() - connect_start
            state["status"] = "connected"
            state["connect_time"] = connect_time

            if update_task:
                update_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await update_task

            if not RICH_AVAILABLE:
                print(f"Connected in {connect_time:.2f}s")

            if live:
                live.update(render())

            if not RICH_AVAILABLE:
                print(f"Streaming audio at {real_time_factor}x realtime...")
            start_time = time.perf_counter()

            def progress_update(sent_bytes: int):
                state["sent_bytes"] = sent_bytes
                state["status"] = "streaming"
                if live:
                    live.update(render())

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
            tokens: list[str] = []
            token_times: list[float] = []
            last_msg_ts = time.perf_counter()
            send_time = 0.0  # Will be updated when send_task completes

            def handle_token(text: str):
                """Process a received token and update state."""
                tokens.append(text)
                token_times.append(last_msg_ts - start_time)
                # Update state directly (avoid closure issues)
                state["recognized_text"] = "".join(tokens)
                state["tokens"] = len(tokens)
                state["bear_idx"] = len(tokens)
                if len(token_times) == 1:
                    state["first_token"] = token_times[0]
                    state["first_token_text"] = text
                if live:
                    live.update(render())

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

                    msg_type = data.get("type")
                    if msg_type == "token":
                        handle_token(data.get("text", ""))
                    elif msg_type == "vad_end":
                        state["vad_end"] = True
                    elif msg_type == "ping":
                        pass  # Server keepalive
                    elif "text" in data:  # Legacy format
                        handle_token(data.get("text", ""))
                    elif data.get("status") == "complete":
                        break
            except asyncio.TimeoutError:
                pass  # Expected for streaming
            except websockets.exceptions.ConnectionClosed:
                pass  # Session ended normally

            if stop_event and stop_event.is_set() and not send_task.done():
                send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                send_time = await send_task
            total_time = time.perf_counter() - start_time
            full_text = "".join(tokens)
            state["send_time"] = send_time
            state["total_time"] = total_time
            state["status"] = "complete"
            if live:
                live.update(render())
    except Exception as e:
        # Connection failed - update state and re-raise
        state["status"] = f"error: {e}"
        if update_task:
            update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await update_task
        if live:
            live.update(render())
        raise
    finally:
        # Only stop the live display if we created it (not external)
        if live and not external_live:
            live.stop()

    if playback_stream:
        with contextlib.suppress(Exception):
            playback_stream.stop()
            playback_stream.close()

    # Only print summary if not using external TUI (it will show results differently)
    if not external_live:
        print(f"\nTranscription: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
        print(f"\nTotal round-trip: {total_time:.3f}s")
        print(f"  Connect: {connect_time:.2f}s")
        print(f"  Send: {send_time:.3f}s")
        if token_times:
            print(f"  First token: {token_times[0]:.3f}s")
        else:
            print("  No tokens received")
        print(f"  Tokens: {len(tokens)}")

    return {
        "connect_time": connect_time,
        "send_time": send_time,
        "first_token_time": token_times[0] if token_times else None,
        "total_time": total_time,
        "token_count": len(tokens),
    }


async def run_sequential_samples(
    uri: str,
    wav_paths: list,
    auth_headers: dict | None,
    real_time_factor: float,
    runs_per_sample: int,
    load_expected_fn: callable,
    stop_event: asyncio.Event | None = None,
    playback: bool = False,
    playback_device: int | None = None,
):
    """Run multiple samples sequentially with a persistent TUI."""
    if not RICH_AVAILABLE:
        # Fall back to simple loop without persistent TUI
        for wav_path in wav_paths:
            file_expected = load_expected_fn(str(wav_path))
            for i in range(runs_per_sample):
                print(f"\n{'='*50}")
                print(f"Test {i+1}/{runs_per_sample} - {wav_path.name}")
                print('='*50)
                await measure_latency(
                    uri,
                    wav_path=str(wav_path),
                    auth_headers=auth_headers,
                    real_time_factor=real_time_factor,
                    expected_text=file_expected,
                    stop_event=stop_event,
                    playback=playback,
                    playback_device=playback_device,
                )
                if stop_event and stop_event.is_set():
                    break
            if stop_event and stop_event.is_set():
                break
        return

    # Persistent TUI mode
    console = Console()
    state = {}
    session = {
        "sample_idx": 0,
        "run_idx": 0,
        "total_samples": len(wav_paths),
        "runs_per_sample": runs_per_sample,
        "current_file": "",
        "results": [],
    }
    bear_frames = ["ʕ•ᴥ•ʔ", "ʕᵔᴥᵔʔ", "ʕ•̀ᴥ•́ʔ✧", "ʕ•ᴥ•ʔ♪"]

    def render():
        def fmt(v):
            return f"[cyan]{v:.2f}s[/cyan]" if v is not None else "[dim]--[/dim]"

        # Status with color
        status = state.get("status", "--")
        if status == "connecting" and state.get("connect_start"):
            elapsed = time.perf_counter() - state["connect_start"]
            status = f"[yellow]connecting ({elapsed:.1f}s)[/yellow]"
        elif status == "streaming":
            status = f"[cyan]{status}[/cyan]"
        elif status == "complete":
            status = f"[green]{status}[/green]"

        # Compact progress bar
        pct = state["sent_bytes"] / state["total_bytes"] if state.get("total_bytes") else 0.0
        bar_len = 20
        filled = int(bar_len * pct)
        bar = f"[cyan]{'█'*filled}[/cyan][dim]{'·'*(bar_len-filled)}[/dim]"

        # Session info
        total_tests = session["total_samples"] * session["runs_per_sample"]
        current_test = session["sample_idx"] * session["runs_per_sample"] + session["run_idx"] + 1

        # Build compact display
        lines = []
        lines.append(f"[bold]{current_test}/{total_tests}[/bold] [white]{session['current_file']}[/white]  [dim]{state.get('audio_len', 0):.1f}s @ {state.get('rtf', 1.0)}x[/dim]")
        lines.append(f"{status}  [dim]connect:[/dim]{fmt(state.get('connect_time'))}  [dim]first:[/dim]{fmt(state.get('first_token'))}  {bar} {pct*100:5.1f}%")

        if state.get("expected_text"):
            expected = state["expected_text"]
            actual = state.get("recognized_text", "")
            lines.append(f"[blue]expect:[/blue] {expected}")
            lines.append(f"[blue]actual:[/blue] {_color_diff(expected, actual)}")

        bear = bear_frames[state.get("bear_idx", 0) % len(bear_frames)]
        content = "\n".join(lines)
        return Panel(content, title=f"[magenta]{bear} STT[/magenta]", border_style="magenta", padding=(0, 1))

    live = Live(render(), console=console, refresh_per_second=12, screen=True)
    live.start()

    try:
        for sample_idx, wav_path in enumerate(wav_paths):
            session["sample_idx"] = sample_idx
            session["current_file"] = wav_path.name
            file_expected = load_expected_fn(str(wav_path))

            for run_idx in range(runs_per_sample):
                session["run_idx"] = run_idx
                live.update(render())

                result = await measure_latency(
                    uri,
                    wav_path=str(wav_path),
                    auth_headers=auth_headers,
                    real_time_factor=real_time_factor,
                    expected_text=file_expected,
                    stop_event=stop_event,
                    playback=playback,
                    playback_device=playback_device,
                    live=live,
                    render_fn=render,
                    state=state,
                )
                session["results"].append({
                    "file": wav_path.name,
                    "run": run_idx + 1,
                    **result,
                })

                if stop_event and stop_event.is_set():
                    break
            if stop_event and stop_event.is_set():
                break
    finally:
        live.stop()

    # Print summary
    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print('='*60)
    for r in session["results"]:
        ft = r.get("first_token_time")
        ft_str = f"{ft:.3f}s" if ft else "--"
        print(f"  {r['file']} run {r['run']}: first_token={ft_str}, tokens={r.get('token_count', 0)}")


async def quick_warmup(uri: str, auth_headers: dict | None) -> bool:
    """Quick warmup: just connect and send minimal audio to wake up the server."""
    try:
        async with websockets.connect(
            uri,
            additional_headers=auth_headers,
            open_timeout=60,
            close_timeout=5,
        ) as ws:
            # Send 0.5s of silence (24kHz, int16)
            silence = np.zeros(12000, dtype=np.int16)
            await ws.send(silence.tobytes())
            # Wait briefly for any response
            try:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        return True
    except Exception as e:
        print(f"Warmup connection failed: {e}")
        return False


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
        print("Warmup...", end=" ", flush=True)
        if await quick_warmup(uri, auth_headers):
            print("done.\n")
        else:
            print("failed (continuing anyway).\n")

    if not RICH_AVAILABLE:
        # Fallback: run without TUI
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
    else:
        # TUI mode: show all parallel streams
        console = Console()
        bear_frames = ["ʕ•ᴥ•ʔ", "ʕᵔᴥᵔʔ", "ʕ•̀ᴥ•́ʔ✧", "ʕ•ᴥ•ʔ♪"]

        # Create a state dict for each stream
        states = [{} for _ in range(num_streams)]

        def render_stream_panel(idx: int, state: dict) -> Panel:
            """Render a single stream's panel."""
            def fmt(v):
                return f"[cyan]{v:.2f}s[/cyan]" if v is not None else "[dim]--[/dim]"

            status = state.get("status", "pending")
            if status == "connecting" and state.get("connect_start"):
                elapsed = time.perf_counter() - state["connect_start"]
                status = f"[yellow]connecting ({elapsed:.1f}s)[/yellow]"
            elif status == "streaming":
                status = f"[cyan]{status}[/cyan]"
            elif status == "complete":
                status = f"[green]{status}[/green]"
            elif status.startswith("error"):
                status = f"[red]{status}[/red]"
            else:
                status = f"[dim]{status}[/dim]"

            pct = state["sent_bytes"] / state["total_bytes"] if state.get("total_bytes") else 0.0
            bar_len = 20
            filled = int(bar_len * pct)
            bar = f"[cyan]{'█'*filled}[/cyan][dim]{'·'*(bar_len-filled)}[/dim]"

            lines = []
            lines.append(f"[dim]{state.get('audio_len', 0):.1f}s @ {state.get('rtf', 1.0)}x[/dim]  {status}  [dim]connect:[/dim]{fmt(state.get('connect_time'))}  [dim]first:[/dim]{fmt(state.get('first_token'))}")
            lines.append(f"{bar} {pct*100:5.1f}%")

            if state.get("expected_text"):
                expected = state["expected_text"]
                actual = state.get("recognized_text", "")
                lines.append(f"[blue]expect:[/blue] {expected}")
                lines.append(f"[blue]actual:[/blue] {_color_diff(expected, actual)}")

            bear = bear_frames[state.get("bear_idx", 0) % len(bear_frames)]
            content = "\n".join(lines)
            return Panel(content, title=f"[magenta]{bear} Stream {idx+1}[/magenta]", border_style="magenta", padding=(0, 1))

        def render_all():
            """Render all stream panels stacked vertically."""
            panels = [render_stream_panel(i, states[i]) for i in range(num_streams)]
            return Group(*panels)

        live = Live(render_all(), console=console, refresh_per_second=12, screen=True)
        live.start()

        # Background task to keep display updated
        update_running = True

        async def update_display():
            while update_running:
                live.update(render_all())
                await asyncio.sleep(0.1)

        update_task = asyncio.create_task(update_display())

        async def labeled_test(idx: int):
            # Create a render function that updates just this stream's panel but refreshes all
            def stream_render():
                return render_all()

            result = await measure_latency(
                uri,
                wav_path=wav_path,
                auth_headers=auth_headers,
                real_time_factor=real_time_factor,
                expected_text=expected_text,
                stop_event=stop_event,
                playback=playback if idx == 0 else False,  # Only first stream plays audio
                playback_device=playback_device,
                live=live,
                render_fn=stream_render,
                state=states[idx],
            )
            return idx, result

        tasks = [labeled_test(i) for i in range(num_streams)]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            update_running = False
            update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await update_task
            live.stop()

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
    print("\nTo clean up test apps:")
    for gpu in gpu_urls:
        print(f"  modal app stop kyutai-stt-{gpu.lower()}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Measure STT transcription latency")
    parser.add_argument("--parallel", "-p", type=int, default=0,
                        help="Number of parallel streams (0 = sequential mode)")
    parser.add_argument("--runs", "-n", type=int, default=None,
                        help="Number of test runs per sample (default: 1 with --all-samples, 3 otherwise)")
    parser.add_argument("--wav", type=str, default="samples/wav24k/latency_example_01.wav",
                        help="Path to test audio file (default: first latency example)")
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
    parser.add_argument("--all-samples", action="store_true",
                        help="Run over all wavs in samples/wav24k instead of a single file")
    args = parser.parse_args()

    uri = f"wss://{MODAL_WORKSPACE}--kyutai-stt-kyutaisttservice-serve.modal.run/v1/stream"

    def load_expected_text(wav_path: str, explicit_text: str | None = None, explicit_file: str | None = None) -> str | None:
        """Load expected transcript text for a wav file."""
        if explicit_text is not None:
            return explicit_text
        candidate = explicit_file
        if candidate is None:
            default_txt = f"{wav_path}.txt"
            candidate = default_txt if os.path.exists(default_txt) else None
        if candidate and os.path.exists(candidate):
            try:
                return Path(candidate).read_text(encoding="utf-8").strip()
            except Exception:
                pass
        return None

    expected_text = load_expected_text(args.wav, args.expected, args.expected_file)

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
        # Gather wav paths first
        wav_paths = []
        if args.all_samples:
            latency_set = sorted(Path("samples/wav24k").glob("latency_example_*.wav"))
            wav_paths = latency_set if latency_set else sorted(Path("samples/wav24k").glob("*.wav"))
            if not wav_paths:
                print("No wav files found in samples/wav24k")
                return
        else:
            wav_paths = [Path(args.wav)]

        if args.compare_gpus:
            gpus = [g.strip() for g in args.compare_gpus.split(",")]
            num_streams = args.parallel if args.parallel > 0 else 3
            await compare_gpus(
                gpus,
                num_streams,
                str(wav_paths[0]),
                auth_headers,
                args.rtf,
                expected_text,
                stop_event=stop_event,
                playback=args.playback,
                playback_device=args.playback_device,
            )
        elif args.parallel > 0:
            # Parallel mode: run streams in parallel, cycling through samples
            for wav_path in wav_paths:
                file_expected = load_expected_text(str(wav_path))
                if len(wav_paths) > 1:
                    print(f"\n{'='*50}")
                    print(f"Sample: {wav_path.name}")
                    print('='*50)
                await run_parallel_test(
                    uri,
                    args.parallel,
                    str(wav_path),
                    auth_headers,
                    args.rtf,
                    file_expected,
                    warmup=not args.no_warmup,
                    stop_event=stop_event,
                    playback=args.playback,
                    playback_device=args.playback_device,
                )
                if stop_event and stop_event.is_set():
                    break
                # Only warmup on first sample
                args.no_warmup = True
        else:
            # Sequential mode
            # Default runs: 1 for --all-samples, 3 otherwise
            runs = args.runs if args.runs is not None else (1 if args.all_samples else 3)

            # Use persistent TUI for multi-sample runs
            if len(wav_paths) > 1 or runs > 1:
                await run_sequential_samples(
                    uri,
                    wav_paths,
                    auth_headers,
                    real_time_factor=args.rtf,
                    runs_per_sample=runs,
                    load_expected_fn=load_expected_text,
                    stop_event=stop_event,
                    playback=args.playback,
                    playback_device=args.playback_device,
                )
            else:
                # Single sample, single run - use simple mode
                await measure_latency(
                    uri,
                    wav_path=str(wav_paths[0]),
                    auth_headers=auth_headers,
                    real_time_factor=args.rtf,
                    expected_text=expected_text,
                    stop_event=stop_event,
                    playback=args.playback,
                    playback_device=args.playback_device,
                )
    finally:
        stop_event.set()
        quit_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await quit_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted, exiting cleanly.")
