#!/usr/bin/env -S uv run python
"""
Real-time Microphone Transcription CLI

Captures audio from your microphone and streams it to the Modal-deployed
Kyutai STT service for real-time transcription with live-updating display.

Features:
- Real-time microphone capture at 24kHz
- Low-latency streaming transcription
- Live-updating terminal display
- Modal proxy authentication support

Usage:
    ./transcribe_cli.py [--device DEVICE_ID]

Environment variables:
    MODAL_KEY     - Modal proxy auth key
    MODAL_SECRET  - Modal proxy auth secret
"""

import argparse
import asyncio
import json
import math
import os
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = Live = Panel = Group = Layout = None
    Text = str  # type: ignore
    print("Note: Install 'rich' for better terminal display: pip install rich")


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MODAL_WORKSPACE = os.environ.get("MODAL_WORKSPACE", "YOUR_WORKSPACE")
DEFAULT_WS_URL = f"wss://{MODAL_WORKSPACE}--kyutai-stt-kyutaisttservice-serve.modal.run/v1/stream"

SAMPLE_RATE = 24000  # Kyutai expects 24kHz audio
CHANNELS = 1
DTYPE = np.float32
CHUNK_DURATION_MS = 80  # Send audio every 80ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
METER_FLOOR_DB = -60.0
METER_WIDTH = 36
METER_HISTORY_SECONDS = 10
SPARK_BUCKETS = 24
STALL_TIMEOUT_SECONDS = 15.0
STALL_AUDIO_GRACE = 5.0

# Buffer settings for text correction
FINALIZE_DELAY_MS = 1800  # Time before text is considered final (slightly longer for accuracy)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def compute_stall_notice(
    last_token_ts: Optional[float],
    last_audio_ts: Optional[float],
    now: Optional[float] = None,
    grace_seconds: float = 5.0,
    stall_seconds: float = 15.0,
) -> Optional[str]:
    """Decide whether to surface a stall notice based on recent audio and tokens."""
    now = now or time.time()
    if last_token_ts is None or last_audio_ts is None:
        return None
    if (now - last_audio_ts) >= grace_seconds:
        return None
    if (now - last_token_ts) <= stall_seconds:
        return None
    return f"No transcripts for {now - last_token_ts:0.1f}s (audio active)"


# ------------------------------------------------------------------------------
# Transcript Manager
# ------------------------------------------------------------------------------


@dataclass
class TranscriptManager:
    """
    Manages the live transcript with support for corrections.

    The transcript is divided into:
    - Final text: Cannot be changed, displayed normally
    - Pending text: Can still be corrected, displayed with indicator
    """

    final_text: str = ""
    pending_tokens: list = field(default_factory=list)
    pending_timestamps: list = field(default_factory=list)
    correction_window_ms: float = FINALIZE_DELAY_MS

    def add_token(self, token: str):
        """Add a new token to the pending buffer."""
        now = time.time() * 1000
        self.pending_tokens.append(token)
        self.pending_timestamps.append(now)
        self._finalize_old_tokens()

    def _finalize_old_tokens(self):
        """Move old tokens from pending to final."""
        now = time.time() * 1000
        finalize_count = 0

        for ts in self.pending_timestamps:
            if now - ts > self.correction_window_ms:
                finalize_count += 1
            else:
                break

        if finalize_count > 0:
            finalized = "".join(self.pending_tokens[:finalize_count])
            self.final_text += finalized
            self.pending_tokens = self.pending_tokens[finalize_count:]
            self.pending_timestamps = self.pending_timestamps[finalize_count:]

    def force_finalize(self):
        """Finalize all pending tokens (e.g., on pause/stop)."""
        self.final_text += "".join(self.pending_tokens)
        self.pending_tokens = []
        self.pending_timestamps = []

    def get_display_text(self) -> tuple[str, str]:
        """Get (final_text, pending_text) for display."""
        self._finalize_old_tokens()
        return self.final_text, "".join(self.pending_tokens)

    def get_full_text(self) -> str:
        """Get the complete transcript."""
        return self.final_text + "".join(self.pending_tokens)

    def clear(self):
        """Clear the transcript."""
        self.final_text = ""
        self.pending_tokens = []
        self.pending_timestamps = []


# ------------------------------------------------------------------------------
# Terminal Display
# ------------------------------------------------------------------------------


class SimpleDisplay:
    """Simple terminal display without rich library."""

    def __init__(self):
        self.last_line_count = 0

    def update(
        self,
        final_text: str,
        pending_text: str,
        status: str = "",
        meter: str | None = None,
        sparkline: str | None = None,
        level_line: str | None = None,
        stall_notice: str | None = None,
    ):
        # Clear previous output
        if self.last_line_count > 0:
            print(f"\033[{self.last_line_count}A\033[J", end="")

        # Build display
        lines = []
        if status:
            lines.append(f"\033[90m[{status}]\033[0m")
        if stall_notice:
            lines.append(f"\033[91m{stall_notice}\033[0m")
        if meter:
            lines.append(meter)
        if level_line:
            lines.append(level_line)
        if sparkline:
            lines.append(sparkline)

        full_text = final_text + f"\033[93m{pending_text}\033[0m"
        if full_text:
            # Word wrap at ~80 chars
            words = full_text.split(" ")
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 > 80:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = f"{current_line} {word}".strip()
            if current_line:
                lines.append(current_line)

        if not lines:
            lines.append("\033[90m(listening...)\033[0m")

        output = "\n".join(lines)
        print(output)
        self.last_line_count = len(lines)

    def stop(self):
        print()


class RichDisplay:
    """Rich terminal display with panels and formatting."""

    def __init__(self):
        self.console = Console()
        self.live = None
        self._status = "Initializing..."

    def start(self):
        self.live = Live(
            self._build_layout("", "", self._status, None, None, None, None),
            console=self.console,
            refresh_per_second=10,
            transient=False,
            screen=True,
        )
        self.live.start()

    def _build_layout(
        self,
        final_text: str,
        pending_text: str,
        status: str,
        meter: Text | None,
        sparkline: Text | None,
        level_line: Text | None,
        stall_notice: Text | None,
    ) -> Layout:
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="transcript", ratio=3),
            Layout(name="meters", size=32),
        )

        # Header with title and live stats
        header_lines = [
            Text("Live Transcription (Kyutai STT)", style="bold cyan"),
            Text(status or "Listening...", style="dim"),
        ]
        if stall_notice is not None:
            header_lines.append(stall_notice)
        layout["header"].update(
            Panel(
                Group(*header_lines),
                border_style="blue",
                padding=(0, 1),
            )
        )

        # Transcript area fills most of the screen
        text = Text()
        if final_text:
            text.append(final_text, style="white")

        if pending_text:
            text.append(pending_text, style="yellow italic")

        if not final_text and not pending_text:
            text.append("(listening...)", style="dim")

        layout["transcript"].update(
            Panel(
                text,
                title="[white]Transcript[/white]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Audio meters on the right
        meter_sections = []
        if meter is not None:
            meter_sections.append(Text("Meter", style="dim"))
            meter_sections.append(meter)
        if level_line is not None:
            meter_sections.append(level_line)
        if sparkline is not None:
            meter_sections.append(Text("Last 10s", style="dim"))
            meter_sections.append(sparkline)
        if not meter_sections:
            meter_sections.append(Text("No audio yet", style="dim"))

        layout["meters"].update(
            Panel(
                Group(*meter_sections),
                title="[white]Audio[/white]",
                border_style="green",
                padding=(1, 1),
            )
        )

        # Footer with controls/help
        footer_lines = [
            Text("Press Ctrl+C or 'q' to quit.", style="dim"),
            Text("Use --list-devices to select an input device.", style="dim"),
        ]
        layout["footer"].update(
            Panel(
                Group(*footer_lines),
                border_style="blue",
                padding=(0, 1),
            )
        )

        return layout

    def update(
        self,
        final_text: str,
        pending_text: str,
        status: str = "",
        meter: Text | None = None,
        sparkline: Text | None = None,
        level_line: Text | None = None,
        stall_notice: Text | None = None,
    ):
        self._status = status
        if self.live:
            self.live.update(
                self._build_layout(
                    final_text,
                    pending_text,
                    status,
                    meter,
                    sparkline,
                    level_line,
                    stall_notice,
                )
            )

    def stop(self):
        if self.live:
            self.live.stop()
            try:
                self.console.line()
            except Exception:
                pass


# ------------------------------------------------------------------------------
# WebSocket Client
# ------------------------------------------------------------------------------


class TranscriptionClient:
    """WebSocket client for streaming audio to the Kyutai STT server."""

    def __init__(
        self,
        ws_url: str,
        auth_headers: dict | None = None,
        debug: bool = False,
    ):
        self.ws_url = ws_url
        self.auth_headers = auth_headers
        self.ws = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.transcript = TranscriptManager()
        self.display = RichDisplay() if RICH_AVAILABLE else SimpleDisplay()

        self.audio_queue = asyncio.Queue()
        self.stop_event: asyncio.Event | None = None
        self.running = False
        self.connected = False
        self.level_dbfs = float("-inf")
        self.peak_dbfs = float("-inf")
        self.level_history = deque()  # (timestamp, dbfs)
        self.debug = debug
        self.last_token_ts: float | None = None
        self.last_audio_ts: float | None = None
        self.reconnect_requested = False
        self.stall_notice: str | None = None

        # Stats
        self.audio_sent_bytes = 0
        self.tokens_received = 0
        self.start_time = None

    def request_stop(self, reason: str | None = None):
        """Signal all loops to stop."""
        if reason:
            print(reason)
        self.running = False
        self.connected = False
        if self.stop_event is not None:
            self.stop_event.set()
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass
        if self.ws and self.loop and self.loop.is_running():

            def _close():
                asyncio.create_task(self._close_ws_fast())

            try:
                self.loop.call_soon_threadsafe(_close)
            except Exception:
                pass

    async def connect(self):
        """Connect to the WebSocket server."""
        print(f"Connecting to {self.ws_url}...")
        if self.auth_headers:
            print("Using Modal proxy authentication")
        print("(First connection may take 30-60s while model loads...)")
        try:
            self.ws = await websockets.connect(
                self.ws_url,
                additional_headers=self.auth_headers,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
                open_timeout=180,  # Allow up to 3 minutes for cold start
            )
            self.connected = True
            print("Connected!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def _dbfs(self, value: float) -> float:
        if value <= 0:
            return float("-inf")
        return 20.0 * math.log10(value)

    def _record_level(self, audio_data: np.ndarray):
        """Update level metrics for UI."""
        if audio_data.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(audio_data))))
        peak = float(np.max(np.abs(audio_data)))
        self.level_dbfs = self._dbfs(rms)
        self.peak_dbfs = self._dbfs(peak)
        now = time.time()
        self.last_audio_ts = now
        self.level_history.append((now, self.level_dbfs))
        cutoff = now - METER_HISTORY_SECONDS
        while self.level_history and self.level_history[0][0] < cutoff:
            self.level_history.popleft()

    async def _trigger_reconnect(self, reason: str):
        """Request a reconnect without shutting down the whole app."""
        if self.reconnect_requested or not self.running:
            return
        print(f"\n{reason} - reconnecting...")
        self.reconnect_requested = True
        self.connected = False
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass

    async def _drain_audio_queue(self):
        """Drop any queued audio chunks to avoid backlog after reconnect."""
        try:
            while True:
                self.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            return

    async def send_audio(self, audio_data: np.ndarray):
        """Queue audio data for sending and update meters."""
        if self.running:
            self._record_level(audio_data)
            await self.audio_queue.put(audio_data)

    async def _quit_watcher(self):
        """Watch stdin for 'q' to quit (works without pressing Enter)."""
        if not sys.stdin.isatty():
            return
        loop = asyncio.get_running_loop()

        use_single_char = False
        restore_termios = None
        fd = None
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            restore_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            use_single_char = True
        except Exception:
            restore_termios = None
            use_single_char = False

        def read_char():
            try:
                return sys.stdin.read(1)
            except Exception:
                return ""

        def read_line():
            try:
                return sys.stdin.readline()
            except Exception:
                return ""

        reader = read_char if use_single_char else read_line

        try:
            while self.running:
                text = await loop.run_in_executor(None, reader)
                if not text:
                    break
                if text.strip().lower() == "q":
                    print("\nQuit requested (q)")
                    self.request_stop()
                    break
        finally:
            if restore_termios is not None and fd is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, restore_termios)
                except Exception:
                    pass

    async def _send_loop(self):
        """Send queued audio to the server."""
        chunks_sent = 0
        while self.running and self.connected:
            try:
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(), timeout=0.1
                )
                if audio_data is None:
                    break
                if self.ws and self.connected:
                    pcm_bytes = np.asarray(audio_data, dtype=np.float32).tobytes()
                    if not pcm_bytes:
                        continue
                    await self.ws.send(pcm_bytes)
                    self.audio_sent_bytes += len(pcm_bytes)
                    chunks_sent += 1
                    if chunks_sent <= 3 and self.debug:
                        print(f"[send] chunk {chunks_sent}: {len(pcm_bytes)} bytes")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    print(f"Send error: {e}")
                    traceback.print_exc()
                await self._trigger_reconnect("Send loop error")
                break

    async def _receive_loop(self):
        """Receive transcription results from the server."""
        while self.running and self.connected:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
                data = json.loads(message)

                if self.debug:
                    print(f"[recv] {data}")

                msg_type = data.get("type")

                # Handle token-by-token streaming
                if msg_type == "token":
                    text = data.get("text", "")
                    if text:
                        self.transcript.add_token(text)
                        self.tokens_received += 1
                        self.last_token_ts = time.time()

                # Handle voice activity end (sentence boundary)
                elif msg_type == "vad_end":
                    self.transcript.force_finalize()
                    self.transcript.add_token("\n")
                    if self.debug:
                        print("[recv] vad_end - sentence boundary")

                # Handle server keepalive pings (just ignore)
                elif msg_type == "ping":
                    pass

                # Legacy format support (text responses)
                elif "text" in data:
                    text = data.get("text", "")
                    if text.strip():
                        self.transcript.add_token(text + " ")
                        self.tokens_received += 1
                        self.last_token_ts = time.time()
                    if data.get("final"):
                        self.transcript.force_finalize()
                        self.transcript.add_token("\n")

                elif data.get("status") == "complete":
                    if self.debug:
                        print("[recv] stream complete")

                elif msg_type == "error" or "error" in data:
                    print(f"\nServer error: {data.get('message') or data.get('error')}")

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                if self.running:
                    await self._trigger_reconnect("Connection closed by server")
                break
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                    traceback.print_exc()
                    await self._trigger_reconnect("Receive loop error")
                break

    def _meter_text(self):
        """Return (rich.Text or str) meter, sparkline, level line for UI."""

        def clamp(db):
            if db == float("-inf"):
                return METER_FLOOR_DB
            return max(METER_FLOOR_DB, min(0.0, db))

        raw_db = self.level_dbfs
        raw_peak = self.peak_dbfs
        db = clamp(raw_db)
        peak_db = clamp(raw_peak)

        def fill(dbfs):
            level = (dbfs - METER_FLOOR_DB) / (0.0 - METER_FLOOR_DB)
            level = max(0.0, min(1.0, level))
            level = level**0.6  # soften low levels
            return int(round(level * METER_WIDTH))

        fill_level = fill(db)
        fill_peak = fill(peak_db)

        if db > -12:
            color = "red"
        elif db > -30:
            color = "yellow"
        else:
            color = "green"

        chars = ["-"] * METER_WIDTH
        for i in range(min(fill_level, METER_WIDTH)):
            chars[i] = "#"
        if METER_WIDTH:
            peak_pos = min(max(fill_peak - 1, 0), METER_WIDTH - 1)
            chars[peak_pos] = "|"

        meter_line = "".join(chars)

        # History sparkline
        now = time.time()
        cutoff = now - METER_HISTORY_SECONDS
        history = [db for ts, db in self.level_history if ts >= cutoff]
        spark = ""
        spark_chars = "........"
        if history:
            buckets = []
            bucket_size = max(1, len(history) // SPARK_BUCKETS)
            for i in range(0, len(history), bucket_size):
                bucket = history[i : i + bucket_size]
                buckets.append(max(bucket))
            for val in buckets[-SPARK_BUCKETS:]:
                norm = (clamp(val) - METER_FLOOR_DB) / (0.0 - METER_FLOOR_DB)
                norm = max(0.0, min(1.0, norm))
                idx = min(int(norm * (len(spark_chars) - 1)), len(spark_chars) - 1)
                spark += spark_chars[idx]

        # Build outputs for rich/non-rich
        level_text = (
            f"Level {raw_db:0.1f} dBFS | Peak {raw_peak:0.1f} dBFS"
            if raw_db != float("-inf")
            else "Level -inf dBFS"
        )

        if RICH_AVAILABLE:
            meter_text = Text(meter_line[:fill_level], style=color) + Text(
                meter_line[fill_level:]
            )
            spark_text = Text(spark or " " * min(8, SPARK_BUCKETS), style="dim")
            level_line = Text(level_text, style="dim")
            return meter_text, spark_text, level_line
        else:
            return meter_line, spark or "", level_text

    async def _display_loop(self):
        """Update the terminal display."""
        if RICH_AVAILABLE:
            self.display.start()

        while self.running:
            if self.reconnect_requested and not self.connected:
                break
            try:
                await asyncio.sleep(0.05)  # 20 FPS

                final, pending = self.transcript.get_display_text()
                meter, spark, level_line = self._meter_text()

                # Build status line
                elapsed = time.time() - self.start_time if self.start_time else 0
                since_token = (
                    f"{time.time() - self.last_token_ts:0.1f}s ago"
                    if self.last_token_ts
                    else "no tokens yet"
                )
                status = (
                    f"Time {elapsed:.1f}s | Sent {self.audio_sent_bytes / 1024:.1f}KB | "
                    f"Tokens {self.tokens_received} (last: {since_token})"
                )
                if not self.connected:
                    status += " | Disconnected"
                if self.reconnect_requested:
                    status += " | Reconnecting..."

                stall_text: Text | str | None = None
                if self.stall_notice:
                    stall_text = (
                        Text(self.stall_notice, style="red")
                        if RICH_AVAILABLE
                        else self.stall_notice
                    )

                self.display.update(
                    final,
                    pending,
                    status,
                    meter,
                    spark,
                    level_line,
                    stall_text,
                )
            except Exception as e:
                print(f"\nDisplay loop error: {e}")
                traceback.print_exc()
                self.request_stop("Display error")
                break

        self.display.stop()

    async def _stall_watcher(self):
        """Detect stalls (audio but no text) and surface a notice."""
        while self.running and not self.reconnect_requested:
            await asyncio.sleep(2.0)
            if not self.connected:
                self.stall_notice = None
                continue
            now = time.time()
            last_audio = self.last_audio_ts
            last_token = self.last_token_ts
            if last_token is None:
                continue
            self.stall_notice = compute_stall_notice(
                last_token, last_audio, now, STALL_AUDIO_GRACE, STALL_TIMEOUT_SECONDS
            )

    async def run(self):
        """Main run loop."""
        self.running = True
        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        if self.start_time is None:
            self.start_time = time.time()

        while self.running:
            self.reconnect_requested = False
            connect_task = asyncio.create_task(self.connect(), name="connect")
            stop_task = asyncio.create_task(self.stop_event.wait(), name="stop_wait")

            done, _ = await asyncio.wait(
                {connect_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if stop_task in done:
                connect_task.cancel()
                await asyncio.gather(connect_task, return_exceptions=True)
                break

            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)

            try:
                if not await connect_task:
                    break
            except asyncio.CancelledError:
                break

            tasks = [
                asyncio.create_task(self._send_loop(), name="send_loop"),
                asyncio.create_task(self._receive_loop(), name="recv_loop"),
                asyncio.create_task(self._display_loop(), name="display_loop"),
                asyncio.create_task(self._stall_watcher(), name="stall_watcher"),
            ]
            if sys.stdin.isatty():
                tasks.append(
                    asyncio.create_task(self._quit_watcher(), name="quit_watcher")
                )

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for task, result in zip(tasks, results):
                    if isinstance(result, asyncio.CancelledError):
                        continue
                    if isinstance(result, Exception):
                        print(f"\nTask {task.get_name()} error: {result}")
                        traceback.print_exception(result)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"\nClient run error: {e}")
                traceback.print_exc()
                self.request_stop()
            finally:
                for task in tasks:
                    task.cancel()
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception:
                    pass
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                self.connected = False

            if self.running and self.reconnect_requested:
                await self._drain_audio_queue()
                continue
            break

    def stop(self):
        """Stop the client."""
        self.request_stop()
        self.transcript.force_finalize()

    async def _close_ws_fast(self):
        """Close websocket if open."""
        if self.ws:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=1.5)
            except Exception:
                traceback.print_exc()
            self.ws = None
            self.connected = False

    async def shutdown(self):
        """Stop client and close websocket."""
        self.stop()
        await self._close_ws_fast()


# ------------------------------------------------------------------------------
# Audio Capture
# ------------------------------------------------------------------------------


class AudioCapture:
    """Capture audio from the microphone."""

    def __init__(
        self,
        device: Optional[int] = None,
        callback=None,
        save_path: Optional[str] = None,
    ):
        self.device = device
        self.callback = callback
        self.stream = None
        self.running = False
        self.save_path = save_path
        self.saved_audio = [] if save_path else None

    def list_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio input devices:")
        print("-" * 50)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                default = " (default)" if i == sd.default.device[0] else ""
                print(f"  [{i}] {device['name']}{default}")
        print("-" * 50)

    def start(self):
        """Start audio capture."""
        self.running = True

        def audio_callback(indata, frames, time_info, status):
            try:
                if status:
                    print(f"Audio status: {status}")
                if self.running:
                    audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                    audio_f32 = audio.astype(np.float32)

                    if self.saved_audio is not None:
                        self.saved_audio.append(audio_f32.copy())

                    if self.callback:
                        self.callback(audio_f32)
            except Exception as e:
                print(f"Audio callback error: {e}")
                traceback.print_exc()

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        )
        self.stream.start()
        print(f"Audio capture started (device: {self.device or 'default'})")
        if self.save_path:
            print(f"Audio will be saved to: {self.save_path}")

    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

        if self.save_path and self.saved_audio:
            import wave

            all_audio = np.concatenate(self.saved_audio)
            audio_int16 = (all_audio * 32767).astype(np.int16)
            with wave.open(self.save_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            print(f"Audio saved to: {self.save_path} ({len(all_audio) / SAMPLE_RATE:.1f}s)")


# ------------------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------------------


async def main(args):
    """Main application entry point."""

    if args.list_devices:
        capture = AudioCapture()
        capture.list_devices()
        return

    # Build auth headers from environment or args
    auth_headers = None
    modal_key = args.modal_key or os.environ.get("MODAL_KEY")
    modal_secret = args.modal_secret or os.environ.get("MODAL_SECRET")
    if modal_key and modal_secret:
        auth_headers = {
            "Modal-Key": modal_key,
            "Modal-Secret": modal_secret,
        }

    client = TranscriptionClient(
        args.url,
        auth_headers=auth_headers,
        debug=args.debug_ws,
    )

    loop = asyncio.get_running_loop()

    def on_audio(audio_data):
        if not client.running:
            return
        if loop.is_closed() or not loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(client.send_audio(audio_data), loop)
        except RuntimeError as e:
            if "closed" not in str(e).lower():
                print(f"Audio dispatch error: {e}")

    capture = AudioCapture(
        device=args.device, callback=on_audio, save_path=args.save_audio
    )

    stop_once = {"done": False}

    def handle_sigint():
        if stop_once["done"]:
            return
        stop_once["done"] = True
        print("\nCtrl+C received, stopping...")
        loop.call_soon_threadsafe(client.request_stop, "Ctrl+C")
        capture.stop()

    try:
        loop.add_signal_handler(signal.SIGINT, handle_sigint)
    except NotImplementedError:
        pass

    print("\n" + "=" * 60)
    print("  Real-time Transcription (Kyutai STT on Modal)")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    capture.start()

    try:
        await client.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nStopping...")
        client.request_stop()
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
    finally:
        await client.shutdown()
        capture.stop()
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except Exception:
            pass
        print("\n\nFinal transcript:")
        print("-" * 40)
        print(client.transcript.get_full_text())
        print("-" * 40)


def run():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription using Kyutai STT on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List audio devices
    ./transcribe_cli.py --list-devices

    # Transcribe with default microphone
    ./transcribe_cli.py

    # Transcribe with specific device
    ./transcribe_cli.py --device 2

Environment variables:
    MODAL_KEY      Modal proxy auth key
    MODAL_SECRET   Modal proxy auth secret
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_WS_URL,
        help=f"WebSocket URL of the STT server (default: {DEFAULT_WS_URL})",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID (use --list-devices to see options)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--save-audio",
        type=str,
        default=None,
        help="Save captured audio to a WAV file (e.g., /tmp/debug.wav)",
    )
    parser.add_argument(
        "--debug-ws",
        action="store_true",
        help="Print raw websocket messages",
    )
    parser.add_argument(
        "--modal-key",
        type=str,
        default=None,
        help="Modal proxy auth key (or set MODAL_KEY env var)",
    )
    parser.add_argument(
        "--modal-secret",
        type=str,
        default=None,
        help="Modal proxy auth secret (or set MODAL_SECRET env var)",
    )

    args = parser.parse_args()

    asyncio.run(main(args))


if __name__ == "__main__":
    run()
