"""Modal deployment for Kyutai STT using the Rust moshi-server.

This uses the production Rust server which can handle 64+ concurrent sessions
per container. A Python proxy accepts PCM audio from clients and forwards
it to the internal Rust server via msgpack.

Architecture:
  Client --[PCM]--> Python proxy --[msgpack]--> Rust moshi-server
                                            <--[Text]--

Deploy with: uvx modal deploy src/stt/modal_app.py
Dev server: uvx modal serve src/stt/modal_app.py
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "kyutai/stt-1b-en_fr")
KYUTAI_GPU = os.getenv("KYUTAI_GPU", "L40S")  # L40S recommended for Rust server
APP_NAME = os.getenv("KYUTAI_APP_NAME", "kyutai-stt-rust")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "30"))  # Concurrent sessions per container
RUST_SERVER_PORT = 8998  # Internal port for Rust server
RUST_LOG_LEVEL = os.getenv("RUST_LOG_LEVEL") or os.getenv("RUST_LOG") or "info"
RUST_LOG_TAIL_BYTES = int(os.getenv("RUST_LOG_TAIL_BYTES", "4000"))
ASR_DELAY_TOKENS_ENV = os.getenv("ASR_DELAY_TOKENS")

MINUTES = 60

# Use CUDA 12.1 to avoid compatibility issues with newer versions
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "curl",
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "git",
        "cmake",
        "libopus-dev",
        "libsndfile1",
        "python3.11-dev",
        "libpython3.11-dev",
    )
    # Install Rust toolchain
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    # Install moshi-server with CUDA support
    # Must compile on same GPU type as runtime to avoid CUDA symbol errors
    .run_commands(
        "export PATH=$HOME/.cargo/bin:$PATH && "
        "cargo install --features cuda moshi-server",
        gpu=KYUTAI_GPU,  # Compile on same GPU type as runtime
    )
    .pip_install(
        "huggingface-hub[hf_transfer]>=0.25.0",
        "fastapi>=0.115.0",
        "websockets>=13.0",
        "numpy<2",
        "opuslib",  # Python bindings for Opus codec
        "msgpack",  # For parsing Rust server responses
        "soundfile",  # For loading bundled test audio
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Volume for caching model weights
hf_cache_vol = modal.Volume.from_name("kyutai-stt-hf-cache", create_if_missing=True)
hf_cache_vol_path = Path("/root/.cache/huggingface")

app = modal.App(APP_NAME, image=image)


def _read_log_tail(path: str, max_bytes: int = RUST_LOG_TAIL_BYTES) -> str:
    """Return the last max_bytes of a log file without reading the whole thing."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - max_bytes, 0))
            return f.read().decode(errors="replace")
    except Exception as exc:  # pragma: no cover - best effort helper
        print(f"Could not read log tail: {exc}")
        return ""


def generate_config(model_name: str, batch_size: int, port: int) -> str:
    """Generate moshi-server config TOML."""
    if "2.6b" in model_name.lower():
        candle_repo = "kyutai/stt-2.6b-en-candle"
        asr_delay = 31
        d_model, num_heads, num_layers, dim_feedforward = 2560, 20, 24, 10240
    else:
        candle_repo = "kyutai/stt-1b-en_fr-candle"
        asr_delay = 6
        d_model, num_heads, num_layers, dim_feedforward = 2048, 16, 16, 8192

    if ASR_DELAY_TOKENS_ENV:
        try:
            override = int(ASR_DELAY_TOKENS_ENV)
            if override > 0:
                print(f"Overriding asr_delay_in_tokens: {asr_delay} -> {override}")
                asr_delay = override
        except ValueError:
            print(f"Invalid ASR_DELAY_TOKENS={ASR_DELAY_TOKENS_ENV}, using default {asr_delay}")

    return f'''static_dir = "/tmp/static/"
log_dir = "/tmp/stt-logs"
instance_name = "stt"
authorized_ids = ["public_token"]

[modules.asr]
path = "/api/asr-streaming"
type = "BatchedAsr"
lm_model_file = "hf://{candle_repo}/model.safetensors"
text_tokenizer_file = "hf://{candle_repo}/tokenizer_en_fr_audio_8000.model"
audio_tokenizer_file = "hf://{candle_repo}/mimi-pytorch-e351c8d8@125.safetensors"
asr_delay_in_tokens = {asr_delay}
batch_size = {batch_size}
conditioning_learnt_padding = true
temperature = 0.0

[modules.asr.model]
audio_vocab_size = 2049
text_in_vocab_size = 8001
text_out_vocab_size = 8000
audio_codebooks = 32

[modules.asr.model.transformer]
d_model = {d_model}
num_heads = {num_heads}
num_layers = {num_layers}
dim_feedforward = {dim_feedforward}
causal = true
norm_first = true
bias_ff = false
bias_attn = false
context = 750
max_period = 100000
use_conv_block = false
use_conv_bias = true
gating = "silu"
norm = "RmsNorm"
positional_embedding = "Rope"
conv_layout = false
conv_kernel_size = 3
kv_repeat = 1
max_seq_len = 40960

[modules.asr.model.extra_heads]
num_heads = 4
dim = 6
'''


@app.cls(
    gpu=KYUTAI_GPU,
    volumes={hf_cache_vol_path: hf_cache_vol},
    timeout=30 * MINUTES,
    scaledown_window=120,
    max_containers=10,
    min_containers=0,
)
@modal.concurrent(max_inputs=BATCH_SIZE)
class KyutaiSTTRustService:
    """Proxy service: accepts PCM, encodes to Opus, forwards to Rust server."""

    @modal.enter()
    def start_rust_server(self):
        """Start the Rust moshi-server on container startup."""
        # Generate config
        config_content = generate_config(MODEL_NAME, BATCH_SIZE, RUST_SERVER_PORT)
        config_path = "/tmp/stt-config.toml"
        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"Config written to {config_path}")

        os.makedirs("/tmp/stt-logs", exist_ok=True)
        os.makedirs("/tmp/static", exist_ok=True)

        # Start Rust server
        cmd = [
            "/root/.cargo/bin/moshi-server",
            "worker",
            "--config", config_path,
            "--port", str(RUST_SERVER_PORT),
        ]
        print(f"Starting Rust server: {' '.join(cmd)}")

        env = os.environ.copy()
        env["RUST_LOG"] = RUST_LOG_LEVEL
        env["RUST_BACKTRACE"] = "1"
        env["HF_HOME"] = str(hf_cache_vol_path)
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print(f"Using RUST_LOG={env['RUST_LOG']}")

        # Write Rust server output to a log file we can read
        self.rust_log_file = open("/tmp/rust-server.log", "w")
        self.rust_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.rust_log_file,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be fully ready (model loaded)
        print("Waiting for Rust server to start and load model...")
        start_time = time.monotonic()
        log_content = ""
        while time.monotonic() - start_time < 600:  # 10 min timeout for model loading
            # Read stdout to check logs (non-blocking would be better but this works)
            if self.rust_process.poll() is not None:
                output = self.rust_process.stdout.read().decode() if self.rust_process.stdout else ""
                raise RuntimeError(f"Rust server exited: {output[-3000:]}")

            # Try socket connection check first, then wait for model to warm up
            import socket
            try:
                with socket.create_connection(("127.0.0.1", RUST_SERVER_PORT), timeout=1):
                    # Socket is open - server is listening
                    elapsed = time.monotonic() - start_time
                    if elapsed < 30:
                        # Wait a bit more for model to load (socket opens before model loads)
                        print(f"[{elapsed:.1f}s] Server listening, waiting for model warmup...")
                        time.sleep(5)
                    else:
                        print(f"Rust server ready after {elapsed:.1f}s")
                        break
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                elapsed = time.monotonic() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    print(f"[{elapsed:.0f}s] Still waiting for Rust server... ({e})")
                time.sleep(1)
        else:
            raise RuntimeError("Rust server failed to start within 10 minutes")

        # Store internal WebSocket URL
        self.rust_ws_url = f"ws://127.0.0.1:{RUST_SERVER_PORT}/api/asr-streaming"
        print(f"Rust server URL: {self.rust_ws_url}")

    @modal.exit()
    def stop_rust_server(self):
        """Stop the Rust server on container shutdown."""
        if hasattr(self, "rust_process") and self.rust_process:
            self.rust_process.terminate()
            try:
                self.rust_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rust_process.kill()

    async def _rust_selftest(
        self,
        sample_path: str,
        duration_s: float = 3.0,
        real_time_factor: float = 3.0,
    ) -> dict:
        """Send a bundled audio sample directly to the Rust server and measure first token."""
        import msgpack
        import numpy as np
        import soundfile as sf
        import websockets

        sample_path = sample_path or "samples/wav24k/latency_example_01.wav"
        # Resolve relative to repo root if needed
        if not os.path.isabs(sample_path):
            sample_path = str(Path(__file__).resolve().parents[2] / sample_path)

        audio, sr = sf.read(sample_path, dtype="float32", always_2d=False)
        if sr != 24000:
            raise ValueError(f"Expected 24kHz sample, got {sr}")
        if audio.ndim > 1:
            audio = audio[:, 0]

        if duration_s and duration_s > 0:
            max_samples = int(sr * duration_s)
            audio = audio[:max_samples]

        # Append a bit of silence to let the model flush tokens
        audio = np.concatenate([audio, np.zeros(int(sr * 0.8), dtype=np.float32)])

        frame_size = 1920  # 80ms @24kHz
        rust_headers = {"kyutai-api-key": "public_token"}

        send_start = None
        first_token_time = None
        token_count = 0

        async with websockets.connect(
            self.rust_ws_url,
            additional_headers=rust_headers,
            open_timeout=15,
        ) as rust_ws:
            async def sender():
                nonlocal send_start
                send_start = time.perf_counter()
                for i in range(0, len(audio), frame_size):
                    frame = audio[i:i + frame_size]
                    msg = {"type": "Audio", "pcm": frame.tolist()}
                    await rust_ws.send(msgpack.packb(msg))
                    await asyncio.sleep(len(frame) / sr / max(real_time_factor, 0.25))

            async def receiver():
                nonlocal first_token_time, token_count
                while True:
                    try:
                        msg = await asyncio.wait_for(rust_ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        break
                    except websockets.exceptions.ConnectionClosed:
                        break

                    if isinstance(msg, bytes):
                        try:
                            data = msgpack.unpackb(msg, raw=False)
                        except msgpack.UnpackException:
                            continue
                        if isinstance(data, dict):
                            if "Word" in data or data.get("type") == "Word":
                                token_count += 1
                                if first_token_time is None and send_start:
                                    first_token_time = time.perf_counter() - send_start
                            elif "Marker" in data:
                                break
                    # Ignore string messages

            await asyncio.gather(sender(), receiver())

        return {
            "sample": os.path.basename(sample_path),
            "duration_s": duration_s,
            "rtf": real_time_factor,
            "first_token_s": first_token_time,
            "tokens": token_count,
        }

    @modal.asgi_app(requires_proxy_auth=True)
    def serve(self):
        """Create the ASGI app that proxies to Rust server."""
        import json

        import msgpack
        import numpy as np
        import websockets
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        web_app = FastAPI(title="Kyutai STT (Rust Backend)")

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": MODEL_NAME,
                "gpu": KYUTAI_GPU,
                "backend": "rust",
                "batch_size": BATCH_SIZE,
            }

        @web_app.get("/")
        def root():
            return {
                "service": "Kyutai STT (Rust Backend)",
                "model": MODEL_NAME,
                "status": "ready",
                "endpoints": {
                    "websocket": "/v1/stream",
                    "health": "/health",
                    "rust_selftest": "/internal/rust_selftest",
                },
            }

        @web_app.get("/internal/rust_selftest")
        async def rust_selftest(
            sample: str = "samples/wav24k/latency_example_01.wav",
            duration_s: float = 3.0,
            rtf: float = 3.0,
        ):
            """Hit the Rust server directly to measure first token without Python proxy."""
            return await self._rust_selftest(sample, duration_s, rtf)

        @web_app.websocket("/v1/stream")
        async def transcribe_websocket(ws: WebSocket):
            """WebSocket endpoint - accepts PCM, proxies to Rust server."""
            await ws.accept()
            print("Client session started")

            # Rust server expects 1920 samples per frame (80ms at 24kHz)
            SAMPLE_RATE = 24000
            FRAME_SIZE = 1920  # 80ms frames

            pcm_buffer = np.array([], dtype=np.float32)
            bytes_in = 0
            tokens_sent = 0

            async def send_json(payload: dict) -> bool:
                try:
                    await asyncio.wait_for(ws.send_text(json.dumps(payload)), timeout=5.0)
                    return True
                except Exception:
                    return False

            try:
                # Connect to Rust server
                rust_headers = {"kyutai-api-key": "public_token"}
                print(f"Connecting to Rust server at {self.rust_ws_url}...")
                async with websockets.connect(
                    self.rust_ws_url,
                    additional_headers=rust_headers,
                    open_timeout=30,
                ) as rust_ws:
                    print(f"Connected to Rust backend: {rust_ws.state}")

                    async def receive_from_rust():
                        """Receive transcriptions from Rust server and forward to client."""
                        nonlocal tokens_sent
                        msg_count = 0
                        while True:
                            try:
                                msg = await rust_ws.recv()
                                msg_count += 1
                                if isinstance(msg, bytes):
                                    # Rust server sends msgpack-encoded messages
                                    try:
                                        data = msgpack.unpackb(msg, raw=False)
                                        if msg_count <= 10 or msg_count % 50 == 0:
                                            print(f"Rust msg {msg_count}: {data}")

                                        # Handle different message types
                                        if isinstance(data, dict):
                                            msg_type = data.get("type") or data.get("Word") or list(data.keys())[0] if data else None

                                            # Word message contains transcription
                                            if "Word" in data or msg_type == "Word":
                                                word_data = data.get("Word", data)
                                                text = word_data.get("text", "")
                                                if text:
                                                    # Convert sentencepiece space marker to regular space
                                                    text = text.replace("\u2581", " ")
                                                    # Add leading space if token starts with alphanumeric
                                                    # and doesn't already have a leading space
                                                    if text and text[0].isalnum():
                                                        text = " " + text
                                                    await send_json({"type": "token", "text": text})
                                                    tokens_sent += 1
                                                    if tokens_sent <= 5:
                                                        print(f"Token {tokens_sent}: {repr(text)}")
                                            elif "Step" in data:
                                                # Step message - log occasionally
                                                if msg_count <= 10 or msg_count % 100 == 0:
                                                    print(f"Step msg {msg_count}: step_idx={data['Step'].get('step_idx') if isinstance(data.get('Step'), dict) else data.get('step_idx')}, buffered={data.get('buffered_pcm', data.get('Step', {}).get('buffered_pcm', '?'))}")
                                            elif "Marker" in data:
                                                # End marker
                                                print(f"Received end marker: {data}")
                                            else:
                                                # Unknown message type - always log
                                                print(f"Unknown Rust msg {msg_count}: {data}")
                                    except msgpack.UnpackException as e:
                                        if msg_count <= 10:
                                            print(f"Rust msg {msg_count}: {len(msg)} bytes (not msgpack: {e})")
                                else:
                                    print(f"Rust string msg: {repr(msg[:100] if len(msg) > 100 else msg)}")
                            except websockets.exceptions.ConnectionClosed as e:
                                print(f"Rust connection closed: {e}")
                                break
                            except Exception as e:
                                print(f"Rust recv error: {e}")
                                import traceback
                                traceback.print_exc()
                                break
                        print(f"Rust receiver done: {msg_count} messages, {tokens_sent} tokens")

                    # Start receiving from Rust in background
                    print("Starting receive task...")
                    recv_task = asyncio.create_task(receive_from_rust())
                    print(f"Receive task started: {recv_task}")

                    try:
                        while True:
                            try:
                                # Use generic receive() to handle both text and binary frames
                                # Modal's proxy may convert frame types
                                msg = await asyncio.wait_for(ws.receive(), timeout=30.0)
                            except asyncio.TimeoutError:
                                print("Client timeout")
                                break
                            except WebSocketDisconnect:
                                print("Client disconnected")
                                break

                            msg_type = msg.get("type", "")
                            if msg_type == "websocket.disconnect":
                                print("Client sent disconnect frame")
                                break

                            # Extract data from the message
                            data = msg.get("bytes") or msg.get("text")
                            if bytes_in == 0:
                                print(f"First WS message: type={msg_type}, keys={list(msg.keys())}, "
                                      f"has_bytes={msg.get('bytes') is not None}, "
                                      f"has_text={msg.get('text') is not None}, "
                                      f"data_len={len(data) if data else 0}")

                            if not data:
                                continue

                            # Handle text frames (base64 or raw)
                            if isinstance(data, str):
                                import base64
                                try:
                                    data = base64.b64decode(data)
                                    if bytes_in == 0:
                                        print(f"Decoded base64 text frame: {len(data)} bytes")
                                except Exception:
                                    if bytes_in == 0:
                                        print(f"Text frame (not base64): {repr(data[:100])}")
                                    continue

                            bytes_in += len(data)
                            frames_sent = 0

                            # Convert bytes to float32 PCM
                            pcm = np.frombuffer(data, dtype=np.float32)
                            pcm_buffer = np.concatenate([pcm_buffer, pcm])

                            # Send complete frames as msgpack to Rust server
                            while len(pcm_buffer) >= FRAME_SIZE:
                                frame = pcm_buffer[:FRAME_SIZE]
                                pcm_buffer = pcm_buffer[FRAME_SIZE:]

                                # Send as msgpack: {"type": "Audio", "pcm": [floats]}
                                msg = {"type": "Audio", "pcm": frame.tolist()}
                                data_out = msgpack.packb(msg)
                                await rust_ws.send(data_out)
                                frames_sent += 1

                            if bytes_in <= 80000 or bytes_in % 384000 < 8000:  # Log periodically
                                print(f"Received {len(data)} bytes, sent {frames_sent} PCM frames, total={bytes_in}, pcm_buf={len(pcm_buffer)}")

                    finally:
                        print(f"Cleaning up, recv_task done={recv_task.done()}")
                        recv_task.cancel()
                        try:
                            await recv_task
                        except asyncio.CancelledError:
                            print("Receive task cancelled")

            except Exception as e:
                print(f"Session error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"Session ended: {bytes_in} bytes in, {tokens_sent} tokens out")
                # Print last bit of Rust server log for debugging
                try:
                    self.rust_log_file.flush()
                    content = _read_log_tail("/tmp/rust-server.log")
                    if content:
                        print(f"Rust server log tail ({len(content)} chars):")
                        print(content)
                except Exception as e:
                    print(f"Could not read Rust log: {e}")
                try:
                    await ws.close()
                except Exception:
                    pass

        return web_app
