#!/usr/bin/env -S uv run
"""Local Kyutai STT server for Proxmox LXC deployment.

Adapted from the Modal deployment (src/stt/modal_app.py). Runs a FastAPI proxy
in front of the Rust moshi-server, providing the same WebSocket API.

Architecture:
  Client --[PCM]--> FastAPI proxy --[msgpack]--> Rust moshi-server
                                  <--[Text]--
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import msgpack
import numpy as np
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "kyutai/stt-1b-en_fr")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))  # 4 for 8GB VRAM (RTX 5060)
RUST_SERVER_PORT = 8998
HF_CACHE_PATH = Path(os.getenv("HF_HOME", "/root/.cache/huggingface"))
IDLE_AUDIO_TIMEOUT_SECONDS = float(os.getenv("IDLE_AUDIO_TIMEOUT_SECONDS", "10"))


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


# Global state for the Rust server process
_rust_process = None
_rust_log_file = None
_rust_ws_url = None


def _start_rust_server():
    """Start the Rust moshi-server."""
    global _rust_process, _rust_log_file, _rust_ws_url

    config_content = generate_config(MODEL_NAME, BATCH_SIZE, RUST_SERVER_PORT)
    config_path = "/tmp/stt-config.toml"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Config written to {config_path}")

    os.makedirs("/tmp/stt-logs", exist_ok=True)
    os.makedirs("/tmp/static", exist_ok=True)

    cmd = [
        os.path.expanduser("~/.cargo/bin/moshi-server"),
        "worker",
        "--config", config_path,
        "--port", str(RUST_SERVER_PORT),
    ]
    print(f"Starting Rust server: {' '.join(cmd)}")

    env = os.environ.copy()
    env["RUST_LOG"] = "debug,moshi_server=trace,moshi=trace"
    env["RUST_BACKTRACE"] = "1"
    env["HF_HOME"] = str(HF_CACHE_PATH)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    _rust_log_file = open("/tmp/rust-server.log", "w")
    _rust_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=_rust_log_file,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    print("Waiting for Rust server to start and load model...")
    start_time = time.monotonic()
    while time.monotonic() - start_time < 600:  # 10 min timeout
        if _rust_process.poll() is not None:
            raise RuntimeError("Rust server exited unexpectedly")

        try:
            with socket.create_connection(("127.0.0.1", RUST_SERVER_PORT), timeout=1):
                elapsed = time.monotonic() - start_time
                if elapsed < 30:
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

    _rust_ws_url = f"ws://127.0.0.1:{RUST_SERVER_PORT}/api/asr-streaming"
    print(f"Rust server URL: {_rust_ws_url}")


def _stop_rust_server():
    """Stop the Rust server."""
    global _rust_process, _rust_log_file
    if _rust_process:
        _rust_process.terminate()
        try:
            _rust_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _rust_process.kill()
        _rust_process = None
    if _rust_log_file:
        _rust_log_file.close()
        _rust_log_file = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop the Rust moshi-server."""
    _start_rust_server()
    yield
    _stop_rust_server()


web_app = FastAPI(title="Kyutai STT (Local)", lifespan=lifespan)


@web_app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "gpu": "local",
        "backend": "rust",
        "batch_size": BATCH_SIZE,
    }


@web_app.get("/")
def root():
    return {
        "service": "Kyutai STT (Local)",
        "model": MODEL_NAME,
        "status": "ready",
        "endpoints": {
            "websocket": "/v1/stream",
            "health": "/health",
        },
    }


@web_app.websocket("/v1/stream")
async def transcribe_websocket(ws: WebSocket):
    """WebSocket endpoint - accepts PCM, proxies to Rust server."""
    await ws.accept()
    print("Client session started")

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
        rust_headers = {"kyutai-api-key": "public_token"}
        print(f"Connecting to Rust server at {_rust_ws_url}...")
        async with websockets.connect(
            _rust_ws_url,
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
                            try:
                                data = msgpack.unpackb(msg, raw=False)
                                if msg_count <= 10:
                                    print(f"Rust msg {msg_count}: {data}")

                                if isinstance(data, dict):
                                    msg_type = data.get("type") or data.get("Word") or list(data.keys())[0] if data else None

                                    if "Word" in data or msg_type == "Word":
                                        word_data = data.get("Word", data)
                                        text = word_data.get("text", "")
                                        if text:
                                            text = text.replace("\u2581", " ")
                                            if text and text[0].isalnum():
                                                text = " " + text
                                            await send_json({"type": "token", "text": text})
                                            tokens_sent += 1
                                            if tokens_sent <= 5:
                                                print(f"Token {tokens_sent}: {repr(text)}")
                                    elif "Step" in data:
                                        pass
                                    elif "Marker" in data:
                                        print(f"Received end marker: {data}")
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

            print("Starting receive task...")
            recv_task = asyncio.create_task(receive_from_rust())
            print(f"Receive task started: {recv_task}")

            try:
                while True:
                    try:
                        data = await asyncio.wait_for(
                            ws.receive_bytes(),
                            timeout=IDLE_AUDIO_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        print("Client timeout")
                        break
                    except WebSocketDisconnect:
                        print("Client disconnected")
                        break

                    if not data:
                        continue

                    bytes_in += len(data)
                    frames_sent = 0

                    pcm = np.frombuffer(data, dtype=np.float32)
                    pcm_buffer = np.concatenate([pcm_buffer, pcm])

                    while len(pcm_buffer) >= FRAME_SIZE:
                        frame = pcm_buffer[:FRAME_SIZE]
                        pcm_buffer = pcm_buffer[FRAME_SIZE:]

                        msg = {"type": "Audio", "pcm": frame.tolist()}
                        data_out = msgpack.packb(msg)
                        await rust_ws.send(data_out)
                        frames_sent += 1

                    if bytes_in <= 80000:
                        print(f"Received {len(data)} bytes, sent {frames_sent} PCM frames, total={bytes_in}")

                    if bytes_in == len(data):
                        try:
                            _rust_log_file.flush()
                            with open("/tmp/rust-server.log", "r") as f:
                                content = f.read()
                                print(f"[Rust log: {len(content)} chars]")
                                if content:
                                    print("=== RUST LOG START ===")
                                    print(content[:2000])
                                    print("=== RUST LOG END ===")
                        except Exception as e:
                            print(f"Log read error: {e}")

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
        try:
            _rust_log_file.flush()
            with open("/tmp/rust-server.log", "r") as f:
                content = f.read()
                if content:
                    print(f"Rust server log ({len(content)} chars):")
                    print(content[-2000:] if len(content) > 2000 else content)
        except Exception as e:
            print(f"Could not read Rust log: {e}")
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting Kyutai STT local server on 0.0.0.0:{port}")
    uvicorn.run(web_app, host="0.0.0.0", port=port)
