"""Modal deployment for Kyutai STT with low-latency streaming.

Deploy with: uvx modal deploy src/stt/modal_app.py
Dev server: uvx modal serve src/stt/modal_app.py

This implementation uses moshi's streaming architecture for ~0.5s first-token latency
instead of the batch-based transformers approach (~5s latency).

Authentication:
  The endpoint requires proxy auth tokens. Create tokens in Modal workspace settings.
  Clients must pass Modal-Key and Modal-Secret headers (or query params for WebSocket).
"""

import asyncio
import os
import time
from pathlib import Path

import modal

# Model choice: 1B model for low latency (0.5s first-token vs 2.5s for 2.6B)
MODEL_NAME = os.getenv("MODEL_NAME", "kyutai/stt-1b-en_fr")
KYUTAI_GPU = os.getenv("KYUTAI_GPU", "A100")
APP_NAME = os.getenv("KYUTAI_APP_NAME", "kyutai-stt")  # Allows deploying multiple GPU variants
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "4"))
IDLE_AUDIO_TIMEOUT_SECONDS = float(os.getenv("IDLE_AUDIO_TIMEOUT_SECONDS", "10.0"))  # Close idle connections quickly
MAX_SESSION_SECONDS = float(os.getenv("MAX_SESSION_SECONDS", "3600.0"))  # 1 hour max per session
PING_INTERVAL_SECONDS = float(os.getenv("PING_INTERVAL_SECONDS", "10.0"))  # Detect dead connections
# LM generation parameters (temp=0 = greedy decoding, fastest and deterministic)
LM_TEMP = float(os.getenv("LM_TEMP", "0"))
LM_TEMP_TEXT = float(os.getenv("LM_TEMP_TEXT", "0"))

# Build image with moshi stack
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "moshi==0.2.9",
        "sphn",
        "torch==2.4.0",
        "numpy<2",
        "fastapi>=0.115.0",
        "websockets>=13.0",
        "huggingface-hub[hf_transfer]>=0.25.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

# Volume for caching model weights
hf_cache_vol = modal.Volume.from_name("kyutai-stt-hf-cache", create_if_missing=True)
hf_cache_vol_path = Path("/root/.cache/huggingface")
volumes = {hf_cache_vol_path: hf_cache_vol}

app = modal.App(APP_NAME, image=image)

MINUTES = 60


@app.cls(
    gpu=KYUTAI_GPU,
    volumes=volumes,
    timeout=10 * MINUTES,  # Max 10 min per request (prevents stuck connections)
    scaledown_window=60,  # 1 minute idle before scale down
    max_containers=2,
    min_containers=0,
    buffer_containers=0,  # No buffer containers
    enable_memory_snapshot=False,  # Disabled: moshi streaming state doesn't survive snapshots
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_SESSIONS)
class KyutaiSTTService:
    """Real-time streaming Speech-to-Text service using Kyutai STT with moshi."""

    BATCH_SIZE = 1

    @modal.enter()
    def load_model(self):
        """Load model and warmup GPU on container startup."""
        import torch
        from huggingface_hub import snapshot_download
        from moshi.models import LMGen, loaders

        print(f"Loading Kyutai STT model: {MODEL_NAME}")
        start_time = time.monotonic_ns()

        # Download model weights (cached in volume)
        snapshot_download(MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Enable TF32 for Ampere+ GPUs (A100, A10G, L4, H100)
        # Ignored on older GPUs (T4) - they'll use FP32 automatically
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load model components directly to GPU
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(MODEL_NAME)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.moshi = checkpoint_info.get_moshi(device=self.device)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()

        # torch.compile with CUDA graphs - beneficial on Ampere+ GPUs
        # Disable on T4 or for debugging with TORCH_COMPILE=0
        use_compile = os.getenv("TORCH_COMPILE", "auto")
        if use_compile == "auto":
            # Auto-detect: enable for Ampere+ (compute capability >= 8.0)
            if self.device == "cuda":
                capability = torch.cuda.get_device_capability()
                use_compile = capability[0] >= 8  # Ampere is 8.0, Turing (T4) is 7.5
            else:
                use_compile = False
        else:
            use_compile = use_compile == "1"

        if use_compile:
            print("Compiling mimi encoder with torch.compile (Ampere+ GPU detected)...")
            self.mimi.encoder = torch.compile(self.mimi.encoder, mode="reduce-overhead")

        # Create language model generator
        # temp=0 means greedy decoding (fastest, deterministic)
        # Higher temp (e.g., 0.8) adds randomness, useful for creative tasks
        self.lm_gen = LMGen(self.moshi, temp=LM_TEMP, temp_text=LM_TEMP_TEXT)

        # Model configuration
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        print(
            f"Mimi: sample_rate={self.mimi.sample_rate} "
            f"frame_rate={self.mimi.frame_rate} frame_size={self.frame_size}"
        )

        # Enable streaming mode
        self.mimi.streaming_forever(self.BATCH_SIZE)
        self.lm_gen.streaming_forever(self.BATCH_SIZE)

        # Warmup CUDA kernels
        print("Warming up GPU...")
        for _ in range(4):
            codes = self.mimi.encode(
                torch.zeros(self.BATCH_SIZE, 1, self.frame_size).to(self.device)
            )
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Initialize session semaphore
        self.session_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SESSIONS)

        elapsed = round((time.monotonic_ns() - start_time) / 1e9, 2)
        print(f"Model loaded and warmed up in {elapsed}s")

    def reset_state(self):
        """Reset model state between sessions."""
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    async def transcribe_chunk(self, pcm, all_pcm_data):
        """Process an audio chunk and yield transcription tokens.

        Args:
            pcm: New PCM audio data (numpy array, float32)
            all_pcm_data: Accumulated PCM data from previous chunks

        Yields:
            - str: A transcription token (word fragment)
            - dict: Control messages (e.g., {"type": "vad_end"})
            - numpy.ndarray: Remaining PCM data (yielded last)
        """
        import numpy as np
        import torch

        if pcm is None or len(pcm) == 0:
            yield all_pcm_data
            return

        if pcm.shape[-1] == 0:
            yield all_pcm_data
            return

        if all_pcm_data is None:
            all_pcm_data = pcm
        else:
            all_pcm_data = np.concatenate((all_pcm_data, pcm))

        # Process each complete frame (80ms at 24kHz = 1920 samples)
        while all_pcm_data.shape[-1] >= self.frame_size:
            chunk = all_pcm_data[: self.frame_size]
            all_pcm_data = all_pcm_data[self.frame_size :]

            with torch.no_grad():
                chunk = torch.from_numpy(chunk)
                chunk = chunk.unsqueeze(0).unsqueeze(0)  # (1, 1, frame_size)
                chunk = chunk.expand(self.BATCH_SIZE, -1, -1)
                chunk = chunk.to(device=self.device)

                # Encode audio with mimi
                codes = self.mimi.encode(chunk)

                # Run language model inference
                for c in range(codes.shape[-1]):
                    text_tokens, vad_heads = self.lm_gen.step_with_extra_heads(
                        codes[:, :, c : c + 1]
                    )
                    if text_tokens is None:
                        yield all_pcm_data
                        return

                    # Check voice activity detection
                    if vad_heads:
                        pr_vad = vad_heads[2][0, 0, 0].cpu().item()
                        if pr_vad > 0.5:
                            # End of speech detected
                            yield {"type": "vad_end"}
                            yield all_pcm_data
                            return

                    text_token = text_tokens[0, 0, 0].item()
                    # Token 0 and 3 are special tokens (padding/silence)
                    if text_token not in (0, 3):
                        text = self.text_tokenizer.id_to_piece(text_token)
                        text = text.replace("\u2581", " ")  # Sentencepiece space marker
                        yield text

        yield all_pcm_data

    @modal.asgi_app(requires_proxy_auth=True)
    def serve(self):
        """Create and return the ASGI app."""
        import json
        import traceback

        import numpy as np
        import sphn
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        web_app = FastAPI(title="Kyutai STT Streaming API")

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": MODEL_NAME,
                "gpu": KYUTAI_GPU,
                "sample_rate": int(self.mimi.sample_rate),
                "frame_rate": int(self.mimi.frame_rate),
                "frame_size": self.frame_size,
            }

        @web_app.get("/")
        async def root():
            return {
                "service": "Kyutai STT (Streaming)",
                "model": MODEL_NAME,
                "status": "ready",
                "endpoints": {
                    "websocket": "/v1/stream",
                    "health": "/health",
                },
            }

        @web_app.websocket("/v1/stream")
        async def transcribe_websocket(ws: WebSocket):
            """WebSocket endpoint for streaming audio transcription.

            Protocol:
            - Client sends: Opus-encoded audio bytes
            - Server sends: JSON messages with transcription updates
              - {"type": "token", "text": "word"} - New token
              - {"type": "vad_end"} - Voice activity ended
              - {"type": "ping"} - Keepalive (client should ignore)
              - {"type": "error", "message": "..."} - Error occurred
            """
            await ws.accept()
            print("Session started")

            session_start = time.monotonic()
            bytes_in = 0
            tokens_sent = 0
            all_pcm_data = None
            capture_bytes = bytearray()
            processed_samples = 0
            recv_debug_logged = 0
            connection_dead = asyncio.Event()

            async def send_json(payload: dict) -> bool:
                try:
                    await asyncio.wait_for(ws.send_text(json.dumps(payload)), timeout=5.0)
                    return True
                except Exception:
                    connection_dead.set()
                    return False

            async def ping_task():
                """Send periodic pings to detect dead connections."""
                while not connection_dead.is_set():
                    await asyncio.sleep(PING_INTERVAL_SECONDS)
                    if connection_dead.is_set():
                        break
                    ok = await send_json({"type": "ping"})
                    if not ok:
                        print("[ws] ping failed - connection dead")
                        break

            ping_handle = asyncio.create_task(ping_task())

            try:
                while not connection_dead.is_set():
                    # Enforce max session duration
                    if time.monotonic() - session_start > MAX_SESSION_SECONDS:
                        print(f"[ws] max session duration reached ({MAX_SESSION_SECONDS}s)")
                        break

                    try:
                        data = await asyncio.wait_for(
                            ws.receive_bytes(), timeout=IDLE_AUDIO_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        print("[ws] receive timeout")
                        break
                    except WebSocketDisconnect:
                        print("[ws] client disconnect")
                        break
                    except Exception as e:
                        print(f"[ws] receive error: {e}")
                        traceback.print_exc()
                        break

                    if not data:
                        continue

                    bytes_in += len(data)
                    if len(capture_bytes) < 1_500_000:
                        capture_bytes.extend(data)
                    if recv_debug_logged < 3:
                        recv_debug_logged += 1
                        print(f"[ws] received chunk len={len(data)} total={bytes_in}")

                    # Decode all received Opus bytes and process new PCM samples
                    try:
                        pcm_out, sr_out = sphn.read_opus_bytes(bytes(capture_bytes))
                    except Exception as exc:
                        print(f"[ws] opus decode error: {exc}")
                        continue

                    if pcm_out.ndim > 1:
                        pcm_out = pcm_out[0]

                    # Resample if needed
                    target_sr = self.mimi.sample_rate
                    if sr_out != target_sr:
                        x_old = np.linspace(0, 1, pcm_out.shape[-1])
                        x_new = np.linspace(0, 1, int(pcm_out.shape[-1] * target_sr / sr_out))
                        pcm_out = np.interp(x_new, x_old, pcm_out).astype(np.float32)

                    if processed_samples >= pcm_out.shape[-1]:
                        continue

                    new_pcm = pcm_out[processed_samples:]
                    processed_samples = pcm_out.shape[-1]

                    # Process audio and yield tokens
                    try:
                        async for msg in self.transcribe_chunk(new_pcm, all_pcm_data):
                            if isinstance(msg, str):
                                ok = await send_json({"type": "token", "text": msg})
                                if not ok:
                                    print(f"[ws] send failed for token: {msg}")
                                    raise RuntimeError("send failed")
                                tokens_sent += 1
                                if tokens_sent <= 5:
                                    print(f"[ws] sent token {tokens_sent}: {msg}")
                            elif isinstance(msg, dict):
                                ok = await send_json(msg)
                                if not ok:
                                    print(f"[ws] send failed for msg: {msg}")
                                    raise RuntimeError("send failed")
                            else:
                                all_pcm_data = msg
                    except Exception as e:
                        print(f"[ws] inference/send error: {e}")
                        traceback.print_exc()
                        continue

            except Exception as e:
                print(f"[ws] session error: {e}")
                traceback.print_exc()
            finally:
                # Stop ping task
                connection_dead.set()
                ping_handle.cancel()
                try:
                    await ping_handle
                except asyncio.CancelledError:
                    pass

                print(
                    f"[session] bytes_in={bytes_in} tokens={tokens_sent}"
                )
                self.reset_state()
                try:
                    await ws.close()
                except Exception:
                    pass
                print("Session ended")

        return web_app
