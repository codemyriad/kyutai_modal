"""Modal deployment configuration for Kyutai STT service.

Deploy with: uvx modal deploy src/stt/modal_app.py
Dev server: uvx modal serve src/stt/modal_app.py

Authentication:
  The endpoint requires proxy auth tokens. Create tokens in Modal workspace settings.
  Clients must pass Modal-Key and Modal-Secret headers (or query params for WebSocket).
"""

from pathlib import Path

import modal

MODEL_ID = "kyutai/stt-1b-en_fr-trfs"
MODEL_DIR = Path("/models/stt")

# Minimum audio length for reliable transcription
# ================================================
# The Kyutai model crashes with very short audio (<1 second):
#   IndexError: index -1 is out of bounds for dimension 0 with size 0
# This happens in model.generate() -> prepare_inputs_for_generation()
# when cache_position is empty due to insufficient audio frames.
#
# We require 2 seconds minimum for reliable results.
MIN_SAMPLES = 24000 * 2  # 2 seconds at 24kHz
MIN_BYTES = MIN_SAMPLES * 2  # PCM16 = 2 bytes per sample


def download_model():
    """Download model during image build."""
    import os

    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.md", "*.txt"],
    )


# Build image with model weights baked in
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.53.0",
        "accelerate>=0.33.0",
        "huggingface-hub[hf_transfer]>=0.25.0",
        "torchaudio>=2.4.0",
        "fastapi>=0.115.0",
        "websockets>=13.0",
        "numpy<2",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Disable torch.compile to avoid CUDA graph issues in thread pools
            "TORCH_COMPILE_DISABLE": "1",
            "TORCHDYNAMO_DISABLE": "1",
        }
    )
    .run_function(download_model)
)

app = modal.App("kyutai-stt", image=image)


@app.cls(
    gpu="A100",  # A100 shows most consistent latency (tested L40S, A100, H100)
    memory=32768,  # 32GB system RAM
    timeout=600,
    # Scale to zero when idle - no always-on containers
    # Note: scaledown_window may not account for active WebSocket connections
    # Use longer window for interactive sessions to avoid mid-session cold starts
    min_containers=0,
    buffer_containers=0,
    scaledown_window=300,  # 5-minute idle timeout
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=12, target_inputs=10)
class KyutaiSTTService:
    """Modal service class for Kyutai STT.

    Memory Snapshot Strategy
    ========================
    Modal's memory snapshots speed up cold starts by serializing container state.
    However, CUDA state cannot be serialized - snapshots run on CPU-only workers.

    We split initialization into two phases:

    1. snap=True (load_model): Runs ONCE during snapshot creation
       - Load model weights to CPU
       - Initialize processor
       - NO CUDA calls allowed (will fail with "No CUDA GPUs are available")
       - This state gets serialized to the snapshot

    2. snap=False (warmup_gpu): Runs on EVERY container restore
       - Move model from CPU to GPU
       - Warmup CUDA kernels
       - Initialize any runtime state (locks, etc.)

    Cold start flow:
      [Snapshot restore] -> [warmup_gpu()] -> [Ready to serve]

    Without snapshots, cold start would be:
      [Download model] -> [Load to GPU] -> [Warmup] -> [Ready to serve]

    The snapshot saves ~30-60s of model loading time on cold starts.
    """

    @modal.enter(snap=True)
    def load_model(self):
        """Load model to CPU during snapshot phase.

        WARNING: No CUDA calls here! Snapshots run on CPU-only workers.
        Calling .to("cuda") here causes: RuntimeError: No CUDA GPUs are available
        """
        import torch
        from transformers import (
            KyutaiSpeechToTextForConditionalGeneration,
            KyutaiSpeechToTextProcessor,
        )

        self.processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_DIR)
        # Load to CPU first - will move to GPU after snapshot restore
        self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.model.eval()
        print("Model loaded to CPU for snapshot")

    @modal.enter(snap=False)
    def warmup_gpu(self):
        """Move model to GPU and warmup after snapshot restore."""
        import asyncio

        import numpy as np
        import torch

        # Move model to GPU
        self.model = self.model.to("cuda")

        # Warmup CUDA kernels with a realistic audio length
        dummy = np.random.randn(48000).astype(np.float32) * 0.1  # 2 seconds
        with torch.no_grad():
            inputs = self.processor(
                audio=dummy, sampling_rate=24000, return_tensors="pt"
            )
            input_values = inputs.input_values.to("cuda")
            _ = self.model.generate(input_values=input_values, max_new_tokens=100)

        self._lock = asyncio.Lock()
        print("Model initialized and warmed up on GPU")

    def _transcribe_sync(self, audio_arrays: list) -> list[str]:
        """Synchronous batched transcription."""
        import torch

        with torch.no_grad():
            inputs = self.processor(
                audio=audio_arrays,
                sampling_rate=24000,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to("cuda")

            # Calculate reasonable max_new_tokens based on audio length
            # Roughly 1 token per 40ms of audio
            audio_duration_ms = len(audio_arrays[0]) / 24000 * 1000
            max_tokens = max(10, min(500, int(audio_duration_ms / 40)))

            output_ids = self.model.generate(
                input_values=input_values,
                max_new_tokens=max_tokens
            )
            texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return texts

    @modal.asgi_app(requires_proxy_auth=True)
    def serve(self):
        """Create and return the ASGI app."""
        import asyncio

        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        import numpy as np

        web_app = FastAPI(title="Kyutai STT Service")

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": MODEL_ID,
                "gpu": "L40S",
                "sample_rate": 24000,
                "min_audio_seconds": MIN_SAMPLES / 24000,
            }

        @web_app.websocket("/v1/stream")
        async def stream_transcribe(ws: WebSocket):
            await ws.accept()
            buffer = bytearray()

            try:
                while True:
                    data = await ws.receive_bytes()

                    if data == b"EOS":
                        # Process any remaining audio
                        if len(buffer) >= MIN_BYTES // 2:  # At least 1 second for final
                            audio = (
                                np.frombuffer(buffer, dtype=np.int16).astype(
                                    np.float32
                                )
                                / 32768.0
                            )
                            loop = asyncio.get_event_loop()
                            try:
                                texts = await loop.run_in_executor(
                                    None, self._transcribe_sync, [audio]
                                )
                                if texts and texts[0].strip():
                                    await ws.send_json({"text": texts[0], "final": True})
                            except Exception as e:
                                print(f"Transcription error (final): {e}")
                                import traceback
                                traceback.print_exc()
                                await ws.send_json({"error": str(e)})
                        await ws.send_json({"status": "complete"})
                        break

                    buffer.extend(data)

                    # Only transcribe when we have enough audio (2+ seconds)
                    if len(buffer) >= MIN_BYTES:
                        audio = (
                            np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        buffer.clear()

                        loop = asyncio.get_event_loop()
                        try:
                            texts = await loop.run_in_executor(
                                None, self._transcribe_sync, [audio]
                            )
                            if texts and texts[0].strip():
                                await ws.send_json({"text": texts[0], "final": False})
                        except Exception as e:
                            print(f"Transcription error: {e}")
                            import traceback
                            traceback.print_exc()
                            await ws.send_json({"error": str(e)})

            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"WebSocket error: {e}")
                import traceback
                traceback.print_exc()

        return web_app
