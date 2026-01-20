"""Modal deployment configuration for Kyutai STT service.

Deploy with: uvx modal deploy src/stt/modal_app.py
Dev server: uvx modal serve src/stt/modal_app.py
"""

from pathlib import Path

import modal

MODEL_ID = "kyutai/stt-1b-en_fr-trfs"
MODEL_DIR = Path("/models/stt")


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
    gpu="L40S",
    memory=32768,  # 32GB system RAM
    timeout=600,
    # Scale to zero when idle - no always-on containers
    min_containers=0,
    buffer_containers=0,
    scaledown_window=60,  # 1-minute idle timeout
    enable_memory_snapshot=True,
    # Note: GPU snapshots require experimental flag
    # experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=12, target_inputs=10)
class KyutaiSTTService:
    """Modal service class for Kyutai STT."""

    @modal.enter(snap=True)
    def initialize(self):
        """Initialize model once per container lifecycle."""
        import asyncio

        import numpy as np
        import torch
        from transformers import (
            KyutaiSpeechToTextForConditionalGeneration,
            KyutaiSpeechToTextProcessor,
        )

        self.processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_DIR)
        self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Warmup CUDA kernels
        dummy = np.random.randn(24000).astype(np.float32) * 0.1
        with torch.no_grad():
            inputs = self.processor(
                audio=dummy, sampling_rate=24000, return_tensors="pt"
            )
            input_values = inputs.input_values.to("cuda")
            _ = self.model.generate(input_values=input_values, max_new_tokens=5)

        # Initialize batching infrastructure
        self._lock = asyncio.Lock()
        print("Model initialized and warmed up")

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
            output_ids = self.model.generate(input_values=input_values, max_new_tokens=256)
            texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return texts

    @modal.asgi_app()
    def serve(self):
        """Create and return the ASGI app."""
        import asyncio

        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        import numpy as np

        CHUNK_BYTES = 23040  # 480ms at 24kHz PCM16

        web_app = FastAPI(title="Kyutai STT Service")

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": MODEL_ID,
                "gpu": "L40S",
                "sample_rate": 24000,
            }

        @web_app.websocket("/v1/stream")
        async def stream_transcribe(ws: WebSocket):
            await ws.accept()
            buffer = bytearray()

            try:
                while True:
                    data = await ws.receive_bytes()

                    if data == b"EOS":
                        if buffer:
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

                    if len(buffer) >= CHUNK_BYTES:
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
