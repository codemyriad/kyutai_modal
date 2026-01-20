# Deploying kyutai/stt-1b-en_fr on Modal with WebSocket streaming

The **kyutai/stt-1b-en_fr** model is a 1-billion parameter streaming speech-to-text system built on the Moshi architecture. Deploying it on Modal for **10 concurrent WebSocket audio streams** requires an **L40S GPU** (48GB VRAM) with memory snapshots enabled, achieving cold starts under 5 seconds. The critical configuration: `@modal.concurrent(max_inputs=12)` combined with `min_containers=2` ensures instant availability while the model's 24kHz audio requirement and 80ms frame size dictate your chunking strategy.

This guide provides production-ready code and configurations for building a high-performance streaming STT service.

---

## Model architecture and hardware requirements

The kyutai/stt-1b-en_fr uses a **decoder-only transformer** with Delayed Streams Modeling—fundamentally different from CTC or transducer architectures. The model processes audio through the **Mimi neural codec** at 12.5 Hz (80ms frames), producing text with a fixed **0.5-second latency** due to the temporal offset between audio and text streams.

**Key specifications:**
- **Parameters**: ~1 billion (48 layers, 32 attention heads, 2048 hidden size)
- **Model weights**: ~2.0 GB (safetensors) + 385 MB (Mimi codec)
- **VRAM requirement**: **24 GB minimum** for PyTorch (no quantization available)
- **Audio input**: 24kHz mono, 80ms frames (1920 samples per frame)
- **Throughput benchmarks**: H100 handles 400 concurrent streams; L40S handles 64 streams at 3× real-time

The Transformers implementation (`kyutai/stt-1b-en_fr-trfs`) supports batched inference, making it ideal for serving multiple concurrent streams efficiently.

```python
# Core dependencies (pin versions for reproducibility)
DEPENDENCIES = [
    "torch==2.4.0",
    "transformers>=4.53.0",
    "accelerate>=0.33.0",
    "huggingface-hub[hf_transfer]>=0.25.0",
    "torchaudio>=2.4.0",
    "fastapi>=0.115.0",
    "websockets>=13.0",
]
```

---

## GPU selection for 10 concurrent streams

For **10 concurrent audio streams**, the memory calculation drives GPU selection. The shared model consumes ~4GB, while each active stream requires approximately **200-400 MB** for audio buffers, intermediate activations, and session state.

| GPU | VRAM | Modal Price | Concurrent Streams | Recommendation |
|-----|------|-------------|-------------------|----------------|
| A10G | 24 GB | $0.000306/s | 6-8 streams | Budget option, needs 2 containers |
| **L40S** | 48 GB | $0.000542/s | **10-15 streams** | **Best choice for 10 streams** |
| A100-40GB | 40 GB | $0.000583/s | 10-12 streams | Overkill for inference |

The **L40S** provides the optimal cost-performance ratio: enough headroom for 10 streams plus burst capacity within a single container, eliminating cross-container coordination complexity.

---

## Image building with baked-in model weights

For fastest cold starts, bake model weights directly into the container image rather than downloading at runtime. This approach adds ~3GB to image size but eliminates network latency during startup.

```python
import modal
from pathlib import Path

MODEL_ID = "kyutai/stt-1b-en_fr-trfs"
MODEL_DIR = Path("/models/stt")

def download_model():
    """Download model during image build."""
    from huggingface_hub import snapshot_download
    import os
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
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .run_function(download_model)  # Bake weights into image
)
```

**Layer caching strategy**: Order pip installations by change frequency—stable dependencies first, frequently-updated packages last. Pin exact versions to maximize cache hits.

---

## Container lifecycle with memory snapshots

Modal's **GPU memory snapshots** can reduce cold starts from 45+ seconds to under 5 seconds by capturing the fully-initialized CUDA state. The `@modal.enter()` decorator runs once when a container starts, before accepting any requests.

```python
app = modal.App("stt-service", image=image)

@app.cls(
    gpu="L40S",
    memory=32768,  # 32GB system RAM
    timeout=600,
    # Cold start optimization
    min_containers=2,           # Always keep 2 warm
    buffer_containers=1,        # Extra container for burst traffic
    scaledown_window=300,       # 5-minute idle timeout
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},  # 10x faster restarts
)
@modal.concurrent(max_inputs=12, target_inputs=10)  # 10 target, 12 burst
class STTService:
    
    @modal.enter(snap=True)  # Include in memory snapshot
    def load_model(self):
        """Initialize model once per container lifecycle."""
        import torch
        from transformers import (
            KyutaiSpeechToTextProcessor,
            KyutaiSpeechToTextForConditionalGeneration,
        )
        
        self.device = "cuda"
        self.processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_DIR)
        self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        
        # Warm up CUDA kernels (captured in snapshot)
        dummy_audio = torch.randn(1, 24000, device=self.device)
        with torch.no_grad():
            inputs = self.processor(dummy_audio.cpu().numpy())
            inputs.to(self.device)
            _ = self.model.generate(**inputs, max_new_tokens=10)
        
        print(f"Model loaded on {self.device}, ready for inference")
```

The `snap=True` parameter ensures the fully-warmed model state is captured, including compiled CUDA kernels and memory allocations.

---

## WebSocket server implementation

Modal supports WebSockets via `@modal.asgi_app()`. Each WebSocket connection counts as one concurrent input, making `@modal.concurrent()` essential for handling multiple streams.

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio
import numpy as np

@app.cls(
    gpu="L40S",
    memory=32768,
    min_containers=2,
    buffer_containers=1,
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=12, target_inputs=10)
class STTWebSocketServer:
    
    @modal.enter(snap=True)
    def setup(self):
        import torch
        from transformers import (
            KyutaiSpeechToTextProcessor,
            KyutaiSpeechToTextForConditionalGeneration,
        )
        
        self.processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_DIR)
        self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        
        # Session state for concurrent streams
        self.sessions: Dict[str, "StreamSession"] = {}
        self._lock = asyncio.Lock()
    
    @modal.asgi_app()
    def web(self):
        web_app = FastAPI(title="Kyutai STT Service")
        
        @web_app.get("/health")
        async def health():
            return {"status": "healthy", "model": MODEL_ID}
        
        @web_app.websocket("/v1/transcribe")
        async def transcribe(websocket: WebSocket):
            await websocket.accept()
            session_id = str(id(websocket))
            
            # Initialize stream session
            session = StreamSession(
                sample_rate=24000,
                frame_size=1920,  # 80ms at 24kHz
            )
            
            async with self._lock:
                self.sessions[session_id] = session
            
            try:
                await self._handle_stream(websocket, session)
            except WebSocketDisconnect:
                print(f"Client {session_id} disconnected")
            except Exception as e:
                print(f"Stream error: {e}")
                try:
                    await websocket.close(code=1011, reason=str(e))
                except:
                    pass
            finally:
                async with self._lock:
                    self.sessions.pop(session_id, None)
        
        return web_app
    
    async def _handle_stream(self, websocket: WebSocket, session: "StreamSession"):
        """Process incoming audio chunks and stream transcriptions."""
        import torch
        
        while True:
            # Receive binary audio data (PCM 24kHz 16-bit mono)
            data = await websocket.receive_bytes()
            
            # Handle end-of-stream signal
            if data == b"EOS":
                # Process any remaining audio in buffer
                if session.buffer_samples > 0:
                    text = await self._transcribe(session.flush())
                    if text:
                        await websocket.send_json({"text": text, "final": True})
                await websocket.send_json({"status": "complete"})
                break
            
            # Accumulate audio in session buffer
            session.append(data)
            
            # Process when buffer reaches threshold (480ms = 6 frames)
            if session.buffer_samples >= 11520:  # 480ms at 24kHz
                audio_array = session.flush()
                text = await self._transcribe(audio_array)
                
                if text and text.strip():
                    await websocket.send_json({
                        "text": text,
                        "final": False,
                    })
    
    async def _transcribe(self, audio_array: np.ndarray) -> str:
        """Run inference on audio array."""
        import torch
        
        # Run inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_array,
        )
    
    def _transcribe_sync(self, audio_array: np.ndarray) -> str:
        """Synchronous transcription for executor."""
        import torch
        
        with torch.no_grad():
            inputs = self.processor(audio_array, sampling_rate=24000)
            inputs = inputs.to(self.model.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=256)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        
        return text[0] if text else ""


class StreamSession:
    """Manages audio buffer state for a single WebSocket connection."""
    
    def __init__(self, sample_rate: int = 24000, frame_size: int = 1920):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._buffer = bytearray()
    
    @property
    def buffer_samples(self) -> int:
        return len(self._buffer) // 2  # 16-bit = 2 bytes per sample
    
    def append(self, data: bytes):
        self._buffer.extend(data)
    
    def flush(self) -> np.ndarray:
        """Return buffer as float32 array and clear."""
        audio = np.frombuffer(self._buffer, dtype=np.int16).astype(np.float32)
        audio /= 32768.0  # Normalize to [-1, 1]
        self._buffer.clear()
        return audio
```

---

## Audio chunking strategy for streaming

The kyutai model requires **24kHz audio** (not 16kHz like Whisper). Clients must resample before sending. The 80ms frame size (1920 samples) sets the minimum chunk granularity.

```python
# Client-side audio configuration
AUDIO_CONFIG = {
    "sample_rate": 24000,      # Required by Mimi codec
    "channels": 1,              # Mono only
    "sample_width": 2,          # 16-bit PCM
    "frame_size": 1920,         # 80ms at 24kHz (minimum chunk)
    "recommended_chunk": 11520, # 480ms (6 frames) for batching efficiency
    "max_chunk": 24000,         # 1 second maximum
}

# Chunk size in bytes: samples × 2 bytes = 3840 bytes per 80ms frame
# Recommended: Send 480ms chunks (23040 bytes) for optimal latency/throughput
```

**Chunking recommendations**: Send audio every **200-500ms** to balance latency against network overhead. Smaller chunks (80-100ms) reduce latency but increase packet overhead; larger chunks (500ms+) improve throughput but add perceived delay.

---

## Handling concurrent streams with batching

For maximum throughput, batch requests from multiple streams into single inference calls. The Transformers implementation supports batched inputs via padding.

```python
import asyncio
from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass
class BatchRequest:
    audio: np.ndarray
    future: asyncio.Future
    session_id: str

class BatchingTranscriber:
    """Batches requests from multiple streams for efficient GPU utilization."""
    
    def __init__(self, model, processor, max_batch_size: int = 8, max_wait_ms: int = 50):
        self.model = model
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._running = True
    
    async def start(self):
        """Start the background batch processor."""
        asyncio.create_task(self._batch_loop())
    
    async def transcribe(self, audio: np.ndarray, session_id: str) -> str:
        """Submit audio for batched transcription."""
        future = asyncio.get_event_loop().create_future()
        await self._queue.put(BatchRequest(audio, future, session_id))
        return await future
    
    async def _batch_loop(self):
        """Continuously process batches."""
        while self._running:
            batch: List[BatchRequest] = []
            
            # Wait for first request
            try:
                first = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                batch.append(first)
            except asyncio.TimeoutError:
                continue
            
            # Collect more requests up to batch size or timeout
            deadline = asyncio.get_event_loop().time() + (self.max_wait_ms / 1000)
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=remaining
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
            
            # Process batch
            await self._process_batch(batch)
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """Run batched inference."""
        if not batch:
            return
        
        loop = asyncio.get_event_loop()
        
        def run_inference():
            audio_arrays = [req.audio for req in batch]
            
            with torch.no_grad():
                inputs = self.processor(
                    audio_arrays,
                    sampling_rate=24000,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = inputs.to(self.model.device)
                output_ids = self.model.generate(**inputs, max_new_tokens=256)
                texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)
            
            return texts
        
        try:
            results = await loop.run_in_executor(None, run_inference)
            for req, text in zip(batch, results):
                req.future.set_result(text)
        except Exception as e:
            for req in batch:
                req.future.set_exception(e)
```

---

## Complete production deployment

Here's the full Modal application combining all components:

```python
import modal
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

MODEL_ID = "kyutai/stt-1b-en_fr-trfs"
MODEL_DIR = Path("/models/stt")

def download_model():
    from huggingface_hub import snapshot_download
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)

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
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_function(download_model)
)

app = modal.App("kyutai-stt-production", image=image)

@app.cls(
    gpu="L40S",
    memory=32768,
    timeout=600,
    min_containers=2,
    buffer_containers=1,
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=12, target_inputs=10)
class KyutaiSTTService:
    
    @modal.enter(snap=True)
    def initialize(self):
        import torch
        from transformers import (
            KyutaiSpeechToTextProcessor,
            KyutaiSpeechToTextForConditionalGeneration,
        )
        
        self.processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_DIR)
        self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.sessions: Dict[str, bytearray] = {}
        self._lock = asyncio.Lock()
        
        # Warmup
        dummy = torch.randn(1, 24000)
        with torch.no_grad():
            inputs = self.processor(dummy.numpy(), sampling_rate=24000)
            inputs.to("cuda")
            self.model.generate(**inputs, max_new_tokens=5)
        print("Model initialized and warmed up")
    
    @modal.asgi_app()
    def serve(self):
        web_app = FastAPI()
        
        @web_app.get("/health")
        def health():
            return {"status": "ok", "model": MODEL_ID, "gpu": "L40S"}
        
        @web_app.websocket("/v1/stream")
        async def stream_transcribe(ws: WebSocket):
            await ws.accept()
            sid = str(id(ws))
            buffer = bytearray()
            
            try:
                while True:
                    data = await ws.receive_bytes()
                    if data == b"EOS":
                        if buffer:
                            text = self._infer(buffer)
                            await ws.send_json({"text": text, "final": True})
                        break
                    
                    buffer.extend(data)
                    if len(buffer) >= 23040:  # 480ms
                        text = self._infer(buffer)
                        buffer.clear()
                        if text.strip():
                            await ws.send_json({"text": text, "final": False})
                            
            except WebSocketDisconnect:
                pass
        
        return web_app
    
    def _infer(self, buffer: bytearray) -> str:
        import torch
        audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        with torch.no_grad():
            inputs = self.processor(audio, sampling_rate=24000)
            inputs.to("cuda")
            ids = self.model.generate(**inputs, max_new_tokens=256)
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0]
```

**Deploy with:**
```bash
modal deploy app.py
```

---

## Configuration reference summary

| Parameter | Recommended Value | Purpose |
|-----------|------------------|---------|
| `gpu` | `"L40S"` | 48GB VRAM for 10+ concurrent streams |
| `memory` | `32768` (32GB) | System RAM for audio processing |
| `min_containers` | `2` | Eliminate cold starts |
| `buffer_containers` | `1` | Handle traffic bursts |
| `scaledown_window` | `300` | 5-minute idle before shutdown |
| `max_inputs` | `12` | Maximum concurrent WebSocket connections |
| `target_inputs` | `10` | Autoscaler target (20% headroom) |
| `enable_memory_snapshot` | `True` | 1.5-3× faster cold starts |
| `enable_gpu_snapshot` | `True` | 5-10× faster GPU restoration |

## Conclusion

Deploying kyutai/stt-1b-en_fr on Modal requires understanding three critical constraints: the model's **24GB VRAM footprint** dictates L40S GPU selection, the **24kHz/80ms audio frame requirements** shape your WebSocket chunking strategy, and Modal's **container lifecycle** demands proper use of `@modal.enter()` with memory snapshots for sub-5-second cold starts.

The batching transcriber pattern maximizes GPU utilization across concurrent streams, while `@modal.concurrent(max_inputs=12)` allows the container to handle burst traffic beyond the target 10 streams. For production deployments, the `min_containers=2` configuration eliminates cold start latency entirely at the cost of approximately **$3.90/hour** (2 × L40S)—a reasonable tradeoff for real-time transcription services.