# Kyutai STT - Real-time Speech-to-Text on Modal

Deploy [Kyutai's streaming STT model](https://huggingface.co/kyutai/stt-1b-en_fr) to [Modal](https://modal.com) for real-time speech transcription with **~0.5 second latency**.

```
You: "Hello, how are you today?"
     ↓ ~500ms
Server: {"type": "token", "text": " Hello"}
        {"type": "token", "text": ","}
        {"type": "token", "text": " how"}
        ...
```

## Features

- **Low latency**: First token in ~0.5s using moshi streaming architecture
- **Real-time**: Token-by-token transcription over WebSocket
- **Scalable**: Auto-scales from 0 to handle concurrent sessions
- **Cost-effective**: Pay only for GPU time used (scales to zero when idle)
- **Simple protocol**: Send raw PCM float32 audio, receive JSON tokens

## Prerequisites

- [Modal account](https://modal.com) (free tier available)
- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.11+

## Quick Start

### 1. Install Modal CLI and authenticate

```bash
uvx modal setup
```

### 2. Clone and deploy

```bash
git clone https://github.com/YOUR_USERNAME/kyutai-stt-modal.git
cd kyutai-stt-modal

# Deploy to Modal
uvx modal deploy src/stt/modal_app.py
```

### 3. Set up environment

Go to your [Modal workspace settings](https://modal.com/settings) and create proxy auth tokens. Then set environment variables:

```bash
# Your Modal workspace name (shown in Modal dashboard URL)
export MODAL_WORKSPACE=your-workspace-name

# Proxy auth credentials
export MODAL_KEY=your-key
export MODAL_SECRET=your-secret
```

### 4. Test with your microphone

```bash
# Install dependencies and run
uv run scripts/transcribe_cli.py
```

Speak into your microphone and see real-time transcription.

## Usage

### WebSocket API

Connect to `wss://{MODAL_WORKSPACE}--kyutai-stt-rust-kyutaisttrustservice-serve.modal.run/v1/stream`

**Protocol:**
- **Client sends**: Raw PCM float32 (little-endian) audio bytes (24kHz mono). Send in ~80ms chunks for low latency.
- **Server sends**: JSON messages

```json
{"type": "token", "text": " Hello"}     // Transcription token
{"type": "vad_end"}                      // Voice activity ended (sentence boundary)
{"type": "ping"}                         // Keepalive (ignore)
{"type": "error", "message": "..."}      // Error occurred
```

**Authentication**: Include headers `Modal-Key` and `Modal-Secret` with your credentials.

### Python Client Example

```python
import asyncio
import websockets
import json
import os

async def transcribe():
    workspace = os.environ["MODAL_WORKSPACE"]
    uri = f"wss://{workspace}--kyutai-stt-rust-kyutaisttrustservice-serve.modal.run/v1/stream"
    headers = {
        "Modal-Key": os.environ["MODAL_KEY"],
        "Modal-Secret": os.environ["MODAL_SECRET"],
    }

    async with websockets.connect(uri, additional_headers=headers) as ws:
        # Send raw PCM float32 (LE) audio bytes sampled at 24kHz mono
        # Example: audio, _ = soundfile.read("audio.wav", dtype="float32")
        #          pcm_audio_bytes = audio.astype("float32").tobytes()
        await ws.send(pcm_audio_bytes)

        # Receive tokens
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "token":
                print(data["text"], end="", flush=True)
            elif data["type"] == "vad_end":
                print()  # New line after sentence

asyncio.run(transcribe())
```

### CLI Tools

```bash
# Real-time microphone transcription
uv run scripts/transcribe_cli.py

# List available audio devices
uv run scripts/transcribe_cli.py --list-devices

# Use specific microphone
uv run scripts/transcribe_cli.py --device 2

# Latency benchmark
uv run scripts/latency_test.py -p 4
```

## Configuration

### Client Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODAL_WORKSPACE` | Yes | Your Modal workspace name |
| `MODAL_KEY` | Yes | Modal proxy auth key |
| `MODAL_SECRET` | Yes | Modal proxy auth secret |

### Server Configuration (set when deploying)

| Variable | Default | Description |
|----------|---------|-------------|
| `KYUTAI_GPU` | `L40S` | GPU type (`L4`, `A10G`, `L40S`, `A100`, `H100`) |
| `BATCH_SIZE` | `8` | Max concurrent sessions per container |
| `MODEL_NAME` | `kyutai/stt-1b-en_fr` | Model to use (1B or 2.6B) |

Example:

```bash
KYUTAI_GPU=L4 uvx modal deploy src/stt/modal_app.py
```

## GPU Selection & Pricing

Modal bills per-second. Choose based on your latency and cost requirements:

| GPU | VRAM | Cost/Hour | First Token Latency |
|-----|------|-----------|---------------------|
| **T4** | 16GB | $0.59 | ~0.7s |
| **L4** | 24GB | $0.80 | ~0.6s |
| **A10G** | 24GB | $1.10 | ~0.5s |
| **A100** | 80GB | $2.78 | ~0.5s |

**Benchmarks**: First token latency measured with 8 seconds of audio on warm container.

**Scaling**: Each container handles up to 8 concurrent sessions (configurable via `BATCH_SIZE`). Modal automatically scales containers to handle more concurrent users (up to 10 containers by default).

### Benchmark Different GPUs

```bash
# Deploy and test each GPU (creates separate apps)
uv run scripts/latency_test.py --compare-gpus "T4,L4,A10G,A100" -p 4

# Cleanup after benchmarking
uvx modal app stop kyutai-stt-t4
uvx modal app stop kyutai-stt-l4
uvx modal app stop kyutai-stt-a10g
uvx modal app stop kyutai-stt-a100
```

## How It Works

1. **Audio Capture**: Client captures microphone audio at 24kHz mono and streams raw PCM float32
2. **WebSocket Streaming**: PCM chunks (~80ms) are streamed to the Python proxy over WebSocket
3. **Rust Server**: A Python proxy forwards audio to the internal Rust moshi-server (supports batched inference)
4. **Neural Codec**: Audio is encoded with Mimi neural codec (80ms frames)
5. **Language Model**: Each frame is processed by the streaming transformer for immediate token output
6. **Token Streaming**: Tokens are sent back immediately as they're generated (~0.5s latency)

The key to low latency is the **moshi streaming architecture** - instead of waiting for complete utterances, the model processes audio frame-by-frame and outputs tokens incrementally. The Rust server enables efficient batched processing of multiple concurrent streams.

## Cost Optimization

The deployment is configured for cost efficiency:

- **Scale to zero**: Containers shut down after 60s of no connections
- **Idle timeout**: WebSocket connections close after 10s without audio
- **No buffer containers**: Only spin up containers when needed

Typical costs:
- Cold start: ~20-30s (first request after scale-down)
- Warm request: ~0.5s latency
- GPU time: Only billed while processing

## Troubleshooting

**401 Unauthorized**: Check your `MODAL_KEY` and `MODAL_SECRET` are set correctly.

**Connection timeout**: The service may have scaled to zero. First request takes 20-30s for cold start.

**No transcription**: Ensure you're sending raw PCM float32 24kHz mono audio (little-endian).

**High latency**: Check your GPU selection. T4 is cheapest but slower than A10G/A100.

## License

MIT

## Acknowledgments

- [Kyutai](https://kyutai.org/) for the STT model and moshi library
- [Modal](https://modal.com/) for serverless GPU infrastructure
