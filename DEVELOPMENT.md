# Development Guide

This document covers development setup, testing, and architecture details.

## Project Structure

```
kyutai-stt-modal/
├── src/stt/
│   ├── modal_app.py      # Modal deployment (main entry point)
│   ├── server.py         # Alternative FastAPI server (local dev)
│   ├── engine/           # STT engine implementations
│   │   ├── protocol.py   # Engine interface
│   │   ├── kyutai.py     # Real Kyutai model engine
│   │   └── fake.py       # Mock engine for testing
│   ├── audio.py          # Audio processing utilities
│   ├── batching.py       # Request batching logic
│   └── constants.py      # Shared constants
├── scripts/
│   ├── transcribe_cli.py # Real-time microphone transcription
│   ├── latency_test.py   # Latency benchmarking tool
│   ├── prepare_samples.py # Generate test audio samples
│   └── ws_stress.py      # WebSocket stress testing
├── tests/
│   ├── unit/             # Unit tests (no GPU required)
│   ├── integration/      # Integration tests (requires deployed service)
│   └── integration_gpu/  # GPU integration tests
├── samples/              # Test audio files
└── pyproject.toml        # Project configuration
```

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Modal account and CLI (`uvx modal setup`)

### Install Dependencies

```bash
# Install all dependencies including dev tools
uv sync --extra dev
```

### Environment Variables

Create a `.envrc` file (used by [direnv](https://direnv.net/)):

```bash
export MODAL_WORKSPACE=your-workspace-name
export MODAL_KEY=your-modal-proxy-key
export MODAL_SECRET=your-modal-proxy-secret
```

Or export manually:

```bash
source .envrc
```

## Running Locally

### Development Server (Modal)

```bash
# Hot-reload server (restarts on code changes)
uvx modal serve src/stt/modal_app.py
```

### Local FastAPI Server

For testing without Modal (requires local GPU):

```bash
uv run uvicorn stt.server:app --reload
```

## Testing

### Unit Tests

Unit tests don't require GPU or deployed services:

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific test file
uv run pytest tests/unit/test_server.py -v
```

### Integration Tests

Integration tests run against the deployed Modal service:

```bash
# Requires MODAL_KEY and MODAL_SECRET
uv run pytest tests/integration/ -v

# Run specific test
uv run pytest tests/integration/test_idle_timeout.py -v -s
```

### Critical Tests

The idle timeout test verifies that containers scale down properly:

```bash
uv run pytest tests/integration/test_idle_timeout.py::TestIdleTimeout::test_connection_closes_after_idle_timeout -v -s
```

This test ensures:
- WebSocket connections close after 10s without audio
- Containers can scale down when idle
- Costs are minimized

### GPU Integration Tests

Require a local GPU:

```bash
uv run pytest tests/integration_gpu/ -v -m gpu
```

## Architecture

### Streaming Flow

```
Client                    Python Proxy (Modal)           Rust moshi-server
  │                              │                              │
  │  ──Raw PCM float32───────►   │                              │
  │                              │  ──msgpack {Audio,pcm}────►  │
  │                              │                              │  mimi.encode()
  │                              │                              │  lm_gen.step()
  │                              │  ◄──msgpack {Word,text}────  │
  │  ◄────{"type":"token"}────   │                              │
  │  ◄────{"type":"token"}────   │                              │
```

### Key Components

#### Modal App (`modal_app.py`)

The deployment uses a Python proxy that forwards to an internal Rust moshi-server:
- `@app.cls()` for GPU class with lifecycle management
- `@modal.enter()` starts the Rust server on container startup
- `@modal.asgi_app()` for WebSocket endpoint (Python proxy)
- `@modal.concurrent(max_inputs=BATCH_SIZE)` for multiple sessions per container (default 8)

The Rust server (moshi-server) handles batched inference, enabling multiple concurrent sessions per container.

#### Moshi Streaming

Unlike batch-based transformers, moshi processes audio frame-by-frame:

```
Audio (24kHz) → Mimi Encoder (80ms frames) → LM (streaming) → Tokens
```

This gives ~0.5s first-token latency vs ~5s for batch processing.

#### Audio Format

The server expects raw PCM float32 (little-endian) audio:
- 24kHz mono
- Send ~80ms chunks for low latency
- The server converts byte buffers directly with `numpy.frombuffer`

### Container Lifecycle

```
Request arrives
     │
     ▼
┌─────────────────┐
│ Cold Start      │  ~60-90s (Rust server + model loading)
│ @modal.enter()  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Warm Container  │  Handles up to BATCH_SIZE concurrent
│ (Rust server    │  WebSocket sessions (default 8)
│ with batching)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scale Down      │  120s no connections → container stops
└─────────────────┘
```

### Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `BATCH_SIZE` | 8 | Max concurrent sessions per container |
| `RUST_SERVER_PORT` | 8998 | Internal port for Rust moshi-server |
| `scaledown_window` | 120s | Container idle before shutdown |

### Performance Tuning

The Rust moshi-server is compiled with CUDA support. The image compilation step must run on the same GPU type as the runtime to avoid CUDA symbol errors.

**GPU Compatibility:**

| GPU | Arch | Recommended |
|-----|------|-------------|
| L4 | Ada | ✅ Good balance of cost/performance |
| A10G | Ampere | ✅ |
| L40S | Ada | ✅ Default - high throughput |
| A100 | Ampere | ✅ Best performance |
| H100 | Hopper | ✅ Best performance |

**Note**: T4 GPUs are not recommended - the Rust server compilation requires more recent CUDA features.

## Scripts

### `transcribe_cli.py`

Real-time microphone transcription with rich terminal UI.

```bash
./scripts/transcribe_cli.py --help
./scripts/transcribe_cli.py --device 2  # Specific microphone
./scripts/transcribe_cli.py --debug     # Show raw messages
```

### `latency_test.py`

Benchmark latency with various options:

```bash
# Sequential tests
uv run scripts/latency_test.py -n 5

# Parallel streams (tests container scaling)
uv run scripts/latency_test.py -p 4

# Compare GPUs (deploys each to separate app)
uv run scripts/latency_test.py --compare-gpus "T4,L4,A10G" -p 4
```

### `prepare_samples.py`

Generate test audio samples:

```bash
uv run scripts/prepare_samples.py
```

## Deployment

### Production Deployment

```bash
uvx modal deploy src/stt/modal_app.py
```

### Custom Configuration

```bash
KYUTAI_GPU=A10G \
KYUTAI_APP_NAME=my-stt-service \
uvx modal deploy src/stt/modal_app.py
```

### Multiple GPU Variants

Deploy different GPU configurations to separate apps:

```bash
KYUTAI_GPU=T4 KYUTAI_APP_NAME=kyutai-stt-t4 uvx modal deploy src/stt/modal_app.py
KYUTAI_GPU=A10G KYUTAI_APP_NAME=kyutai-stt-a10g uvx modal deploy src/stt/modal_app.py
```

### Monitoring

```bash
# List running containers
uvx modal container list

# View logs
uvx modal app logs kyutai-stt

# Stop app (all containers)
uvx modal app stop kyutai-stt
```

## Troubleshooting

### CUDA Symbol Errors

If you see `CUDA_ERROR_NOT_FOUND "named symbol not found"`:
- The Rust server was compiled on a different GPU architecture than runtime
- Ensure `gpu=KYUTAI_GPU` is set in the image compilation step
- Redeploy with the correct GPU type

### Container Not Scaling Down

If containers stay alive longer than expected:

1. Check for zombie WebSocket connections
2. Check Modal dashboard for connection status
3. Containers scale down after 120s of no connections

### Audio Payload Issues

The server expects raw PCM float32 (little-endian) audio at 24kHz mono. If transcripts are empty:
- Confirm you're sending float32 samples (not int16) at 24kHz
- Send in ~80ms chunks (1920 samples per frame)

## Contributing

1. Run unit tests before submitting: `uv run pytest tests/unit/ -v`
2. Run integration tests if you changed server logic
3. Update documentation for user-facing changes
4. Follow existing code style
