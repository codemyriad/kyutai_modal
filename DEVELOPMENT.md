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
Client                          Server (Modal)
  │                                  │
  │  ──Opus audio bytes──────────►   │
  │                                  │  sphn.read_opus_bytes()
  │                                  │  ↓
  │                                  │  mimi.encode() (80ms frames)
  │                                  │  ↓
  │                                  │  lm_gen.step() (streaming LM)
  │                                  │  ↓
  │  ◄────{"type":"token"}────────   │
  │  ◄────{"type":"token"}────────   │
  │  ◄────{"type":"vad_end"}──────   │
```

### Key Components

#### Modal App (`modal_app.py`)

The main deployment uses:
- `@app.cls()` for GPU class with lifecycle management
- `@modal.enter()` for model loading on container startup
- `@modal.asgi_app()` for WebSocket endpoint
- `@modal.concurrent()` for handling multiple sessions per container

#### Moshi Streaming

Unlike batch-based transformers, moshi processes audio frame-by-frame:

```python
# Enable streaming mode
self.mimi.streaming_forever(batch_size=1)
self.lm_gen.streaming_forever(batch_size=1)

# Process each 80ms frame
codes = self.mimi.encode(audio_frame)  # Neural codec
tokens = self.lm_gen.step(codes)        # LM inference
```

This gives ~0.5s first-token latency vs ~5s for batch processing.

#### Opus Audio

The server expects Opus-encoded audio (not raw PCM):
- Client encodes with `sphn.OpusStreamWriter`
- Server decodes with `sphn.read_opus_bytes()`
- 24kHz mono audio

### Container Lifecycle

```
Request arrives
     │
     ▼
┌─────────────────┐
│ Cold Start      │  ~20-30s (model loading)
│ @modal.enter()  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Warm Container  │  Handles WebSocket sessions
│ Multiple        │  Up to MAX_CONCURRENT_SESSIONS
│ concurrent      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Idle Timeout    │  10s no audio → close connection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scale Down      │  60s no connections → container stops
└─────────────────┘
```

### Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `IDLE_AUDIO_TIMEOUT_SECONDS` | 10s | Close WebSocket after silence |
| `PING_INTERVAL_SECONDS` | 10s | Detect dead connections |
| `MAX_SESSION_SECONDS` | 3600s | Maximum session duration |
| `scaledown_window` | 60s | Container idle before shutdown |

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

# Parallel streams
uv run scripts/latency_test.py -p 4

# Compare GPUs (deploys each to separate app)
uv run scripts/latency_test.py --compare-gpus "T4,L4,A10G" -p 4

# Skip warmup
uv run scripts/latency_test.py -p 4 --no-warmup
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
MAX_CONCURRENT_SESSIONS=8 \
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

### Container Not Scaling Down

If containers stay alive longer than expected:

1. Check for zombie WebSocket connections
2. Verify `IDLE_AUDIO_TIMEOUT_SECONDS` is set correctly
3. Run the idle timeout test to verify behavior:
   ```bash
   uv run pytest tests/integration/test_idle_timeout.py -v -s
   ```

### Memory Issues

The moshi streaming state doesn't survive memory snapshots, so `enable_memory_snapshot=False` is required. Cold starts take ~20-30s as a result.

### Opus Decoding Errors

The server needs complete Ogg pages to decode. The CLI buffers ~4KB before sending to ensure decodability.

## Contributing

1. Run unit tests before submitting: `uv run pytest tests/unit/ -v`
2. Run integration tests if you changed server logic
3. Update documentation for user-facing changes
4. Follow existing code style
