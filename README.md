# Kyutai STT on Modal

Real-time streaming speech-to-text using [Kyutai's STT model](https://huggingface.co/kyutai/stt-1b-en_fr) deployed on Modal with ~0.5s first-token latency.

## Quick Start

```bash
# Deploy
uvx modal deploy src/stt/modal_app.py

# Test with microphone
./scripts/transcribe_cli.py

# Run latency test
uv run scripts/latency_test.py -p 4
```

## GPU Selection & Pricing

Modal bills per-second. Choose based on your latency/cost requirements:

| GPU | Hourly Cost | Notes |
|-----|-------------|-------|
| **T4** | $0.59/hr | Budget option, may have higher latency |
| **L4** | $0.80/hr | Good balance |
| **A10G** | $1.10/hr | Recommended for production |
| **L40S** | $1.95/hr | High throughput |
| **A100 40GB** | $2.10/hr | Overkill for single-stream |

Set GPU via environment variable:
```bash
KYUTAI_GPU=A10G uvx modal deploy src/stt/modal_app.py
```

### GPU Comparison Tool

Benchmark different GPUs with parallel streams:

```bash
# Compare GPUs (deploys each to separate app)
uv run scripts/latency_test.py --compare-gpus "T4,L4,A10G,A100" -p 4

# Cleanup test apps after
modal app stop kyutai-stt-t4
modal app stop kyutai-stt-l4
modal app stop kyutai-stt-a10g
modal app stop kyutai-stt-a100
```

## Architecture

- **Model**: `kyutai/stt-1b-en_fr` (1B parameters, moshi streaming)
- **Audio**: 24kHz, Opus-encoded over WebSocket
- **Protocol**: Client sends Opus bytes, server streams `{"type": "token", "text": "..."}` JSON

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `KYUTAI_GPU` | `A100` | GPU type |
| `KYUTAI_APP_NAME` | `kyutai-stt` | Modal app name |
| `MAX_CONCURRENT_SESSIONS` | `4` | Sessions per container |
| `IDLE_AUDIO_TIMEOUT_SECONDS` | `30` | Close after silence |
| `MAX_SESSION_SECONDS` | `3600` | Max session duration |

## Authentication

The endpoint uses Modal proxy auth. Set credentials:

```bash
export MODAL_KEY=your-key
export MODAL_SECRET=your-secret
```

Create tokens in Modal workspace settings.

## Files

- `src/stt/modal_app.py` - Modal deployment
- `scripts/transcribe_cli.py` - Real-time microphone transcription
- `scripts/latency_test.py` - Latency benchmarking tool
