# Proxmox LXC Deployment Log

## Goal

Deploy the Kyutai streaming STT service locally in a Proxmox LXC container (ID 101) with GPU passthrough to the RTX 5060 (8GB VRAM), replacing the current ComfyUI container (ID 100). The service should expose the same WebSocket API as the Modal deployment (`/v1/stream`, `/health`, `/`).

Architecture:
```
Internet/Tailscale ──:8000──▶ Host iptables ──▶ Container 101 (:8000)
                                                  Python proxy (FastAPI/uvicorn)
                                                       │ msgpack
                                                  Rust moshi-server (:8998)
                                                       │ CUDA
                                                  RTX 5060 GPU
```

## What was done

### Files committed to the repo (`deploy/proxmox/`)

| File | Purpose |
|------|---------|
| `local_server.py` | FastAPI proxy server adapted from `src/stt/modal_app.py`, with `lifespan` context manager replacing Modal's `@enter`/`@exit`, identical WebSocket proxy logic, `BATCH_SIZE=4` default |
| `kyutai-stt.service` | systemd unit file for the service |
| `smoke-test.py` | WebSocket test script (checks `/health`, sends 3s of silence, prints tokens) |

Installation-specific scripts were written to `/root/deploy-stt/` on the host (not in the repo):
- `setup-container.sh` — creates LXC 101
- `provision.sh` — installs dependencies inside the container
- `host-iptables.sh` — port forwarding rules

### Container creation (done)

- Stopped container 100, disabled its autostart
- Created container 101: Debian 13, 10 cores, 12GB RAM, 4GB swap, 50GB disk
- Static IP 192.168.1.101/24, gateway 192.168.1.254
- GPU passthrough configured (cgroup 195/508, bind mounts for nvidia device nodes)
- Container started and verified running

### NVIDIA userspace libraries (done)

- Used `/usr/local/share/downloads/NVIDIA-Linux-x86_64-580.126.09.run` (the host's driver installer)
- Ran with `--no-kernel-module --silent` inside the container
- Verified: `nvidia-smi` shows RTX 5060, 8151MiB VRAM, driver 580.126.09, CUDA 13.0

### System packages (done)

- Installed: build-essential, pkg-config, libssl-dev, git, cmake, libopus-dev, python3-dev, wget, curl, ca-certificates, gnupg

### CUDA toolkit 12.8 (done, with workarounds)

**Problem 1:** NVIDIA's apt repo for Debian 12 uses a SHA1-signed key. Debian 13's `sqv` rejects SHA1 signatures as of 2026-02-01. The standard `cuda-keyring` approach fails.

**Solution:** Used `apt-get -o Acquire::AllowInsecureRepositories=true update` and `apt-get --allow-unauthenticated install -y cuda-toolkit-12-8`. This is acceptable for a local deployment.

**Problem 2:** The CUDA toolkit `.run` file alternative also needed workarounds — `pct exec` doesn't allocate a TTY (needed by the Makeself installer), and `/tmp` is a 7.4GB tmpfs that's too small for extraction. Used `script -qc` for a pseudo-tty and `--tmpdir=/opt/tmp` for extraction space.

**Problem 3:** glibc 2.41 (Debian 13) declares `cospi`/`sinpi`/`cospif`/`sinpif` with `noexcept(true)`, but CUDA 12.8's `math_functions.h` declares them without. This causes compile errors.

**Solution:** Patched `/usr/local/cuda/include/crt/math_functions.h` to add `noexcept(true)` to the four function declarations (lines 2556, 2579, 2601, 2623), plus `sincospi`/`sincospif` which got caught by the same sed pattern.

### Rust toolchain (done)

- Installed via rustup: Rust 1.93.1 stable

### uv (done)

- The `curl`-based installer failed inside the container (pipe write error)
- Downloaded on the host, pushed binary via `pct push` to `/usr/local/bin/uv`
- Verified: `uv 0.10.4`

### moshi-server build (in progress)

Two builds running in parallel with different compute capabilities:
- `CUDA_COMPUTE_CAP=100` (Blackwell base) — task `b9d10fa`
- `CUDA_COMPUTE_CAP=89` (Ada Lovelace, forward-compatible) — task `bedc2c5`

The previous build with `CUDA_COMPUTE_CAP=120` failed because candle-kernels' MOE WMMA cuda code doesn't support sm_120 yet. Trying 100 first (native Blackwell without the newest features), falling back to 89 (Ada, which runs on Blackwell via forward compatibility).

### Python venv (not yet done)

Will be set up at `/opt/kyutai-stt/.venv` with: fastapi, uvicorn, websockets, msgpack, numpy, huggingface-hub.

## What remains to be done

1. **Wait for moshi-server build** — whichever compute cap succeeds first, use that binary
2. **Set up Python venv** — `uv venv && uv pip install ...` at `/opt/kyutai-stt/`
3. **Copy local_server.py** — already pushed to `/tmp/local_server.py`, needs to go to `/opt/kyutai-stt/`
4. **Install systemd service** — already pushed to `/tmp/kyutai-stt.service`, needs `systemctl daemon-reload && enable`
5. **Start the service** — `systemctl start kyutai-stt` (first start downloads model weights from HuggingFace)
6. **Set up host iptables** — run `host-iptables.sh` to forward port 8000 on tailscale0 to 192.168.1.101:8000
7. **Smoke test** — run `smoke-test.py`
8. **Integration tests** — `WS_URL=ws://192.168.1.101:8000/v1/stream uv run pytest tests/integration/ -v`
