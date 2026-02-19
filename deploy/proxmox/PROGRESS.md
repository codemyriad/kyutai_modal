# Local Kyutai STT Deployment — Progress Log

## Goal

Deploy the Kyutai streaming STT service locally in a Proxmox LXC container (CT 101) with GPU passthrough to the RTX 5060 (8GB VRAM), replacing the current ComfyUI container (CT 100). The service should expose the same WebSocket API as the Modal deployment (`/v1/stream`, `/health`, `/`).

## What was done

### Files committed to the repo (`deploy/proxmox/`)

| File | Status | Purpose |
|------|--------|---------|
| `local_server.py` | Done | FastAPI proxy adapted from `src/stt/modal_app.py`. Uses lifespan context manager instead of Modal decorators. Same WebSocket proxy logic, same `generate_config()`. BATCH_SIZE=4 default for 8GB VRAM. |
| `kyutai-stt.service` | Done | systemd unit file for the service |
| `smoke-test.py` | Done | WebSocket test script with `uv run` shebble |

Installation-specific scripts were written to `/root/deploy-stt/` on the host (not in the repo, per your feedback that they're ephemeral to this installation).

### Container setup (complete)

1. Stopped container 100, disabled its onboot
2. Created container 101: Debian 13, 10 cores, 12GB RAM, 4GB swap, 50GB disk, static IP 192.168.1.101
3. Added GPU passthrough (cgroup 195 + 508, bind mounts for nvidia device nodes — matching CT 100's config)
4. Container boots and `nvidia-smi` works: RTX 5060, 8GB VRAM, driver 580.126.09

### Inside the container (complete)

1. **System packages**: build-essential, pkg-config, libssl-dev, git, cmake, libopus-dev, python3-dev, curl, etc.
2. **NVIDIA userspace libs**: Installed via the `/usr/local/share/downloads/NVIDIA-Linux-x86_64-580.126.09.run` installer with `--no-kernel-module --silent`
3. **CUDA toolkit 12.8**: Installed via apt with `--allow-unauthenticated` (NVIDIA's repo key uses SHA1, which Debian 13's `sqv` rejects since Feb 2026)
4. **Patched CUDA headers**: See "Obstacles encountered" below — required 12 declarations patched
5. **Rust toolchain**: Installed via rustup (1.93.1)
6. **uv**: Downloaded on host, pushed binary to `/usr/local/bin/uv` (the `curl | sh` installer failed inside the container)
7. **Python venv**: Created at `/opt/kyutai-stt/.venv` with fastapi, uvicorn, websockets, msgpack, numpy, huggingface-hub
8. **systemd service**: Installed and enabled at `/etc/systemd/system/kyutai-stt.service`
9. **Application files**: `local_server.py` copied to `/opt/kyutai-stt/`

### moshi-server build (complete)

Successfully built with `CUDA_COMPUTE_CAP=100` (Blackwell base). Binary at `/root/.cargo/bin/moshi-server` (moshi-server v0.6.4).

Build history:
- `CUDA_COMPUTE_CAP=120` (sm_120): Failed — candle-kernels MOE WMMA code doesn't support sm_120
- `CUDA_COMPUTE_CAP=89` (Ada): Failed — hit the `sincospi`/`sincospif` header issue (built before second round of patches)
- `CUDA_COMPUTE_CAP=100` (Blackwell base): **Succeeded** after patching both the `extern` declarations AND the `__func__` macro redeclarations

### Service started (running, model loading)

The service was started with `systemctl start kyutai-stt`. Both processes are running:
- Python proxy (FastAPI/uvicorn) on port 8000
- Rust moshi-server on port 8998

First start downloads model weights (~2GB) from Hugging Face. Check progress:
```bash
pct exec 101 -- journalctl -u kyutai-stt -f
```

## What remains to be done

### Step-by-step to finish

1. **Wait for model to load** (first start only — downloads ~2GB, takes 2-5 min):
   ```bash
   pct exec 101 -- journalctl -u kyutai-stt -f
   ```
   Look for "Rust server ready after Xs"

2. **Verify health endpoint**:
   ```bash
   curl http://192.168.1.101:8000/health
   ```

3. **Set up iptables port forwarding** (on the host):
   ```bash
   bash /root/deploy-stt/host-iptables.sh
   ```

4. **Run smoke test**:
   ```bash
   uv run deploy/proxmox/smoke-test.py
   ```

5. **Run full integration tests**:
   ```bash
   WS_URL=ws://192.168.1.101:8000/v1/stream uv run pytest tests/integration/ -v
   ```

### If the service needs restarting or rebuilding

```bash
# Restart
pct exec 101 -- systemctl restart kyutai-stt

# Rebuild moshi-server (if needed after CUDA updates, etc.)
pct exec 101 -- bash -c '
export PATH="/root/.cargo/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
export CUDA_COMPUTE_CAP=100
cargo install --features cuda moshi-server
'

# Re-setup Python venv
pct exec 101 -- bash -c 'cd /opt/kyutai-stt && /usr/local/bin/uv venv --clear && /usr/local/bin/uv pip install fastapi uvicorn websockets msgpack "numpy<2" "huggingface-hub[hf_transfer]"'
```

## Obstacles encountered

| Problem | Root cause | Solution |
|---------|-----------|----------|
| CUDA apt repo rejected | Debian 13's `sqv` rejects SHA1 keys (deprecated Feb 2026) | `apt-get -o Acquire::AllowInsecureRepositories=true update` + `--allow-unauthenticated` |
| CUDA .run installer "cannot create /dev/tty" | Makeself archive requires a terminal; `pct exec` has no controlling tty | Used `script -qc '...' /dev/null` for pseudo-tty |
| CUDA .run extraction "no space left" | `/tmp` is 7.4GB tmpfs; runfile + extraction exceeded it | `--tmpdir=/opt/tmp` to use real filesystem |
| uv installer fails inside container | `curl` pipe failure | Downloaded on host, pushed binary via `pct push` |
| moshi-server: cospi/sinpi exception spec | glibc 2.41 declares these with `noexcept(true)`; CUDA 12.8 headers don't | Patched `math_functions.h` (see below) |
| moshi-server: sincospi/sincospif `__func__` conflict | The `__func__` macro redeclarations (lines ~5990-6014) also lack `noexcept` | Patched those too |
| moshi-server: sm_120 not supported | candle-kernels MOE WMMA code doesn't support Blackwell sm_120 | Used `CUDA_COMPUTE_CAP=100` instead |

### Full CUDA header patch details

File: `/usr/local/cuda/include/crt/math_functions.h`

12 declarations needed `noexcept(true)` added:

**Section 1: `extern` declarations (~line 2556-2649)**
- Line 2556: `sinpi(double x)`
- Line 2579: `sinpif(float x)`
- Line 2601: `cospi(double x)`
- Line 2623: `cospif(float x)`
- Line 2636: `sincospi(double x, double *sptr, double *cptr)`
- Line 2649: `sincospif(float x, float *sptr, float *cptr)`

**Section 2: `__func__` macro redeclarations (~line 5990-6014)**
- Line 5990: `__func__(double sinpi(double a))`
- Line 5992: `__func__(double cospi(double a))`
- Line 5994: `__func__(void sincospi(double a, double *sptr, double *cptr))`
- Line 6010: `__func__(float sinpif(float a))`
- Line 6012: `__func__(float cospif(float a))`
- Line 6014: `__func__(void sincospif(float a, float *sptr, float *cptr))`

All had `noexcept(true)` appended before the semicolon.

## Update (2026-02-19)

- Applied host iptables forwarding and confirmed `/health` OK on `http://192.168.1.101:8000/health`.
- Smoke test passed.
- Integration tests now pass after adding `IDLE_AUDIO_TIMEOUT_SECONDS` (default `10`) to the local server idle timeout.
- Updated `scripts/transcribe_cli.py` with `--service local` to target the Proxmox service.
