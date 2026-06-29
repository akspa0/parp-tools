# RunPod Integration Cookbook

**Everything we learned making cloud GPU training work without losing our minds.**

This is a *generalized* document. Replace `your_project` with your project name. The commands, structures, and pitfalls apply to any model-training project on RunPod.

---

## Architecture

Two scripts, one workflow:

```
packager  →  produces a .tar bundle (derived data only, no game/client/raw assets)
setup     →  creates network volume + Pod, transfers bundle, bootstraps training
```

Both are Python scripts, run from your workstation. The Pod runs Linux shell scripts (`set -euo pipefail`).

**Do not combine these.** The packager runs offline and validates your bundle independently. The setup helper never builds the bundle inline unless you say so. Separation of concerns means you can fix packaging without touching Pod creation, and vice versa.

---

## The Packager

### What goes in the bundle

ONLY derived training artifacts:

- Python source code and `pyproject.toml`
- Training scripts, test files
- Precomputed Zarr/NPZ/Parquet datasets
- `requirements-runpod.txt` (not your full dev `requirements.txt` — no Windows-only deps)
- Pod-side helper scripts in `runpod/` subdirectory
- `manifest.json` — includes `contains_game_client_files: false` as a safety claim
- `README_RunPod.md` — the Pod operator docs

NEVER in the bundle:

- Game client files, MPQ/CASC archives, raw asset trees
- `RUNPOD_API_KEY` or any credentials
- `.venv/` or `__pycache__/`
- Model checkpoints (those live on the network volume, not in the bundle)

### Bundle structure

```
your_bundle_name/
  data-harvester/
    src/...
    scripts/...
    pyproject.toml
    uv.lock
  data/
    teacher-prior/
      <build>.zarr/...
    v18/
      <build>.zarr/...
    curation-manifest/
      kept_tiles.parquet
  runpod/
    install_deps.sh
    verify_bundle.sh
    verify_bundle.py
    smoke.sh
    train.sh
  requirements-runpod.txt
  README_RunPod.md
  manifest.json
```

### Validation checklist

Before you ship a bundle:

1. `manifest.json` has `contains_game_client_files: false`
2. No paths under the bundle resolve to game-client roots
3. All `runpod/*.sh` scripts are present and executable
4. `requirements-runpod.txt` installs cleanly on Linux
5. The archive fits within your target Pod's disk budget

### Archive format

Use `.tar` (not `.tar.gz`, not `.zip`). RunPod's `runpodctl receive` works with raw `.tar`. Compression buys you little when the bottleneck is network latency and the bundle is mostly Zarr arrays that are already dense.

---

## The Setup Helper

### Argument design patterns

Every flag should have a `--no-` counterpart:

| Flag | Opposite | Why |
|------|----------|-----|
| `--use-network-volume` | `--no-network-volume` | Default to network volumes — they persist across Pod restarts |
| `--auto-transfer` | `--no-auto-transfer` | Default on, but `runpodctl` might not be on PATH |
| `--auto-start-training` | `--no-auto-start-training` | Default on, but sometimes you want to verify first |

Other critical flags:

- `--dry-run` — builds the bundle and payload JSON but never calls the RunPod API
- `--gpu-types` — explicit ordered GPU list (overrides cost-target mode)
- `--cloud-type` — `COMMUNITY` (default) or `SECURE`
- `--data-centers` — ordered datacenter fallback list
- `--no-cost-target` — disable cost-based filtering, use exact `--gpu-type`
- `--transfer-code` — custom `runpodctl send/receive` relay code
- `--network-volume-id` — attach an existing volume instead of creating one

### Environment variables

- `RUNPOD_API_KEY` — required for all REST API calls. NEVER stored in the bundle manifest.
- `RUNPOD_API_KEY` has Pod/volume management permissions but does NOT give you S3 file upload access to network volumes (separate credentials).

---

## Creating Pods via REST API

### The minimal known-working payload

```json
{
  "name": "your-project-pod",
  "cloudType": "COMMUNITY",
  "gpuTypeIds": ["NVIDIA RTX 4000 Ada Generation"],
  "gpuCount": 1,
  "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "containerDiskInGb": 50,
  "volumeInGb": 150,
  "volumeMountPath": "/workspace",
  "ports": ["22/tcp", "8888/http"],
  "supportPublicIp": true
}
```

With a network volume, add:

```json
{
  "networkVolumeId": "<volume-id>",
  "dataCenterId": "US-KS-2"
}
```

With a bootstrap command:

```json
{
  "dockerStartCmd": ["bash", "-lc", "set -euo pipefail; cd /workspace; ..."]
}
```

### Candidate iteration pattern

Always iterate over (GPU, datacenter) pairs instead of trying once:

```
for (gpu_type, data_center) in candidates:
    try:
        create_network_volume(datacenter=data_center)
        create_pod(gpu_type=gpu_type, data_center=data_center, network_volume_id=...)
        break
    except RetryableError:
        delete_orphaned_network_volume()
        continue
```

This handles the "no instances currently available" error that RunPod returns even when the web UI shows availability.

### Known API quirks

- `POST /pods` can return "no instances currently available" for GPU/datacenter combos that the web UI shows as available. This is the most common failure mode.
- The web UI (my.runpod.io) is *more reliable* than the REST API. When the API fails, create the Pod manually and pass `--network-volume-id --data-center` to attach.
- `runpodctl gpu list` does NOT expose pricing. The fields are `memoryInGb`, `available`, `communityCloud`, `secureCloud`, `stockStatus` — no price fields. Use hardcoded approximate prices for filtering.
- `runpodctl datacenter list` returns datacenters that may NOT support network volumes. You must filter against a known-good set.

---

## Network Volumes

### Critical constraint

**Network volume + Pod MUST share the same concrete datacenter.** You cannot create a volume in US-KS-2 and a Pod in EU-FR-1. The `POST /networkvolumes` API does not accept `"auto"` as a datacenter.

### Known network-volume-capable datacenters

This set was validated against actual API errors. If you get "not found or does not support network volumes", add the datacenter ID to this list:

```
AP-IN-2, AP-JP-1, CA-MTL-3, CA-MTL-4,
EU-CZ-1, EU-FR-1, EU-NL-1, EU-RO-1, EU-SE-1,
EUR-IS-1, EUR-IS-3, EUR-NO-1, EUR-NO-2,
US-CA-2, US-GA-2, US-IL-1, US-KS-2,
US-MO-2, US-NC-1, US-NC-2, US-NE-1, US-TX-3, US-WA-1
```

### Cleanup on failure

When a Pod creation attempt fails for a candidate, the auto-created network volume (for that attempt only) must be deleted. RunPod does not auto-clean orphaned volumes, and they count against your volume quota.

Rule: if you created the volume and the Pod attempt failed, delete the volume. If the volume was pre-existing (`--network-volume-id`), leave it alone.

### Volume vs Pod-local

| Feature | Network Volume | Pod-local (`volumeInGb`) |
|---------|---------------|--------------------------|
| Persists after Pod stop | Yes | No |
| Shareable across Pods | Yes | No |
| Requires concrete datacenter | Yes | No |
| Slower I/O | Slightly | Native |
| Good for | Long training, checkpoint persistence | Throwaway runs |

Default: network volume. Only use Pod-local for smoke tests or short runs.

---

## Data Transfer

### Preferred path: `runpodctl send/receive`

1. Pod-side bootstrap starts `runpodctl receive <code>` in `/workspace`
2. Local script starts `runpodctl send <archive.tar> --code <code>`
3. Pod receives the archive, extracts it, runs `install_deps.sh`

Warning: `runpodctl send` uses a relay server. If the local `send` doesn't start (e.g., `runpodctl` not on PATH), the Pod-side `receive` gets a "Malformed relay" error and exits. The Pod container may restart but the relay won't reconnect. This is the #1 bootstrap failure mode — handle it by printing manual instructions instead of crashing.

### Fallback: SCP

```bash
scp -P <port> <archive.tar> root@<pod-ip>:/workspace/
```

SSH key setup is required first (see below).

### Fallback: rsync

```bash
rsync -avzP <archive.tar> root@<pod-ip>:/workspace/
```

Better for incremental updates. Needs SSH key setup.

### What NOT to use

- RunPod S3 API for network volume uploads — requires separate S3 credentials that are NOT the `RUNPOD_API_KEY`. Avoid unless you have a specific reason.
- Direct HTTP download — fragile, requires a publicly accessible URL or auth headers in the Pod env.

---

## SSH Setup

### The one weird trick

**SSH keys must be registered via the RunPod web console.** There is no API or CLI command to do this.

1. `ssh-keygen -t ed25519` on your workstation
2. Go to my.runpod.io → Settings → SSH Keys
3. Paste your public key
4. Wait a minute for propagation

`runpodctl ssh info <pod-id>` shows the SSH info (IP, port) but does NOT set up keys.

### After Pod creation

Wait for the Pod to get a public IP:

```python
# Poll /pods/<id> until publicIp and portMappings are non-empty
```

Then print:

```
ssh root@<publicIp> -p <sshPort>
scp -P <sshPort> <archive> root@<publicIp>:/workspace/
```

---

## Pod-Side Bootstrap Scripts

### Template

Every shell script must start with `set -euo pipefail`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ... do work ...
```

### Script chain

The bootstrap command runs sequentially:

```
install_deps.sh → verify_bundle.sh → [smoke.sh → train.sh]
```

Each script should be independently runnable. `install_deps.sh` must be idempotent.

### install_deps.sh pattern

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Check if torch is installed with CUDA; upgrade only if too old
python - <<'PY'
import subprocess, sys
try:
    import torch
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split("."))
except Exception:
    version = (0, 0)
if version < (2, 5):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade",
        "--index-url", "https://download.pytorch.org/whl/cu124", "torch>=2.5,<3"])
PY

# Install project deps
python -m pip install -r requirements-runpod.txt

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
```

Key insight: Use `requirements-runpod.txt` instead of `uv sync` because `pyproject.toml` may include Windows-only dependencies that break on Linux.

### Bootstrap command sentinel

Log everything to `/workspace/bootstrap.log`:

```bash
exec > >(tee -a /workspace/bootstrap.log) 2>&1
date -u
```

---

## GPU Selection and Cost Targeting

### Default behavior

- Default cloud type: **COMMUNITY** — higher availability, lower cost than SECURE
- Default GPU: **NVIDIA RTX 4000 Ada Generation** — 20GB VRAM, ~$0.35/hr, widely available
- Datacenter GPUs (A100, H100, H200, B200, L4, L40, A40, Tesla, RTX PRO, AMD): **excluded by default**
- Fallback GPUs are opt-in only (`--gpu-fallback`)

### Cost-target mode

The helper automatically filters GPUs by:

1. Exclude datacenter/pro cards
2. Minimum VRAM (default 12GB)
3. Maximum cost per hour (default $1.00)
4. Must be available in COMMUNITY cloud
5. Sort by price (cheapest first)

Price data is hardcoded because `runpodctl` does not expose pricing. Keep the table updated:

| GPU | VRAM | Approx $/hr |
|-----|------|------------|
| RTX 2000 Ada | 16GB | $0.12 |
| RTX 4000 Ada | 20GB | $0.35 |
| RTX 4000 SFF Ada | 20GB | $0.30 |
| RTX A4500 | 20GB | $0.31 |
| RTX A4000 | 16GB | $0.27 |
| RTX A5000 | 24GB | $0.38 |
| RTX A6000 | 48GB | $0.55 |
| RTX 6000 Ada | 48GB | $0.65 |
| RTX 3060 | 12GB | $0.15 |
| RTX 3090 | 24GB | $0.34 |
| RTX 4090 | 24GB | $0.40 |
| RTX 5090 | 32GB | $0.50 |

### runpodctl quirks

- `--include-unavailable` flag may not exist in `runpodctl` v2.x. Call `["gpu", "list"]` without it and handle no-results gracefully.
- `runpodctl` v2.x uses positional args, not flags: `runpodctl ssh info <pod-id>`, not `runpodctl ssh --pod <pod-id>`.
- `runpodctl` on Windows: download `runpodctl-windows-amd64.exe` and place it somewhere on PATH (e.g., `.venv/Scripts/runpodctl.exe` or `C:\Windows\System32\`).
- `shutil.which("runpodctl")` works on both Windows and Linux.

---

## Error Handling

### Classify API errors

```python
def is_retryable(ex):
    """Retry on: no instances, currently available, server errors >= 500."""
    if not isinstance(ex, RunPodApiError):
        return False
    if ex.status in (400, 401, 403, 404):
        return False  # fatal — bad request, unauthorized, forbidden, not found
    text = ex.detail.lower()
    return (
        "no instances" in text
        or "currently available" in text
        or "not found" in text
        or "does not support" in text
        or ex.status >= 500
    )
```

### Fatal errors (do not retry)

- 400 Bad Request — malformed payload
- 401 Unauthorized — bad API key
- 403 Forbidden — API key lacks permissions
- 404 Not Found — resource doesn't exist (unless it's GPU availability)

### Graceful degradation

When the REST API fails for all candidates:

```python
print("Manual Pod creation URL: https://www.runpod.io/console/pods")
print("Recommended settings:")
print(f"  GPU: {gpu_type}")
print(f"  Image: {image_name}")
print(f"  Volume: {volume_gb}GB at /workspace")
```

---

## Monitoring Training

After the Pod starts training:

- SSH in periodically and check `bootstrap.log` in `/workspace/`
- Training outputs go to a run directory under `models/<project>/<run-name>/`
- Key files to monitor:
  - `*_metrics.json` — loss curves
  - `*_preview.png` — visual previews
  - `*_validation_previews/` — per-epoch validation
  - `*_best.pt`, `*_latest.pt` — checkpoints

### Downloading results

```bash
scp -P <port> root@<pod-ip>:/workspace/<bundle>/models/<project>/<run>/*_{metrics.json,best.pt,latest.pt,preview.png} ./
```

Or for the whole run directory:

```bash
rsync -avzP -e "ssh -p <port>" root@<pod-ip>:/workspace/<bundle>/models/<project>/<run>/ ./local_run_dir/
```

---

## Cleanup

```bash
# Stop Pod (keeps it for later)
runpodctl pod stop <pod-id>

# Terminate Pod (destroys it, but network volume persists)
runpodctl pod terminate <pod-id>

# Delete network volume (do this AFTER terminating all Pods using it)
curl -X DELETE https://rest.runpod.io/v1/networkvolumes/<volume-id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

**Always check for orphaned network volumes.** RunPod does not auto-clean them when a Pod is terminated. Orphaned volumes count against quotas and cost money.

---

## Serverless Migration Path

### When to migrate

When the training command is stable and you want repeatable, triggerable runs without manual Pod management.

### How

1. Keep the bundle on a network volume
2. Write a Python handler that reads from `/runpod-volume/<bundle>/` and runs the training script
3. Create a Serverless endpoint with the handler image
4. Attach the network volume to the endpoint
5. Trigger the endpoint

### Flash (`@Endpoint`)

Flash runs local `@Endpoint` functions on Serverless workers and mounts network volumes at `/runpod-volume/`. This is simpler than a full Docker-based Serverless worker but less flexible.

### Advice

Serverless is better for API-style inference and repeatable training jobs. For the first week of interactive model development and preview inspection, a Pod is simpler. Migrate to Serverless only after the training command is frozen.

---

## Checklist for a New Project

1. **Install runpodctl**: Download binary, put on PATH. Verify with `runpodctl version`.
2. **Set API key**: `export RUNPOD_API_KEY=...` (or PowerShell `$env:RUNPOD_API_KEY = "..."`)
3. **Set up SSH keys**: Generate `ssh-keygen -t ed25519`, paste `.pub` at my.runpod.io → Settings → SSH Keys.
4. **Write the packager**: Copy the template from your previous project. Update source paths, exclude patterns, and derived-data Zarr locations.
5. **Write the setup helper**: Copy the template. Update Pod name, GPU defaults, image name, and training script paths.
6. **Write pod-side helpers**: `install_deps.sh`, `verify_bundle.sh`, `smoke.sh`, `train.sh`.
7. **Test with --dry-run**: Verify payloads are correct without spending money.
8. **Do a smoke run**: Create a Pod with `--no-auto-start-training`, SSH in, run the smoke script manually.
9. **Full training**: If smoke passes, start the full training run.
10. **Monitor**: Check metrics and previews periodically.
11. **Download results**: Get checkpoints, metrics, and previews before terminating the Pod.
12. **Clean up**: Delete the Pod and orphaned network volumes.

Total setup time for a project that already has this cookbook and the templates: **under 10 minutes**.

---

## Quick Start (Absolute Minimum)

Given this cookbook, existing packager + setup helper templates, and existing pod-side scripts:

```bash
# 1. Package your derived data
python packager.py --output-root ./output/cloud-packages

# 2. Create Pod, transfer, start training
export RUNPOD_API_KEY="..."
python setup.py --dry-run                   # verify first
python setup.py                             # do it for real
```

That's it. Two commands. Everything else in this document is what to do when something goes wrong.
