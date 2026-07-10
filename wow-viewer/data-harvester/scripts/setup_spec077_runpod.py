"""Create a RunPod training Pod for the Spec 077 cloud bundle.

This script automates the parts RunPod exposes through the normal REST API:

* validate/build the derived-data-only Spec 077 training package
* create or attach a network volume by default
* create an RTX 4000 Ada Pod with enough RAM/storage for the training run
* write a local setup manifest and bootstrap the runpodctl transfer when available

The normal RunPod API key can manage Pods and volumes, but direct network-volume
file upload uses RunPod's separate S3-compatible API credentials. The default
handoff avoids extra S3 credentials by starting Pod-side `runpodctl receive` and
local `runpodctl send` with a shared code for the generated tar.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import package_spec077_runpod  # noqa: E402


RUNPOD_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_GPU_TYPE = "NVIDIA RTX 4000 Ada Generation"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_DATA_CENTER = "auto"
DEFAULT_MAX_COST_PER_HOUR = 1.00
DEFAULT_MIN_GPU_VRAM_GB = 12

# All valid RunPod datacenters that support network volumes (from RunPod API error)
VALID_NETWORK_VOLUME_DATACENTERS = frozenset({
    "AP-IN-2", "AP-JP-1", "CA-MTL-3", "CA-MTL-4",
    "EU-CZ-1", "EU-FR-1", "EU-NL-1", "EU-RO-1", "EU-SE-1",
    "EUR-IS-1", "EUR-IS-3", "EUR-NO-1", "EUR-NO-2",
    "US-CA-2", "US-GA-2", "US-IL-1", "US-KS-2",
    "US-MO-2", "US-NC-1", "US-NC-2", "US-NE-1", "US-TX-3", "US-WA-1",
})

# US-only datacenters. EU/AP/CA datacenters often have different public-IP policies,
# slower transfer speeds from US-based workstations, and may not support SCP properly.
# Only US datacenters are included in the default fallback list.
DEFAULT_FALLBACK_DATA_CENTERS = (
    "US-CA-2",
    "US-GA-2",
    "US-IL-1",
    "US-KS-2",
    "US-MO-2",
    "US-NC-1",
    "US-NC-2",
    "US-NE-1",
    "US-TX-3",
    "US-WA-1",
)

# Consumer/workstation GPUs only — no datacenter or pro cards
EXCLUDED_GPU_PREFIXES = (
    "NVIDIA A100",
    "NVIDIA H100",
    "NVIDIA H200",
    "NVIDIA B200",
    "NVIDIA B300",
    "NVIDIA L4",
    "NVIDIA L40",
    "NVIDIA A40",
    "NVIDIA Tesla",
    "NVIDIA RTX PRO",
    "AMD",
)

# Hardcoded fallback GPU info (approximate pricing) — runpodctl doesn't expose pricing
FALLBACK_GPU_INFOS = {
    "NVIDIA RTX 2000 Ada Generation": {"vram_gb": 16, "price_per_hour": 0.12},
    "NVIDIA RTX 4000 Ada Generation": {"vram_gb": 20, "price_per_hour": 0.35},
    "NVIDIA RTX 4000 SFF Ada Generation": {"vram_gb": 20, "price_per_hour": 0.30},
    "NVIDIA RTX A4500": {"vram_gb": 20, "price_per_hour": 0.31},
    "NVIDIA RTX A4000": {"vram_gb": 16, "price_per_hour": 0.27},
    "NVIDIA RTX A5000": {"vram_gb": 24, "price_per_hour": 0.38},
    "NVIDIA RTX A6000": {"vram_gb": 48, "price_per_hour": 0.55},
    "NVIDIA RTX 6000 Ada Generation": {"vram_gb": 48, "price_per_hour": 0.65},
    "NVIDIA GeForce RTX 3060": {"vram_gb": 12, "price_per_hour": 0.15},
    "NVIDIA GeForce RTX 3090": {"vram_gb": 24, "price_per_hour": 0.34},
    "NVIDIA GeForce RTX 4090": {"vram_gb": 24, "price_per_hour": 0.40},
    "NVIDIA GeForce RTX 5090": {"vram_gb": 32, "price_per_hour": 0.50},
}


@dataclass(frozen=True)
class PackageResult:
    package_name: str
    bundle_dir: Path
    archive_path: Path


class RunPodApiError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, detail: str) -> None:
        super().__init__(f"RunPod API {method} {path} failed: HTTP {status}: {detail}")
        self.method = method
        self.path = path
        self.status = int(status)
        self.detail = detail


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _default_wow_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _api_key_arg(value: str | None) -> str | None:
    if value:
        return value
    return os.environ.get("RUNPOD_API_KEY")


def _request_json(
    method: str,
    path: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    url = f"{RUNPOD_API_BASE}{path}"
    body = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as ex:
        detail = ex.read().decode("utf-8", errors="replace")
        raise RunPodApiError(method, path, ex.code, detail) from ex


def _run_packager(args: argparse.Namespace, package_name: str) -> PackageResult:
    wow_root = args.wow_root.resolve()
    output_root = (args.output_root or (wow_root / "output" / "cloud-packages")).resolve()
    package_args = [
        "--wow-root", str(wow_root),
        "--output-root", str(output_root),
        "--package-name", package_name,
        "--archive-format", "tar",
    ]
    if args.overwrite_package:
        package_args.append("--overwrite")
    if args.no_package_tests:
        package_args.append("--no-tests")
    if args.full_v18_stores:
        package_args.append("--full-v18-stores")
    if args.builds:
        package_args.append("--builds")
        package_args.extend(args.builds)
    if args.curation_manifest is not None:
        package_args.extend(["--curation-manifest", str(args.curation_manifest)])
    package_args.extend([
        "--run-name", args.run_name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--target-vram-gb", str(args.target_vram_gb),
    ])
    exit_code = package_spec077_runpod.main(package_args)
    if exit_code != 0:
        raise RuntimeError(f"package_spec077_runpod.py failed with exit code {exit_code}")
    return PackageResult(
        package_name=package_name,
        bundle_dir=output_root / package_name,
        archive_path=(output_root / package_name).with_suffix(".tar"),
    )


def _build_network_volume_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.data_center or str(args.data_center).lower() == "auto":
        raise ValueError("RunPod network volumes require a concrete datacenter id, not 'auto'.")
    return {
        "name": args.network_volume_name,
        "size": int(args.network_volume_gb),
        "dataCenterId": args.data_center,
    }


def _build_pod_payload(args: argparse.Namespace, *, network_volume_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": args.pod_name,
        "cloudType": args.cloud_type,
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": 1,
        "imageName": args.image_name,
        "containerDiskInGb": int(args.container_disk_gb),
        "volumeInGb": int(args.volume_gb),
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp", "8888/http"],
        "supportPublicIp": True,
    }
    if args.allowed_cuda_versions:
        payload["allowedCudaVersions"] = list(args.allowed_cuda_versions)
    start_cmd = _build_bootstrap_start_cmd(args)
    if start_cmd is not None:
        payload["dockerStartCmd"] = ["bash", "-lc", start_cmd]
    if network_volume_id:
        if not args.data_center or str(args.data_center).lower() == "auto":
            raise ValueError("RunPod Pods with network volumes require a concrete datacenter id, not 'auto'.")
        payload["networkVolumeId"] = network_volume_id
        payload["dataCenterId"] = args.data_center
    return payload


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _resolve_transfer_mode(args: argparse.Namespace) -> str:
    """Return 'url', 'scp', or 'relay' based on args and runpodctl availability."""
    # --download-url overrides everything
    download_url = getattr(args, "download_url", None)
    if download_url:
        return "url"
    mode = str(getattr(args, "transfer_mode", "auto")).lower()
    if mode == "relay":
        return "relay"
    if mode == "scp":
        return "scp"
    # Auto: prefer SCP if runpodctl is missing (since relay won't work)
    if shutil.which("runpodctl") is None:
        return "scp"
    return "relay"


def _build_bootstrap_start_cmd(args: argparse.Namespace) -> str | None:
    """Build the Docker start command for the pod.

    The bootstrap always runs extract → install_deps → verify → [smoke → train].
    The transfer-wait step (runpodctl receive, SCP poll, wget) is only included
    when ``--auto-transfer`` is set. When using a network volume (``--network-volume-id``),
    the bundle is expected to already exist on the volume at /workspace/.
    """
    package_name = str(getattr(args, "_resolved_package_name", "spec077_runpod_bundle"))
    archive_name = f"{package_name}.tar"
    transfer_code = str(getattr(args, "_resolved_transfer_code", args.transfer_code or "spec077-transfer"))
    transfer_mode = _resolve_transfer_mode(args)

    lines = [
        "set -euo pipefail",
        "cd /workspace",
        "exec > >(tee -a /workspace/bootstrap.log) 2>&1",
        "date -u",
    ]

    if args.auto_transfer:
        if transfer_mode == "url":
            download_url = str(getattr(args, "download_url", ""))
            lines.append(_build_download_url_bootstrap_cmd(archive_name, download_url))
        elif transfer_mode == "scp":
            lines.append(_build_scp_bootstrap_wait_cmd(archive_name))
        else:
            lines.extend([
                f"echo Waiting for package via runpodctl code {_shell_quote(transfer_code)}",
                f"if ! runpodctl receive {_shell_quote(transfer_code)}; then",
                "  echo",
                "  echo 'runpodctl receive failed.'",
                "  echo 'Manual transfer options:'",
                f'  echo "  scp -P <port> {archive_name} root@<pod-ip>:/workspace/"',
                f'  echo "  rsync -avzP {archive_name} root@<pod-ip>:/workspace/"',
                "  exit 1",
                "fi",
            ])
    else:
        lines.append(f"echo 'Network volume mode: bundle expected at /workspace/{archive_name}'")

    lines.extend([
        f"test -f {_shell_quote(archive_name)}",
        f"tar -xf {_shell_quote(archive_name)}",
        f"cd {_shell_quote(package_name)}",
        "bash runpod/install_deps.sh",
        "bash runpod/verify_bundle.sh",
    ])
    if args.auto_start_training:
        lines.extend([
            "bash runpod/smoke_spec077.sh",
            "bash runpod/train_spec077.sh",
        ])
    else:
        lines.append("echo Auto-start training disabled; run bash runpod/smoke_spec077.sh then bash runpod/train_spec077.sh manually.")
    lines.append("date -u")
    return "\n".join(lines)


def _send_package_with_runpodctl(package: PackageResult, transfer_code: str) -> bool:
    exe = shutil.which("runpodctl")
    if exe is None:
        print("runpodctl was not found on PATH; package transfer was not started.", file=sys.stderr)
        print(f"Install runpodctl or run manually: runpodctl send \"{package.archive_path}\" --code {transfer_code}", file=sys.stderr)
        return False
    command = [exe, "send", str(package.archive_path), "--code", transfer_code]
    print(f"Starting package transfer: {' '.join(command)}")
    subprocess.run(command, check=True)
    return True


def _pod_scp_info(pod: dict[str, Any]) -> tuple[str, int] | None:
    """Extract (public_ip, ssh_port) from a polled pod dict, or None if not ready."""
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22") or port_mappings.get(22)
    if not public_ip or not ssh_port:
        return None
    try:
        return str(public_ip), int(ssh_port)
    except (TypeError, ValueError):
        return None


def _send_package_with_scp(package: PackageResult, pod: dict[str, Any]) -> bool:
    """SCP the bundle archive to the Pod at /workspace/. Returns True on success."""
    scp_info = _pod_scp_info(pod)
    if scp_info is None:
        print("Pod does not have public IP/port mappings yet; cannot SCP.", file=sys.stderr)
        return False

    public_ip, ssh_port = scp_info
    exe = shutil.which("scp")
    if exe is None:
        print("scp was not found on PATH; cannot transfer via SCP.", file=sys.stderr)
        return False

    archive = str(package.archive_path.resolve())
    dest = f"root@{public_ip}:/workspace/"
    # -o StrictHostKeyChecking=accept-new: auto-accept unknown host keys (Windows OpenSSH
    #   has no TTY to prompt, so it closes the connection when it encounters an unknown key)
    # -o ServerAliveInterval=30: send keepalive every 30s to prevent connection drops
    #   during long transfers (RunPod pods sometimes drop idle SSH connections).
    command = [exe, "-P", str(ssh_port),
               "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ServerAliveInterval=30",
               archive, dest]
    archive_mb = package.archive_path.stat().st_size / 1_000_000
    print(f"Starting SCP transfer: scp -P {ssh_port} <archive> root@{public_ip}:/workspace/")
    print(f"  Archive size: {archive_mb:.1f} MB")
    try:
        subprocess.run(command, check=True)
        print("SCP transfer completed.")
        return True
    except subprocess.CalledProcessError as ex:
        print(f"SCP transfer failed (exit code {ex.returncode}).", file=sys.stderr)
        return False
    except OSError as ex:
        print(f"SCP transfer failed: {ex}", file=sys.stderr)
        return False


def _is_scp_available() -> bool:
    """Check if SCP is available on PATH and the pod has SCP info."""
    return shutil.which("scp") is not None


def _build_scp_bootstrap_wait_cmd(archive_name: str) -> str:
    """Generate bash code that polls for a tar file delivered via SCP instead of runpodctl receive."""
    escaped = _shell_quote(archive_name)
    return (
        "echo 'Waiting for bundle via SCP...'\n"
        f"while [ ! -f {escaped} ]; do\n"
        "  sleep 10\n"
        "done\n"
        f"test -f {escaped}\n"
    )


def _build_download_url_bootstrap_cmd(archive_name: str, url: str) -> str:
    """Generate bash code that downloads the bundle from a URL via wget."""
    escaped_archive = _shell_quote(archive_name)
    escaped_url = _shell_quote(url)
    return (
        f"echo 'Downloading bundle from URL: {escaped_url}'\n"
        f"wget -q {escaped_url} -O {escaped_archive}\n"
        f"test -f {escaped_archive}\n"
    )


def _runpodctl_json(args: list[str]) -> Any | None:
    exe = shutil.which("runpodctl")
    if exe is None:
        return None
    try:
        completed = subprocess.run(
            [exe, *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = completed.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _requested_gpu_types(args: argparse.Namespace) -> list[str]:
    # Explicit GPU list overrides everything
    if args.gpu_types:
        seen: set[str] = set()
        out: list[str] = []
        for gpu in args.gpu_types:
            if gpu not in seen:
                out.append(str(gpu))
                seen.add(gpu)
        return out

    # Cost-target mode: filter by VRAM, cloud availability, exclude datacenter cards
    if not args.no_cost_target:
        gpu_infos = _gpu_full_info()
        min_vram = int(args.min_gpu_vram_gb)
        cloud_type = args.cloud_type
        preferred_gpu_ids = [str(item) for item in getattr(args, "preferred_gpu_ids", [])]
        preferred_rank = {gpu_id: index for index, gpu_id in enumerate(preferred_gpu_ids)}

        qualifying = []
        for name, info in gpu_infos.items():
            if _is_excluded_gpu(name):
                continue
            if info["vram_gb"] < min_vram:
                continue
            if not info.get("available", True):
                continue
            if cloud_type == "COMMUNITY" and not info.get("community_cloud", False):
                continue
            price = float(info.get("price_per_hour", 0))
            if args.max_cost_per_hour and price > 0 and price > float(args.max_cost_per_hour):
                continue
            rank = preferred_rank.get(name, len(preferred_rank))
            qualifying.append((name, price, info["vram_gb"], rank))

        # Prefer explicit GPU ids first, then sort the remainder by known price/VRAM.
        qualifying.sort(key=lambda x: (x[3], x[1] if x[1] > 0 else 999, x[2]))
        if qualifying:
            print("GPU candidates (cheapest known first):")
            for name, price, vram, rank in qualifying:
                if price > 0:
                    prefix = "* " if rank < len(preferred_rank) else "  "
                    print(f"{prefix}{name}: ~${price:.2f}/hr, {int(vram)}GB VRAM")
                else:
                    prefix = "* " if rank < len(preferred_rank) else "  "
                    print(f"{prefix}{name}: price unknown, {int(vram)}GB VRAM")
            return [name for name, _, _, _ in qualifying]
        print(f"No GPUs found matching criteria; falling back to {DEFAULT_GPU_TYPE}.", file=sys.stderr)

    # Specific GPU type mode
    requested = [str(args.gpu_type)]
    if args.gpu_fallback:
        requested.extend(FALLBACK_GPU_INFOS.keys())
    seen2: set[str] = set()
    out2: list[str] = []
    for gpu in requested:
        if gpu not in seen2:
            out2.append(gpu)
            seen2.add(gpu)
    return out2


def _requested_data_centers(args: argparse.Namespace) -> list[str]:
    if args.data_centers:
        return [str(item) for item in args.data_centers]
    if args.data_center and str(args.data_center).lower() != "auto":
        return [str(args.data_center)]
    return list(DEFAULT_FALLBACK_DATA_CENTERS)


def _gpu_memory_map() -> dict[str, int]:
    rows = _runpodctl_json(["gpu", "list", "--include-unavailable"])
    out: dict[str, int] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("gpuId"):
                try:
                    out[str(row["gpuId"])] = int(row.get("memoryInGb", 0))
                except (TypeError, ValueError):
                    pass
    return out


def _gpu_full_info() -> dict[str, dict[str, float | bool]]:
    """Query runpodctl for GPU info. Merges with hardcoded pricing for display."""
    rows = _runpodctl_json(["gpu", "list", "--include-unavailable"])
    out: dict[str, dict[str, float | bool]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpuId") or row.get("id") or row.get("displayName") or "")
            if not gpu_id:
                continue
            mem_gb = float(row.get("memoryInGb", 0))
            available = bool(row.get("available", False))
            community = bool(row.get("communityCloud", False))
            fallback = FALLBACK_GPU_INFOS.get(gpu_id, {})
            price = float(fallback.get("price_per_hour", 0))
            out[gpu_id] = {
                "vram_gb": mem_gb,
                "price_per_hour": price,
                "available": available,
                "community_cloud": community,
            }
    if not out:
        for name, info in FALLBACK_GPU_INFOS.items():
            out[name] = {
                "vram_gb": info["vram_gb"],
                "price_per_hour": info["price_per_hour"],
                "available": True,
                "community_cloud": True,
            }
    return out


def _is_excluded_gpu(gpu_id: str) -> bool:
    return any(gpu_id.startswith(prefix) for prefix in EXCLUDED_GPU_PREFIXES)


def _availability_candidates(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Build (GPU, datacenter) candidate pairs.

    Only US datacenters are considered because non-US datacenters frequently
    don't provide public IPs needed for SCP/SSH, have slower transfer speeds
    from US-based workstations, and cause connection instability."""
    requested_gpus = _requested_gpu_types(args)
    requested_dcs = _requested_data_centers(args)

    # Only US datacenters can work reliably — non-US datacenters frequently
    # don't provide public IPs and cause SSH/SCP connection drops.
    requested_dcs = [dc for dc in requested_dcs if dc.startswith("US-")]
    if not requested_dcs:
        requested_dcs = [dc for dc in DEFAULT_FALLBACK_DATA_CENTERS if dc.startswith("US-")]

    # Filter out datacenters that don't support network volumes
    requested_dcs = [dc for dc in requested_dcs if dc in VALID_NETWORK_VOLUME_DATACENTERS]
    requested_dc_set = set(requested_dcs)
    auto_dc = bool(args.data_center and str(args.data_center).lower() == "auto" and not args.data_centers)
    min_vram = int(args.min_gpu_vram_gb)
    memory_by_gpu = _gpu_memory_map()

    datacenters = _runpodctl_json(["datacenter", "list"])
    candidates: list[tuple[str, str]] = []
    if isinstance(datacenters, list):
        for dc_row in datacenters:
            if not isinstance(dc_row, dict):
                continue
            dc_id = str(dc_row.get("id") or dc_row.get("name") or "")
            if not dc_id or dc_id not in VALID_NETWORK_VOLUME_DATACENTERS:
                continue
            # Non-US datacenters frequently don't provide public IPs needed for
            # SCP/SSH and cause connection drops. Always skip them.
            if not dc_id.startswith("US-"):
                continue
            if not auto_dc and dc_id not in requested_dc_set:
                continue
            availability = dc_row.get("gpuAvailability") or []
            available_gpu_ids = {
                str(item.get("gpuId"))
                for item in availability
                if isinstance(item, dict)
                and item.get("gpuId")
                and str(item.get("stockStatus", "")).lower() not in {"none", "unavailable", "outofstock", "out_of_stock"}
            }
            for gpu in requested_gpus:
                if gpu not in available_gpu_ids:
                    continue
                if memory_by_gpu.get(gpu, min_vram) < min_vram:
                    continue
                candidates.append((gpu, dc_id))

    if not candidates:
        for dc_id in requested_dcs:
            for gpu in requested_gpus:
                if memory_by_gpu.get(gpu, min_vram) < min_vram:
                    continue
                candidates.append((gpu, dc_id))

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _is_retryable_error(ex: Exception) -> bool:
    if not isinstance(ex, RunPodApiError):
        return False
    if ex.status in (400, 401, 403, 404):
        return False
    text = ex.detail.lower()
    return (
        "no instances" in text
        or "currently available" in text
        or "not found" in text
        or "does not support" in text
        or ex.status >= 500
    )


def _delete_network_volume(api_key: str, volume_id: str) -> None:
    try:
        _request_json("DELETE", f"/networkvolumes/{volume_id}", api_key=api_key)
    except Exception as ex:  # noqa: BLE001
        print(f"Warning: failed to delete unused network volume {volume_id}: {ex}", file=sys.stderr)


def _poll_pod(api_key: str, pod_id: str, *, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    latest: dict[str, Any] = {}
    while time.time() < deadline:
        latest = _request_json("GET", f"/pods/{pod_id}", api_key=api_key)
        public_ip = latest.get("publicIp")
        port_mappings = latest.get("portMappings") or {}
        if public_ip and port_mappings:
            return latest
        time.sleep(10)
    return latest


def _ssh_hint(pod: dict[str, Any]) -> str | None:
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22") or port_mappings.get(22)
    if not public_ip or not ssh_port:
        return None
    return f"ssh root@{public_ip} -p {ssh_port}"


def _write_setup_manifest(
    output_root: Path,
    *,
    package: PackageResult,
    pod_payload: dict[str, Any],
    pod: dict[str, Any] | None,
    network_volume_payload: dict[str, Any] | None,
    network_volume: dict[str, Any] | None,
    dry_run: bool,
    transfer_code: str,
    auto_start_training: bool,
    attempts: list[dict[str, Any]],
) -> Path:
    path = output_root / f"runpod_setup_{package.package_name}.json"
    payload = {
        "schema": "spec-077-runpod-setup",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "package": {
            "name": package.package_name,
            "bundle_dir": str(package.bundle_dir),
            "archive_path": str(package.archive_path),
        },
        "pod_create_payload": pod_payload,
        "pod": pod,
        "network_volume_create_payload": network_volume_payload,
        "network_volume": network_volume,
        "transfer": {
            "mode": "runpodctl" if package.archive_path.exists() else "none",
            "code": transfer_code,
            "archive_name": package.archive_path.name,
            "auto_start_training": bool(auto_start_training),
        },
        "availability_attempts": attempts,
        "api_key_stored": False,
        "notes": [
            "RUNPOD_API_KEY is used only for REST calls and is not written to this manifest.",
            "Normal RunPod API keys do not upload files to network volumes; use runpodctl/rsync or separate S3 API credentials.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _print_next_steps(package: PackageResult, pod: dict[str, Any] | None, transfer_code: str) -> None:
    print("\nNext steps")
    print("----------")
    print(f"Package archive: {package.archive_path}")
    if pod:
        print(f"Pod ID: {pod.get('id')}")
        print(f"Pod status: desired={pod.get('desiredStatus')} ip={pod.get('publicIp')} ports={pod.get('portMappings')}")
        ssh = _ssh_hint(pod)
        if ssh:
            print(f"SSH hint: {ssh}")
    print("\nTransfer options (if auto-transfer failed or was skipped):")
    print(f"  scp -P <ssh-port> \"{package.archive_path}\" root@<pod-ip>:/workspace/")
    print(f"  rsync -avzP \"{package.archive_path}\" root@<pod-ip>:/workspace/")
    print("\nPod-side manual steps (SSH in first):")
    print("  tar -xf <archive-name>.tar")
    print("  cd <bundle-name>")
    print("  bash runpod/install_deps.sh")
    print("  bash runpod/verify_bundle.sh")
    print("  bash runpod/smoke_spec077.sh   # or smoke.sh / smoke_v24.sh")
    print("  bash runpod/train_spec077.sh    # or train.sh")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a RunPod RTX 4000 Ada Pod for Spec 077 training.")
    parser.add_argument("--api-key", type=str, default=None, help="RunPod API key. Defaults to RUNPOD_API_KEY env var.")
    parser.add_argument("--wow-root", type=Path, default=_default_wow_root())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--package-name", type=str, default=None)
    parser.add_argument("--overwrite-package", action="store_true", default=False)
    parser.add_argument("--skip-package", action="store_true", default=False,
                        help="Use an existing package archive named by --package-name instead of rebuilding.")
    parser.add_argument("--no-package-tests", action="store_true", default=False)
    parser.add_argument("--full-v18-stores", action="store_true", default=False)
    parser.add_argument("--builds", nargs="*", default=None)
    parser.add_argument("--curation-manifest", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="cuda_albedo_group_nearest")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-vram-gb", type=float, default=12.0)
    parser.add_argument("--pod-name", type=str, default="spec077-rtx4000ada")
    parser.add_argument("--gpu-type", type=str, default=DEFAULT_GPU_TYPE,
                        help="Specific GPU type. Ignored in cost-target mode (default) unless --no-cost-target is set.")
    parser.add_argument("--gpu-types", nargs="*", default=None,
                        help="Explicit ordered GPU list. Overrides cost-target and --gpu-type.")
    parser.add_argument("--gpu-fallback", dest="gpu_fallback", action="store_true", default=False,
                        help="Opt in to built-in fallback GPUs after --gpu-type. Default is off.")
    parser.add_argument("--no-gpu-fallback", dest="gpu_fallback", action="store_false",
                        help="Only try --gpu-type (default).")
    parser.add_argument("--min-gpu-vram-gb", type=int, default=DEFAULT_MIN_GPU_VRAM_GB,
                        help="Minimum GPU VRAM in GB for cost-target filtering (default 12).")
    parser.add_argument("--max-cost-per-hour", type=float, default=DEFAULT_MAX_COST_PER_HOUR,
                        help="Maximum cost per hour for cost-target GPU filtering (default $0.50).")
    parser.add_argument("--no-cost-target", dest="no_cost_target", action="store_true", default=False,
                        help="Disable cost-target mode; use --gpu-type instead.")
    parser.add_argument("--image-name", type=str, default=DEFAULT_IMAGE,
                        help="RunPod/PyTorch image. Override if your account has a newer preferred template image.")
    parser.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE",
                        help="RunPod cloud type. SECURE always gets a public IP (default, needed for SCP/SSH). "
                             "COMMUNITY is cheaper but pods may not get public IPs, breaking SCP transfer.")
    parser.add_argument("--data-center", type=str, default=DEFAULT_DATA_CENTER,
                        help="Datacenter for the Pod/network volume, or 'auto' to select from availability (default).")
    parser.add_argument("--data-centers", nargs="*", default=None,
                        help="Ordered datacenter fallback list. Overrides --data-center.")
    parser.add_argument("--container-disk-gb", type=int, default=50)
    parser.add_argument("--volume-gb", type=int, default=150,
                        help="Persistent Pod volume size when not using a network volume.")
    parser.add_argument("--min-ram-gb", type=int, default=24,
                        help="Minimum system RAM per GPU in GB (default 24).")
    parser.add_argument("--min-vcpu", type=int, default=8)
    parser.add_argument("--allowed-cuda-versions", nargs="*", default=None)
    parser.add_argument("--interruptible", action="store_true", default=False)
    parser.add_argument("--use-network-volume", dest="use_network_volume", action="store_true", default=True,
                        help="Create/attach a RunPod network volume mounted at /workspace (default).")
    parser.add_argument("--no-network-volume", dest="use_network_volume", action="store_false",
                        help="Use a Pod-local persistent volume instead of a RunPod network volume.")
    parser.add_argument("--network-volume-id", type=str, default=None,
                        help="Attach an existing network volume instead of creating one.")
    parser.add_argument("--network-volume-name", type=str, default="spec077-training")
    parser.add_argument("--network-volume-gb", type=int, default=150)
    parser.add_argument("--auto-transfer", dest="auto_transfer", action="store_true", default=True,
                        help="Start the Pod waiting for runpodctl receive, then run runpodctl send locally (default).")
    parser.add_argument("--no-auto-transfer", dest="auto_transfer", action="store_false",
                        help="Create the Pod/volume but do not set a bootstrap receive command or send data.")
    parser.add_argument("--transfer-mode", type=str, default="auto",
                        choices=["auto", "relay", "scp"],
                        help="Transfer method: 'auto' (detect: SCP if runpodctl missing, else relay), "
                             "'relay' (runpodctl send/receive), 'scp' (SCP to Pod IP). Default: auto. "
                             "Set --download-url instead to skip all local transfer and have the Pod download via wget.")
    parser.add_argument("--download-url", type=str, default=None,
                        help="Have the Pod download the bundle from this URL via wget instead of local transfer. "
                             "The URL must be publicly accessible. No SCP, no runpodctl relay needed. "
                             "Example: --download-url https://my-server.example.com/bundle.tar")
    parser.add_argument("--transfer-code", type=str, default=None,
                        help="Custom runpodctl send/receive code. Defaults to a generated spec077-<token> code. "
                             "Not used in SCP mode (SCP uses Pod IP/port from the created Pod).")
    parser.add_argument("--auto-start-training", dest="auto_start_training", action="store_true", default=True,
                        help="After receiving the package, install deps, verify, smoke, and start full training (default).")
    parser.add_argument("--no-auto-start-training", dest="auto_start_training", action="store_false",
                        help="Receive/unpack/install/verify only; do not run smoke/full training automatically.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Build/package and write payloads, but do not call RunPod API.")
    parser.add_argument("--no-wait", action="store_true", default=False,
                        help="Do not poll for public IP/ports after creating the Pod.")
    parser.add_argument("--wait-timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-failed-volumes", action="store_true", default=False,
                        help="Do not delete auto-created network volumes when a candidate Pod create fails for no availability.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    wow_root = args.wow_root.resolve()
    output_root = (args.output_root or (wow_root / "output" / "cloud-packages")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    package_name = args.package_name or f"spec077_runpod_bundle_{_utc_stamp()}"
    args._resolved_package_name = package_name
    args._resolved_transfer_code = args.transfer_code or f"spec077-{secrets.token_hex(4)}"

    if args.skip_package:
        package = PackageResult(
            package_name=package_name,
            bundle_dir=output_root / package_name,
            archive_path=(output_root / package_name).with_suffix(".tar"),
        )
        if not package.archive_path.exists():
            print(f"Existing package archive not found: {package.archive_path}", file=sys.stderr)
            return 2
    else:
        package = _run_packager(args, package_name)

    api_key = _api_key_arg(args.api_key)
    if not api_key and not args.dry_run:
        print("RUNPOD_API_KEY is required unless --dry-run is used.", file=sys.stderr)
        return 2

    candidates = _availability_candidates(args)
    if not candidates:
        print("No RunPod GPU/datacenter candidates matched the request.", file=sys.stderr)
        return 2

    pod_payload: dict[str, Any] = {}
    pod: dict[str, Any] | None = None
    network_volume_payload: dict[str, Any] | None = None
    network_volume: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []

    for gpu_type, data_center in candidates:
        args.gpu_type = gpu_type
        args.data_center = data_center
        attempt: dict[str, Any] = {
            "gpu_type": gpu_type,
            "data_center": data_center,
            "status": "started",
        }
        candidate_volume_payload: dict[str, Any] | None = None
        candidate_volume: dict[str, Any] | None = None
        candidate_volume_id = args.network_volume_id
        created_volume_id: str | None = None

        try:
            if args.use_network_volume and not candidate_volume_id:
                candidate_volume_payload = _build_network_volume_payload(args)
                if args.dry_run:
                    candidate_volume = {"dry_run": True, **candidate_volume_payload}
                    candidate_volume_id = "DRY_RUN_NETWORK_VOLUME_ID"
                else:
                    assert api_key is not None
                    candidate_volume = _request_json(
                        "POST",
                        "/networkvolumes",
                        api_key=api_key,
                        payload=candidate_volume_payload,
                    )
                    candidate_volume_id = str(candidate_volume.get("id"))
                    if not candidate_volume_id:
                        raise RuntimeError(f"RunPod did not return a network volume id: {candidate_volume}")
                    created_volume_id = candidate_volume_id
                    attempt["network_volume_id"] = candidate_volume_id
            elif candidate_volume_id:
                candidate_volume = {"id": candidate_volume_id, "existing": True}
                attempt["network_volume_id"] = candidate_volume_id

            candidate_pod_payload = _build_pod_payload(args, network_volume_id=candidate_volume_id)
            pod_payload = candidate_pod_payload
            network_volume_payload = candidate_volume_payload
            network_volume = candidate_volume

            if args.dry_run:
                pod = {"dry_run": True, "id": "DRY_RUN_POD_ID"}
                attempt["status"] = "dry_run"
                attempts.append(attempt)
                break

            assert api_key is not None
            pod = _request_json("POST", "/pods", api_key=api_key, payload=candidate_pod_payload)
            pod_id = str(pod.get("id"))
            if not pod_id:
                raise RuntimeError(f"RunPod did not return a pod id: {pod}")
            attempt["status"] = "created"
            attempt["pod_id"] = pod_id
            attempts.append(attempt)
            if not args.no_wait:
                print(f"Created Pod {pod_id}; waiting for public IP/port mappings...")
                pod = _poll_pod(api_key, pod_id, timeout_seconds=int(args.wait_timeout_seconds))
                # SCP mode requires a public IP. If _poll_pod timed out without one,
                # terminate the pod so it doesn't keep billing, then retry next candidate.
                if _resolve_transfer_mode(args) == "scp" and _pod_scp_info(pod) is None:
                    print(f"Pod {pod_id} has no public IP after {args.wait_timeout_seconds}s; terminating.", file=sys.stderr)
                    try:
                        _request_json("DELETE", f"/pods/{pod_id}", api_key=api_key)
                    except Exception:
                        pass
                    raise RunPodApiError("SCP_FAIL", f"/pods/{pod_id}", 502,
                        f"Pod {pod_id} ({gpu_type}/{data_center}) has no public IP. "
                        f"SECURE pods in this datacenter/GPU combo may not support public IPs. "
                        f"Use --data-center to pick a known working datacenter.")
            break
        except Exception as ex:  # noqa: BLE001
            attempt["status"] = "failed"
            attempt["error"] = str(ex)
            if created_volume_id and not args.keep_failed_volumes:
                assert api_key is not None
                _delete_network_volume(api_key, created_volume_id)
                attempt["network_volume_deleted"] = True
            attempts.append(attempt)
            if _is_retryable_error(ex):
                print(f"RunPod candidate {gpu_type} / {data_center} failed: {ex}; trying next.", file=sys.stderr)
                continue
            raise

    if pod is None:
        print("RunPod could not create a Pod for any requested candidate.", file=sys.stderr)
        for attempt in attempts:
            print(f"  {attempt['gpu_type']} / {attempt['data_center']}: {attempt.get('error', attempt['status'])}", file=sys.stderr)
        manifest_path = _write_setup_manifest(
            output_root,
            package=package,
            pod_payload=pod_payload,
            pod=None,
            network_volume_payload=network_volume_payload,
            network_volume=network_volume,
            dry_run=bool(args.dry_run),
            transfer_code=str(args._resolved_transfer_code),
            auto_start_training=bool(args.auto_start_training),
            attempts=attempts,
        )
        print(f"Wrote setup manifest: {manifest_path}")
        return 1

    manifest_path = _write_setup_manifest(
        output_root,
        package=package,
        pod_payload=pod_payload,
        pod=pod,
        network_volume_payload=network_volume_payload,
        network_volume=network_volume,
        dry_run=bool(args.dry_run),
        transfer_code=str(args._resolved_transfer_code),
        auto_start_training=bool(args.auto_start_training),
        attempts=attempts,
    )
    print(f"Wrote setup manifest: {manifest_path}")

    transfer_mode = _resolve_transfer_mode(args)
    transferred = False
    if args.auto_transfer and not args.dry_run:
        if transfer_mode == "scp":
            print(f"Transfer mode: SCP (using Pod IP {pod.get('publicIp', '?')})")
            transferred = _send_package_with_scp(package, pod) if pod else False
        else:
            print("Transfer mode: runpodctl relay")
            transferred = _send_package_with_runpodctl(package, str(args._resolved_transfer_code))
        if transferred:
            print("Package transfer completed. The Pod bootstrap should now unpack and continue.")
    elif args.auto_transfer:
        if transfer_mode == "scp":
            print(f"Dry run SCP command: scp -P <ssh-port> \"{package.archive_path}\" root@<pod-ip>:/workspace/")
        else:
            print(f"Dry run transfer command: runpodctl send \"{package.archive_path}\" --code {args._resolved_transfer_code}")
    _print_next_steps(package, pod, str(args._resolved_transfer_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
