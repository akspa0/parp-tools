"""Create a RunPod Pod for the Spec 089 V23 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import package_v23_runpod  # noqa: E402
import setup_spec077_runpod as base  # noqa: E402


DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output" / "cloud-packages" / "v23"
DEFAULT_PREFERRED_GPU_IDS = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
]


def _default_wow_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run_packager(args: argparse.Namespace, package_name: str) -> base.PackageResult:
    wow_root = args.wow_root.resolve()
    output_root = args.output_root.resolve()
    package_args = [
        "--bundle-name",
        package_name,
        "--dataset-dir",
        str(args.dataset_dir.resolve()),
        "--builds",
        *args.builds,
        "--include-v22-subset-tiles",
        str(args.include_v22_subset_tiles),
        "--output-root",
        str(output_root),
        "--archive-format",
        "tar",
    ]
    if args.tileset_prune_table is not None:
        package_args.extend(["--tileset-prune-table", str(args.tileset_prune_table.resolve())])
    if args.overwrite_package:
        package_args.append("--overwrite")
    if args.no_package_tests:
        package_args.append("--no-tests")
    exit_code = package_v23_runpod.main(package_args)
    if exit_code != 0:
        raise RuntimeError(f"package_v23_runpod.py failed with exit code {exit_code}")
    return base.PackageResult(
        package_name=package_name,
        bundle_dir=output_root / package_name,
        archive_path=(output_root / package_name).with_suffix(".tar"),
    )


def _build_bootstrap_start_cmd(args: argparse.Namespace) -> str | None:
    if not args.auto_transfer:
        return None
    package_name = str(getattr(args, "_resolved_package_name", "v23_bundle"))
    archive_name = f"{package_name}.tar"
    transfer_code = str(getattr(args, "_resolved_transfer_code", args.transfer_code or "v23-transfer"))
    build_flags = " ".join(_shell_quote(build) for build in args.builds)
    lines = [
        "set -euo pipefail",
        "cd /workspace",
        "exec > >(tee -a /workspace/bootstrap.log) 2>&1",
        "date -u",
        f"echo Waiting for Spec 089 package via runpodctl code {_shell_quote(transfer_code)}",
        f"if ! runpodctl receive {_shell_quote(transfer_code)}; then",
        "  echo",
        "  echo 'runpodctl receive failed.'",
        "  echo 'Manual transfer options:'",
        f'  echo "  scp -P <port> {archive_name} root@<pod-ip>:/workspace/"',
        f'  echo "  rsync -avzP {archive_name} root@<pod-ip>:/workspace/"',
        "  exit 1",
        "fi",
        f"test -f {_shell_quote(archive_name)}",
        f"tar -xf {_shell_quote(archive_name)}",
        f"cd {_shell_quote(package_name)}",
        "bash runpod/v23/install_deps.sh",
        "bash runpod/v23/verify_bundle.sh",
    ]
    if args.auto_start_training:
        lines.extend(
            [
                "bash runpod/v23/smoke.sh",
                (
                    "bash runpod/v23/train.sh "
                    f"--dataset-dir data/v22 --builds {build_flags} --device cuda "
                    f"--target-vram-gb {float(args.target_vram_gb):g} --batch-size {int(args.batch_size)} "
                    f"--epochs {int(args.epochs)} --gpct-K {int(args.gpct_K)} "
                    f"--gpct-weight {float(args.gpct_weight):g} "
                    f"--bias-free-mask-ratio {float(args.bias_free_mask_ratio):g} "
                    f"--run-name {_shell_quote(args.run_name)} "
                    "--output-dir models/v23/height/runs/auto_train"
                ),
            ]
        )
    else:
        lines.append("echo Auto-start training disabled; smoke/train can be run from runpod/v23 manually.")
    lines.append("date -u")
    return "\n".join(lines)


def _build_pod_payload(args: argparse.Namespace, *, network_volume_id: str | None) -> dict[str, object]:
    payload = {
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


def _write_setup_manifest(
    output_root: Path,
    *,
    package: base.PackageResult,
    pod_payload: dict[str, object],
    pod: dict[str, object] | None,
    network_volume_payload: dict[str, object] | None,
    network_volume: dict[str, object] | None,
    dry_run: bool,
    transfer_code: str,
    auto_start_training: bool,
    attempts: list[dict[str, object]],
) -> Path:
    path = output_root / f"runpod_setup_{package.package_name}.json"
    payload = {
        "schema": "spec-089-runpod-setup",
        "created_utc": base.datetime.now(base.timezone.utc).isoformat(),
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
            "Bundle bootstrap path is runpod/v23/install_deps.sh -> verify_bundle.sh -> smoke.sh -> train.sh.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a RunPod Pod for Spec 089 V23 training.")
    parser.add_argument("--api-key", type=str, default=None, help="RunPod API key. Defaults to RUNPOD_API_KEY.")
    parser.add_argument("--wow-root", type=Path, default=_default_wow_root())
    parser.add_argument("--dataset-dir", type=Path, default=_default_wow_root() / "output" / "datasets" / "v22")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--package-name", type=str, default=None)
    parser.add_argument("--overwrite-package", action="store_true", default=False)
    parser.add_argument("--skip-package", action="store_true", default=False)
    parser.add_argument("--no-package-tests", action="store_true", default=False)
    parser.add_argument("--builds", nargs="+", default=["0_5_3_3368", "3_3_5_12340"])
    parser.add_argument("--tileset-prune-table", type=Path, default=None)
    parser.add_argument("--include-v22-subset-tiles", type=int, default=8)
    parser.add_argument("--pod-name", type=str, default="spec089-v23")
    parser.add_argument("--gpu-type", type=str, default=DEFAULT_GPU_TYPE)
    parser.add_argument("--gpu-types", nargs="*", default=None)
    parser.add_argument("--gpu-fallback", dest="gpu_fallback", action="store_true", default=False)
    parser.add_argument("--no-gpu-fallback", dest="gpu_fallback", action="store_false")
    parser.add_argument("--min-gpu-vram-gb", type=int, default=20)
    parser.add_argument("--max-cost-per-hour", type=float, default=1.00)
    parser.add_argument("--no-cost-target", dest="no_cost_target", action="store_true", default=False)
    parser.add_argument("--image-name", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="COMMUNITY")
    parser.add_argument("--data-center", type=str, default="auto")
    parser.add_argument("--data-centers", nargs="*", default=None)
    parser.add_argument("--container-disk-gb", type=int, default=50)
    parser.add_argument("--volume-gb", type=int, default=150)
    parser.add_argument("--min-ram-gb", type=int, default=24)
    parser.add_argument("--min-vcpu", type=int, default=8)
    parser.add_argument("--allowed-cuda-versions", nargs="*", default=None)
    parser.add_argument("--use-network-volume", dest="use_network_volume", action="store_true", default=True)
    parser.add_argument("--no-network-volume", dest="use_network_volume", action="store_false")
    parser.add_argument("--network-volume-id", type=str, default=None)
    parser.add_argument("--network-volume-name", type=str, default="spec089-v23")
    parser.add_argument("--network-volume-gb", type=int, default=150)
    parser.add_argument("--auto-transfer", dest="auto_transfer", action="store_true", default=True)
    parser.add_argument("--no-auto-transfer", dest="auto_transfer", action="store_false")
    parser.add_argument("--transfer-code", type=str, default=None)
    parser.add_argument("--auto-start-training", dest="auto_start_training", action="store_true", default=True)
    parser.add_argument("--no-auto-start-training", dest="auto_start_training", action="store_false")
    parser.add_argument("--run-name", type=str, default="v23_height_bundle_train")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-vram-gb", type=float, default=22.0)
    parser.add_argument("--gpct-K", dest="gpct_K", type=int, default=4)
    parser.add_argument("--gpct-weight", type=float, default=0.1)
    parser.add_argument("--bias-free-mask-ratio", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--no-wait", action="store_true", default=False)
    parser.add_argument("--wait-timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-failed-volumes", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    wow_root = args.wow_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    package_name = args.package_name or f"spec089_v23_bundle_{base._utc_stamp()}"
    args._resolved_package_name = package_name
    args._resolved_transfer_code = args.transfer_code or f"spec089-{secrets.token_hex(4)}"

    if args.skip_package:
        package = base.PackageResult(
            package_name=package_name,
            bundle_dir=output_root / package_name,
            archive_path=(output_root / package_name).with_suffix(".tar"),
        )
        if not package.archive_path.exists():
            print(f"Existing package archive not found: {package.archive_path}", file=sys.stderr)
            return 2
    else:
        package = _run_packager(args, package_name)

    api_key = base._api_key_arg(args.api_key)
    if not api_key and not args.dry_run:
        print("RUNPOD_API_KEY is required unless --dry-run is used.", file=sys.stderr)
        return 2

    if not args.gpu_types:
        args.preferred_gpu_ids = list(DEFAULT_PREFERRED_GPU_IDS)

    candidates = base._availability_candidates(args)
    if not candidates:
        print("No RunPod GPU/datacenter candidates matched the request.", file=sys.stderr)
        return 2

    pod_payload: dict[str, object] = {}
    pod: dict[str, object] | None = None
    network_volume_payload: dict[str, object] | None = None
    network_volume: dict[str, object] | None = None
    attempts: list[dict[str, object]] = []

    for gpu_type, data_center in candidates:
        args.gpu_type = gpu_type
        args.data_center = data_center
        attempt: dict[str, object] = {
            "gpu_type": gpu_type,
            "data_center": data_center,
            "status": "started",
        }
        candidate_volume_payload: dict[str, object] | None = None
        candidate_volume: dict[str, object] | None = None
        candidate_volume_id = args.network_volume_id
        created_volume_id: str | None = None

        try:
            if args.use_network_volume and not candidate_volume_id:
                candidate_volume_payload = base._build_network_volume_payload(args)
                if args.dry_run:
                    candidate_volume = {"dry_run": True, **candidate_volume_payload}
                    candidate_volume_id = "DRY_RUN_NETWORK_VOLUME_ID"
                else:
                    assert api_key is not None
                    candidate_volume = base._request_json(
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
            pod = base._request_json("POST", "/pods", api_key=api_key, payload=candidate_pod_payload)
            pod_id = str(pod.get("id"))
            if not pod_id:
                raise RuntimeError(f"RunPod did not return a pod id: {pod}")
            attempt["status"] = "created"
            attempt["pod_id"] = pod_id
            attempts.append(attempt)
            if not args.no_wait:
                print(f"Created Pod {pod_id}; waiting for public IP/port mappings...")
                pod = base._poll_pod(api_key, pod_id, timeout_seconds=int(args.wait_timeout_seconds))
            break
        except Exception as ex:  # noqa: BLE001
            attempt["status"] = "failed"
            attempt["error"] = str(ex)
            if created_volume_id and not args.keep_failed_volumes:
                assert api_key is not None
                base._delete_network_volume(api_key, created_volume_id)
                attempt["network_volume_deleted"] = True
            attempts.append(attempt)
            if base._is_retryable_error(ex):
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
    if args.auto_transfer and not args.dry_run:
        sent = base._send_package_with_runpodctl(package, str(args._resolved_transfer_code))
        if sent:
            print("Package transfer completed. The Pod bootstrap command should now unpack and continue.")
    elif args.auto_transfer:
        print(f"Dry run transfer command: runpodctl send \"{package.archive_path}\" --code {args._resolved_transfer_code}")
    base._print_next_steps(package, pod, str(args._resolved_transfer_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
