from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RUNPOD_PODS_API = "https://rest.runpod.io/v1/pods"
DEFAULT_ALLOWED_CUDA_VERSION = "12.8"
DEFAULT_PORTS = ["22/tcp", "8888/http"]


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_key_value(raw_text: str) -> tuple[str, str]:
    if "=" not in raw_text:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {raw_text}")
    key, value = raw_text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Expected a non-empty KEY in: {raw_text}")
    return key, value


def parse_host_env(var_name: str) -> tuple[str, str]:
    value = os.environ.get(var_name)
    if value is None:
        raise SystemExit(f"Host environment variable not set: {var_name}")
    return var_name, value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a Runpod Pod for v9 training with the GHCR trainer image and portable bundle contract."
    )
    parser.add_argument("--api-key", help="Runpod API key. Prefer --api-key-env or RUNPOD_API_KEY.")
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY", help="Host environment variable containing the Runpod API key. Default: RUNPOD_API_KEY")
    parser.add_argument("--api-url", default=RUNPOD_PODS_API, help=f"Runpod Pods REST endpoint. Default: {RUNPOD_PODS_API}")
    parser.add_argument("--name", default=f"v9-trainer-{utc_now_compact()}", help="Pod name.")
    parser.add_argument("--image", required=True, help="Container image tag, usually a GHCR tag.")
    parser.add_argument("--container-registry-auth-id", help="Runpod container registry auth ID for private GHCR pulls.")
    parser.add_argument("--gpu-type", action="append", required=True, help="Repeatable GPU type preference. First value is most preferred when --gpu-type-priority custom is used.")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs to attach. Default: 1")
    parser.add_argument("--gpu-type-priority", choices=["availability", "custom"], default="availability", help="How Runpod should interpret the ordered GPU type list.")
    parser.add_argument("--allowed-cuda-version", action="append", default=[DEFAULT_ALLOWED_CUDA_VERSION], help=f"Repeatable acceptable CUDA version for the Pod. Default: {DEFAULT_ALLOWED_CUDA_VERSION}")
    parser.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    parser.add_argument("--interruptible", action=argparse.BooleanOptionalAction, default=False, help="Use an interruptible or spot Pod.")
    parser.add_argument("--global-networking", action=argparse.BooleanOptionalAction, default=False, help="Enable global networking when supported.")
    parser.add_argument("--support-public-ip", action=argparse.BooleanOptionalAction, default=None, help="Request a public IP when relevant.")
    parser.add_argument("--data-center-id", action="append", help="Repeatable datacenter preference.")
    parser.add_argument("--data-center-priority", choices=["availability", "custom"], default="availability")
    parser.add_argument("--network-volume-id", help="Attach an existing Runpod network volume.")
    parser.add_argument("--volume-in-gb", type=int, default=200, help="Persistent Pod volume size in GB when not using a network volume. Default: 200")
    parser.add_argument("--container-disk-in-gb", type=int, default=50, help="Ephemeral container disk size in GB. Default: 50")
    parser.add_argument("--volume-mount-path", default="/workspace", help="Mount path for the Pod or network volume. Default: /workspace")
    parser.add_argument("--port", action="append", default=None, help="Repeatable exposed port in Runpod format, e.g. 22/tcp or 8888/http.")

    parser.add_argument("--run-name", help="Logical training run name. Defaults to the Pod name.")
    parser.add_argument("--bundle-root", default="/workspace/data/v9_bundle", help="Bundle mount root inside the container. Default: /workspace/data/v9_bundle")
    parser.add_argument("--bundle-download-url", help="Optional URL for a bundle archive to download inside the Pod when the bundle is not already mounted.")
    parser.add_argument("--bundle-header", action="append", default=None, help="Repeatable HTTP header for bundle download requests, formatted exactly as 'Header-Name: value'.")
    parser.add_argument("--output-root", default="/workspace/runs", help="Container root directory for training outputs. Default: /workspace/runs")
    parser.add_argument("--output-dir", help="Explicit output directory inside the container. Defaults to <output-root>/<run-name>.")
    parser.add_argument("--train-manifest", help="Explicit training manifest path inside the container. Defaults to <bundle-root>/manifests/train_manifest.json")
    parser.add_argument("--dev-manifest", help="Explicit dev manifest path inside the container. Defaults to <bundle-root>/manifests/dev_holdout_manifest.json")
    parser.add_argument("--no-dev-manifest", action="store_true", help="Disable the dev-eval manifest even if the bundle contains one.")

    parser.add_argument("--epochs", type=int, default=120, help="Trainer epoch budget. Default: 120")
    parser.add_argument("--batch-size", type=int, default=4, help="Trainer batch size. Default: 4")
    parser.add_argument("--train-workers", type=int, default=1, help="Trainer DataLoader workers. Default: 1")
    parser.add_argument("--val-workers", type=int, default=1, help="Validation DataLoader workers. Default: 1")
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=True, help="Pass --use-compile true|false to the trainer. Default: true")
    parser.add_argument("--require-minimap", action=argparse.BooleanOptionalAction, default=False, help="Require minimap-gated samples. Default: false")
    parser.add_argument("--require-wdl", action=argparse.BooleanOptionalAction, default=False, help="Require WDL-gated samples. Default: false")
    parser.add_argument("--selection-metric", help="Explicit trainer selection metric. Default: dev_global_mae when using a dev manifest, otherwise auto.")
    parser.add_argument("--target-curated-samples", type=int, help="Optional trainer curated sample cap.")
    parser.add_argument("--trainer-arg", action="append", default=None, help="Repeatable raw extra trainer argument token. Example: --trainer-arg=--resume-from --trainer-arg=/workspace/runs/foo/last_checkpoint.pt")

    parser.add_argument("--env", action="append", type=parse_key_value, default=None, help="Repeatable extra Pod environment entry, formatted KEY=VALUE.")
    parser.add_argument("--env-from-host", action="append", default=None, help="Repeatable host environment variable name to copy into the Pod environment.")
    parser.add_argument("--docker-entrypoint", action="append", help="Optional dockerEntrypoint override token. Repeat for each token.")
    parser.add_argument("--docker-start-cmd", action="append", help="Optional dockerStartCmd override token. Repeat for each token.")
    parser.add_argument("--dry-run", action="store_true", help="Print the request payload instead of creating the Pod.")
    return parser


def require_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    value = os.environ.get(args.api_key_env)
    if value:
        return value
    raise SystemExit(
        f"Runpod API key missing. Set {args.api_key_env} in the host environment or pass --api-key directly."
    )


def resolve_run_name(args: argparse.Namespace) -> str:
    return args.run_name or args.name


def resolve_train_manifest(args: argparse.Namespace) -> str:
    return args.train_manifest or f"{args.bundle_root.rstrip('/')}/manifests/train_manifest.json"


def resolve_dev_manifest(args: argparse.Namespace) -> str:
    if args.no_dev_manifest:
        return ""
    return args.dev_manifest or f"{args.bundle_root.rstrip('/')}/manifests/dev_holdout_manifest.json"


def build_trainer_args(args: argparse.Namespace, *, dev_manifest_enabled: bool) -> list[str]:
    trainer_args: list[str] = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--train-workers",
        str(args.train_workers),
        "--val-workers",
        str(args.val_workers),
        "--use-compile",
        "true" if args.use_compile else "false",
    ]

    if not args.require_minimap:
        trainer_args.append("--no-require-minimap")
    if not args.require_wdl:
        trainer_args.append("--no-require-wdl")

    selection_metric = args.selection_metric
    if not selection_metric:
        selection_metric = "dev_global_mae" if dev_manifest_enabled else "auto"
    trainer_args.extend(["--selection-metric", selection_metric])

    if args.target_curated_samples:
        trainer_args.extend(["--target-curated-samples", str(args.target_curated_samples)])
    if args.trainer_arg:
        trainer_args.extend(args.trainer_arg)

    return trainer_args


def build_pod_env(args: argparse.Namespace) -> dict[str, str]:
    run_name = resolve_run_name(args)
    output_dir = args.output_dir or f"{args.output_root.rstrip('/')}/{run_name}"
    train_manifest = resolve_train_manifest(args)
    dev_manifest = resolve_dev_manifest(args)
    dev_manifest_enabled = bool(dev_manifest)
    trainer_args = build_trainer_args(args, dev_manifest_enabled=dev_manifest_enabled)

    env: dict[str, str] = {
        "V9_RUN_NAME": run_name,
        "V9_BUNDLE_ROOT": args.bundle_root,
        "V9_TRAIN_MANIFEST": train_manifest,
        "V9_OUTPUT_ROOT": args.output_root,
        "V9_OUTPUT_DIR": output_dir,
        "V9_TRAINER_ARGS_JSON": json.dumps(trainer_args),
    }
    if dev_manifest_enabled:
        env["V9_DEV_EVAL_MANIFEST"] = dev_manifest
    if args.bundle_download_url:
        env["V9_BUNDLE_DOWNLOAD_URL"] = args.bundle_download_url
    if args.bundle_header:
        env["V9_BUNDLE_HEADERS_JSON"] = json.dumps(args.bundle_header)

    if args.env:
        for key, value in args.env:
            env[key] = value
    if args.env_from_host:
        for host_var in args.env_from_host:
            key, value = parse_host_env(host_var)
            env[key] = value

    return env


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": args.name,
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "imageName": args.image,
        "gpuCount": args.gpu_count,
        "gpuTypeIds": args.gpu_type,
        "gpuTypePriority": args.gpu_type_priority,
        "allowedCudaVersions": list(dict.fromkeys(args.allowed_cuda_version)),
        "containerDiskInGb": args.container_disk_in_gb,
        "volumeMountPath": args.volume_mount_path,
        "interruptible": args.interruptible,
        "globalNetworking": args.global_networking,
        "env": build_pod_env(args),
        "ports": args.port or list(DEFAULT_PORTS),
    }

    if args.container_registry_auth_id:
        payload["containerRegistryAuthId"] = args.container_registry_auth_id
    if args.support_public_ip is not None:
        payload["supportPublicIp"] = args.support_public_ip
    if args.data_center_id:
        payload["dataCenterIds"] = args.data_center_id
        payload["dataCenterPriority"] = args.data_center_priority
    if args.network_volume_id:
        payload["networkVolumeId"] = args.network_volume_id
    else:
        payload["volumeInGb"] = args.volume_in_gb
    if args.docker_entrypoint:
        payload["dockerEntrypoint"] = args.docker_entrypoint
    if args.docker_start_cmd:
        payload["dockerStartCmd"] = args.docker_start_cmd

    return payload


def print_result(response: dict[str, Any]) -> None:
    pod_id = response.get("id", "<unknown>")
    name = response.get("name", "<unnamed>")
    desired_status = response.get("desiredStatus", "<unknown>")
    image_name = response.get("imageName", "<unknown>")
    print(f"Runpod Pod created: {pod_id}")
    print(f"  name: {name}")
    print(f"  desiredStatus: {desired_status}")
    print(f"  image: {image_name}")
    public_ip = response.get("publicIp")
    if public_ip:
        print(f"  publicIp: {public_ip}")


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = build_payload(args)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    api_key = require_api_key(args)
    request = Request(
        args.api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request) as response:
            payload_text = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Runpod Pod creation failed ({exc.code}): {body}") from exc
    except URLError as exc:
        raise SystemExit(f"Runpod Pod creation failed: {exc}") from exc

    try:
        response_payload = json.loads(payload_text)
    except json.JSONDecodeError:
        print(payload_text)
        raise SystemExit("Runpod returned a non-JSON response.")

    if not isinstance(response_payload, dict):
        print(json.dumps(response_payload, indent=2))
        raise SystemExit("Unexpected Runpod response shape.")

    print_result(response_payload)
    print("")
    print("Next steps:")
    print("  1. Wait for the Pod to finish pulling the image and exposing its shell ports.")
    print("  2. Confirm /workspace contains the expected bundle or that the bundle download completed.")
    print("  3. Relaunch the same Pod or command to resume from last_checkpoint.pt if training is interrupted.")


if __name__ == "__main__":
    main()
