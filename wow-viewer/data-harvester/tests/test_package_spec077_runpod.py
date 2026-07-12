"""Tests for the Spec 077 RunPod packaging helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import package_spec077_runpod  # noqa: E402
import setup_spec077_runpod  # noqa: E402


def _write_text(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_zarr_store(path: Path, arrays: list[str], files: list[str] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_text(path / "zarr.json", "{}")
    for array in arrays:
        array_dir = path / array
        array_dir.mkdir(parents=True, exist_ok=True)
        _write_text(array_dir / "zarr.json", "{}")
        _write_text(array_dir / "c" / "0", "chunk")
    for filename in files or []:
        _write_text(path / filename, "file")


def _fake_harvester_root(path: Path) -> None:
    _write_text(path / "data-harvester" / "src" / "harvester" / "__init__.py", "")
    for script_name in (
        "train_height_only_prior.py",
        "train_height_coarse_prior.py",
        "train_height_residual_prior.py",
        "build_albedo_dataset.py",
        "package_spec077_runpod.py",
    ):
        _write_text(path / "data-harvester" / "scripts" / script_name, "# placeholder\n")
    _write_text(path / "data-harvester" / "pyproject.toml", "[project]\nname='x'\n")


def test_package_spec077_runpod_copies_slim_v18_required_arrays(tmp_path: Path) -> None:
    wow_root = tmp_path / "wow-viewer"
    output_root = tmp_path / "packages"
    build = "0_5_3_3368"
    _fake_harvester_root(wow_root)
    _fake_zarr_store(
        wow_root / "output" / "datasets" / "teacher-prior" / f"{build}.zarr",
        ["processed_minimap_prior_256", "teacher_object_mask_256", "teacher_object_confidence_256", "raw_minimap_rgb_256"],
        ["tiles.parquet"],
    )
    _fake_zarr_store(
        wow_root / "output" / "datasets" / "v18" / f"{build}.zarr",
        ["height_257", "object_precise_mask", "object_filtered_mask", "normal_xyz", "normal_mask", "alpha_256"],
        ["index.parquet"],
    )
    _fake_zarr_store(
        wow_root / "output" / "datasets" / "albedo" / f"{build}.zarr",
        ["albedo_rgb_256"],
        ["tiles.parquet", "metadata.json"],
    )
    curation = wow_root / "output" / "analysis" / "teacher-prior" / "visibility-audit" / "two_build"
    _write_text(curation / "kept_tiles.parquet", "parquet")

    exit_code = package_spec077_runpod.main(
        [
            "--wow-root", str(wow_root),
            "--output-root", str(output_root),
            "--package-name", "pkg",
            "--builds", build,
            "--archive-format", "none",
            "--no-tests",
        ]
    )

    assert exit_code == 0
    bundle = output_root / "pkg"
    assert (bundle / "README_RunPod.md").exists()
    assert (bundle / "runpod" / "train_spec077.sh").exists()
    v18_dest = bundle / "data" / "v18" / f"{build}.zarr"
    assert (v18_dest / "height_257").exists()
    assert (v18_dest / "object_precise_mask").exists()
    assert (v18_dest / "object_filtered_mask").exists()
    assert (v18_dest / "normal_xyz").exists()
    assert (v18_dest / "normal_mask").exists()
    assert not (v18_dest / "alpha_256").exists()
    assert not (v18_dest / "index.parquet").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_game_client_files"] is False
    assert manifest["full_v18_stores"] is False
    assert manifest["training"]["entrypoint"] == "bash runpod/train_spec077.sh"


def test_setup_spec077_runpod_defaults_to_cost_target() -> None:
    args = setup_spec077_runpod._parse_args(["--dry-run"])
    assert args.max_cost_per_hour == 1.00
    assert args.min_gpu_vram_gb == 12
    assert args.min_ram_gb == 24
    assert args.cloud_type == "SECURE"
    assert args.no_cost_target is False
    assert args.use_network_volume is True
    payload = setup_spec077_runpod._build_pod_payload(args, network_volume_id=None)
    assert payload["cloudType"] == "SECURE"
    assert payload["gpuTypeIds"] == ["NVIDIA RTX 4000 Ada Generation"]
    assert payload["gpuCount"] == 1
    assert payload["containerDiskInGb"] == 50
    assert payload["ports"] == ["22/tcp", "8888/http"]
    assert payload["volumeMountPath"] == "/workspace"
    assert payload["volumeInGb"] == 150
    assert "dataCenterId" not in payload
    assert "dataCenterIds" not in payload
    assert "gpuType" not in payload
    assert "gpuTypePriority" not in payload
    assert "minRAMPerGPU" not in payload
    assert "minVCPUPerGPU" not in payload


def test_setup_spec077_runpod_excludes_datacenter_gpus() -> None:
    assert setup_spec077_runpod._is_excluded_gpu("NVIDIA A100 80GB PCIe")
    assert setup_spec077_runpod._is_excluded_gpu("NVIDIA H100 80GB HBM3")
    assert setup_spec077_runpod._is_excluded_gpu("NVIDIA B200")
    assert setup_spec077_runpod._is_excluded_gpu("NVIDIA L40S")
    assert setup_spec077_runpod._is_excluded_gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert not setup_spec077_runpod._is_excluded_gpu("NVIDIA GeForce RTX 3090")
    assert not setup_spec077_runpod._is_excluded_gpu("NVIDIA RTX 4000 Ada Generation")
    assert not setup_spec077_runpod._is_excluded_gpu("NVIDIA GeForce RTX 5090")


def test_setup_spec077_runpod_no_cost_target_uses_gpu_type() -> None:
    args = setup_spec077_runpod._parse_args(["--dry-run", "--no-cost-target"])
    requested = setup_spec077_runpod._requested_gpu_types(args)
    assert requested == ["NVIDIA RTX 4000 Ada Generation"]
    assert args.gpu_fallback is False


def test_setup_spec077_runpod_gpu_fallback_is_explicit_opt_in() -> None:
    args = setup_spec077_runpod._parse_args(["--dry-run", "--no-cost-target", "--gpu-fallback"])
    requested = setup_spec077_runpod._requested_gpu_types(args)
    assert requested[0] == "NVIDIA RTX 4000 Ada Generation"


def test_setup_spec077_runpod_network_volume_pins_datacenter() -> None:
    args = setup_spec077_runpod._parse_args(["--dry-run", "--use-network-volume", "--data-center", "US-KS-2"])
    payload = setup_spec077_runpod._build_pod_payload(args, network_volume_id="vol_test")
    assert payload["networkVolumeId"] == "vol_test"
    assert payload["dataCenterId"] == "US-KS-2"
    assert "dataCenterIds" not in payload
    assert payload["volumeInGb"] == 150


def test_setup_spec077_runpod_network_volume_rejects_auto_datacenter() -> None:
    args = setup_spec077_runpod._parse_args(["--dry-run", "--use-network-volume"])
    try:
        setup_spec077_runpod._build_network_volume_payload(args)
    except ValueError as ex:
        assert "auto" in str(ex)
    else:
        raise AssertionError("expected auto datacenter to be rejected for network-volume creation")
    try:
        setup_spec077_runpod._build_pod_payload(args, network_volume_id="vol_test")
    except ValueError as ex:
        assert "auto" in str(ex)
    else:
        raise AssertionError("expected auto datacenter to be rejected for network-volume Pod payload")


def test_setup_spec077_runpod_bootstrap_handles_receive_failure() -> None:
    args = setup_spec077_runpod._parse_args(["--dry-run", "--transfer-mode", "relay"])
    args._resolved_package_name = "pkg"
    args._resolved_transfer_code = "spec077-test"

    start_cmd = setup_spec077_runpod._build_bootstrap_start_cmd(args)

    assert start_cmd is not None
    assert "/workspace/bootstrap.log" in start_cmd
    assert "if ! runpodctl receive 'spec077-test'; then" in start_cmd
    assert "runpodctl receive failed." in start_cmd
    assert 'scp -P <port> pkg.tar root@<pod-ip>:/workspace/' in start_cmd
    assert 'rsync -avzP pkg.tar root@<pod-ip>:/workspace/' in start_cmd
    assert "bash runpod/install_deps.sh" in start_cmd


def test_setup_spec077_runpod_retries_no_capacity_and_deletes_failed_volume(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "packages"
    output_root.mkdir()
    _write_text(output_root / "pkg.tar", "bundle")
    calls: list[tuple[str, str, dict | None]] = []

    def fake_request(method: str, path: str, *, api_key: str, payload: dict | None = None, timeout: int = 60) -> dict:
        calls.append((method, path, payload))
        if method == "POST" and path == "/networkvolumes":
            data_center = str((payload or {}).get("dataCenterId"))
            return {"id": f"vol_{data_center}", "dataCenterId": data_center}
        if method == "POST" and path == "/pods":
            data_center = str((payload or {}).get("dataCenterId", ""))
            if data_center == "US-KS-2":
                raise setup_spec077_runpod.RunPodApiError("POST", "/pods", 500, "There are no instances currently available")
            return {"id": "pod_ok", "desiredStatus": "RUNNING"}
        if method == "DELETE" and path.startswith("/networkvolumes/"):
            return {}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(setup_spec077_runpod, "_request_json", fake_request)
    monkeypatch.setattr(setup_spec077_runpod, "_availability_candidates", lambda args: [
        ("NVIDIA RTX 4000 Ada Generation", "US-KS-2"),
        ("NVIDIA RTX A4500", "US-GA-2"),
    ])

    exit_code = setup_spec077_runpod.main([
        "--api-key", "test_key",
        "--wow-root", str(tmp_path / "wow-viewer"),
        "--output-root", str(output_root),
        "--package-name", "pkg",
        "--skip-package",
        "--no-auto-transfer",
        "--no-wait",
    ])

    assert exit_code == 0
    assert ("DELETE", "/networkvolumes/vol_US-KS-2", None) in calls
    manifest = json.loads((output_root / "runpod_setup_pkg.json").read_text(encoding="utf-8"))
    assert manifest["pod"]["id"] == "pod_ok"
    assert manifest["pod_create_payload"]["gpuTypeIds"] == ["NVIDIA RTX A4500"]
    assert manifest["pod_create_payload"]["dataCenterId"] == "US-GA-2"
    assert manifest["pod_create_payload"]["networkVolumeId"] == "vol_US-GA-2"
    assert "dockerStartCmd" in manifest["pod_create_payload"]
    assert manifest["availability_attempts"][0]["status"] == "failed"
    assert manifest["availability_attempts"][0]["network_volume_deleted"] is True
    assert manifest["availability_attempts"][1]["status"] == "created"


def test_setup_spec077_runpod_retries_datacenter_not_found(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "packages"
    output_root.mkdir()
    _write_text(output_root / "pkg.tar", "bundle")

    def fake_request(method: str, path: str, *, api_key: str, payload: dict | None = None, timeout: int = 60) -> dict:
        if method == "POST" and path == "/networkvolumes":
            data_center = str((payload or {}).get("dataCenterId"))
            if data_center == "EUR-IS-2":
                raise setup_spec077_runpod.RunPodApiError(
                    "POST", "/networkvolumes", 500,
                    'create network volume: Data center "EUR-IS-2" not found or does not support network volumes.',
                )
            return {"id": f"vol_{data_center}", "dataCenterId": data_center}
        if method == "POST" and path == "/pods":
            return {"id": "pod_ok", "desiredStatus": "RUNNING"}
        if method == "DELETE" and path.startswith("/networkvolumes/"):
            return {}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(setup_spec077_runpod, "_request_json", fake_request)
    monkeypatch.setattr(setup_spec077_runpod, "_availability_candidates", lambda args: [
        ("NVIDIA RTX 4000 Ada Generation", "EUR-IS-2"),
        ("NVIDIA RTX 4000 Ada Generation", "US-KS-2"),
    ])

    exit_code = setup_spec077_runpod.main([
        "--api-key", "test_key",
        "--wow-root", str(tmp_path / "wow-viewer"),
        "--output-root", str(output_root),
        "--package-name", "pkg",
        "--skip-package",
        "--no-auto-transfer",
        "--no-wait",
    ])

    assert exit_code == 0
    manifest = json.loads((output_root / "runpod_setup_pkg.json").read_text(encoding="utf-8"))
    assert manifest["pod"]["id"] == "pod_ok"
    assert manifest["pod_create_payload"]["dataCenterId"] == "US-KS-2"
    assert manifest["availability_attempts"][0]["status"] == "failed"
    assert "not found" in manifest["availability_attempts"][0]["error"].lower()
    assert manifest["availability_attempts"][1]["status"] == "created"

