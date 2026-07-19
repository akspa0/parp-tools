"""Spec 114 T056: deployment inference contract — unseen tiles, auditable manifest, no truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from harvester.v50.direct_geometry_infer import (
    InferenceContractError,
    discover_tiles,
    load_geometry_checkpoint,
    load_tile_rgb,
    main,
    relief_to_uint16,
    run_inference,
)
from harvester.v50.direct_geometry_model import DIRECT_CNN_ID, build_geometry_model
from harvester.v50.height_relative_model import TARGET_CONTRACT_VERSION


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(114)
    model, _ = build_geometry_model(DIRECT_CNN_ID)
    path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {
            "model_variant": DIRECT_CNN_ID,
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "model": model.state_dict(),
            "epoch": 3,
            "val_mae": 0.19,
        },
        path,
    )
    return path


@pytest.fixture()
def tile(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    path = tmp_path / "tile_12_34.png"
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), mode="RGB").save(path)
    return path


def test_dry_run_predicts_and_writes_nothing(checkpoint: Path, tile: Path, tmp_path: Path) -> None:
    output = tmp_path / "infer"
    manifest = run_inference(
        checkpoint_path=checkpoint, inputs=[tile], output=output, device="cpu", write=False
    )
    assert not output.exists()
    assert manifest["schema"] == "v114-direct-geometry-inference-v1"
    assert manifest["tile_count"] == 1
    assert manifest["deployment_contract"]["relative_only"] is True
    entry = manifest["tiles"][0]
    assert len(entry["input_sha256"]) == 64
    assert "output" not in entry
    assert 0.0 <= entry["relief_min"] <= entry["relief_max"] <= 1.0


def test_write_persists_relief_sheet_and_manifest(
    checkpoint: Path, tile: Path, tmp_path: Path
) -> None:
    output = tmp_path / "infer"
    manifest = run_inference(
        checkpoint_path=checkpoint, inputs=[tile], output=output, device="cpu", write=True
    )
    entry = manifest["tiles"][0]
    relief_path = Path(entry["output"])
    assert relief_path.is_file()
    with Image.open(relief_path) as image:
        relief = np.asarray(image)
    assert relief.shape == (257, 257)
    assert relief.dtype in (np.uint16, np.int32)  # PIL I;16 reads back as int32 on some builds
    assert (output / "review_sheet.png").is_file()
    persisted = json.loads((output / "inference_manifest.json").read_text(encoding="utf-8"))
    assert persisted["checkpoint"]["sha256"] == manifest["checkpoint"]["sha256"]
    assert len(entry["output_sha256"]) == 64


def test_inference_is_deterministic(checkpoint: Path, tile: Path, tmp_path: Path) -> None:
    first = run_inference(
        checkpoint_path=checkpoint, inputs=[tile], output=tmp_path / "a", device="cpu", write=False
    )
    second = run_inference(
        checkpoint_path=checkpoint, inputs=[tile], output=tmp_path / "b", device="cpu", write=False
    )
    assert first["tiles"][0]["relief_mean"] == second["tiles"][0]["relief_mean"]
    assert first["tiles"][0]["relief_min"] == second["tiles"][0]["relief_min"]


def test_wrong_size_tile_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB").save(path)
    with pytest.raises(InferenceContractError, match="exactly 256x256"):
        load_tile_rgb(path)


def test_checkpoint_identity_gates(tmp_path: Path, checkpoint: Path) -> None:
    with pytest.raises(InferenceContractError, match="checkpoint not found"):
        load_geometry_checkpoint(tmp_path / "absent.pt", device="cpu")
    bad = tmp_path / "bad.pt"
    torch.save({"model_variant": DIRECT_CNN_ID, "target_contract_version": "v999",
                "model": {}}, bad)
    with pytest.raises(InferenceContractError, match="target contract"):
        load_geometry_checkpoint(bad, device="cpu")
    model, ckpt, identity = load_geometry_checkpoint(checkpoint, device="cpu")
    assert ckpt["epoch"] == 3
    assert identity["architecture"]["id"] == DIRECT_CNN_ID


def test_discovery_refuses_unsupported_and_empty(tmp_path: Path, tile: Path) -> None:
    assert discover_tiles([tile]) == [tile]
    assert discover_tiles([tmp_path]) == [tile]  # folder discovery finds the png
    with pytest.raises(InferenceContractError, match="unsupported or missing"):
        discover_tiles([tmp_path / "notes.txt"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InferenceContractError, match="no decodable"):
        discover_tiles([empty])


def test_relief_uint16_roundtrip_bounds() -> None:
    assert relief_to_uint16(np.array([[0.0, 1.0]], dtype=np.float32)).tolist() == [[0, 65535]]
    assert relief_to_uint16(np.array([[-1.0, 2.0]], dtype=np.float32)).tolist() == [[0, 65535]]


def test_cli_dry_run(checkpoint: Path, tile: Path, tmp_path: Path, capsys) -> None:
    exit_code = main(
        ["--checkpoint", str(checkpoint), "--input", str(tile),
         "--output", str(tmp_path / "infer"), "--device", "cpu"]
    )
    assert exit_code == 0
    assert "DRY RUN ONLY" in capsys.readouterr().out
    assert not (tmp_path / "infer").exists()
