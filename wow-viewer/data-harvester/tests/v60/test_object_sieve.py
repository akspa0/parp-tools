from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from harvester.v60.object_sieve import validate_object_sieve_corpus


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def _write_row(root: Path, row_id: str, regime: str, value: float, boundary: bool = False) -> dict:
    clean = np.full((256, 256), value, dtype=np.float32)
    contaminated = clean.copy()
    mask = np.zeros((256, 256), dtype=np.float32)
    if regime != "none":
        if boundary:
            mask[:, :4] = 1.0
        else:
            mask[96:112, 96:112] = 1.0
        contaminated[mask > 0.5] = 0.15
    name = f"{row_id}.npz"
    np.savez(
        root / name,
        objectified_terrain_shadow_256=contaminated,
        terrain_shadow_256=clean,
        object_contamination_mask_256=mask,
    )
    return {
        "row_id": row_id,
        "terrain_control_row_id": "ridge-v00",
        "terrain_control_family": "ridge",
        "object_family": "tree",
        "placement_regime": regime,
        "input": "objectified_terrain_shadow_256",
        "targets": ["terrain_shadow_256", "object_contamination_mask_256"],
        "split": "train",
        "placement_metadata": {"placement_count": 0 if regime == "none" else 1},
        "input_sha256": _hash(contaminated),
        "terrain_target_sha256": _hash(clean),
        "contamination_target_sha256": _hash(mask),
        "npz": name,
    }


def test_object_sieve_validates_all_placement_regimes(tmp_path: Path) -> None:
    regimes = ["none", "sparse", "dense", "overlap", "boundary_crossing"]
    rows = [_write_row(tmp_path, f"row-{regime}", regime, 0.4, regime == "boundary_crossing") for regime in regimes]
    (tmp_path / "object_sieve_manifest.json").write_text(
        json.dumps(
            {
                "schema": "v60-object-sieve-control-v1",
                "signal_contract": [
                    "objectified_terrain_shadow_256",
                    "terrain_shadow_256",
                    "object_contamination_mask_256",
                ],
                "row_count": len(rows),
                "terrain_row_count": 1,
                "object_families": ["tree"],
                "placement_regimes": regimes,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    report = validate_object_sieve_corpus(tmp_path)

    assert report["valid"] is True
    assert report["regime_counts"] == dict.fromkeys(regimes, 1)
    assert report["boundary_touch_counts"]["boundary_crossing"] == 1


def test_object_sieve_rejects_boundary_regime_without_boundary_pixels(tmp_path: Path) -> None:
    row = _write_row(tmp_path, "boundary", "boundary_crossing", 0.4, boundary=False)
    (tmp_path / "object_sieve_manifest.json").write_text(
        json.dumps(
            {
                "schema": "v60-object-sieve-control-v1",
                "signal_contract": [
                    "objectified_terrain_shadow_256",
                    "terrain_shadow_256",
                    "object_contamination_mask_256",
                ],
                "row_count": 1,
                "rows": [row],
            }
        ),
        encoding="utf-8",
    )

    report = validate_object_sieve_corpus(tmp_path)

    assert report["valid"] is False
    assert any("does not touch a tile boundary" in failure for failure in report["failures"])
