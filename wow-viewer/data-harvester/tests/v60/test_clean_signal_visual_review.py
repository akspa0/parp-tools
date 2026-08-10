from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harvester.v60.clean_signal_corpus import build_clean_signal_corpus
from harvester.v60.clean_signal_visual_review import (
    VISUAL_REVIEW_SCHEMA,
    render_clean_signal_review,
)
from harvester.v60.control_corpus import CONTROL_FAMILY_BUCKETS


def _write_control_corpus(root: Path) -> None:
    rows = []
    source_rows = [("flat", "train", None), ("ridge", "validation", None)]
    source_rows.extend(
        ("cross_tile_lightning", "train", (tile_x, tile_y))
        for tile_x, tile_y in ((0, 0), (1, 0), (0, 1), (1, 1))
    )
    for index, (family, split, tile_position) in enumerate(source_rows):
        shadow = np.full((256, 256), 0.2 + index * 0.1, dtype=np.float32)
        y, x = np.mgrid[0:257, 0:257]
        height = (x + y + index).astype(np.float32)
        name = f"{family}-{index:02d}.npz"
        np.savez(root / name, terrain_shadow_256=shadow, height_257=height)
        row = {
                "row_id": f"{family}-v{index:02d}",
                "control_family": family,
                "complexity_bucket": CONTROL_FAMILY_BUCKETS[family],
                "source_group_id": f"group-{index}",
                "variant": index,
                "split": split,
                "npz": name,
            }
        if tile_position is not None:
            row.update(
                {
                    "pattern_id": "cross-lightning-00",
                    "pattern_tile_x": tile_position[0],
                    "pattern_tile_y": tile_position[1],
                    "pattern_tile_span": 2,
                    "pattern_continuity": "continuous_global_2x2",
                }
            )
        rows.append(row)
    import hashlib

    for row in rows:
        with np.load(root / row["npz"], allow_pickle=False) as payload:
            row["input_sha256"] = hashlib.sha256(
                np.ascontiguousarray(payload["terrain_shadow_256"], dtype="<f4").tobytes()
            ).hexdigest()
            row["target_sha256"] = hashlib.sha256(
                np.ascontiguousarray(payload["height_257"], dtype="<f4").tobytes()
            ).hexdigest()
    (root / "control_manifest.json").write_text(
        json.dumps({"schema": "v60-control-corpus-v1", "row_count": len(rows), "rows": rows}),
        encoding="utf-8",
    )


def test_clean_signal_visual_review_writes_family_and_variant_atlases(tmp_path: Path) -> None:
    source = tmp_path / "control"
    source.mkdir()
    _write_control_corpus(source)
    corpus = tmp_path / "clean"
    build_clean_signal_corpus(source, corpus)
    output = tmp_path / "visual-review"

    report = render_clean_signal_review(corpus, output, rows_per_family=2)

    assert report["schema"] == VISUAL_REVIEW_SCHEMA
    assert report["family_count"] == 3
    assert (output / "clean-signal-family-atlas.png").is_file()
    assert (output / "clean-signal-variant-atlas.png").is_file()
    assert report["outputs"]["cross_tile"]["available"]
    assert (output / "clean-signal-cross-tile-atlas.png").is_file()
    persisted = json.loads((output / "clean-signal-visual-review.json").read_text(encoding="utf-8"))
    assert persisted["row_count"] == 6
