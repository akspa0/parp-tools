from __future__ import annotations

# The test imports the CLI module through its script path to exercise the real entrypoint.
# ruff: noqa: I001

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from v60_visualize_control_corpus import render_visual_review  # noqa: E402

from test_control_corpus import _write_manifest, _write_row  # noqa: E402


def test_visual_review_writes_family_and_variant_atlases(tmp_path: Path) -> None:
    rows = [
        _write_row(tmp_path, "ridge-v00", "ridge", "train", 0.4),
        _write_row(tmp_path, "ridge-v01", "ridge", "train", 0.5),
        _write_row(tmp_path, "noise-v00", "noise", "validation", 0.6),
    ]
    for variant, (tile_x, tile_y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        row = _write_row(tmp_path, f"cross-v{variant:02d}", "cross_tile_lightning", "train", 0.2 + (0.1 * variant))
        row.update(
            {
                "pattern_id": "cross_tile_lightning-pattern-00",
                "pattern_tile_x": tile_x,
                "pattern_tile_y": tile_y,
                "pattern_tile_span": 2,
                "pattern_continuity": "continuous_global_2x2",
            }
        )
        rows.append(row)
    _write_manifest(tmp_path, rows)

    output = tmp_path / "visual-review"
    report = render_visual_review(tmp_path, output, variants_per_family=2)

    assert report["signals_rendered"] == [
        "height_257",
        "terrain_shadow_256",
        "mcnr_normal_xyz",
        "height_edges",
    ]
    assert report["coverage_complete"] is False
    assert report["cross_tile_complete"] is True
    assert (output / "control-family-atlas.png").is_file()
    assert (output / "control-variant-atlas.png").is_file()
    assert (output / "control-cross-tile-atlas.png").is_file()
    saved = json.loads((output / "control-visual-review.json").read_text(encoding="utf-8"))
    assert saved["family_count"] == 3
