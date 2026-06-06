from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(module_name: str, relative_path: str):
    script_path = _ROOT / relative_path
    sys.path.insert(0, str(script_path.parent))
    sys.path.insert(0, str(_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tiny_manifest = _load_script_module("build_v18_tiny_manifest_test", "scripts/build_v18_tiny_manifest.py")


def _row(
    *,
    build: str,
    bucket: str,
    map_name: str,
    tile_id: int,
    quality: float,
) -> dict[str, object]:
    return {
        "keep": True,
        "build": build,
        "difficulty_bucket": bucket,
        "difficulty_rank": _tiny_manifest.DIFFICULTY_BUCKETS.index(bucket),
        "map": map_name,
        "tile_id": tile_id,
        "quality_score": quality,
        "usefulness_score": quality,
        "difficulty_score": quality,
    }


def test_resolve_target_count_uses_the_tighter_cap() -> None:
    assert _tiny_manifest._resolve_target_count(
        10,
        samples_per_bucket_per_build=3,
        fraction_per_bucket_per_build=0.2,
    ) == 2


def test_select_diverse_rows_round_robins_across_maps() -> None:
    rows = [
        _row(build="0_5_3_3368", bucket="hard", map_name="A", tile_id=1, quality=0.95),
        _row(build="0_5_3_3368", bucket="hard", map_name="A", tile_id=2, quality=0.90),
        _row(build="0_5_3_3368", bucket="hard", map_name="A", tile_id=3, quality=0.85),
        _row(build="0_5_3_3368", bucket="hard", map_name="B", tile_id=4, quality=0.92),
        _row(build="0_5_3_3368", bucket="hard", map_name="B", tile_id=5, quality=0.80),
    ]

    chosen = _tiny_manifest._select_diverse_rows(rows, target=3, seed=0)

    assert [row["map"] for row in chosen[:2]] == ["A", "B"]
    assert len(chosen) == 3


def test_build_tiny_manifest_rows_balances_each_build_bucket_stratum() -> None:
    rows = [
        _row(build="0_5_3_3368", bucket="easy", map_name="A", tile_id=1, quality=0.9),
        _row(build="0_5_3_3368", bucket="easy", map_name="B", tile_id=2, quality=0.8),
        _row(build="0_5_3_3368", bucket="hard", map_name="A", tile_id=3, quality=0.7),
        _row(build="0_5_3_3368", bucket="hard", map_name="B", tile_id=4, quality=0.6),
        _row(build="3_3_5_12340", bucket="easy", map_name="C", tile_id=5, quality=0.9),
        _row(build="3_3_5_12340", bucket="easy", map_name="D", tile_id=6, quality=0.85),
        _row(build="3_3_5_12340", bucket="hard", map_name="C", tile_id=7, quality=0.8),
        _row(build="3_3_5_12340", bucket="hard", map_name="D", tile_id=8, quality=0.75),
    ]

    selected, summary = _tiny_manifest.build_tiny_manifest_rows(
        rows,
        builds=["0_5_3_3368", "3_3_5_12340"],
        samples_per_bucket_per_build=1,
        fraction_per_bucket_per_build=1.0,
        sample_seed=7,
    )

    assert len(selected) == 4
    assert summary["build_bucket_counts"] == {
        "0_5_3_3368": {"easy": 1, "medium": 0, "hard": 1, "pathological": 0},
        "3_3_5_12340": {"easy": 1, "medium": 0, "hard": 1, "pathological": 0},
    }
