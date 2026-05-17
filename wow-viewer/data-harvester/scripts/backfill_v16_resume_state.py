from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pyarrow.parquet as pq
import zarr
import zarr.storage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _store_length(zarr_path: Path) -> int:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
            category=UserWarning,
        )
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
        try:
            return int(root["height_257"].shape[0])
        finally:
            store.close()


def _build_state_from_final_store(build: str, zarr_path: Path) -> dict[str, object]:
    index_path = zarr_path / "index.parquet"
    if not index_path.exists():
        raise RuntimeError(f"{zarr_path} is missing index.parquet")

    table = pq.read_table(str(index_path), columns=["map"])
    maps = _ordered_unique([str(value.as_py()) for value in table.column("map")])
    valid_tiles = table.num_rows
    array_length = _store_length(zarr_path)
    if array_length != valid_tiles:
        raise RuntimeError(
            f"{zarr_path} looks incomplete: array length {array_length} != index rows {valid_tiles}"
        )

    return {
        "build": build,
        "requested_maps": maps,
        "completed_maps": maps,
        "valid_tiles": valid_tiles,
        "skipped_zero_usable_maps": 0,
        "rejected_tile_count": 0,
        "codec": "backfilled-final-store",
        "clevel": -1,
        "shuffle": "backfilled-final-store",
        "capacity": valid_tiles,
        "finalized": True,
        "backfilled_from_final_store": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill _resume_state.json into completed V16 final stores")
    parser.add_argument("--build", type=str, help="Single build key (e.g. 3_3_5_12340)")
    parser.add_argument("--builds", nargs="+", help="Multiple build keys")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing _resume_state.json in final stores")
    args = parser.parse_args()

    builds = args.builds or ([args.build] if args.build else [])
    if not builds:
        parser.error("Provide --build or --builds")

    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: no final store at {zarr_path}")
            continue

        state_path = zarr_path / "_resume_state.json"
        if state_path.exists() and not args.overwrite:
            print(f"SKIP {build}: {state_path} already exists")
            continue

        state = _build_state_from_final_store(build, zarr_path)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"Wrote {state_path}")


if __name__ == "__main__":
    main()
