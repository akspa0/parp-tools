"""Spec 119 family-isolated held-out split (T006/T007, research D-01, FR-004).

The object library has no spatial coordinates, so Spec 116's spatial isolation does not apply.
The library's locality unit is the asset's parent directory: numbered near-duplicate variants
(``castle01``/``castle02``, ``name_000``/``name_001``) live in one directory, so holding out
entire families (directories) provably keeps variants on one side of the split.

The leakage check is mandatory and a refusal condition, not a warning: after building the split,
``leakage_check`` enumerates numeric-suffix variant stems and asserts none straddle train/held-out
(``verified_violation_count`` must be 0 before the CLI will ``--write``).
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harvester.spec119.object_library_contract import (
    derive_asset_family,
    variant_stem,
)

SPLIT_SCHEMA = "v119-family-split-v1"


class SplitError(ValueError):
    """Raised when a split cannot be built or violates the leakage contract."""


def build_family_split(
    rows: Sequence[dict[str, Any]],
    held_out_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Group rows by asset family and hold out entire families until ~``held_out_fraction`` of rows.

    ``rows`` are asset dicts carrying ``normalized_asset_path``. Deterministic: families are
    sorted before the seeded shuffle, so the same rows + seed always yield the same split.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise SplitError(f"held_out_fraction must be in (0, 1); got {held_out_fraction}")
    if not rows:
        raise SplitError("no rows to split")

    families: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        family = derive_asset_family(str(row["normalized_asset_path"]))
        families.setdefault(family, []).append(index)
    if len(families) < 2:
        raise SplitError(
            f"need at least 2 distinct asset families to hold one out; found {len(families)}"
        )

    ordered = sorted(families)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)

    target = max(1, round(len(rows) * held_out_fraction))
    held_out_families: list[str] = []
    held_out_rows = 0
    # Walk the shuffled families and take whole families until the row target is met, always
    # leaving at least one family (and at least one row) on the train side.
    for family in shuffled:
        remaining = len(shuffled) - len(held_out_families)
        if remaining <= 1:
            break
        if held_out_rows >= target:
            break
        held_out_families.append(family)
        held_out_rows += len(families[family])
    if not held_out_families:
        held_out_families.append(shuffled[0])
        held_out_rows = len(families[shuffled[0]])

    held_out_set = set(held_out_families)
    train_families = [family for family in ordered if family not in held_out_set]
    train_rows = sum(len(families[family]) for family in train_families)

    split = {
        "schema": SPLIT_SCHEMA,
        "seed": int(seed),
        "held_out_fraction": float(held_out_fraction),
        "train_families": train_families,
        "held_out_families": sorted(held_out_families),
        "train_row_count": int(train_rows),
        "held_out_row_count": int(held_out_rows),
    }
    train_idx, held_out_idx = apply_family_split(rows, split)
    violations = leakage_check(rows, train_idx, held_out_idx)
    split["verified_violation_count"] = int(violations)
    return split


def apply_family_split(
    rows: Sequence[dict[str, Any]], split: dict[str, Any]
) -> tuple[list[int], list[int]]:
    """Resolve a split document to (train row indices, held-out row indices)."""
    held_out = set(split["held_out_families"])
    train_idx: list[int] = []
    held_out_idx: list[int] = []
    for index, row in enumerate(rows):
        family = derive_asset_family(str(row["normalized_asset_path"]))
        (held_out_idx if family in held_out else train_idx).append(index)
    return train_idx, held_out_idx


def leakage_check(
    rows: Sequence[dict[str, Any]],
    train_idx: Sequence[int],
    held_out_idx: Sequence[int],
) -> int:
    """Count near-duplicate variant stems that straddle the split (must be 0 — FR-004).

    A violation is one variant stem (filename minus a numeric suffix) that appears on BOTH
    sides of the split, meaning numbered near-duplicates leaked across train/held-out.
    """
    sides: dict[str, set[str]] = {}
    for side, indices in (("train", train_idx), ("held_out", held_out_idx)):
        for index in indices:
            path = str(rows[index]["normalized_asset_path"])
            key = f"{derive_asset_family(path)}::{variant_stem(path)}"
            sides.setdefault(key, set()).add(side)
    return sum(1 for key_sides in sides.values() if len(key_sides) > 1)


def load_split(path: Path) -> dict[str, Any]:
    """Load and minimally validate a split document; refuse a leaky one (FR-004)."""
    if not Path(path).is_file():
        raise SplitError(f"{path}: split file does not exist (build it with spec119_build_split.py)")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SPLIT_SCHEMA:
        raise SplitError(f"{path}: schema must be {SPLIT_SCHEMA!r}, got {doc.get('schema')!r}")
    violations = int(doc.get("verified_violation_count", -1))
    if violations != 0:
        raise SplitError(
            f"{path}: split has verified_violation_count={violations}; a leaky split is an "
            "error, not a warning (FR-004) — rebuild it with spec119_build_split.py"
        )
    return doc


def read_asset_rows(store: Path) -> list[dict[str, Any]]:
    """Read ``assets.parquet``; only ``captured`` rows are trainable (data-model.md)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(Path(store) / "assets.parquet").to_pylist()
    return [row for row in rows if row.get("capture_status") == "captured"]


def main() -> int:
    """CLI per contracts/cli-contract.md §1 (dry-run-first; refuses to write a leaky split)."""
    ap = argparse.ArgumentParser(
        description="Spec 119 family-isolated held-out split builder (FR-004; dry-run-first)"
    )
    ap.add_argument("--store", required=True, type=Path, help="object-library zarr (read-only)")
    ap.add_argument("--output", required=True, type=Path, help="split JSON path")
    ap.add_argument("--held-out-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true",
                    help="write the split JSON; without this flag only print the plan")
    args = ap.parse_args()

    if not (args.store / "assets.parquet").is_file():
        raise SystemExit(f"{args.store}: missing assets.parquet (not an object-library store)")
    rows = read_asset_rows(args.store)
    try:
        split = build_family_split(rows, held_out_fraction=args.held_out_fraction, seed=args.seed)
    except SplitError as exc:
        raise SystemExit(str(exc)) from exc

    summary = {
        "schema": "v119-split-plan-v1",
        "store": str(args.store.resolve()),
        "captured_rows": len(rows),
        "family_count": len(split["train_families"]) + len(split["held_out_families"]),
        "train_families": len(split["train_families"]),
        "held_out_families": len(split["held_out_families"]),
        "train_row_count": split["train_row_count"],
        "held_out_row_count": split["held_out_row_count"],
        "verified_violation_count": split["verified_violation_count"],
        "seed": split["seed"],
        "held_out_fraction": split["held_out_fraction"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if split["verified_violation_count"] > 0:
        raise SystemExit(
            "leakage check FAILED: verified_violation_count="
            f"{split['verified_violation_count']} (FR-004); refusing to write"
        )
    if not args.write:
        print("DRY RUN ONLY: add --write to write the split JSON.", flush=True)
        return 0
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; refusing to overwrite an immutable split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SPLIT_SCHEMA",
    "SplitError",
    "apply_family_split",
    "build_family_split",
    "leakage_check",
    "load_split",
    "main",
    "read_asset_rows",
]
