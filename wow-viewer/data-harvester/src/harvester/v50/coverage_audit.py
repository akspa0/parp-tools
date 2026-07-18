"""Per-signal coverage audit for rebuilt v50 stores (Spec 112 T010, SC-001/SC-002).

Full-store scan, never sampled — the 2026-07-18 discovery pass used 50-row samples, which was fine
for finding gaps but is not proof of their absence. The report conforms to
``specs/112-v50-height-model/contracts/coverage-audit-report.schema.json``:

- every signal the store manifest declares is measured row-by-row (a row counts as populated when
  any element is nonzero);
- a declared signal at 0% population with no recorded unavailability reason is the failing
  ``zero_coverage_unexplained`` state SC-001 forbids;
- signals the manifest records as unavailable are classified by their reason-prefix vocabulary
  (``era_unavailable:`` -> ``era_unavailable``; anything else -> ``no_source_data_expected``);
- the ``minimap_resolution_parity`` block lists exactly which rows have a 256px minimap but no
  1024px counterpart (and vice versa) — SC-002 requires both lists empty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from harvester.v50.contracts import REASON_ERA_UNAVAILABLE, classify_reason

AUDIT_SCHEMA = "v112-coverage-audit-v1"


def _populated_rows(array, row_count: int, batch: int = 64) -> list[int]:
    populated: list[int] = []
    for start in range(0, row_count, batch):
        stop = min(row_count, start + batch)
        block = np.asarray(array[start:stop])
        flat = block.reshape(block.shape[0], -1)
        for offset, row_any in enumerate(flat.any(axis=1)):
            if row_any:
                populated.append(start + offset)
    return populated


def audit_store(store_path: Path) -> dict:
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store_path), mode="r")
    attrs = dict(group.attrs)
    declared_signals = list(attrs.get("signals", []))
    unavailable = {u["name"]: str(u["reason"]) for u in attrs.get("unavailable_signals", [])}
    row_count = int(attrs.get("row_count", 0))
    if row_count <= 0:
        raise ValueError(f"store {store_path} declares row_count={row_count}; nothing to audit")

    index = pq.read_table(store_path / "index.parquet").to_pylist()
    maps = sorted({str(row.get("map", "")) for row in index})
    if len(maps) != 1:
        raise ValueError(f"expected a single-map per-build store, found maps={maps}")

    signal_reports: list[dict] = []
    populated_by_signal: dict[str, list[int]] = {}

    for signal in declared_signals:
        name = str(signal["name"])
        if name not in group:
            signal_reports.append(
                {
                    "name": name,
                    "declared": True,
                    "populated_rows": 0,
                    "population_fraction": 0.0,
                    "status": "zero_coverage_unexplained",
                    "unavailable_reason_sample": None,
                }
            )
            continue
        rows = _populated_rows(group[name], row_count)
        populated_by_signal[name] = rows
        if rows:
            status = "populated"
        elif name in unavailable:
            status = "era_unavailable" if classify_reason(unavailable[name]) == REASON_ERA_UNAVAILABLE else "no_source_data_expected"
        else:
            status = "zero_coverage_unexplained"
        signal_reports.append(
            {
                "name": name,
                "declared": True,
                "populated_rows": len(rows),
                "population_fraction": len(rows) / row_count,
                "status": status,
                "unavailable_reason_sample": unavailable.get(name),
            }
        )

    for name, reason in unavailable.items():
        if any(entry["name"] == name for entry in signal_reports):
            continue
        signal_reports.append(
            {
                "name": name,
                "declared": False,
                "populated_rows": 0,
                "population_fraction": 0.0,
                "status": "era_unavailable" if classify_reason(reason) == REASON_ERA_UNAVAILABLE else "no_source_data_expected",
                "unavailable_reason_sample": reason,
            }
        )

    rows_256 = set(populated_by_signal.get("minimap_rgb", []))
    rows_1024 = set(populated_by_signal.get("minimap_rgb_1024", []))
    only_256 = sorted(rows_256 - rows_1024)
    only_1024 = sorted(rows_1024 - rows_256)

    return {
        "schema": AUDIT_SCHEMA,
        "store": str(store_path.resolve()),
        "build_id": str(attrs.get("build_id", "")),
        "map": maps[0],
        "row_count": row_count,
        "signals": signal_reports,
        "minimap_resolution_parity": {
            "minimap_rgb_rows": len(rows_256),
            "minimap_rgb_1024_rows": len(rows_1024),
            "rows_only_in_256": only_256,
            "rows_only_in_1024": only_1024,
            "parity": not only_256 and not only_1024,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-store per-signal coverage audit (Spec 112 SC-001/SC-002)")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    report = audit_store(args.store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    unexplained = [entry["name"] for entry in report["signals"] if entry["status"] == "zero_coverage_unexplained"]
    parity = report["minimap_resolution_parity"]["parity"]
    print(f"coverage audit: {report['map']} rows={report['row_count']} -> {args.output}")
    for entry in report["signals"]:
        print(f"  {entry['name']:28s} {entry['population_fraction']:6.2%}  {entry['status']}")
    print(f"  minimap 256/1024 parity: {parity}")
    if unexplained:
        print(f"FAIL (SC-001): unexplained zero-coverage signals: {unexplained}")
    if not parity:
        print("FAIL (SC-002): 256px and 1024px minimap row sets differ")
    return 0 if not unexplained and parity else 1
