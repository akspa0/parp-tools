"""Create the explicit 3.3.5-only M0 build-local split from frozen curation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from harvester.spec102.m0_scope import build_m0_build_local_manifest
from harvester.spec102.signal_audit import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Spec 102 3.3.5-only M0 split")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-map", default="Northrend")
    parser.add_argument("--test-map", default="Kalimdor")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite split manifest: {args.output}")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_index_path = args.store / "index.parquet"
    source_index = pq.read_table(source_index_path).to_pylist()
    manifest = build_m0_build_local_manifest(
        source_manifest,
        source_index=source_index,
        validation_map=args.validation_map,
        test_map=args.test_map,
        source_manifest_path=str(args.source_manifest.resolve()),
        source_manifest_sha256=sha256_file(args.source_manifest),
        source_store=str(args.store.resolve()),
        source_index_sha256=sha256_file(source_index_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "schema": manifest["schema"],
        "scope": manifest["m0_training_scope"],
        "counts": manifest["counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
