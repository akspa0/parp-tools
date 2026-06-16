"""CLI: Import C# PM4 asset corpus export into a Zarr store.

Usage::

    uv run python scripts/pm4_import_asset_corpus.py --input corpus.json --output store.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harvester.pm4_asset_matching import (
    import_asset_corpus,
    write_asset_references_zarr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import PM4 asset corpus to Zarr")
    parser.add_argument("--input", "-i", required=True, help="C# asset corpus JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output Zarr store path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing store")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    assets = import_asset_corpus(input_path)
    write_asset_references_zarr(output_path, assets, overwrite=args.overwrite)
    print(f"Wrote {len(assets)} asset references to {output_path}")


if __name__ == "__main__":
    main()
