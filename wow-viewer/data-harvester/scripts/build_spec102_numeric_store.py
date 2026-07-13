"""CLI for the minimal identity-checked Spec 102 numeric store."""

from __future__ import annotations

import argparse
from pathlib import Path

from harvester.spec102.numeric_store import build_numeric_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Spec 102 numeric-only store")
    parser.add_argument("--selection-store", required=True, type=Path)
    parser.add_argument("--v18-store", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = build_numeric_store(
        selection_store=args.selection_store,
        v18_stores=args.v18_store,
        output=args.output,
    )
    print(f"Spec 102 numeric store ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
