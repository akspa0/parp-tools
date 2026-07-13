"""CLI for the minimal identity-checked Spec 102 numeric store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.spec102.numeric_store import SELECTION_MODES, build_numeric_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Spec 102 numeric-only store")
    parser.add_argument("--selection-store", required=True, type=Path)
    parser.add_argument("--v18-store", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="curated_selection",
        help=(
            "curated_selection requires explicit v18_row provenance; raw_v18_identity copies one "
            "single-build V18 store only after exact ordinal identity checks"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only a partial raw_v18_identity output with a matching hash-bound progress record",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="copy at most this many rows, write a resume checkpoint, and exit cleanly",
    )
    args = parser.parse_args()
    output = build_numeric_store(
        selection_store=args.selection_store,
        v18_stores=args.v18_store,
        output=args.output,
        selection_mode=args.selection_mode,
        resume=args.resume,
        max_rows=args.max_rows,
    )
    progress_path = output / "numeric_build_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    if (output / "contract.json").is_file():
        print(f"Spec 102 numeric store ready: {output}")
    else:
        print(json.dumps({
            "status": "partial",
            "output": str(output.resolve()),
            "next_row": progress.get("next_row"),
            "total_rows": progress.get("total_rows"),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
