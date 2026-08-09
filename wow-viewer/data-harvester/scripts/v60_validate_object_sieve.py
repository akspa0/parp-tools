from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v60.object_sieve import validate_object_sieve_corpus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the v60 synthetic object-sieve corpus.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)
    report = validate_object_sieve_corpus(args.corpus)
    if args.write_report:
        (args.corpus / "object_sieve_validation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
