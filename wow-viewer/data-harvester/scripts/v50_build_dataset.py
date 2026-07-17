"""Fail-closed V50 dataset command until the clean-room store owner exists.

The former wrapper delegated to Spec 108's mixed-copy builder, which can stamp
historical rows as V50 without V50 provenance, row lineage, or liquid-source
proof. Do not restore that delegation: Spec 109's future canonical store builder
owns this command once it has fixture-backed migration and fresh-extraction paths.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V50 complete-store build (not available until the clean-room owner is implemented)."
    )
    parser.parse_args(argv)
    print(
        "V50 build is intentionally unavailable: the legacy Spec 108 mixed builder cannot create "
        "a clean-room V50 store. Implement Spec 109's verified migration and client-backed "
        "extraction path first."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
