"""Canonical CLI entry point for v50 datastore visual review."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.store_visual_review import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
