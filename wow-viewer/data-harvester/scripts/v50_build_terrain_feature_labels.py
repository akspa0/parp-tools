"""Spec 115 entry point: derive terrain-feature labels for a v50 curriculum (dry run by default)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.terrain_feature_label_build import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
