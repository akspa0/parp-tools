"""Canonical v50 entry point for residual-extractor training (USER runs CUDA)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.residual_extractor_train import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
