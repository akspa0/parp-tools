#!/usr/bin/env python
"""Thin user-owned CLI for Spec 114 universal relief training."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.universal_relief_train import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
