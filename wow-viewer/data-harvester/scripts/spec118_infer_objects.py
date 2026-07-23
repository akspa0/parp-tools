#!/usr/bin/env python3
"""Thin CLI for Spec 118 visible-object inference (mirrors spec116_infer_structure.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.spec118.object_segment_infer import main

if __name__ == "__main__":
    raise SystemExit(main())
