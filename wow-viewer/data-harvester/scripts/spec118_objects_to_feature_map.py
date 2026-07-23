#!/usr/bin/env python3
"""Thin CLI for the Spec 118 object feature bridge (mirrors spec117_lattice_to_feature_map.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.spec118.object_feature_bridge import main

if __name__ == "__main__":
    raise SystemExit(main())
