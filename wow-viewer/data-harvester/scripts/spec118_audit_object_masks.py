#!/usr/bin/env python3
"""Thin CLI for Spec 118 US1's object-mask audit (mirrors the Spec 117 script pattern)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.spec118.object_mask_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
