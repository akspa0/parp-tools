#!/usr/bin/env python3
"""Thin CLI for the Spec 119 family-isolated held-out split builder (cli-contract §1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.spec119.split import main

if __name__ == "__main__":
    raise SystemExit(main())
