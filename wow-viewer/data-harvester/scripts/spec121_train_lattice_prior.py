#!/usr/bin/env python3
"""Spec 121 Stage A CLI: minimap RGB -> 545-point WDL lattice prior.

Thin wrapper over ``harvester.spec121.lattice_backbone_train``. Dry-run-first: without
``--confirm-run`` it prints the validated plan and exits. All training is user-run (RULE 0).

Usage (from wow-viewer/data-harvester):
    uv run python scripts/spec121_train_lattice_prior.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec121.lattice_backbone_train import main

if __name__ == "__main__":
    raise SystemExit(main())
