#!/usr/bin/env python3
"""Spec 121 Stage A flatline diagnostic (read-only; CPU by default).

Answers "train loss drops, val flat — WHY?" with train-vs-val MAE, per-map val breakdown, and
prediction-variance collapse metrics. See harvester/spec121/diagnose.py.

Usage (from wow-viewer/data-harvester):
    uv run python scripts/spec121_diagnose_stage_a.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec121.diagnose import main

if __name__ == "__main__":
    raise SystemExit(main())
