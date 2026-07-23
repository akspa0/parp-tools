#!/usr/bin/env python3
"""Thin CLI for the Spec 119 object-library classifier trainer (cli-contract §2)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.spec119.classifier_train import main

if __name__ == "__main__":
    raise SystemExit(main())
