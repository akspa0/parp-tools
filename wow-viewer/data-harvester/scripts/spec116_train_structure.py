#!/usr/bin/env python3
"""Spec 116 US3 CLI: per-slot structure classifier training (dry-run by default).

Thin wrapper around ``harvester.spec116.structure_train.main``. Prints a training plan
(architecture, split counts, class weights, time/memory estimate) and exits. Add
``--confirm-run`` to launch user-owned CUDA training (FR-018).

Run from ``wow-viewer/data-harvester/``::

    uv run python scripts/spec116_train_structure.py \\
        --store <v50 curriculum store> \\
        --split <held-out split directory> \\
        --dumps <texture-name dump json> \\
        --vocabulary <US1 family_slot_consistency report json> \\
        --output <run output directory> \\
        --slot 1 \\
        [--epochs 60] [--batch 16] [--base 32] [--lr 3e-4] \\
        [--device cuda] [--confirm-run]
"""

from __future__ import annotations

import sys

from harvester.spec116.structure_train import main

if __name__ == "__main__":
    sys.exit(main())
