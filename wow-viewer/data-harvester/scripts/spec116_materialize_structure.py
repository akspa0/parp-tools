#!/usr/bin/env python3
"""Spec 116 US5 CLI: materialize predicted structure into a derived store (dry-run by default).

Thin wrapper around ``harvester.spec116.structure_materialize.main``. Runs a frozen
``StructureSlotNet`` checkpoint over a v50 store and writes a derived structure store
(``structure_family``, ``structure_confidence``, ``structure_legal``) row-aligned with the
source curriculum.

Run from ``wow-viewer/data-harvester/``::

    uv run python scripts/spec116_materialize_structure.py \\
        --store <v50 curriculum store> \\
        --checkpoint <checkpoint_best.pt> \\
        --output <derived structure store> \\
        --dumps <texture-name dump json> \\
        --slot 1 \\
        [--write]
"""

from __future__ import annotations

import sys

from harvester.spec116.structure_materialize import main

if __name__ == "__main__":
    sys.exit(main())
