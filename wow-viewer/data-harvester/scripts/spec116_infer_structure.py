#!/usr/bin/env python3
"""Spec 116 US3 CLI: structure inference from minimap (legality resolver + OOD audit).

Thin wrapper around ``harvester.spec116.structure_infer.main``. Runs a frozen
``StructureSlotNet`` and emits a ``v50-structure-infer-v1`` audit record. Two mutually
exclusive input modes:

Batch mode, over an existing v50 store (resolves legality per tile via its own texture-name
dump)::

    uv run python scripts/spec116_infer_structure.py \\
        --checkpoint <checkpoint_best.pt> \\
        --store <v50 curriculum store> \\
        --dumps <texture-name dump json> \\
        --slot 1 \\
        --output <audit.json> [--write]

Loose-image mode, including a hand-painted OOD tile with no store backing (spec US3 acceptance
4) -- omit ``--tile-table`` for the OOD path::

    uv run python scripts/spec116_infer_structure.py \\
        --checkpoint <checkpoint_best.pt> \\
        --inputs <tile.png> <hand-painted.png> \\
        [--tile-table <MTEX table json>] \\
        --slot 1 \\
        --output <audit.json> [--write]
"""

from __future__ import annotations

import sys

from harvester.spec116.structure_infer import main

if __name__ == "__main__":
    sys.exit(main())
