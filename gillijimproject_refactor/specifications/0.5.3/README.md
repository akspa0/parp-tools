# WoW 0.5.3.3368 Chunk Research (Ghidra)

This folder now contains a **0.7.0-style deep-dive chunk tree** for 0.5.3 under `chunks/` plus the earlier first-pass notes.

## Primary output
- `chunks/ADT/`
- `chunks/WDT/`
- `chunks/WMO/`
- `chunks/MDX/`
- `NETWORK_VIEWER_OPCODE_PLAN.md`
- `NETWORK_OPCODE_DEEP_DIVE.md`
- `network_opcode_inventory.json`
- `NETWORK_OPCODE_BEHAVIOR_REFERENCE.md`
- `network_server_smsg_inventory.json`
- `NETWORK_LAYER_IMPLEMENTATION_MATRIX.md`
- `network_opcode_tracker.json`
- `network_opcode_tracker.csv`
- `SQL_WORLD_POPULATION_PLAN.md`

## What changed in the deep dive
- Promoted 0.5.3 docs from token-only notes to per-chunk pages with confidence and structure sections.
- Captured 0.5.3 as a **monolithic WDT↔ADT parse domain** in `chunks/WDT/MONOLITHIC_LAYOUT.md`.
- Added broader optional WMO group chunk coverage (`MOBN/MOBR/MOLR/MODR/MOCV/MPB*`).

## Evidence basis
- Parser/assertion token strings in the loaded 0.5.3 binary (Ghidra).
- MDX section-type symbols and `MDLX` magic check (`'XLDM'` LE compare).

## Constraint reminder
- Without full xref-driven decompile traversal in this pass, unknown field-level details remain explicitly marked `???`.

## Next pass targets
- Recover exact parser loop bodies for `MPHD`, `MARE`, `MAOF`, and `MCNK` subchunk offsets.
- Confirm `MCLQ`/`MCVT`/`MCNR`/`MCAL` in 0.5.3 with direct decompile + hex correlation.
