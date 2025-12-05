# Product Context

## Users
- Technical artists and engineers maintaining Alpha-to-Lich King world data conversions.

## Problem
- LK ADT → Alpha WDT packing loses data: missing MCLQ (MH2O not detected), missing MCSE, and empty MCRF on busy tiles due to source-tile attribution without destination re-bucketing.

## Desired Experience
- A minimal, deterministic CLI toolchain to:
  - Export LK ADTs from an original Alpha WDT (`export-lk`).
  - Pack Alpha WDTs from LK ADTs (`pack-alpha`) with toggles to isolate issues (`--no-coord-xform`, `--dest-rebucket`, `--emit-mclq`, `--scan-mcse`).
  - Compare tiles between original and generated WDTs (`tile-diff`).
  - Inspect LK ADT subchunks quickly (`dump-adt`).
  - Run a dedicated standalone LK→Alpha converter that consumes those LK ADTs plus AlphaWDTAnalysisTool crosswalk CSVs in reverse (3.3.5→0.5.3), with no asset gating and full MDDF/MODF/WMO UID preservation.
- Consistent CSV diagnostics for sizes and placements to remove guesswork.
