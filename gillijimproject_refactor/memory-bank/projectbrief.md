# Project Brief

## Mission
Build a standalone .NET 9 console toolchain ('AlphaWdtInspector' and a dedicated LK→Alpha converter) to diagnose and fix LK ADT → Alpha WDT packing issues (MCRF destination re-bucketing, MCLQ from MH2O, MCSE extraction) and provide a canonical LK→Alpha WDT writer independent of WoWRollback, mirroring AlphaWDTAnalysisTool semantics.

## Objectives
- Implement CLI commands: `export-lk`, `pack-alpha` [--no-coord-xform] [--dest-rebucket] [--emit-mclq] [--scan-mcse], `tile-diff`, `dump-adt`.
- Emit CSVs: `mcnk_sizes.csv`, `placements_mddf.csv`, `placements_modf.csv`, `mcrf_diag.csv`.
- Keep the tool self-contained (no WoWRollback dependencies) and deterministic.
- Provide a minimal standalone LK→Alpha converter that uses AlphaWDTAnalysisTool crosswalk CSVs in reverse (3.3.5→0.5.3), with no asset gating and full MDDF/MODF/WMO UID preservation.

## Success Criteria
- `pack-alpha` produces Alpha WDTs with non-zero MCLQ/MCSE where present and populated MCRF on busy tiles (non-zero refs).
- `tile-diff` shows expected subchunk presence/size/offset parity; raw subchunks are dumped for inspection.
- `dump-adt` confirms MH2O and MDDF/MODF presence on source LK ADTs.
- `export-lk` and `pack-alpha` paths are deterministic and reproducible.
