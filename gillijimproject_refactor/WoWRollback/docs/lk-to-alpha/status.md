# LK → Alpha Status and Roadmap

This document summarizes current capabilities for converting LK ADT data back into Alpha WDT/ADT structures, and the planned roadmap.

## Implemented
- Liquids (MH2O)
  - Extraction from LK ADTs and rehydration into Alpha structures.
- Placements
  - MMDX/MMID (M2 filename/offset tables)
  - MWMO/MWID (WMO filename/offset tables)
  - MDDF (M2 placements)
  - MODF (WMO placements)
- Builders
  - LkAdtBuilder writes MH2O and FDDM/FDOM (format-bridge chunks used internally for data flow).
- Converters
  - PlacementConverter rehydrates Alpha data from LK sources.

## Partially Implemented / In Progress
- Terrain parity (heightmaps, normals, layers) — tracked in converter modules; parity checks wiring TBD.
- Integrity validation utilities (roundtrip)
  - `AdtLk.ValidateIntegrity()` planned to verify MHDR/MCIN offsets and overall layout.
  - CLI planned for byte-level compare vs legacy exporter (focus: MCLQ offsets/sizes parity).

## CLI and Examples
- Patch LK ADTs using Alpha terrain logic (bury, holes, shadows):
```powershell
# Patch LK ADTs in place to iterate on results
dotnet run --project WoWRollback.Cli -- \
  lk-to-alpha \
  --lk-adts-dir .\wrb_out\lk_adts\World\Maps\Azeroth \
  --map Azeroth \
  --max-uniqueid 43000 \
  --fix-holes --disable-mcsh \
  --out .\patched_lk_az
```

- Standalone terrain converter (WoWRollback.AdtConverter): see `WoWRollback.AdtConverter/README.md` for commands like `pack-monolithic`, `inspect-alpha`, and `convert-map-terrain`.

## Roadmap
- Roundtrip suite:
  - `--adt-lk`, `--out`, `--compare`, `--report-mclq` to validate offsets/sizes and byte-equivalence.
- Expand terrain parity coverage and tests.
- Export/Import MFBO/MTXF when available.

## References
- LkToAlphaModule tests and builders (source) for liquid and placement workflows.
