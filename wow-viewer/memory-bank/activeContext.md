# Active Context — wow-viewer

**Last updated**: 2026-06-15 | **Focus**: Spec 046 — PM4 → ADT placement restoration

## Direction
WoW viewer. Libraries bridge to Unreal Engine.

## Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## Current: 046 — PM4 Asset Matching → ADT Writing

### What works end-to-end
- `Pm4ObjectSegmentBuilder` — groups MSUR surfaces by CK24, builds typed segments
- `Pm4AssetMatchScorer` — scores segments against asset references (shape, TypeFlags, footprint)
- `Pm4MatchSupport.Run` — full PM4→placement matching pipeline (M2 + WMO)
- `Pm4AdtWriter` — converts match results to `LkAdtData` for ADT writing
- `LkAdtWriter` — writes complete LK ADT files (MVER/MHDR/MCIN/MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF/MCNK)
- CLI: `pm4 write-adt --input <pm4> --archive-root <client> --placements <obj0.adt> --output <out.adt>`
- Tested: 10 M2 + 15 WMO placements written to valid ADT (verified with `map inspect`)

### Key data paths
- PM4 test data: `gillijimproject_refactor/test_data/NOT THE RIGHT FOLDER/World/Maps/development/`
- _obj0.adt placements: same directory
- Staged clients: `output/tmp/wowarchive-clients/`
- User guide: `docs/PM4-ADT-RESTORATION.md`

## That's it
Everything else (001, 029, 030/031/032, 038/040, 042, 045, 049, 053, 055, 056, 057) is not started or research only.
