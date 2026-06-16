# Active Context — wow-viewer

**Last updated**: 2026-06-16 | **Focus**: Spec 046 — PM4 match report (ADT writing removed)

## Direction
WoW viewer. Libraries bridge to Unreal Engine.

## Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## Current: 046 — PM4 Asset Matching → Match Reports

### What works end-to-end
- `Pm4ObjectSegmentBuilder` — groups MSUR surfaces by CK24, builds typed segments
- `Pm4AssetMatchScorer` — scores segments against asset references (shape, TypeFlags, footprint)
- `Pm4MatchSupport.Run` — full PM4→placement matching pipeline (M2 + WMO)
- CLI: `pm4 match-report --input <pm4> --archive-root <client> --placements <obj0.adt> --output <report.md>`

### What was removed (2026-06-16)
- `Pm4AdtWriter`, `Pm4BinaryAdtPatcher`, `Pm4AdtM2Placement`, `Pm4AdtWmoPlacement` — all deleted
- `pm4 write-adt` CLI command — replaced by `pm4 match-report`
- `docs/PM4-ADT-RESTORATION.md` — deleted
- ADT writing is out of scope for the PM4 matcher. `LkAdtWriter` is untouched.

### Key data paths
- PM4 test data: `gillijimproject_refactor/test_data/NOT THE RIGHT FOLDER/World/Maps/development/`
- _obj0.adt placements: same directory
- Staged clients: `output/tmp/wowarchive-clients/`

## That's it
Everything else (001, 029, 030/031/032, 038/040, 042, 045, 049, 053, 055, 056, 057) is not started or research only.