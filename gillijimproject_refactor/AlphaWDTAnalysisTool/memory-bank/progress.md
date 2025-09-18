# Progress — AlphaWDTAnalysisTool

## What works
- Exports WotLK WDT/ADT files with high fidelity.
- Strict CSV-only AreaID patching using DBCTool.V2 `compare/v2/` outputs.
- Per-tile verify CSVs in verbose mode confirm what was written and why.
- Asset fixups (textures/models) with capacity-aware in-place patching and logged diagnostics.

## What’s left
- DeadminesInstance: some `src_areaNumber` remain 0 due to missing per-map rows. Requires DBCTool.V2 CSV updates.
- Optional diagnostic: map each `patch_csv_num` hit back to its source CSV file (for auditing).

## Known issues
- Compound LK names (e.g., `western plaguelands: hearthglen`) may cause non-matches. Resolution is expected via DBCTool.V2 aliases/fuzzy or explicit per-map rows.

## Current status
- Tooling is stable in strict CSV-only mode; documentation and memory bank updated to reflect current behavior.
