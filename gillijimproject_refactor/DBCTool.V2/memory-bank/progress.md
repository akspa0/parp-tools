# Progress — DBCTool.V2

## What works
- Deterministic, map-locked AreaTable crosswalks from Alpha/Classic to 3.3.5.
- Clear `match_method` tagging for exact, rename, and fuzzy resolutions.
- Per-map patch CSVs and fallback/unmatched streams.
- Dumps and diagnostics (source and target) to aid investigations.

## What’s left
- Add/curate aliases for compounded LK names (e.g., `western plaguelands: hearthglen`) where safe.
- Golden-file tests for representative maps to lock CSV schema and content.
- Improve `zone_missing_in_target_map_*` diagnostics.

## Known issues
- Early content duplication causes expected 0↔1 churn; filtered in anomalies.
- Some in-instance numerics (e.g., DeadminesInstance `196608`) may need explicit per-map rows when strict matching yields 0.

## Current status
- Readme/docs updated; outputs stable. Consumers (AlphaWDTAnalysisTool) rely exclusively on per-map patch CSVs.
