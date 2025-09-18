# Active Context — DBCTool.V2

## Current focus
- Produce deterministic, map-locked crosswalks from Alpha/Classic to 3.3.5 with clear `match_method` tagging.
- Keep CSV schema stable across sessions; per-map crosswalks are the authoritative inputs for consumers.
- Document edge cases where LK names compound parent + child (e.g., `western plaguelands: hearthglen`).

## Recent changes
- README updated with compound zone string guidance and guarantees around map-lock + unique-only fallbacks.
- `docs/alpha-area-decode-v2.md` expanded: ADT v18 packing reference, validation rules, chain construction, outputs, and consumption guidance.

## Next steps
- Evaluate alias expansions to normalize compounded LK names when safe.
- Add golden-file tests for representative maps to catch regressions.
- Tighten diagnostics for `zone_missing_in_target_map_*` to aid investigation.

## Coordination with consumers
- AlphaWDTAnalysisTool now consumes only per-map numeric crosswalks; no heuristics on their side.
- Ensure crosswalks provide non‑zero `tgt_areaID` for known in-instance values (e.g., DeadminesInstance `196608`) where appropriate.
