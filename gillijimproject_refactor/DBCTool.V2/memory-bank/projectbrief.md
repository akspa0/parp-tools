# DBCTool.V2 — Project Brief

Goal: Provide a deterministic, reusable mapping from Alpha-era AreaTable (0.5.x/0.6.0) area numbers (zone<<16|sub) to WotLK 3.3.5 AreaTable IDs, and generate auditable CSVs and patch files.

Scope:
- Map-lock to the source row’s ContinentID (via Map.dbc crosswalk).
- Zone-only matching, then optional 0.6.0 pivot, then rename/fuzzy fallbacks (unique-only).
- Parent-agnostic patch emission; fallback0 files for unmatched.
- Programmatic API (AreaIdMapperV2) for other tools to consume.

Non-goals:
- Editing or patching DBCs in-place.
- UI world map changes or cross-continent reassignments by default.

Deliverables:
- CLI CSVs under `compare/` and `compare/v2/`.
- API: `DBCTool.V2.Core/AreaIdMapperV2`.
- Documentation and memory bank.
