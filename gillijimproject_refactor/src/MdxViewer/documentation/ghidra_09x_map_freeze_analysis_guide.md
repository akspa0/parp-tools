# Ghidra Investigation Guide: WoW 0.9.x Map-Load Freeze

## Goal
Find the exact native client logic in WoW 0.9.x (focus build 0.9.1.3810) that explains why map loading freezes in our viewer path, then produce implementation constraints we can mirror in `StandardTerrainAdapter` / streaming code.

This guide is designed for handing to another LLM running Ghidra analysis.

---

## Known Symptom (Viewer)
When loading some 0.9.x maps, the app appears to hard-freeze during map load.

Primary C# code paths to correlate with native behavior:
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - `LoadTileWithPlacements(...)`
  - `ParseAdt(...)`
  - `ParseMh2o(...)`
  - `CollectPlacementsViaMhdr(...)`
- `src/MdxViewer/Terrain/TerrainManager.cs`
  - `UpdateAOI(...)`
  - `SubmitPendingTiles(...)`

---

## High-Probability Root Cause Buckets
1. **Chunk offset/size trust issue**
   - Invalid or version-specific offsets cause oversized loops, repeated scans, or pathological allocations.
2. **Wrong feature gating by build/version**
   - Parsing chunk families not valid for 0.9.x in hot path (or missing required 0.9.x-specific handling).
3. **Tile/object loading state machine mismatch**
   - Client may defer or skip specific sections in 0.9.x that our parser assumes mandatory.
4. **Placement parsing over malformed tables**
   - MMID/MMDX/MDDF/MODF index interpretation mismatch can trigger expensive failure paths.
5. **Render-thread saturation from load pattern**
   - Native client may stage/limit work differently than our per-tile parse + upload approach.

---

## Inputs for Analysis
- WoW 0.9.x binary (`WoW.exe`, ideally 0.9.1.3810)
- ADT/WDT samples that freeze in viewer
- Existing reference notes:
  - `specifications/0.9.1/Windows_EXE_vs_macOS_0.9.1.3810_Ghidra_Verification.md`

---

## Investigation Plan (Ghidra)

### Phase 1 — Locate map/tile load entrypoints
Search strings:
- `"MapChunk.cpp"`
- `"MapObjRead.cpp"`
- `"MCNK"`, `"MCLQ"`, `"MHDR"`, `"MCIN"`, `"MDDF"`, `"MODF"`, `"MMID"`, `"MMDX"`
- `"World\\Maps\\"`

Actions:
1. Identify top-level world/map loading function chain.
2. Identify function boundaries for:
   - WDT parsing
   - ADT root parsing
   - per-chunk MCNK setup
   - object placement table parsing
3. Name all key functions in Ghidra for call-graph readability.

Expected output:
- Call graph from map-open -> tile parse -> chunk parse -> placement build.

### Phase 2 — Verify ADT parse invariants for 0.9.x
For each target function, extract:
- Required chunks and strict order (if any)
- Offset origin rules (absolute vs MHDR-relative)
- Bounds checks before dereference
- Failure behavior: continue, skip chunk, or abort tile

Focus checks:
- `MCIN` entry count and bounds validation
- `MCNK` size validation before any field reads
- `MCLQ` optional/required behavior in 0.9.x
- Any guard on `MH2O` (likely absent for this era)

Expected output:
- Pseudocode for native safety checks with exact constants/offsets.

### Phase 3 — Placement table semantics
Trace native handling of:
- `MMDX` string block
- `MMID` indirection table
- `MDDF` records
- `MWMO`/`MODF` records

Verify:
- index units (byte offsets vs element indices)
- record sizes
- any dedupe/validation logic before instance creation
- behavior on invalid name index

Expected output:
- Exact mapping chain pseudocode and rejection conditions.

### Phase 4 — Freeze-risk loops
Find loops where iteration bounds depend on file data.
For each, capture:
- loop bound source
- max sanity clamp (if present)
- break conditions
- fallback/skip paths

Prioritize:
- chunk scan loops
- subchunk scan loops
- placement iteration loops
- optional chunk dispatch loops

Expected output:
- List of loops that can become pathological if unbounded; include mitigations used by native code.

### Phase 5 — Loader state machine + throttling
Locate native logic that gates work per frame/tick during loading.
Look for:
- staged loading percentages
- cooperative yielding
- tile queue prioritization
- max in-flight work

Expected output:
- Native work-scheduling strategy that avoids hard stalls.

---

## What to Hand Back (Required Deliverables)
1. **Named function map** (address + inferred role).
2. **Validated pseudocode** for ADT/MCNK/placement parsing with all guard conditions.
3. **Mismatch table**: native behavior vs current C# behavior.
4. **Top 3 freeze candidates** with confidence scores.
5. **Patch guidance**: minimal C# changes to align parser/streaming behavior.

---

## Comparison Checklist Against Current C#
Use this exact checklist while reviewing output:

- [ ] Are all chunk offsets bounds-checked before access?
- [ ] Is MCNK chunk count strictly capped at 256?
- [ ] Are chunk sizes validated against remaining file bytes?
- [ ] Are optional chunks only parsed when era-appropriate?
- [ ] Are placement indices validated before name-table dereference?
- [ ] Are failure paths cheap (skip) instead of expensive (retry/scan)?
- [ ] Is per-frame work bounded similarly to native scheduling?

---

## Suggested Prompt for the Other LLM
"Analyze WoW 0.9.1.3810 in Ghidra to identify map-load freeze risks and produce native-verified parser invariants for ADT/MCNK and placement chunks (MMDX/MMID/MDDF/MODF). Focus on bounds checks, optional chunk gating, and loop clamps. Compare findings against this C# pipeline: `StandardTerrainAdapter.LoadTileWithPlacements -> ParseAdt -> ParseMh2o/CollectPlacementsViaMhdr` and list minimal fixes with file+line targets."

---

## Notes
- In this era, treat `MH2O` assumptions with caution; prioritize what binary usage proves.
- Native code usually contains strict guard rails; mirror those guard rails before adding heavy fallback behavior.
