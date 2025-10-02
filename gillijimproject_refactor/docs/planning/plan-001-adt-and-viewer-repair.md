# Plan 001 — LK ADT Parsing Audit and Viewer Repair

## Background
- We process Alpha/LK ADT data to build placement CSVs and viewer overlays.
- Current symptoms:
  - 0.6.0 overlays show world as (0,0,0) for many MDDF/MODF entries → suggests parse/transform failed or chunks not found.
  - `tile.html` page loads but does not render a single-tile view (only the back link works).
- We must reuse the working LK ADT parsing logic used for 0.5.x and ensure on‑disk FourCC reversal is consistently handled.

## Core Rules and Constraints
- On‑disk chunk FourCCs are reversed. Documentation and in‑memory identifiers are forward (e.g., `MDDF` in docs, but disk bytes are `FDDM`).
- Always treat FourCCs in code as forward symbols; normalize I/O to handle reversed storage during read/write.
- Do not diverge parsing logic between 0.5.x and 0.6.0 — use one implementation path with clear version toggles only when truly required.

## Hypothesized Root Causes
- MDDF/MODF not found due to FourCC detection mismatch when scanning raw ADT files.
- Name tables (MMDX/MWMO) not loaded → nameIndex deref fails, masking issues downstream.
- Coordinate mapping correct in principle, but validation never executed because no placements were ingested.
- Viewer `tile.html` appears broken because overlays are empty and/or minimap URL mismatches weren’t surfaced via diagnostics.

## Objectives
1. Unify LK ADT parsing used for 0.5.x and 0.6.0.
2. Guarantee MDDF/MODF and name tables are discovered on disk with reversed tags.
3. Validate coordinate transforms with concrete tiles.
4. Ensure viewer tile page renders a single minimap tile and shows per‑tile objects.
5. Add diagnostics and CLI tools to prevent regressions.

## Deliverables
- Working parser producing non‑zero world coordinates for 0.6.0 placements.
- `assets_*.csv` with non‑zero world_x/world_y/world_z for sampled tiles.
- `viewer/overlays/<map>/tile_r<row>_c<col>.json` containing populated `world` and `pixel` for markers.
- `tile.html` renders minimap and per‑tile objects; diff mode works if selected.
- Short docs update summarizing the FourCC rule and verification steps.

## Work Plan

### A) Parser Unification and Audit
- [ ] Standardize LK ADT read path so both 0.5.x and 0.6.0 call the same reader for MDDF/MODF/MMDX/MWMO.
- [ ] Implement robust chunk search that supports both forward and reversed byte sequences but documents that disk is reversed.
- [ ] Add structured logs: which chunks found, sizes, entry counts per ADT.

### B) Quick Validation CLI
- [ ] Add `analyze-adt-dump --adt <file> [--limit N]`:
  - Prints sizes of MMDX/MWMO name tables.
  - Prints MDDF/MODF counts and sample entries with world coords.
  - This enables spot‑checking tiles rapidly.

### C) Coordinate Transform Verification
- [ ] Select 3 ADT tiles under `test_data/0.6.0/tree/World/Maps/<map>/` and verify:
  - MDDF/MODF counts > 0.
  - For 5 samples/tile: `ComputeTileIndices(x,y)` equals the tile’s row/col and `ComputeLocalCoordinates` is within [0..1].
- [ ] If axis/handedness requires adjustment, update the mapping and re‑verify.

### D) CSV and Overlay Production
- [ ] Confirm `RangeCsvWriter.WriteAssetLedgerCsv(...)` writes non‑zero world coords for 0.6.0.
- [ ] Ensure `VersionComparisonService.BuildAssetTimelineDetailed(...)` passes through world coords unchanged.
- [ ] Ensure `OverlayBuilder.BuildOverlayJson(...)` emits `world`, `local`, and `pixel` for every point and does not zero out data.

### E) Viewer Tile Page Repair
- [ ] Confirm minimap path convention:
  - Writer: `viewer/minimap/<version>/<map>/<map>_<col>_<row>.<ext>`
  - Tile page: `state.getMinimapPath(map,row,col,version)` must match.
- [ ] Confirm overlays are loaded from `viewer/overlays/<map>/tile_r<row>_c<col>.json`.
- [ ] Add explicit error messages if minimap or overlay fetch fails in `tile.js` (console + small banner).
- [ ] Verify markers draw when overlays contain points; ensure fit/flip options don’t hide them.

### F) Diagnostics and Tests
- [ ] Add optional per‑tile diagnostics JSON: counts per type and 5 sample objects.
- [ ] Unit smoke test: Asset world coords are preserved into `AssetTimelineDetailedEntry`.
- [ ] Golden‑file sanity: For one known tile, confirm overlay contains expected object count bounds.

### G) Documentation Updates
- [ ] `COORDINATES.md`: clarify on‑disk reversed FourCC rule; keep forward names in docs/code.
- [ ] `FIXES_APPLIED.md`/`CHANGELOG_COORDINATES.md`: note parser unification, diagnostics CLI, and viewer fixes.
- [ ] `README.md`: add `analyze-adt-dump` usage and troubleshooting steps for missing tile overlays.

## Execution Order (Milestones)
1. Parser audit + CLI dump (A, B).
2. Tile validations and transform confirm (C).
3. CSV/Overlay correctness pass (D).
4. Viewer tile page reliability (E).
5. Diagnostics/tests (F) and docs (G).

## Validation & Acceptance Criteria
- On at least two 0.6.0 maps:
  - `analyze-adt-dump` shows non‑zero MDDF/MODF counts and world coords.
  - `assets_*.csv` rows contain non‑zero world columns for samples.
  - `tile.html` loads minimap and renders markers; object list populated.
- No regressions for 0.5.x maps: previous tiles still render.

## Rollback Plan
- Keep previous working build (tag/commit) as fallback.
- The new path is additive; toggles can be disabled to revert to the prior viewer if necessary.

## Notes
- We will continue referring to chunks with forward names in docs/code (MDDF/MODF/MMDX/MWMO), and explicitly state the disk representation is reversed.
- All new logs are scoped and can be turned off for release builds if needed.
