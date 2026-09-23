# Tasks: 046 — PM4 Asset Matching

**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)

## Phase 1: Data Models and JSON Import ✅

- [x] T001: Create `data-harvester/src/harvester/pm4_asset_matching/models.py`
- [x] T002: Create `data-harvester/src/harvester/pm4_asset_matching/json_import.py`
- [x] T003: Create `data-harvester/src/harvester/pm4_asset_matching/__init__.py`
- [x] T004: Create `data-harvester/src/harvester/test_pm4_asset_matching.py`

## Phase 2: Zarr Signal Store ✅

- [x] T005: Create `signal_store.py` — Zarr v3 read/write
- [x] T006: Add store metadata attrs
- [x] T007: Test write/read round-trip

## Phase 3: Python Scorer ✅

- [x] T008: Create `scorer.py` — port Pm4AssetMatchScorer
- [x] T009: Implement TypeFlags profile matching
- [x] T010: Test scorer against C# match report JSON

## Phase 4: Placement Synthesizer ✅

- [x] T011: Create `placement_synthesizer.py` — port Pm4ReplacementPlacementSynthesizer
- [x] T012: Test proposal synthesis against C# proposal JSON

## Phase 5: Validation Script and Known-Tile Proof ✅

- [x] T013: Create `pm4_import_segment_signals.py`
- [x] T014: Create `pm4_import_asset_corpus.py`
- [x] T015: Create `pm4_validate_proposals.py`
- [x] T016: Run end-to-end validation on dev tile (48_37)

## Phase 7: Fix Broken Matching Pipeline (P1 blockers)

- [ ] T022: Fix coordinate mismatch in `Pm4MatchSupport` — convert PM4 object world coordinates to WoW world coordinates (or ADT placements to raw-ADT) before spatial comparison in `BuildPm4ObjectMatches` and `BuildMatches`
- [ ] T023: Validate T022 on tile 0_0 — `pm4 match-report` shows non-zero candidate counts for ≥5 placements (coordinate conversion verified in correlate-models)
- [ ] T024: Add CK24-grouped scoring mode to `Pm4AssetMatchScorer` — merge all segments sharing same CK24 into one combined shape before scoring against asset references
- [ ] T025: Validate T024 on tile 0_0 — at least 1 WMO placement scores ≥0.45 via `pm4 match-assets` with CK24 grouping
- [ ] T026: Build CK24 identity table on tile 0_0 — correlate known placements to PM4 CK24 groups, output (CK24, type, objectId) → (model path, confidence)
- [ ] T027: Validate on tile 22_18 — `pm4 match-report` shows non-zero candidates; `pm4 match-assets` with corpus produces top-3 candidate that includes snowball-fort WMO for at least 1 CK24 group

## Phase 8: Correlation Research (P1 — NEW)

- [x] T028: Add `pm4 correlate-models` CLI command — reads WMO/M2 collision geometry from archive, transforms to world space, computes volumetric overlap against PM4 CK24 group bounds
- [x] T029: Run correlate-models on development tile 0_0 — **HIT**: CK24 0x421809 ↔ ND_IRONDWARF_LARGEBUILDING.WMO at 61.8% overlap
- [x] T030: Run correlate-models on development tiles 22_18, 14_36, 0_1 — **0 correlations**: ADT placements don't overlap PM4 geometry (B4: PM4 contains objects not in MODF)
- [ ] T031: Run correlate-models on additional tiles to expand correlation dataset — target ≥5 tiles with ≥1 correlation each
- [ ] T032: Investigate CK24 types 0x3E/0x3F/0xC0/0xC1/0xC2 — determine what asset types they represent and how to match them
- [ ] T033: Fix M2 collision vertex reading (currently returns 0) — use M2 collision mesh bounds instead of falling back to placement position ±2
- [ ] T034: Research M2 top-surface geometry in PM4 — determine how M2 collision surfaces map to PM4 surfaces (elevated/offset from placement position)
- [ ] T035: Document CK24→material/group mapping — compare PM4 surface attributes (MSUR flags, MSLK TypeFlags) against WMO/M2 material properties to find correlation

- [x] All 48 unit tests pass
- [x] Scorer produces identical scores to C# on dev tile — 65/65 segments match within 0.005
- [x] Proposal IDs match C# output
- [x] No references to `H:\CLIENTS` in any new files
- [x] `spec.md` status updated

## Phase 6: Match Report (Replaces ADT Writing)

**ADT writing was removed 2026-06-16.** `Pm4AdtWriter`, `Pm4BinaryAdtPatcher`, and the `pm4 write-adt` command produced corrupted output. The matcher now produces human-readable markdown reports instead.

- [x] T017: ~~Create Pm4AdtWriter~~ REMOVED — replaced by `pm4 match-report` markdown output
- [x] T018: ~~Add pm4 write-adt CLI~~ REMOVED — replaced by `pm4 match-report` CLI command
- [x] T019: Add `pm4 match-report` CLI command to inspect tool — outputs human-readable markdown with match data per PM4 tile
- [ ] T020: Run `pm4 match-report` on development PM4 corpus and verify output renders correctly
- [ ] T021: Run `pm4 match-report` on 3.3.5 real map data and verify