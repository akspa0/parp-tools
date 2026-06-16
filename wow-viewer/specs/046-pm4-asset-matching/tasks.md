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

## Completion Checklist ✅

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