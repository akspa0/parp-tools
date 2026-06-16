# Tasks: 046 — PM4 Asset Matching Python/Zarr Lane

**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)

## Phase 1: Data Models and JSON Import

- [x] T001: Create `data-harvester/src/harvester/pm4_asset_matching/models.py` — dataclasses for `Pm4Bounds3`, `Pm4SegmentHeightStats`, `Pm4SegmentTopologyStats`, `Pm4SegmentAnchorSignals`, `Pm4SegmentSignalRecord`, `Pm4AssetReferenceSignalRecord`, `Pm4SegmentMatchResult`, `Pm4AssetMatchCandidate`, `Pm4ReplacementPlacementProposal`
- [x] T002: Create `data-harvester/src/harvester/pm4_asset_matching/json_import.py` — `import_segment_signals(json_path) -> list[Pm4SegmentSignalRecord]` and `import_asset_corpus(json_path) -> list[Pm4AssetReferenceSignalRecord]` that parse C# JSON export format (camelCase, nested Vector2/Vector3, dictionary fields)
- [x] T003: Create `data-harvester/src/harvester/pm4_asset_matching/__init__.py` — public API re-exports
- [x] T004: Create `data-harvester/src/harvester/test_pm4_asset_matching.py` — test round-trip import of a C# segment signal JSON and asset corpus JSON

## Phase 2: Zarr Signal Store

- [x] T005: Create `data-harvester/src/harvester/pm4_asset_matching/signal_store.py` — `write_segment_signals_zarr(store_path, segments)` and `read_segment_signals_zarr(store_path) -> list[Pm4SegmentSignalRecord]`; same for asset references. Use Zarr v3 LocalStore with blosc compression, following `zarr_io.py` patterns
- [x] T006: Add store metadata: `signal_version`, `run_id`, `source_path`, `segment_count`/`asset_count` as group attrs
- [x] T007: Test write/read round-trip for segment signals and asset references

## Phase 3: Python Scorer

- [x] T008: Create `data-harvester/src/harvester/pm4_asset_matching/scorer.py` — port `Pm4AssetMatchScorer` logic: `score_segments(segments, asset_refs, max_candidates=10) -> list[Pm4SegmentMatchResult]`. Key functions: `compute_bounds_overlap_ratio`, `score_ratio`, `score_distance`, `evaluate_typed_candidate`, `resolve_expected_asset_kind`
- [x] T009: Implement TypeFlags profile matching: m2-top (0x03), interior-floor (0x10), exterior-solid (0x12) with expected-kind resolution (ck24Type 0x42/0x43→wmo, 0x40/0x41/0xC0-0xC3→m2)
- [x] T010: Test scorer against C# match report JSON — verify scores match to 0.001 tolerance

## Phase 4: Placement Synthesizer

- [x] T011: Create `data-harvester/src/harvester/pm4_asset_matching/placement_synthesizer.py` — port `Pm4ReplacementPlacementSynthesizer.Synthesize`: `synthesize_placements(match_results, asset_refs, target_tiles=None) -> list[Pm4ReplacementPlacementProposal]`. Must produce identical proposal IDs (SHA256-based)
- [x] T012: Test proposal synthesis against C# proposal JSON — verify proposal IDs match

## Phase 5: Validation Script and Known-Tile Proof

- [x] T013: Create `data-harvester/scripts/pm4_import_segment_signals.py` — CLI: `--input <segments.json> --output <store.zarr>` — imports C# segment export into Zarr store
- [x] T014: Create `data-harvester/scripts/pm4_import_asset_corpus.py` — CLI: `--input <corpus.json> --output <store.zarr>` — imports C# asset corpus into Zarr store
- [x] T015: Create `data-harvester/scripts/pm4_validate_proposals.py` — CLI: `--segments <json> --corpus <json> --expected <csharp_report.json>` — runs Python scorer, compares against C# ground truth, reports pass/fail with diffs
- [x] T016: Run end-to-end validation on dev tile (48_37) — export C# JSONs, run Python pipeline, verify all statuses match

## Completion Checklist

- [x] All 48 unit tests pass (`uv run pytest src/harvester/test_pm4_asset_matching.py`)
- [x] Scorer produces identical scores to C# on dev tile — 65/65 segments match within 0.005
- [x] Proposal IDs match C# output
- [x] No references to `H:\CLIENTS` in any new files
- [x] `spec.md` status updated

## Phase 6: ADT Placement Writing

- [x] T017: Create `Core.PM4/Matching/Pm4AdtWriter.cs` — converts PM4 match results to `LkAdtData` with MDDF (M2) + MODF (WMO) entries, string tables, flat terrain chunks
- [x] T018: Add `pm4 write-adt` CLI command to inspect tool — takes `--input <pm4> --archive-root <client> --placements <obj0.adt> --output <out.adt> [--map-name <name>]`
- [x] T019: Test end-to-end on development_00_00.pm4 — 10 M2 + 15 WMO placements written, ADT verified with `map inspect`
- [ ] T020: Test written ADT opens in viewer with visible placements
- [ ] T021: Test with 3.3.5 real map data (archive-backed M2/WMO resolution)
