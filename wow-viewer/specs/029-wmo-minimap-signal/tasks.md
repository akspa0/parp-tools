# Tasks: WMO Minimap BLP Harvest and Asset Signal

**Input**: Design documents from `/specs/029-wmo-minimap-signal/`

**Prerequisites**: plan.md (required), spec.md (required)

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to

## Phase 1: WMO Minimap BLP Discovery and Filename Parsing (US1)

**Goal**: Enumerate and parse WMO minimap BLP filenames from MPQ archives

**Independent Test**: Run discovery on staged `3_3_5_12340`, confirm entries found matching the Ghidra-confirmed pattern

- [x] T001 [US1] Create `WmoMinimapBlpReader.cs` in `wow-viewer/src/core/WowViewer.Core.IO/Wmo/` with a `DiscoverMinimapBlps(string clientRoot)` method that enumerates MPQ entries under `Textures\Minimap\`, filters by filename pattern matching `<stem>_<digits>_<digits>_<digits>.blp`, and returns a list of `WmoMinimapBlpEntry` records
- [x] T002 [US1] Add filename parsing logic to `WmoMinimapBlpReader.cs` that extracts `wmo_stem`, `group_index`, `quad_y`, `quad_x` from each matched filename, handling 2-digit and 3-digit group index padding variants
- [x] T003 [US1] Add `WmoMinimapBlpEntry` record type in `WmoMinimapBlpReader.cs`: fields `BlpPath`, `WmoStem`, `GroupIndex`, `QuadY`, `QuadX`, `Build`
- [x] T004 [US1] Add unit tests in `wow-viewer/tests/WowViewer.Core.Tests/` for filename parsing: verify that `deadmines_000_00_00.blp` parses to stem=`deadmines`, group=0, quadY=0, quadX=0; verify 3-digit padding like `stockades_001_01_02.blp`; verify non-matching filenames are rejected
- [x] T005 [US1] Add `WmoMinimapAssetResolver.cs` in `wow-viewer/src/core/WowViewer.Core.IO/Wmo/` with a `BuildStemToAssetPathMap(string clientRoot)` method that scans the MPQ file list for `.wmo` root files and builds a `Dictionary<string, string>` mapping stem names to full asset paths (e.g., `deadmines` → `World\wmo\dungeon\deadmines\deadmines.wmo`)

**Checkpoint**: Discovery can enumerate BLPs and parse filenames; stem-to-asset mapping works on staged client

---

## Phase 2: BLP Decode and Harvest Command (US1)

**Goal**: Decode discovered BLPs and expose as a harvest CLI command

**Independent Test**: Run `WowViewer.Tool.Harvest wmo-minimap-harvest` on staged `3_3_5_12340`, get decoded RGB images and metadata

- [ ] T006 [US1] Add BLP decode step to `WmoMinimapBlpReader.cs`: for each discovered entry, open the BLP from MPQ, decode to `byte[]` RGB using existing BLP reader, collect `image_width`, `image_height`, and pixel data
- [ ] T007 [US1] Add error handling: skip BLPs that fail to open or decode, log errors to a `wmo_minimap_errors.jsonl` file, continue processing remaining entries
- [ ] T008 [US1] Add `WmoMinimapHarvestCommand.cs` in `wow-viewer/src/tools/harvest/WowViewer.Tool.Harvest/Commands/` that accepts `--client-root` and `--build`, runs discovery + decode + asset resolution, and writes results to Zarr + Parquet output
- [ ] T009 [US1] Add Zarr output writing to the harvest command: write `wmo_minimap_rgb` array (N, H_max, W_max, 3) uint8 with Blosc lz4 compression, padding smaller images to maximum dimensions
- [ ] T010 [US1] Add Parquet metadata output writing to the harvest command: `wmo_minimap_metadata.parquet` with columns `asset_path`, `wmo_stem`, `group_index`, `quad_y`, `quad_x`, `blp_path`, `image_width`, `image_height`, `source` (`discovered`), `build`

**Checkpoint**: Full US1 pipeline works end-to-end: discover → decode → resolve → write Zarr + Parquet

---

## Phase 3: Zarr-Compatible Storage and Group Aggregation (US2)

**Goal**: Per-group aggregation records and joinable metadata

**Independent Test**: Load Parquet in Python, join with `placements.parquet` on `asset_path`

- [ ] T011 [US2] Add per-group aggregation output to the harvest command: `wmo_group_composites.parquet` with columns `asset_path`, `group_index`, `quad_count`, `quad_blp_paths` (semicolon-delimited), `total_width`, `total_height`
- [ ] T012 [US2] Verify Parquet `asset_path` column format matches `placements.parquet` convention (e.g., `World\wmo\dungeon\deadmines\deadmines.wmo` with backslashes normalized)
- [ ] T013 [US2] Add harvest command `--output-root` flag defaulting to `wow-viewer/output/datasets/object_roof_library/<build>/wmo_minimap/`
- [ ] T014 [US2] Write a Python load test in `inspect_wmo_minimap_harvest.py`: open the metadata Parquet, open `placements.parquet` from the V16 dataset, join on `asset_path`, print match count and sample matches

**Checkpoint**: US2 complete — metadata is queryable and joinable with placement data

---

## Phase 4: QA Inspection and Cross-Build Validation (US2 + US3)

**Goal**: Visual QA script and harvest across all 6 builds

**Independent Test**: QA script renders WMO minimap BLPs; per-build summaries show non-zero counts

- [ ] T015 [US2] Expand `inspect_wmo_minimap_harvest.py` to render: sampled WMO minimap BLP images in a grid, group composite mosaics (stitch quads for multi-quad groups), and a summary table of per-build statistics
- [ ] T016 [US2] Run harvest on staged `3_3_5_12340`, verify at least 100 BLPs found and at least 5 distinct WMO stems; save summary to `wmo_minimap_harvest_summary.json`
- [ ] T017 [US3] Add WMO footprint overlap logic to the inspection script: for a WMO placement from `placements.parquet`, project the group AABB onto terrain coordinates and identify the overlapping terrain minimap tile region
- [ ] T018 [US3] Run harvest on remaining builds (`0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_0_1_8303`, `4_0_0_11927`), collect per-build summaries, report BLP counts and unique WMO counts per build

**Checkpoint**: All builds harvested, QA artifacts available for review

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1** → **Phase 2**: Discovery must exist before decoding
- **Phase 2** → **Phase 3**: Harvest command must write basic output before adding aggregation
- **Phase 3** → **Phase 4**: Zarr + Parquet must be complete before QA and cross-build runs

### Parallel Opportunities

- T004 (filename parse tests) and T005 (asset resolver) can run in parallel after T001-T003
- T009 (Zarr writing) and T010 (Parquet writing) can run in parallel as separate output paths
- T016 (335 harvest) and T017 (footprint overlap) can run in parallel
- T018 individual build harvests can run in parallel

### Implementation Strategy

1. Complete Phase 1 → validate discovery on `3_3_5_12340`
2. Complete Phase 2 → validate full harvest on `3_3_5_12340`
3. Complete Phase 3 → validate Parquet join with placements
4. Complete Phase 4 → cross-build runs and QA
