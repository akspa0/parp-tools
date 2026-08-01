# Tasks: V22 Consolidated Dataset

**Input**: `specs/086-v22-consolidated-dataset/spec.md`, `specs/086-v22-consolidated-dataset/plan.md`

---

## Phase 1: Schema Freeze And Inventory Proof

**Purpose**: Lock the exact V22 store surface before touching the C# harvester or Python Zarr writer.

- [x] T001 Create `docs/architecture/v22-dataset-signals-2026-06-30.md` with the final root-array list, model-library layout, tileset-library layout, placement-array fields, metadata/audit tables, and dataset read contract.
- [x] T002 Cross-check every V18 base, patched, promoted, placement, model, and tileset surface from `spec.md` FR-001 through FR-028 has an explicit V22 home or is marked audit-only.
- [x] T003 Record the stream-message boundary: tile blobs stay regular, while model and tileset libraries are separate per-build message types and are not duplicated per tile.
- [x] T004 Record the fixed-key V22 dataset contract, including zero-fill and empty-array behavior for missing tile-local data.
- [x] T005 Confirm Phase 1 exit criteria in this file: no unresolved store-location questions remain; tests in later phases can pin the documented schema.

**Checkpoint**: Phase 1 complete — V22 schema is stable enough for Phase 2 stream and Phase 3 writer tests to pin.

---

## Phase 2: V22 Stream Contract Expansion (C#)

**Purpose**: Make `WowViewer.Tool.Harvest` emit every V22 input in one pass without post-build patch scripts.

- [x] T006 Add `RawArraySerializer.StreamProfile.V22` for V22 tile records without changing existing V16/full stream semantics.
- [x] T007 Emit C#-derived V22 tile arrays for `mcnr_mask_257`, `liquid_type_256`, `ground_intent_height_257`, `model_focus_mask` (renamed from `object_filtered_mask`), and `model_above_terrain_mask` (Z-level culled) in the V22 stream profile; renderer-truth arrays remain pending until the Python writer owns capture ingestion.
- [x] T008 Emit MDDF and MODF placement rows, unique IDs, count arrays, provisional model-id arrays, explicit per-placement asset paths, and tile-local MTEX texture paths in the V22 stream profile.
- [ ] T009 Emit unique M2 model payloads once per build session, including geometry, skin triangles, render flags, blend modes, texture references, bone lookups, bounds, and load-error markers.
- [ ] T010 Emit unique WMO model payloads once per build session, including merged geometry, group offsets, materials, portals, doodad-set paths, bounds, flags, version, and load-error markers.
- [ ] T011 Emit unique terrain tileset payloads once per build session, including decoded BLP RGB, original texture shape, path, and load-error marker.
- [ ] T012 Add bounded stream-dump validation for one object-rich tile and prove repeated placements do not duplicate model payloads.

**Checkpoint**: Phase 2 complete — a bounded stream dump contains all V22 message types with de-duplicated model and tileset payloads.

---

## Phase 3: Python Zarr Writer And Store Layout

**Purpose**: Write the expanded decoded stream into the frozen V22 Zarr dataset layout using the Python package.

- [ ] T013 Create a V22 signal derivation helper for any tile signals that are not already emitted by `RawArraySerializer.StreamProfile.V22`.
+- [x] T013A Add `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py` with `V22ZarrWriter`, `V22Dataset`, `V22TileRecord`, fixed-key contract, and full V22 array/group constants. Write `wow-viewer/data-harvester/scripts/build_v22_dataset.py` as the one-pass V22 stream → Zarr builder. Add `tests/test_v22_zarr_io.py` covering the writer/reader round-trip and fixed-key contract on synthetic records.
- [ ] T014 Create placement accumulation/name-remap logic with per-tile offsets and fixed-shape output arrays.
- [ ] T015 Create model-library contracts with payload validation, load-error handling, and cache metadata writers.
- [ ] T016 Create tileset-library contracts with BLP RGB payload validation, MTEX path remapping, and `mcly_tileset_ids` helpers.
- [x] T017 Create `data-harvester/scripts/build_v22_dataset.py` as the one-pass Python Zarr writer fed by the decoded C# V22 stream. (Initial V22 stream → Zarr writer shipped; pending real-data bounded proof.)
- [ ] T018 Keep index, placement, decoded metadata, and asset inventory outputs as audit reports, not training-only side paths.
- [x] T019 Add synthetic Zarr writer tests for root arrays, placement offsets, model entries, tileset entries, resume behavior, and source-placement parity. (Initial synthetic round-trip test landed; resume + parity proofs remain.)

**Checkpoint**: Phase 3 complete — a synthetic V22 Zarr store round-trips every documented schema group.

---

## Phase 4: Consumer Contract

**Purpose**: Define the single Zarr-backed dataset API used by downstream consumers.

- [x] T020 Add `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py` with `V22Dataset`, fixed batch keys, zero-filled missing tile signals, and zero-length empty placement arrays. (Initial reader shipped; collate/cache enhancements remain.)
- [ ] T021 Expose cached `models` and `tilesets` properties without inlining giant geometry or texture blobs into every tile batch.
- [ ] T022 Create batch/collate helpers for placement/model/tileset reference batches, including empty-placement tiles.
- [x] T023 Add synthetic store tests for fixed keys, shapes, dtypes, empty tiles, load-error entries, model cache hits, tileset cache hits, and multi-worker-safe cache reads. (Initial fixed-key/dtype/empty-tile test shipped; cache/multi-worker tests remain.)
- [ ] T024 Add a compatibility smoke showing selected downstream input paths can read V22 root signals without sidecar or MPQ side paths.

**Checkpoint**: Phase 4 complete — consumers can read tile signals, placements, model refs, and tileset refs from V22 alone.

---

## Phase 5: Bounded Real-Data Proof Build

**Purpose**: Prove V22 on staged real clients before broad rebuild.

- [ ] T025 Build a bounded `3_3_5_12340` Azeroth V22 store from a staged client under `output/tmp/wowarchive-clients/`.
- [ ] T026 Build a bounded `0_5_3_3368` Azeroth V22 store from a staged client under `output/tmp/wowarchive-clients/`.
- [ ] T027 Build a bounded `4_0_0_11927` development-map V22 store from a staged client under `output/tmp/wowarchive-clients/`, covering Cata-only development-map assets.
- [ ] T028 Validate precise-mask coverage, placement-array parity, model-library completeness, tileset-library completeness, WMO mask parity vs V18, and signal coverage thresholds.
- [ ] T029 Record the exact staged client roots and output store paths used for all three bounded proofs.

**Checkpoint**: Phase 5 complete — bounded real-data stores pass coverage and parity gates with no silent fallbacks.

---

## Phase 6: Learnability Gates

**Purpose**: Prove the richer store improves supervision quality or stop before broad migration.

- [ ] T030 Run a tiny-overfit proof on 8-32 V22 tiles and compare against the same V18 route.
- [ ] T031 Run mask-consistency proof reconstructing masks from stored model geometry and compare to `object_precise_mask`.
- [ ] T032 Run an asset-reference retrieval baseline using `mddf_model_ids` plus stored geometry and compare to coarse footprint-only matching.
- [ ] T033 Run a tileset proof that reproduces synthetic minimap/albedo from stored tilesets without external BLP reads.
- [ ] T034 Decide whether V22 passes learnability gates; if not, document the failing signal class before any full rebuild.

**Checkpoint**: Phase 6 complete — at least one bounded route shows improved fit or lower error on object-confused cases, or the failure is diagnosed.

---

## Phase 7: Three-Build Rebuild And Consumer Migration

**Purpose**: Rebuild only the scoped `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927` stores and migrate consumers only after bounded proof and learnability gates pass.

- [ ] T035 Rebuild `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927` V22 stores only. Other staged clients remain out of scope unless Spec 086 is explicitly reopened.
- [ ] T036 Migrate selected downstream consumers to the V22 Zarr contract without MPQ reparse or sidecar-only core semantics.
- [ ] T037 Deprecate post-build patch scripts only after grep confirms no active consumer still needs them.
- [ ] T038 Publish migration notes and update memory-bank completion status.

**Checkpoint**: Phase 7 complete — V22 is the canonical dataset route and old patch/promotion side paths are deprecated.
