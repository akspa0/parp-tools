# Tasks: v65 Decoded Terrain and Placement Interchange

**Planning only: 0/62 tasks complete.** Read [spec.md](spec.md), [plan.md](plan.md),
[data-model.md](data-model.md), [coverage/transport contract](contracts/terrain-interchange.md)
and [durable appearance contract](contracts/appearance-retention.md).
The [per-layer opacity contract](contracts/layer-opacity-report.md) adds source/decode/contribution
metadata. Previous final tasks T050–T054 move to T058–T062; all remain pending.
Paths relative to `wow-viewer/`. Maximum ten tasks per phase. `[P]` means disjoint work after
prerequisites. Every checkbox needs files, exact commands/exits and criterion→real-output receipt.

## Phase 1 — Setup and contract freeze

- [ ] T001 Inventory selected real MPQ/loose and CASC corpus identities and extraction→stream→store→export coverage in `specs/285-v65-terrain-interchange/evidence/source-coverage.md`; include exact CASC product/build/config and required FileDataID companions, without broad harvesting or old Converter assumptions.
- [ ] T002 Record native field/coordinate/dtype/layout rules and operator-confirmed placement-in/geometry-mask-out boundary in `specs/285-v65-terrain-interchange/contracts/terrain-interchange.md`; classify each observed source chunk occurrence without invented decodes.
- [ ] T003 Record original conversion algorithms worth retaining and modern library ownership in `specs/285-v65-terrain-interchange/evidence/algorithm-lineage.md`; legacy CLI compilation/execution is neither prerequisite nor acceptance proof.

**Gate**: Selected corpus and coverage baseline real, contract frozen. User owns broad data gathering.

## Phase 2 — Foundation

- [ ] T004 Define native records, typed placement records, availability and coverage contracts under `src/core/WowViewer.Core/Harvest/`, including `V65TerrainRecord.cs` and `V65PlacementRecord.cs`; no object geometry/mask fields.
- [ ] T005 [P] Define strict Python schema/layout/field allowlists in `data-harvester/src/harvester/v65/contracts.py`, including ragged offsets, exact integer IDs and no silent absent-row substitution.
- [ ] T006 Specify versioned forward/reverse framing validation in `src/core/WowViewer.Core.IO/Harvest/V65StreamContract.cs`, reusing existing ARRY primitives with explicit supported dtypes/limits/completion and separating logs from binary stdout.
- [ ] T007 Verify shared field/dtype/availability/round-trip contracts in `tests/WowViewer.Core.Tests/Harvest/V65/V65ContractTests.cs` and `data-harvester/tests/v65/test_contracts.py`; record exact checks in `specs/285-v65-terrain-interchange/evidence/foundation.md`.

**Gate**: Contract coherence; no Zarr access added in C# or hidden geometry fields.

## Phase 3 — US1: current harvester → v65 native store

**Independent validation**: One bounded native terrain selection survives ingestion/readback,
with placement coverage explicitly pending until Phase 5. Depends on foundation.

- [ ] T008 [US1] Extract current MPQ/loose/CASC catalog and terrain harvest orchestration into `src/core/WowViewer.Core.IO/Harvest/TerrainHarvestService.cs`; retain CASC product/build/FileDataID and companion resolution, default-off CDN fill, and delegate existing `tools/harvest/WowViewer.Tool.Harvest/Program.cs` calls without changing proven readers/default profiles.
- [ ] T009 [US1] Add the opt-in geometry/mask-free projection in `src/core/WowViewer.Core.IO/Harvest/V65TerrainProjector.cs`; reuse existing decoded terrain/native lattices and skip object enrichment work.
- [ ] T010 [US1] Implement `src/core/WowViewer.Core.IO/Harvest/V65TerrainStreamWriter.cs` and `harvest-stream --stream-profile v65` delegation; emit metadata/availability/native fields before reductions, with explicit completion.
- [ ] T011 [US1] Implement bounded ingest and finalized native/ragged Zarr storage in `data-harvester/src/harvester/v65/ingest.py` and `store.py`; preserve row identity, metadata/name tables, hashes and absence reasons.
- [ ] T012 [US1] Add thin `inspect/ingest/validate` entry points in `data-harvester/scripts/v65_dataset.py` and typed reader in `data-harvester/src/harvester/v65/reader.py`; do not use old v60 first-shape stacking.
- [ ] T013 [US1] Verify available/absent fields, legitimate zeros, ragged layer shapes, row alignment and finalization in `data-harvester/tests/v65/test_store.py` using bounded actual source-derived records.
- [ ] T014 [US1] Record bounded actual CASC source identity/FileDataID/companion and native terrain readback alongside MPQ/loose controls in `specs/285-v65-terrain-interchange/evidence/native-store-pilot.md`; placement/SC-006 completion waits for Phase 5, missing native families for US2, with no mock/build-only claim or broad run.

**Gate/MVP**: Finalized native-store pilot, not full target conversion. No training or broad run.

## Phase 4 — US2: missing chunk semantics and complete coverage

**Independent validation**: Every observed chunk occurrence has a real per-stage disposition;
required native semantics no longer disappear through pack/stream/store projection.

- [ ] T015 [US2] Implement full occurrence/owner/tag/hash classification in `src/core/WowViewer.Core.IO/Harvest/V65ChunkCoverageCollector.cs`; distinguish decoded/derived/opaque/placement/structural/geometry-mask-excluded/unsupported at each stage.
- [ ] T016 [US2] Add opt-in uncapped native layer/material/full-resolution alpha extraction in `src/core/WowViewer.Core.IO/Harvest/V65LayerHandler.cs`, before existing four-layer training projection; preserve order, flags/effects and source identities.
- [ ] T017 [P] [US2] Add exact native low/high-resolution holes, terrain flags/area and normal packing provenance in `src/core/WowViewer.Core.IO/Harvest/V65TopologyHandler.cs`; chunk-has-hole booleans are derived only.
- [ ] T018 [P] [US2] Add native per-layer liquid family/coverage/type/height/depth/flag handling in `src/core/WowViewer.Core.IO/Harvest/V65LiquidHandler.cs`; unified/cropped raster cannot substitute for native layers.
- [ ] T019 [US2] Complete decoded shading/colors/material parameters/flight bounds/terrain emitters/WDT/WDL fields in `src/core/WowViewer.Core.IO/Harvest/V65NativeTerrainHandlers.cs`, and extend typed stream/store schemas for each verified missing field.
- [ ] T020 [US2] Implement retained terrain-opaque/unknown metadata classification in `data-harvester/src/harvester/v65/validate.py`; keep unknown terrain bytes/index provenance, exclude known geometry/masks, and never count raw payloads as decoded coverage.
- [ ] T021 [US2] Verify native semantic edge cases and decode→stream→store coverage in `tests/WowViewer.Core.Tests/Harvest/V65/V65NativeCoverageTests.cs` and `data-harvester/tests/v65/test_coverage.py`; use real source bytes and explicit unsupported cases.
- [ ] T022 [US2] Update `specs/285-v65-terrain-interchange/evidence/source-coverage.md` with real per-occurrence outputs and remaining handlers; block preserve-features export for unresolved required pilot semantics.

**Gate**: SC-001/002 native/coverage proof for the selected scope, not universal parser support.

## Phase 5 — US3: exact placement metadata; no geometry or masks

**Independent validation**: IDs over `2^24`, flags/transforms and references survive exactly;
geometry/mask field audit is empty. Depends on native schema and source record identities.

- [ ] T023 [US3] Extract typed MDDF/MODF/global-placement data and asset identities directly from source reader records in `src/core/WowViewer.Core.IO/Harvest/V65PlacementMetadataExtractor.cs`; never recover IDs from float32 pack columns.
- [ ] T024 [US3] Preserve MCRF/MCRD/MCRW/reference ownership and index-versus-ID semantics in `src/core/WowViewer.Core.IO/Harvest/V65PlacementMetadataExtractor.cs`, including repeated cross-tile refs and reused original IDs.
- [ ] T025 [US3] Persist checked ragged placement/reference tables and asset locator metadata inside v65 through `data-harvester/src/harvester/v65/store.py` and `reader.py`; optional manifests are audit mirrors only.
- [ ] T026 [US3] Enforce geometry/model-library/mask/fragment-trace exclusion in both stream projection and `data-harvester/src/harvester/v65/validate.py`; retain metadata without running WMO/M2 mesh enrichment.
- [ ] T027 [US3] Verify exact IDs/flags, empty/nonempty ragged spans, metadata joins and forbidden-field refusal in `tests/WowViewer.Core.Tests/Harvest/V65/V65PlacementTests.cs` and `data-harvester/tests/v65/test_placements.py`; record real readback in `specs/285-v65-terrain-interchange/evidence/placement-custody.md`.

**Gate**: SC-003 plus the placement portion of SC-006; v65 stores source IDs while target
re-itemization remains Spec 284. Actual CASC evidence must be returned to close SC-006.

## Phase 6 — US4: reverse interchange and converter handoff

**Independent validation**: Source-detached v65 read/export reaches modern C# materialization
without old Converter or a C# Zarr reader. Depends on native/placement/coverage gates.

- [ ] T028 [US4] Implement selected finalized-store export in `data-harvester/src/harvester/v65/export_stream.py`, including native terrain, placement metadata, coverage/dependency manifests and versioned terminal digests.
- [ ] T029 [US4] Implement strict C# reverse stream reading in `src/core/WowViewer.Core.IO/Harvest/V65TerrainStreamReader.cs`; reject unknown dtype/schema/length/hash/truncation and incomplete finalization.
- [ ] T030 [US4] Add the Python process/store bridge contract in `src/core/WowViewer.Core.Editor/Conversion/V65DatasetBridge.cs`, accepting Spec 285 native selection/stream DTOs and supervising uv in `data-harvester/` with progress/cancellation; no dependency on Spec 284 request implementation, and target export logic stays in Core.IO.
- [ ] T031 [US4] Define finalized dataset selection/digest/coverage/placement inputs for Spec 284 in `specs/284-map-object-converter-revitalization/contracts/conversion-tools.md`; Map CLI/Editor consume this same contract and never re-decode missing source terrain silently.
- [ ] T032 [US4] Verify forward/store/reverse native DTO values, exact placement IDs and strict failure cases in `tests/WowViewer.Core.Tests/Harvest/V65/V65InterchangeTests.cs` and `data-harvester/tests/v65/test_export_stream.py`; target-file serialization is a later joint Spec 284 gate.
- [ ] T033 [US4] Record real finalized-store→reverse-stream/native-record readiness in `specs/285-v65-terrain-interchange/evidence/interchange-pilot.md`; hand off to Spec 284 T024–T029 for target-adapter implementation, with SC-004 target-output acceptance still open.

**Gate**: Native transport/store readiness unblocks Spec 284 T024–T029. SC-004 is a **joint later
acceptance gate** requiring those target outputs; it is not a prerequisite for implementing the
target adapter. Full target appearance/behavior remains Spec 284/operator-owned.

## Phase 7 — US5: durable terrain evidence and query indexes

**Independent validation**: With source unmounted, retrieve native/unknown bytes and metadata by
index and add an interpretation/view without modifying committed source records. Depends on native store.

- [ ] T034 [US5] Capture all source terrain headers/chunks/unknown regions, including already-decoded chunks and all contributing companion bands, in `src/core/WowViewer.Core.IO/Harvest/V65RetainedEvidenceCollector.cs`; retain occurrences/order/offsets/raw bytes/hashes and exclude known object geometry/masks.
- [ ] T035 [US5] Add durable raw/native/resource revision storage and non-destructive resume in `data-harvester/src/harvester/v65/retained_evidence.py`; exporting must not evict or overwrite committed evidence.
- [ ] T036 [US5] Index source/backend/build/file/band/tag/occurrence/offset/coordinates and unknown metadata in `data-harvester/src/harvester/v65/indexes.py`, with extension keys for not-yet-decoded fields and canonical store provenance.
- [ ] T037 [US5] Add versioned interpretation/view lineage and raw/native metadata lookup in `data-harvester/src/harvester/v65/reader.py` and `retained_evidence.py`; new semantic decodes link to old bytes rather than replacing them.
- [ ] T038 [US5] Add thin `query-chunks`/metadata inspection commands in `data-harvester/scripts/v65_dataset.py`; return source values/bytes/availability instead of invented semantic labels.
- [ ] T039 [US5] Validate retained-byte/index/occurrence coverage against source inventory in `data-harvester/src/harvester/v65/validate.py`; classify only known excluded geometry/masks as exclusions, not arbitrary unknown terrain data.
- [ ] T040 [US5] Verify unknown/duplicate-tag/decoded-metadata retention, source-detached lookup, revision addition and post-export immutability in `data-harvester/tests/v65/test_retained_evidence.py` using real terrain chunks.
- [ ] T041 [US5] Record native/raw retained hashes and query/immutability proof in `specs/285-v65-terrain-interchange/evidence/retention-witness.md`; no claim of semantic decoding from byte retention alone.

**Gate**: SC-007/010 retained evidence and future interpretation capability, with no automatic eviction.

## Phase 8 — US6: cell/layer appearance and complete PNG texture resources

**Independent validation**: Resolve real cell→layers→texture→PNG and inverse texture usages, with
native masks/UV/material metadata, all referenced texture PNGs and explicit decode failures.

- [ ] T042 [US6] Extract full native texture slot/layer/flags/effect/UV/animation/height-blend metadata and relevant appearance table rows in `src/core/WowViewer.Core.IO/Harvest/V65AppearanceMetadataExtractor.cs`; preserve unknown MTXP/MHID/MTXF values and source identity rather than guessing interpretations.
- [ ] T043 [US6] Implement cell topology/basis/alpha-sample-footprint and ordered layer/resource relationships in `data-harvester/src/harvester/v65/appearance.py`; dominant labels and averaged weights are optional versioned views, not source material definitions.
- [ ] T044 [US6] Collect every referenced terrain diffuse/auxiliary/height/liquid texture and dependency identity from MPQ/loose/CASC in `src/core/WowViewer.Core.IO/Harvest/V65TerrainTextureCollector.cs`; retain original bytes, aliases/FDIDs, format/header/mips/dimensions/channel/color/alpha metadata and per-resource failures.
- [ ] T045 [US6] Derive native-resolution RGBA PNGs for source mips through `src/core/WowViewer.Core.IO/Harvest/V65TerrainTextureCollector.cs` using existing decode primitives; preserve alpha/data channels and never reuse opaque-alpha/resized/proxy exports as source truth.
- [ ] T046 [US6] Persist content-addressed source/PNG resources and cell/layer→texture plus inverse usage indexes in `data-harvester/src/harvester/v65/texture_resources.py` and `indexes.py`; retain build aliases and metadata while deduplicating bytes.
- [ ] T047 [US6] Add `query-cell`, `query-texture`, deterministic `export-textures` and versioned `model-view` operations in `data-harvester/scripts/v65_dataset.py` and `v65/model_views.py`; carry source hashes/normalization/coordinates without mutating records or training a model.
- [ ] T048 [US6] Verify real MPQ/CASC multi-layer cell mapping, mask orientation, duplicate basenames, full resource enumeration, RGBA/native dimensions/mips, decoded pixel hashes and missing texture states in `data-harvester/tests/v65/test_appearance.py` and `tests/WowViewer.Core.Tests/Harvest/V65/V65TextureTests.cs`.
- [ ] T049 [US6] Record cell↔texture query outputs, complete source/PNG texture inventory and unresolved appearance parameters in `specs/285-v65-terrain-interchange/evidence/appearance-witness.md`; SC-008/009 stay open for missing/failed required resources and unknown required semantics.

**Gate**: Durable source/native/PNG appearance reference ready for future viewers/models and full
target-export acceptance. Creating all target writers or model training is outside this phase.

## Phase 9 — US7: every-layer opacity and metadata reporting

**Independent validation**: One report row per declared layer; distinct source/decoded/effective
maps and statistics with validity; source reporting does not wait for target writer implementation.

- [ ] T050 [US7] Define every-layer report and common retained-field metadata contracts in `src/core/WowViewer.Core/Harvest/V65LayerOpacityReport.cs` and `data-harvester/src/harvester/v65/contracts.py`, following the opacity contract without merging same-texture layers.
- [ ] T051 [US7] Implement raw MCLY/MCAL custody and native decoded-alpha analysis in `src/core/WowViewer.Core.IO/Harvest/V65LayerOpacityAnalyzer.cs`; record source ranges/encoding/decoder decisions/fixups and observed/derived/unknown sample validity, never treating decoder fill as source evidence.
- [ ] T052 [US7] Implement named ordinary alpha-over contribution/base-residual maps and per-cell summaries in `src/core/WowViewer.Core.IO/Harvest/V65LayerOpacityAnalyzer.cs`; retain zero/overpainted layers and mark unsupported/missing/height-blend semantics partial or unknown instead of normalizing raw alpha.
- [ ] T053 [US7] Persist/index report maps, per-layer statistics/histograms/coverage/validity and field provenance in `data-harvester/src/harvester/v65/layer_opacity.py`, tied to immutable source and decode/composition digests.
- [ ] T054 [US7] Add thin `report-opacity` and layer/cell inspection/export operations in `data-harvester/scripts/v65_dataset.py`; JSON/Parquet/optional preview artifacts read the same retained records and enumerate every layer/unused texture slot with explicit states.
- [ ] T055 [US7] Verify implicit base, ordered overlap, repeated textures, zero/full/overpainted layers, incomplete payloads, fixups and unknown blending in `tests/WowViewer.Core.Tests/Harvest/V65/V65LayerOpacityTests.cs` and `data-harvester/tests/v65/test_layer_opacity.py`; counts must measure records, not per-pixel failures.
- [ ] T056 [US7] Define source→target layer report joining and quantization/coverage/spatial-difference comparison in `data-harvester/src/harvester/v65/layer_opacity.py`; source analysis is independently ready, while target comparisons run only after Spec 284 output exists.
- [ ] T057 [US7] Record bounded real MPQ/loose/CASC source reports with complete layer accounting and known-model numeric checks in `specs/285-v65-terrain-interchange/evidence/layer-opacity-witness.md`; leave target readback/visual claims pending until the corresponding outputs/witnesses exist.

**Gate**: SC-011 source report readiness; target comparison closes jointly with Spec 284's writer
readback. Decoded MCAL, effective contribution and rendered material opacity remain distinct.

## Phase 10 — Migration and final handoff

- [ ] T058 Implement audited read-only old-store inspection/copy into new v65 stores in `data-harvester/src/harvester/v65/migrate.py`; classify preserved/missing/rejected/re-harvest fields and exclude old geometry/masks without mutating old data.
- [ ] T059 Verify flat/dedup layout detection, row/metadata integrity and re-harvest refusal in `data-harvester/tests/v65/test_migrate.py`; do not infer exact holes/layers/IDs/textures from lossy tensors.
- [ ] T060 Operator-owned: execute the bounded real-client harvest/resource/opacity/readback commands prepared in `specs/285-v65-terrain-interchange/quickstart.md`; record builds/roots/hashes, retained metadata/PNG/layer coverage and per-field results in `evidence/operator-native-witness.md`; no full corpus run before pilot acceptance.
- [ ] T061 Record exact focused Python/C# checks, solution build and criterion→real-evidence matrix in `specs/285-v65-terrain-interchange/evidence/acceptance-matrix.md`; keep missing parser/texture/report/transport/runtime proof open and close SC-004/011 jointly after Spec 284's writer output exists.
- [ ] T062 Update receipted checks in `specs/285-v65-terrain-interchange/tasks.md`, routing in `specs/STATUS.md` and `memory-bank/activeContext.md`, and Spec 284's retained-appearance/writer/report handoff; preserve unrelated UI/trainer work.

## Dependencies, parallel work and delivery

Setup → foundation → US1 store → US2 native coverage → US3 metadata/exclusion → US4 reverse stream
→ US5 retained evidence → US6 cell/PNG appearance → US7 layer-opacity source reports → migration/real proof. T005 and T006 are separate Python/protocol scopes after T004; T017/T018 can
proceed independently after native contracts, then integrate at T019–T022. US3 metadata extraction
may be prepared against frozen records while US2 runs, with one owner for store schema integration.
US4 Python export and C# consumer preparation are disjoint once the wire contract is frozen.

Complete one validated phase at a time; no phase closes before its evidence. MVP is US1 native
store/readback, followed by required coverage and placement gates before claiming export readiness.
Full appearance export acceptance additionally requires US5/US6 coverage; adapters may be prepared
after native/placement/protocol readiness without waiting on their own eventual output acceptance.
Task count: setup 3, foundation 4, US1 7, US2 8, US3 5, US4 6, US5 8, US6 8, US7 8, final 5 = **62**.
