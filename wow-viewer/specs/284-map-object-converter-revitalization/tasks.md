# Tasks: Map and Object Converter Revitalization

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts](contracts/conversion-tools.md), [quickstart.md](quickstart.md).  
**Status**: Planning only. **0/74 implementation/acceptance tasks complete.**

**Operator rebase, 2026-10-21**: [Spec 285 v65](../285-v65-terrain-interchange/tasks.md) is the
dataset prerequisite: current harvester including CASC reads → native terrain + typed placement
metadata → target export. Geometry/masks are excluded from v65. Old Converter compilation/parity
is not a baseline; original logic is recovered as algorithms and rebased on modern tooling.

Paths are relative to `wow-viewer/`. `[P]` means disjoint work after the named prerequisite gate,
not permission to skip validation. Each phase has at most ten tasks. Verification tasks exercise
requirements and real evidence; no blanket TDD or unrelated test campaign is prescribed.

Every completed checkbox requires a receipt under this spec's `evidence/`: changed files, exact
commands/exit status, criterion → actual output, source/target root/build/hash where applicable.
Compilation cannot close visual/client criteria. Operator-owned tasks prepare commands and wait
for operator evidence; agents never run heavy conversions, training/GPU or real-client proof.

## Phase 1 — Setup: recover baseline and choose a bounded corpus

**Goal**: Establish what works today without touching protected readers or unrelated UI WIP.

- [ ] T001 Reconcile current harvest and actual viewer New Map writer paths in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`, `src/viewer/WoWViewer/Workbench/Services/NewMapCreatorService.cs` and `specs/284-map-object-converter-revitalization/evidence/baseline.md`; name canonical ADT/WDT/WDL writers, known defaulted metadata and Spec 285 prerequisites, with no old CLI functional assumption.
- [ ] T002 [P] Prepare the operator-selected corpus manifest in `specs/284-map-object-converter-revitalization/evidence/corpus-manifest.json`: real CASC 1.60.1 product/build/config, 0.5.3 target, MPQ/loose control, finalized v65 identities, asymmetric tiles and representative objects; record configured roots/hashes without broad scans.
- [ ] T003 [P] Document original conversion algorithm lineage and modern owners in `specs/284-map-object-converter-revitalization/evidence/algorithm-lineage.md`; recover coordinate/layer/reference transformations as reference logic, with no mandatory old CLI flags/wrappers or functional claim.

**Gate**: Baseline and corpus selection documented. Missing operator data is explicit; no fictional hashes.

## Phase 2 — Foundation: shared contracts and independent hosts

**Goal**: Freeze library boundaries and thin executable shells. Depends on Phase 1.

- [ ] T004 Add immutable client/profile/request/report/placement/candidate contracts under `src/core/WowViewer.Core/Conversion/` following `data-model.md`; define schema/version/digest and typed loss/progress states without UI dependencies.
- [ ] T005 Implement contained staging, no-client-root/no-overwrite publication and partial-run reporting in `src/core/WowViewer.Core.IO/Conversion/ConversionOutputPublisher.cs`; metadata alone must not claim completed content.
- [ ] T006 [P] Add the three small CLI projects at `tools/map-convert/WowViewer.Tool.MapConvert/WowViewer.Tool.MapConvert.csproj`, `tools/model-convert/WowViewer.Tool.ModelConvert/WowViewer.Tool.ModelConvert.csproj` and `tools/wmo-convert/WowViewer.Tool.WmoConvert/WowViewer.Tool.WmoConvert.csproj`; add them to `WowViewer.slnx`, sharing contracts rather than growing the old dispatcher.
- [ ] T007 [P] Implement typed request/job/progress/cancellation coordination in `src/core/WowViewer.Core.Editor/Conversion/ConversionJobCoordinator.cs`; consume domain interfaces, never viewer globals or format logic.
- [ ] T008 Add canonical requests/digests and new CLI envelope/exit handling in `src/core/WowViewer.Core.IO/Conversion/ConversionRequestSerializer.cs` and new hosts' `Program.cs`; Map requests include v65 schema/layout/finalization/placement digests and Spec 285 coverage gates.
- [ ] T009 Verify foundation containment, deterministic request round-trip and cancellation states in `tests/WowViewer.Core.Tests/Conversion/ConversionRequestTests.cs` and `tests/WowViewer.Core.Editor.Tests/Conversion/ConversionJobCoordinatorTests.cs`; record exact focused checks/build exits in `specs/284-map-object-converter-revitalization/evidence/foundation.md`.

**Gate**: Libraries/hosts build; contract checks pass; no reader or app/scene-member growth.

## Phase 3 — US1: client selection and automatic route preview (P1)

**Independent verification**: Real selected clients in either slot; conflicts, missing roots and
unavailable reverse writers remain explicit. Depends on foundation.

- [ ] T010 [US1] Implement selected-client MPQ/loose/CASC identity probing in `src/core/WowViewer.Core.IO/Conversion/ClientCapabilityProbe.cs`, including CASC product/build/config, overlay order and fingerprints; bind source selection to matching v65 data or an explicit current-harvester ingestion request.
- [ ] T011 [US1] Implement per-domain source/target capability resolution in `src/core/WowViewer.Core.IO/Conversion/ConversionRouteResolver.cs`; infer transforms without version-string ordering and prevent profile overrides from enabling unavailable writers.
- [ ] T012 [P] [US1] Adapt existing saved known-good entries and actual mounted-source snapshots in `src/viewer/WoWViewer/Workbench/Services/Converters/ConversionClientSelectionAdapter.cs`; preserve settings and distinguish last-folder fallback from a mount.
- [ ] T013 [US1] Implement explicit source/target roles, loaded-as-source default, loaded-as-target and swap semantics in `src/core/WowViewer.Core.Editor/Conversion/ConversionClientSelection.cs`; do not change the global load dialog.
- [ ] T014 [US1] Add shared `inspect`/client selectors to the new hosts' `Program.cs`, including explicit `--viewer-settings` known-good lookup, profile diagnostics and captured request input; CLI never assumes a running viewer.
- [ ] T015 [US1] Verify conflicting signatures, short-version ambiguity, missing known-good entries, changed roots and disabled split output in `tests/WowViewer.Core.Tests/Conversion/ConversionRouteResolverTests.cs`; use real corpus evidence for supported-profile claims.
- [ ] T016 [US1] Record both orientations of the real 1.60.1/0.5.3 pair, actual route decisions and unresolved writer gaps in `specs/284-map-object-converter-revitalization/evidence/client-routes.md`; do not claim reverse conversion support merely because direction is recognized.

**Gate**: SC-001 route evidence and FR-002/003 client behavior accounted for.

## Phase 4 — US2: placement identity and deterministic re-itemization (P1)

**Independent verification**: Real M2/WMO placement graph has complete old→new mapping, no distinct
instance collisions and no stale references. Depends on client/profile contracts.

- [ ] T017 [US2] Build record/instance identity from v65 typed placement/reference tables in `src/core/WowViewer.Core.IO/Conversion/PlacementIdentityGraphBuilder.cs`; preserve exact original integer IDs, cross-tile evidence and ambiguous/reused-ID cases without modifying source dataset IDs.
- [ ] T018 [P] [US2] Implement stable map-scope M2/WMO allocation in `src/core/WowViewer.Core.IO/Conversion/PlacementIdAllocator.cs`; honor target/signed DTO limits, requested start, reserved IDs and overflow without changing original provenance.
- [ ] T019 [US2] Add retained-map reservation coverage checks and manifest loading in `src/core/WowViewer.Core.IO/Conversion/PlacementReservationReader.cs`; partial-map output requires complete retained-scope evidence.
- [ ] T020 [US2] Implement target-aware placement/reference rewriting in `src/core/WowViewer.Core.IO/Conversion/PlacementReferenceRewriter.cs`, including preserved Alpha uniqueID arrays and index-based MCRF; exclude WMO group/MODD-local namespaces.
- [ ] T021 [US2] Serialize many-record source→instance→output mappings and one-to-many child identity in `src/core/WowViewer.Core/Conversion/PlacementIdMap.cs` and `src/core/WowViewer.Core.IO/Conversion/PlacementIdMapWriter.cs`.
- [ ] T022 [US2] Verify exact IDs above float32 precision, shuffled-order determinism, cross-kind collisions, cross-tile repeats, reused IDs, reservations/overflow and reference integrity in `tests/WowViewer.Core.Tests/Conversion/PlacementReitemizationTests.cs` using typed v65 records.
- [ ] T023 [US2] Run bounded real-placement readback through unchanged LK/Alpha writers/readers and record mappings, collision counts and reference counts in `specs/284-map-object-converter-revitalization/evidence/reitemization.md`; do not mutate original files or chronology analyses.

**Gate**: SC-002 supported by real output plus focused invariants. Frozen writers remain unchanged.

## Phase 5 — US3-A: separate map tool exporting decoded v65 (P1, MVP)

**Independent verification**: Finalized native v65 exports through the standalone Map tool with
the source terrain detached. Depends on Phases 3–4 and Spec 285 native/placement/reverse-stream gates.

Adapter implementation may start at those gates. T028–T029 comparisons and T032/T036 reduction/
readback additionally require Spec 285 source-report readiness (T050–T055/T057). Source analysis
is independent; target comparison occurs only after actual writer output, without a circular gate.

- [ ] T024 [US3] Implement `src/core/WowViewer.Core.IO/Conversion/Maps/MapConversionService.cs` with a small `Maps/MapTargetMaterializer.cs` facade over the canonical `LkAdtWriter`, `LkWdtWriter`, `WdlWriter` and public Alpha boundary used by current tooling; require finalized v65 data and explicit writer field dispositions, not a second serializer or old CLI dependency.
- [ ] T025 [US3] Integrate stored typed-placement re-itemization and separate asset dependency inventory in `src/core/WowViewer.Core.IO/Conversion/Maps/MapConversionService.cs`; source v65 IDs remain immutable.
- [ ] T026 [US3] Wire Map `plan/run/validate`, `--dataset` and source-client-to-ingest selection in `tools/map-convert/WowViewer.Tool.MapConvert/Program.cs`; expose actual writer/coverage availability before execution.
- [ ] T027 [US3] Map v65 native coordinates/layers/references into current writer documents in `src/core/WowViewer.Core.IO/Conversion/Maps/V65TerrainExportAdapter.cs`; audit DTO/defaulted field gaps, select explicit preserved/derived WDL and fresh-write/height-patch modes, and add only bounded opt-in field support or explicit loss/block decisions.
- [ ] T028 [US3] Verify source-detached v65 export and a small current New Map control through the same canonical writer libraries in `tests/WowViewer.Core.Tests/Conversion/V65MapExportTests.cs`; independently read output and compare coordinates/metadata/refs and mapped per-layer opacity reports, without claiming synthetic output is real-client proof.
- [ ] T029 [US3] Record named writer path/version/mode, source field→DTO→readback capability table, dataset hashes and layer-opacity comparisons in `specs/284-map-object-converter-revitalization/evidence/v65-map-export.md`; New Map delegation, if consolidated, preserves existing defaults/load callback, and visual/target proof remains open.

**Gate/MVP**: Separate Map CLI with native v65 export, client preview and deterministic IDs; no old CLI parity prerequisite.

## Phase 6 — US3-B: modern terrain and map/object dependency planning (P1)

**Independent verification**: Bounded real CASC-backed 1.60.1 → v65 → Alpha terrain staging and
full dependency inventory; object blockers stay visible until US4/US5 close them. Depends on v65 export.

- [ ] T030 [US3] Consume finalized real CASC-backed native terrain/placement records in `src/core/WowViewer.Core.IO/Conversion/Maps/V65TerrainExportAdapter.cs`; missing required source chunks return Spec 285 coverage blockers, never local re-decoding fallback.
- [ ] T031 [P] [US3] Implement v65 asset-locator→separate MPQ/loose/CASC geometry resolution, companions, deterministic target paths/collisions/cycles and proven reuse in `src/core/WowViewer.Core.IO/Conversion/ConversionDependencyResolver.cs`; no embedded dataset meshes, unrelated-client fallback or implicit CDN fill.
- [ ] T032 [US3] Implement target layer-budget reduction in `src/core/WowViewer.Core.IO/Conversion/Maps/LegacyLayerReducer.cs` using known effective contributions from per-layer reports and target alpha re-encoding; never normalize raw MCAL channels or silently drop zero/occluded layers, and record every mapping/loss/quantization or unsupported blend state.
- [ ] T033 [US3] Encode measured holes/liquids/shading/flags/area/modern-feature dispositions in `src/core/WowViewer.Core.IO/Conversion/Maps/ModernTerrainDisposition.cs`; unknown semantics block rather than invent data.
- [ ] T034 [US3] Materialize audited target documents and map-scope references through `src/core/WowViewer.Core.IO/Conversion/Maps/MapTargetMaterializer.cs`; use current canonical viewer new-map libraries, keep height-only patching separate, and expose MTXF/MCSE/flags/WDT/WDL/defaulted-field limitations without frozen writer/reader changes.
- [ ] T035 [US3] Compose Model/Wmo dependency subrequests and required-dependency completion rules in `src/core/WowViewer.Core.IO/Conversion/Maps/MapConversionService.cs`; terrain-only output is labelled partial and cannot satisfy SC-003.
- [ ] T036 [US3] Verify source→target layer metadata/opacity maps including base/repeated/zero/overpainted layers, boundary coordinates and hole/liquid/WDT/WDL dispositions in `tests/WowViewer.Core.Tests/Conversion/ModernMapConversionTests.cs`; use independent writer readback and explicit loss/blocked cases.
- [ ] T037 [US3] Record real CASC source→v65→Alpha terrain/placement identity, FileDataID companion coverage, hashes and outstanding object dependencies in `specs/284-map-object-converter-revitalization/evidence/modern-map-pilot.md`; prepare operator commands without whole-map execution.

**Gate**: Terrain staging accounted for. Full 1.60.1 → 0.5.3 acceptance remains open until objects and target witness.

## Phase 7 — US4: separate model conversion tool (P1)

**Independent verification**: Real static prop converted independently; animated/skinned limitations
reported precisely. Depends on foundation/client/dependency contracts.

- [ ] T038 [US4] Implement M2/MDX source/target feature inventory and profiles in `src/core/WowViewer.Core.IO/Conversion/Models/ModelConversionService.cs`; retain current Core.IO conversion where valid, leaving renderer-used duplicates untouched.
- [ ] T039 [US4] Preserve multi-material geometry, normals, UVs, winding, pivot, bounds and texture/replaceable references in `src/core/WowViewer.Core.IO/Conversion/Models/TargetModelWriter.cs`; inspect target bytes instead of accepting extension/signature alone.
- [ ] T040 [US4] Add companion resolution and supported-track/skinning checks with explicit blocked or chosen static-pose fallback in `src/core/WowViewer.Core.IO/Conversion/Models/ModelConversionService.cs`; never silently claim animation fidelity.
- [ ] T041 [US4] Wire standalone Model `inspect/plan/run/validate` in `tools/model-convert/WowViewer.Tool.ModelConvert/Program.cs`, with complete reports and no map-session requirement.
- [ ] T042 [US4] Recover useful original M2/MDX transformation semantics into `src/core/WowViewer.Core.IO/Conversion/Models/ModelConversionService.cs` and `Models/TargetModelWriter.cs`; record algorithm lineage without old CLI wrappers/dependencies or rendering-path changes.
- [ ] T043 [US4] Verify real static/multi-material output and explicit animated/skinned behavior in `tests/WowViewer.Core.Tests/Conversion/ModelConversionTests.cs`; compare geometry/material/dependency accounting and reverse-route availability separately.
- [ ] T044 [US4] Record standalone real-prop output and animated-model disposition in `specs/284-map-object-converter-revitalization/evidence/model-pilot.md`; prepare operator side-by-side visual commands, leaving appearance/animation acceptance open until witnessed.

**Gate**: Independent model route has real-file evidence and honest feature support; static prop can close its map dependency.

## Phase 8 — US5: separate WMO conversion and regrouping tool (P1)

**Independent verification**: Real multi-group/oversized WMO conversion with a topology map and
semantic dispositions. Depends on client/dependency contracts and model dependency support.

- [ ] T045 [US5] Implement `src/core/WowViewer.Core.IO/Conversion/Wmo/WmoConversionService.cs` over canonical `Core.IO/Wmo` converters, with target limits and source-group inventory; do not change `Core.IO/Converters` loading behavior.
- [ ] T046 [US5] Expose and record existing split/compaction/merge choices and source→output group mappings in `src/core/WowViewer.Core.IO/Conversion/Wmo/WmoTopologyMapBuilder.cs`; evidence-check current 384/0xBFFF constraints for the selected target.
- [ ] T047 [US5] Validate root/group counts, batches, bounds, portal remaps, BSP/collision, liquids, lighting and doodad sets/references in `src/core/WowViewer.Core.IO/Conversion/Wmo/WmoTopologyValidator.cs`; block or report current converter losses by preset.
- [ ] T048 [US5] Resolve WMO root/group/LOD and embedded model/texture closure in `src/core/WowViewer.Core.IO/Conversion/Wmo/WmoConversionService.cs`; distinguish root regrouping from real placement splitting and require explicit lossy shell selection.
- [ ] T049 [US5] Wire independent WMO operations in `tools/wmo-convert/WowViewer.Tool.WmoConvert/Program.cs` over current library services; retain original group-conversion algorithm lineage with no dependency on obsolete CLI commands or renderer changes.
- [ ] T050 [US5] Verify real WMO topology/reference preservation and split/merge boundary failures in `tests/WowViewer.Core.Tests/Conversion/WmoConversionTests.cs`; extend evidence beyond existing synthetic regrouping tests.
- [ ] T051 [US5] Record real WMO output, group maps and semantic losses in `specs/284-map-object-converter-revitalization/evidence/wmo-pilot.md`, then rerun the bounded map dependency closure in `evidence/modern-map-pilot.md`; prepare operator interior/exterior/collision checks.

**Gate**: WMO CLI is independent; bounded map has actual prop/WMO files with structural closure.

## Phase 9 — US6: separate, guided Editor tools (P1)

**Independent verification**: Same request/route/IDs/content through CLI and Editor; actual
Select → Options → Review → Run → Result walkthrough. Depends on domain services; preserve Spec 283 WIP.

- [ ] T052 [US6] Implement shared client-role picker/review/progress widgets using `SharedUiWidgets` in `src/viewer/WoWViewer/Workbench/Services/Converters/ConverterToolWidgets.cs`; concise text is authoritative and optional emoji have glyph fallback.
- [ ] T053 [P] [US6] Implement Map preset/scope/dependency/ID workflow in `src/viewer/WoWViewer/Workbench/Services/Converters/MapConverterToolService.cs`, consuming the shared request/coordinator with loaded map defaults.
- [ ] T054 [P] [US6] Implement independent Model workflow with selected-asset defaults and explicit static fallback in `src/viewer/WoWViewer/Workbench/Services/Converters/ModelConverterToolService.cs`.
- [ ] T055 [P] [US6] Implement independent WMO workflow with regrouping review in `src/viewer/WoWViewer/Workbench/Services/Converters/WmoConverterToolService.cs`; keep domain options separate from Map/Model.
- [ ] T056 [US6] Compose the three tools through `src/viewer/WoWViewer/Workbench/Pages/EditorConvertersPage.cs`, `Services/MainMenu/MainMenuBarService.Content.cs` and the existing `ViewerApp_Editor.cs` injection call; retire duplicate converter execution surfaces without new god-class state or base UI redesign.
- [ ] T057 [US6] Register controls in `eng/ui/ui-controls.json` and the inventory from `eng/ui/Get-UiControlInventory.ps1`, regenerate `docs/WoWViewer/UI-REFERENCE.md` through `eng/ui/Update-UiReference.ps1`, and verify Editor-only ownership against Spec 283; do not rerun the one-off registry seeder over authored entries.
- [ ] T058 [US6] Verify CLI/Editor canonical requests, output decisions, cancellation and source-switch behavior in `tests/WowViewer.Core.Editor.Tests/Conversion/ConverterParityTests.cs`; compare deterministic content while excluding run timestamps.
- [ ] T059 [US6] Operator-owned: execute the three Editor walkthroughs, verify readable emoji fallback, and inspect the real modern→Alpha pilot; record captures, observed behavior and open target-client gaps in `specs/284-map-object-converter-revitalization/evidence/editor-witness.md`.

**Gate**: SC-004/006 witnesses returned. SC-003 requires separate actual target acceptance.

## Phase 10 — US7: source-derived object generation attempt (P2)

**Independent verification**: One static replacement and one WMO candidate with inspectable
before/after geometry and explicit review. Depends on accepted converter foundations.

- [ ] T060 [US7] Record a bounded CPU remeshing/simplification algorithm and dependency/licensing decision in `specs/284-map-object-converter-revitalization/evidence/generation-feasibility.md`; inspect read-only legacy geometry writers as references and establish measurable per-candidate error budgets before code.
- [ ] T061 [US7] Implement source visual mesh extraction/lineage with basis, units, pivot, UV/material seams and component identity in `src/core/WowViewer.Core.IO/Conversion/Geometry/ConversionMesh.cs`; collision-only geometry cannot masquerade as a visual mesh.
- [ ] T062 [US7] Implement deterministic cleanup, partitioning and error-bounded simplification in `src/core/WowViewer.Core.IO/Conversion/Geometry/TargetMeshProcessor.cs`; report topology/bounds/material error and stop on unrepresentable limits.
- [ ] T063 [US7] Emit source-derived static prop and WMO candidates through `src/core/WowViewer.Core.IO/Conversion/Models/TargetModelWriter.cs` and `Geometry/TargetWmoMeshWriter.cs`; reuse target profiles and semantics validators without importing legacy runtime dependencies.
- [ ] T064 [US7] Implement explicit candidate accept/reject/substitute lineage and placement-child ID behavior in `src/core/WowViewer.Core.IO/Conversion/Geometry/ObjectCandidateGenerator.cs`; rejected/unvalidated output must not replace map objects.
- [ ] T065 [US7] Add explicit generation options to the separate Model/WMO hosts and `src/viewer/WoWViewer/Workbench/Services/Converters/ModelConverterToolService.cs` / `WmoConverterToolService.cs`, preserving default-off behavior and domain tool separation.
- [ ] T066 [US7] Verify seam/material/component preservation, target budgets, error reporting, missing geometry refusal and rejected-candidate behavior in `tests/WowViewer.Core.Tests/Conversion/ObjectCandidateTests.cs`; retain real source/output geometry metrics.
- [ ] T067 [US7] Operator-owned: run bounded generation and review one static prop plus one WMO candidate; record accepted/rejected decisions and visual/collision/lineage evidence in `specs/284-map-object-converter-revitalization/evidence/generated-object-witness.md`; failed attempts remain explicit.

**Gate**: SC-005 attempt and review evidenced. Successful generation support requires accepted output;
a documented failed candidate does not become a working feature.

## Phase 11 — Cross-cutting validation and final handoff

**Goal**: Close the exact operator request with layered evidence; only then scale. Depends on
the relevant story gates, with no invented target-client packaging.

- [ ] T068 Recheck harvester/v65 route integration, no old Converter project/runtime dependency, frozen-file diffs and no new `ViewerApp`/`WorldScene` members; record the containment audit in `specs/284-map-object-converter-revitalization/evidence/final-validation.md`.
- [ ] T069 Run the preferred solution build/test and UI-reference verifier commands from `quickstart.md`, compare failures with the captured baseline, and record exact commands/exits in `specs/284-map-object-converter-revitalization/evidence/final-validation.md`; do not infer runtime acceptance from green checks.
- [ ] T070 Prepare exact PowerShell 7 commands and a target-loading checklist in `specs/284-map-object-converter-revitalization/quickstart.md` using the actual measured profiles, configured roots and supported loose-content workflow; if no original-client witness path exists, record that gap instead of adding packaging/install work.
- [ ] T071 Operator-owned: load and inspect the bounded 1.60.1 → 0.5.3 result in the viewer and actual target environment, including terrain/layers/holes/liquids, object transforms/textures, WMO interior/collision and placement counts; record the build/hashes and separate witness layers in `specs/284-map-object-converter-revitalization/evidence/target-client-witness.md`.
- [ ] T072 Operator-owned: after T071 passes, run one full-map conversion and a multi-map batch through the same independent CLI, verifying deterministic mapping replay, cancellation/partial-failure reporting and dependency closure; record results in `specs/284-map-object-converter-revitalization/evidence/map-scale-witness.md`.
- [ ] T073 Publish the evidence-qualified route/feature matrix and criterion→receipt table in `specs/284-map-object-converter-revitalization/evidence/acceptance-matrix.md`; distinguish implemented/readback/viewer/target support, unavailable reverse writers and successful versus failed generation.
- [ ] T074 Reconcile only receipted task checks in `specs/284-map-object-converter-revitalization/tasks.md`, update `specs/STATUS.md`, `memory-bank/activeContext.md`, `memory-bank/progress.md` and the converter help entry under `docs/WoWViewer/`; leave unresolved runtime or target gates open.

## Dependencies and execution order

```text
Setup/Foundation (parallel with Spec 285 contracts) → US1 clients → US2 IDs
Spec 285 native/placement/transport readiness → US3-A v65 export → US3-B CASC terrain
                                                        ↓
                       US4 model → US5 WMO → complete map dependency pilot
                                           ↓
                                  US6 Editor → US7 generation
                                           ↓
                              target-client witness → map/batch scale → acceptance
```

Model/WMO service preparation can be isolated against frozen foundation contracts, but a phase is
not closed before its own evidence gate. Do not run later heavy/visual phases as an agent.

## Parallel opportunities (disjoint scopes only)

| Story | Safe bounded split after prerequisites |
|---|---|
| US1 | T012 saved/mounted selection adapter while T010–T011 implement core evidence/routing. |
| US2 | T018 allocator against frozen graph contracts while T017 builds the graph; integrate before T020. |
| US3 | T031 dependency resolver alongside T030 normalizer; both converge at T034–T035. |
| US4 | Independent read-only real-model feature survey alongside T038; all writer changes stay with one owner. |
| US5 | Independent read-only real-WMO semantic survey alongside T045; no overlapping converter edits. |
| US6 | T053/T054/T055 tool services after T052 widget contracts; one owner handles T056 composition. |
| US7 | Read-only candidate metric review alongside the processor implementation; no parallel heavy runs. |

## Implementation strategy and handoff

Start the next chat with `specs/STATUS.md` → `memory-bank/activeContext.md` → this spec/plan/tasks,
then inspect Git state. Start T001–T009 alongside Spec 285 contracts; require its native/placement/
transport readiness for map export. Spec 285 SC-004 closes jointly after this adapter exists,
not before adapter implementation. Deliver the v65 Map MVP before calling any
modern route supported. Close the bounded map/object pilot before scale; add generation as an
explicit lane. Preserve unrelated Spec 283 UI edits. Do not reopen other Epic 248/250/251 backlog
items or modify protected readers to make a converter test pass.

Task distribution: setup 3; foundation 6; US1 7; US2 7; US3 14 across two phases; US4 7; US5 7;
US6 8; US7 8; final validation/handoff 7. Total **74**, all pending.
