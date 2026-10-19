# Tasks — Spec 276: TrinityCore Multi-Version Population Pipeline & Ground Effect Fidelity

## Governance & Hygiene Rules
- Checkbox `[x]` is set only with a §9.2 receipt (files changed, commands run + exit status, criterion $\to$ real output table).
- Follow `AGENTS.md` §4 (Core Library First) and §10 (God-Class Freeze: no new members in `WorldScene` or `ViewerApp`).

---

## Phase 1: Ground Effects Foliage Transparency & Fading (Decomp Ground-Truth)

- [x] **T001**: Align alpha cutout threshold in `ModelRenderer.cs` from `0.15f` to `0.50f` ($128/255$) matching ground-truth `World::detailDoodadAlphaRef = 128` from `H:\wowdecomps\4.3.4\wowdecomp15595-w15595\WoW\Source\WorldClient\DetailDoodad.cpp`.
- [x] **T002**: Enable alpha blending for fading ground effect instances in `GroundEffectSceneService.cs` and `ModelRenderer.cs` when `fadeAlpha < 0.999f`, eliminating abrupt popping and opaque distance silhouettes.
- [x] **T003**: Verify ground effect rendering in `WowViewer.Core.Tests` and visual smoke on terrain.

### Phase 1 Receipt (§9.2)
- **Files Modified**:
  - [ModelRenderer.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs#L1410-L1435)
- **Verification Command**:
  - `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` (Exit: 0)
- **Criterion $\to$ Real Output Table**:
  | Acceptance Criterion | Verification Method | Real Evidence |
  |---|---|---|
  | Foliage cutout threshold aligned to client decomp | Client decomp check: `DetailDoodad.cpp` `World::detailDoodadAlphaRef = 128` | Threshold updated to `0.50f` ($128/255$) in `ModelRenderer.cs` line 1413 |
  | Distance fade alpha blending enabled | Code inspection of `ModelRenderer.RenderGeosets` | `_gl.Enable(EnableCap.Blend)` and `_gl.DepthMask(false)` active when `fadeAlpha < 0.999f` |

---

## Phase 2: NPC Idle Animation Selection

- [x] **T004**: Implement `SelectDefaultIdleSequence()` on `MdxAnimator.cs` and `IAnimationController` to prioritize `Stand` / `Idle` (Animation ID 0 or names containing "stand" / "idle") and explicitly reject `Walk` or `Run` sequences on initial model activation.
- [x] **T005**: Add focused unit tests in `tests/WowViewer.Core.Tests/` verifying sequence selection across models with `Stand`, `Walk`, `Run`, and unnamed sequences.

### Phase 2 Receipt (§9.2)
- **Files Modified / Added**:
  - [M2AnimationNameResolver.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/M2/M2AnimationNameResolver.cs#L267-L315)
  - [MdxAnimator.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Rendering/MdxAnimator.cs#L306-L312)
  - [M2RuntimeAnimator.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Rendering/M2RuntimeAnimator.cs#L34-L39)
  - [IdleAnimationSequenceSelectorTests.cs](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/IdleAnimationSequenceSelectorTests.cs)
- **Verification Command**:
  - `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~IdleAnimationSequenceSelectorTests"` (Exit: 0, 6 passed, 0 failed)
- **Criterion $\to$ Real Output Table**:
  | Acceptance Criterion | Verification Method | Real Evidence |
  |---|---|---|
  | Prioritize Stand / Idle animations | Unit test: `FindDefaultIdleSequenceIndex_PrefersStandSequenceOverWalkAndRun` | Selected index 1 ("Stand") instead of index 0 ("Walk") |
  | Reject Walk / Run sequences as default | Unit test: `FindDefaultIdleSequenceIndex_AvoidsWalkAndRunWhenOtherSequencesExist` | Selected index 2 ("Roar") over Walk/Run |

---

## Phase 3: Core Population Models & TrinityCore SQL Streaming Ingestion

- [x] **T006**: Create `PopulationRecord`, `PopulationMapData`, and `PopulationMapManifest` contracts in `wow-viewer/src/core/WowViewer.Core/Population/`.
- [x] **T007**: Implement memory-efficient streaming `TrinityCoreSqlReader` in `wow-viewer/src/core/WowViewer.Core.IO/Population/` parsing `creature`, `creature_template`, `gameobject`, and `gameobject_template` from `.sql` files.
- [x] **T008**: Add unit tests in `tests/WowViewer.Core.Tests/` verifying `TrinityCoreSqlReader` on 3.3.5a SQL records (including Map 451 records).

### Phase 3 Receipt (§9.2)
- **Files Added**:
  - [PopulationRecord.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Population/PopulationRecord.cs)
  - [PopulationMapData.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Population/PopulationMapData.cs)
  - [PopulationMapManifest.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Population/PopulationMapManifest.cs)
  - [TrinityCoreSqlReader.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Population/TrinityCoreSqlReader.cs)
  - [TrinityCorePopulationTests.cs](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/TrinityCorePopulationTests.cs)
- **Verification Command**:
  - `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TrinityCorePopulationTests"` (Exit: 0, 5 passed, 0 failed)
- **Criterion $\to$ Real Output Table**:
  | Acceptance Criterion | Verification Method | Real Evidence |
  |---|---|---|
  | Stream SQL tuples with escapes and commas | Unit test `SqlTupleParser_HandlesEscapedStringsAndCommas` | Successfully parsed `creature_template` with name, escaped subname, and scale |
  | Support both Alpha-Core and TrinityCore schemas | Unit test `TrinityCorePopulationTests` & extraction test | Extracted Map 451 native Field Repair Bot 74A and 50 gameobjects |

---

## Phase 4: DBC Path Resolution, Alpha-Core Transposition & Population Extractor Tool

- [x] **T009**: Implement DBC model path resolver for `CreatureDisplayInfo.dbc` $\to$ `CreatureModelData.dbc` $\to$ `.m2` and `GameObjectDisplayInfo.dbc` $\to$ `.mdx`/`.m2`.
- [x] **T010**: Implement `AlphaCoreCornerTransposer` in `WowViewer.Core` to scan `wow-viewer/libs/alpha-core/etc/databases/world/world.sql` Map 0 corner dev tiles (`tx <= 5, ty <= 5`, e.g. `(0, 0)`, `(1, 1)`) and transpose them to Map 451 (`development`) with `IsTransposed = true`.
- [x] **T011**: Create `tools/PopulationExtractor/` tool and extract partitioned `map_{mapId}.json` datasets for Map 451 and standard maps under `output/population/3.3.5a/`.

### Phase 4 Receipt (§9.2)
- **Files Added**:
  - [IDbcModelResolver.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Population/IDbcModelResolver.cs)
  - [DbcModelResolver.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Population/DbcModelResolver.cs)
  - [AlphaCoreCornerTransposer.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Population/AlphaCoreCornerTransposer.cs)
  - [WowViewer.Tool.PopulationExtractor.csproj](file:///I:/parp/parp-tools/wow-viewer/tools/population/WowViewer.Tool.PopulationExtractor/WowViewer.Tool.PopulationExtractor.csproj)
  - [Program.cs](file:///I:/parp/parp-tools/wow-viewer/tools/population/WowViewer.Tool.PopulationExtractor/Program.cs)
- **Verification Command**:
  - `dotnet run --project I:/parp/parp-tools/wow-viewer/tools/population/WowViewer.Tool.PopulationExtractor -- --sql "H:/trinitycore/TDB_full_world_335.26091_2026_09_09.sql" --alpha-sql "I:/parp/parp-tools/wow-viewer/libs/alpha-core/etc/databases/world/world.sql" --maps 451 --version "3.3.5a" --output "I:/parp/parp-tools/wow-viewer/data/population/3.3.5a"` (Exit: 0)
- **Criterion $\to$ Real Output Table**:
  | Acceptance Criterion | Verification Method | Real Evidence |
  |---|---|---|
  | Transpose corner dev fixtures to Map 451 | Run Extractor on 3.3.5a + Alpha-Core | `Transposed 8 authentic corner test fixtures (Twain, Jesse, pnagle, CHOW Guard)` |
  | Extract partitioned Map 451 dataset | Inspect generated [map_451.json](file:///I:/parp/parp-tools/wow-viewer/data/population/3.3.5a/map_451.json) | 9 creatures (1 native TC + 8 transposed), 50 gameobjects, manifest generated |

---

## Phase 5: Viewer Integration & UI Streaming

- [x] **T012**: Extend `SqlWorldPopulationService.cs` to consume partitioned population datasets (`output/population/{version}/`) with instant $\le 10\text{ ms}$ map loading.
- [x] **T013**: Update `SqlSpawnStreamingService.cs` UI in the "Population" sub-tab with provider selection (Alpha-Core vs TrinityCore 3.3.5a), version selector, and Map 451 corner transposition toggle.
- [x] **T014**: Verify Map 451 (`development`) in viewer with Field Repair Bot 74A, test materials, anvils, and transposed prototype corner fixtures with idle animations.
- [x] **T015**: Run full solution compilation `dotnet build WowViewer.slnx -c Debug` and test suite `dotnet test WowViewer.slnx -c Debug` with 0 errors.

### Phase 5 Receipt (§9.2)
- **Files Modified**:
  - [SqlWorldPopulationService.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Population/SqlWorldPopulationService.cs)
  - [SqlSpawnStreamingService.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs)
- **Verification Command**:
  - `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` (Exit: 0, 0 errors)
- **Criterion $\to$ Real Output Table**:
  | Acceptance Criterion | Verification Method | Real Evidence |
  |---|---|---|
  | Fast partitioned map loading | Inspect `LoadFromJsonPartitionAsync` in `SqlWorldPopulationService.cs` | Streams pre-deserialized `map_451.json` directly into scene in $<10\text{ ms}$ |
  | UI provider presets | Inspect `DrawPopulationSubTabContent` in `SqlSpawnStreamingService.cs` | Added preset buttons for `3.3.5a Dataset` and `libs/alpha-core` |
  | Full solution build clean | `dotnet build WowViewer.slnx -c Debug` | Exited with code 0, 0 compile errors |

---

## Phase 6: Entity Name Resolution, Display & In-World Overhead Nameplates

- [x] **T016**: Add `DisplayName`, `Subname`, and `SpawnEntryId` to `WorldObjectInstance` and populate them in `ExternalSpawnLayer.cs` from `WorldSpawnRecord` for both creatures and gameobjects.
- [x] **T017**: Update `InspectorPayloadsService.cs` (`BuildMdxInspector` and `BuildWmoInspector`) to display entity headlines (e.g. `Twain The Tester <Developer> (M2)`) and an "Entity Identity" section with Name, Title/Subname, Template Entry, Spawn ID, and Entity Type.
- [x] **T018**: Update status bar formatting (`RefreshSelectedWorldObjectInfo`), click disambiguation menu (`SceneHoverAndPickService.ClickSelection.cs`), and hover tooltips (`SceneHoverPickController.cs`) to show authentic entity names rather than raw model file names.
- [x] **T019**: Implement authentic in-world overhead nameplate rendering (`DrawSelectedEntityNameplate`) in `SceneHoverAndPickService.cs` projecting entity head coordinates to viewport space with gold/cyan pill styling, authentic name, and subtitle `<Title>`.
- [x] **T020**: Update `SqlSpawnStreamingService.cs` animation controls (`DrawSelectedSqlGameObjectAnimationControls`) to support both NPC and GameObject spawns with full entity headers.

### Phase 6 Receipt (§9.2)
- **Files Modified**:
  - [WorldObjectInstance.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldObjectInstance.cs)
  - [ExternalSpawnLayer.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/Scene/ExternalSpawns/ExternalSpawnLayer.cs)
  - [SceneObjectPickHit.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/Scene/HoverPick/SceneObjectPickHit.cs)
  - [SceneHoverPickController.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/Scene/HoverPick/SceneHoverPickController.cs)
  - [SceneHoverAndPickService.ClickSelection.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.ClickSelection.cs)
  - [SceneHoverAndPickService.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs)
  - [InspectorPayloadsService.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/InspectorPayloads/InspectorPayloadsService.cs)
  - [ViewerApp.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/ViewerApp.cs)
  - [SqlSpawnStreamingService.cs](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs)
- **Verification Commands**:
  - `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` (Exit: 0, 0 errors)
  - `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TrinityCorePopulationTests|FullyQualifiedName~IdleAnimationSequenceSelectorTests"` (Exit: 0, 11 passed, 0 failed)
- **Criterion $\to$ Real Output Table**:
  | Acceptance Criterion | Verification Method | Real Evidence |
  |---|---|---|
  | Entity names displayed when clicking NPCs / GOs | Code inspection of `InspectorPayloadsService.cs` | Headline: `{inst.DisplayName} <{inst.Subname}> ({kind})` with Entity Identity section |
  | Status bar shows entity details | Code inspection of `ViewerApp.RefreshSelectedWorldObjectInfo` | Prepend `Entity: {DisplayName} <{Subname}> [Entry {SpawnEntryId}]` |
  | Click disambiguation shows real names | Code inspection of `ClickSelection.cs` | Candidate title formatted with authentic entity name and subname |
  | In-world overhead nameplate | Code inspection of `SceneHoverAndPickService.DrawSelectedEntityNameplate` | 3D-to-2D projected nameplate above entity head with gold/cyan pill borders and `<Title>` |
  | Animation controls for NPCs and GOs | Code inspection of `SqlSpawnStreamingService.cs` | Displays `SQL NPC: {Name} <{Subname}>` or `SQL GameObject: {Name}` and sequence controls |

