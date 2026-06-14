---
description: "Task list for the Unreal Engine bridge feature"
---

# Tasks: Unreal Engine Bridge

**Input**: Design documents from `/specs/055-unreal-engine-bridge/`
- `spec.md` — Feature specification
- `plan.md` — Implementation plan
- `analysis.md` — Risk and gap analysis

**Prerequisites**: `plan.md` (required), `spec.md` (required for user stories)

**Tests**: The spec explicitly requests real-data validation at every phase. Test tasks are included per story.

**Organization**: Tasks are grouped by phase. Each phase ends with a checkpoint that validates the phase's user stories (where applicable).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4, US5, US6)
- **[Foundation]**: Foundational work that blocks all user stories
- **[Polish]**: Cross-cutting work that depends on user stories
- Include exact file paths in descriptions

## Path Conventions

- Bridge C# code: `wow-viewer/src/core/WowViewer.Core.Bridge/`
- Bridge C# tests: `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/`
- UE plugin: `wow-viewer/external/ue-bridge/`
- Contracts/docs: `wow-viewer/specs/055-unreal-engine-bridge/contracts/`
- Research outputs: `wow-viewer/specs/055-unreal-engine-bridge/research/`

---

## Phase 1: Pre-Work Spike (Foundation De-Risking)

**Purpose**: Resolve unknowns that could derail the architecture: AOT compatibility, UE plugin development, UE version pinning, test map matrix.

- [ ] T001 [Foundation] Audit AOT compatibility of all `WowViewer.Core.*` projects — document any reflection-emit, dynamic loading, or runtime code generation usage in `wow-viewer/specs/055-unreal-engine-bridge/research/aot-audit.md`
- [ ] T002 [Foundation] [P] Pin UE 5.4 as the target version, document the rationale and the installed engine path in `wow-viewer/specs/055-unreal-engine-bridge/research/ue-version.md`
- [ ] T003 [Foundation] [P] Define the test map matrix (which staged clients and which maps serve as the validation surface for each user story) in `wow-viewer/specs/055-unreal-engine-bridge/contracts/test-maps.md`
- [ ] T004 [Foundation] Create a UE 5.4 plugin spike — minimal C++ module that loads a tiny Native AOT-compiled C# DLL and calls one exported function, logging the result. Code under `wow-viewer/external/ue-bridge/Spike/`. Report in `research/ue-spike.md`
- [ ] T005 [Foundation] Verify the AOT publish flow on a minimal C# project — confirm `dotnet publish -c Release -r win-x64` produces a self-contained native DLL that loads from C/C++ via `LoadLibrary` and `GetProcAddress`. Document the build flags used in `research/aot-audit.md`

**Checkpoint**: Pre-work spike complete. Architecture risks (AOT, UE plugin development, version lock) are de-risked. Go/no-go decision recorded in `research/ue-spike.md`.

---

## Phase 2: Bridge Foundation (C API + AOT Build + UE Plugin Skeleton)

**Purpose**: Establish the structural skeleton of the bridge. Blocks all user stories (US1-US6).

- [ ] T006 [Foundation] Create the `WowViewer.Core.Bridge` C# project at `wow-viewer/src/core/WowViewer.Core.Bridge/WowViewer.Core.Bridge.csproj` targeting `net10.0` with Native AOT enabled
- [ ] T007 [Foundation] [P] Create the xUnit test project at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/WowViewer.Core.Bridge.Tests.csproj` referencing the bridge project
- [ ] T008 [Foundation] [P] Create the UE 5.4 plugin skeleton at `wow-viewer/external/ue-bridge/WowViewerBridge.uplugin` and the module folder `Source/WowViewerBridge/` with `WowViewerBridge.Build.cs`
- [ ] T009 [Foundation] Define the C API surface (function signatures, POD structs, handle types, error codes) in `wow-viewer/specs/055-unreal-engine-bridge/contracts/c-api.md`
- [ ] T010 [Foundation] Define the memory ownership contract (who allocates, who frees, when) in `wow-viewer/specs/055-unreal-engine-bridge/contracts/memory.md`
- [ ] T011 [Foundation] Implement `BridgeExports` class at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/BridgeExports.cs` with `[UnmanagedCallersOnly]` stubs for `BridgeInit`, `BridgeShutdown`, `BridgeGetVersion`
- [ ] T012 [Foundation] Implement `BridgeHandles` at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/BridgeHandles.cs` — handle → object dictionary with reference counting
- [ ] T013 [Foundation] Implement `BridgeErrors` at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/BridgeErrors.cs` — error code enum + UTF-8 error string buffer pattern
- [ ] T014 [Foundation] Implement `BridgeLoader` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/BridgeLoader.h` and `Private/BridgeLoader.cpp` — `LoadLibrary` + `GetProcAddress` with safe trampolines
- [ ] T015 [Foundation] Implement `BridgeSettings` UE subsystem at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Settings/BridgeSettings.h` — DLL path configuration
- [ ] T016 [Foundation] Wire plugin lifecycle (`StartupModule` / `ShutdownModule`) to call `BridgeInit` / `BridgeShutdown`
- [ ] T017 [Foundation] [P] Add the `WowViewerBridgeEditor` editor module with a placeholder `WowViewer.Status` console command that prints the bridge version

**Checkpoint**: UE plugin loads the AOT DLL, calls `BridgeInit`, `WowViewer.Status` console command works. AOT DLL builds in < 5 min. Zero memory leaks across plugin enable/disable cycles.

---

## Phase 3: BLP Texture Bridge (Foundation for All Visual Assets)

**Purpose**: Establish the texture decode pipeline. Terrain, WMO, M2 all consume BLP textures.

- [ ] T018 [P] [Foundation] Implement `BlpExports` C API functions (`BlpLoadFile`, `BlpGetInfo`, `BlpDecodeMip`) at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/BlpExports.cs` calling into existing `WowViewer.Core.IO` BLP readers
- [ ] T019 [P] [Foundation] Define the `BlpTexture` POD struct in `wow-viewer/specs/055-unreal-engine-bridge/contracts/c-api.md` (width, height, mip count, format flags, RGBA8 mip data pointer)
- [ ] T020 [P] [Foundation] Add xUnit tests for `BlpExports` at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/BlpExportsTests.cs` — decode 1x1, 256x256, 1024x1024, DXT1/DXT3/DXT5, palettized against `output/tmp/wowarchive-clients/0.5.3-development/Textures/`
- [ ] T021 [Foundation] Implement `BlpConverter` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Converters/BlpConverter.h/.cpp` — convert C struct → `UTexture2D`, upload via `FTexture2DMipMap` + `UpdateResource()`
- [ ] T022 [Foundation] Implement texture caching subsystem (LRU cache for decoded `UTexture2D` instances) at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Subsystems/TextureCache.h/.cpp`
- [ ] T023 [Foundation] Add UE Automation tests for `BlpConverter` at `wow-viewer/external/ue-bridge/Tests/BlpConverterTests.cpp` — decode + upload + verify dimensions match
- [ ] T024 [Foundation] Add the `WowViewer.DecodeBlp <path>` console command at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Private/Commands/DecodeBlpCommand.cpp` — loads a BLP and displays it in a UE UI widget for visual confirmation

**Checkpoint**: A BLP file from `0.5.3/Textures/Minimap/map0.blp` loads via console command, displays as a `UTexture2D` in a UE widget. 100 BLP decodes complete in < 30s.

---

## Phase 4: User Story 1 — Terrain Explorer

**Goal**: Load a full WoW map's terrain into UE Landscape with correct heightmaps, alpha layers, and hole masks.
**Independent Test**: `WowViewer.GenerateTerrain 0.5.3-development/development` produces a complete UE Landscape matching the source terrain within visual inspection tolerance.

- [ ] T025 [P] [US1] Implement `TerrainExports` C API (`TerrainLoadMap`, `TerrainGetTileCount`, `TerrainGetTile`, `TerrainFreeTile`) at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/TerrainExports.cs`
- [ ] T026 [P] [US1] Define the `TerrainTile` POD struct in `wow-viewer/specs/055-unreal-engine-bridge/contracts/c-api.md` (145x145 heightmap, 64x64 alpha layers, hole mask, layer count, texture layer assignments)
- [ ] T027 [P] [US1] Define the WoW-to-UE material translation table (terrain layer flags, blend modes, two-sided, etc.) in `wow-viewer/specs/055-unreal-engine-bridge/contracts/materials.md`
- [ ] T028 [P] [US1] Add xUnit tests for `TerrainExports` at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/TerrainExportsTests.cs` — terrain tile decode against `output/tmp/wowarchive-clients/0.5.3-development/Maps/development/`
- [ ] T029 [US1] Implement `TerrainConverter` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Converters/TerrainConverter.h/.cpp` — convert C struct → UE Landscape height/weight data
- [ ] T030 [US1] Implement `WoWTileActor` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Actors/WoWTileActor.h/.cpp` — UE actor class wrapping `ALandscapeProxy`
- [ ] T031 [US1] Implement the `WowViewer.GenerateTerrain` console command at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Private/Commands/GenerateTerrainCommand.cpp`
- [ ] T032 [US1] Add UE Automation tests for `TerrainConverter` at `wow-viewer/external/ue-bridge/Tests/TerrainConverterTests.cpp` — heightmap + alpha layer round-trip verification
- [ ] T033 [US1] End-to-end validation: load `0.5.3-development/development` map terrain in UE, screenshot, visually verify against reference. Document in `wow-viewer/specs/055-unreal-engine-bridge/research/us1-validation.md`

**Checkpoint**: US1 acceptance scenarios 1-3 pass. Terrain loads, heightmap correct, alpha-blended textures visible, hole masks present.

---

## Phase 5: User Story 2 — Model Inspector (Static Meshes: WMO)

**Goal**: Load WMO group files as UE static mesh actors with correct geometry, materials, and portals.
**Independent Test**: `WowViewer.SpawnWmo <path>` produces a static mesh actor with correct vertex count and material assignments for any WMO from `0.5.3`.

- [ ] T034 [P] [US2] Implement `WmoExports` C API (`WmoLoadFile`, `WmoGetGroupCount`, `WmoGetGroup`, `WmoFreeGroup`) at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/WmoExports.cs`
- [ ] T035 [P] [US2] Define the `WmoGroup` POD struct in `wow-viewer/specs/055-unreal-engine-bridge/contracts/c-api.md` (vertex array, index array, UV array, normal array, material slots, portal data)
- [ ] T036 [P] [US2] Add xUnit tests for `WmoExports` at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/WmoExportsTests.cs` — WMO group decode against test maps
- [ ] T037 [US2] Implement `WmoConverter` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Converters/WmoConverter.h/.cpp` — convert C struct → `UStaticMesh`
- [ ] T038 [US2] Implement `WoWWmoActor` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Actors/WoWWmoActor.h/.cpp` — UE actor with `UStaticMeshComponent` per group
- [ ] T039 [US2] Implement WMO material-to-UE-material conversion (extend `contracts/materials.md` with WMO-specific flags, blends, dual-texture, two-sided)
- [ ] T040 [US2] Implement the `WowViewer.SpawnWmo` console command at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Private/Commands/SpawnWmoCommand.cpp`
- [ ] T041 [US2] Add UE Automation tests for `WmoConverter` at `wow-viewer/external/ue-bridge/Tests/WmoConverterTests.cpp` — mesh conversion + material round-trip
- [ ] T042 [US2] End-to-end validation: spawn a known WMO from `0.5.3-development` (e.g., a Stormwind building), verify vertex count and material count. Document in `research/us2-wmo-validation.md`

**Checkpoint**: US2 acceptance scenarios 1-3 (WMO portion) pass. WMO loads as static mesh, geometry correct, materials applied, BLP textures visible.

---

## Phase 6: User Story 4 — Animation Playback (M2/MDX)

**Goal**: Load M2/MDX models as UE skeletal mesh actors with animation playback.
**Independent Test**: `WowViewer.SpawnM2 <path> --anim 0` loads any M2 from `0.5.3` and plays the requested animation at correct bone transforms.

- [ ] T043 [P] [US2/US4] Implement `M2Exports` C API (`M2LoadFile`, `M2GetSkinCount`, `M2GetSkin`, `M2GetAnimationCount`, `M2GetAnimation`, `M2SampleAnimation`, `M2FreeModel`) at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/M2Exports.cs`
- [ ] T044 [P] [US2/US4] Define `M2Model`, `M2Skin`, `M2Animation` POD structs in `wow-viewer/specs/055-unreal-engine-bridge/contracts/c-api.md`
- [ ] T045 [US4] Implement animation sampling in C# (Bezier interpolation, compressed track expansion) at `wow-viewer/src/core/WowViewer.Core.Bridge/Animation/M2AnimationSampler.cs`
- [ ] T046 [P] [US4] Define the animation fidelity contract (sampling rate, keyframe accuracy, raw-vs-baked comparison) in `wow-viewer/specs/055-unreal-engine-bridge/contracts/animation.md`
- [ ] T047 [P] [US2/US4] Add xUnit tests for `M2Exports` and animation sampling at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/M2ExportsTests.cs` — fidelity test: baked C# output vs raw interpolated values from a known reference frame
- [ ] T048 [US2/US4] Implement `M2Converter` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Converters/M2Converter.h/.cpp` — skin → `USkeletalMesh`, animation → `UAnimSequence`
- [ ] T049 [US2/US4] Implement `WoWM2Actor` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Actors/WoWM2Actor.h/.cpp` — UE actor with `USkeletalMeshComponent`
- [ ] T050 [US2/US4] Implement the `WowViewer.SpawnM2` console command at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Private/Commands/SpawnM2Command.cpp` (with `--anim <id>` flag)
- [ ] T051 [US2/US4] Add UE Automation tests for `M2Converter` at `wow-viewer/external/ue-bridge/Tests/M2ConverterTests.cpp` — skeletal mesh + animation playback
- [ ] T052 [US2/US4] End-to-end validation: spawn a known M2 from `0.5.3` (e.g., a humanoid or creature), play animation 0, verify bone transforms. Document in `research/us2-m2-validation.md` and `research/us4-validation.md`

**Checkpoint**: US2 fully complete (WMO + M2). US4 acceptance scenarios 1-2 pass. M2 loads, animations play, bone transforms correct, skin deformation matches reference.

---

## Phase 7: User Story 3 — World Composition

**Goal**: Load a complete WoW map (terrain + WMO + M2 placements + liquids) into UE as a single explorable world.
**Independent Test**: `WowViewer.LoadMap 0.5.3-development/development` produces a complete world with all terrain, models, and liquids correctly positioned.

- [ ] T053 [P] [US3] Implement `LiquidExports` C API (`LiquidGetMapLiquids`, `LiquidGetChunk`, `LiquidFreeChunk`) at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/LiquidExports.cs`
- [ ] T054 [P] [US3] Define the `LiquidChunk` POD struct in `wow-viewer/specs/055-unreal-engine-bridge/contracts/c-api.md` (vertex heights, render flags, liquid type ID, geometry type)
- [ ] T055 [P] [US3] Add xUnit tests for `LiquidExports` at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/LiquidExportsTests.cs` — MH2O and MCLQ decoding
- [ ] T056 [US3] Implement `LiquidConverter` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Converters/LiquidConverter.h/.cpp` — chunk → UE water plane or spline mesh
- [ ] T057 [US3] Implement `MapExports` (`MapLoad`, `MapGetObjectPlacements`) C API at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/MapExports.cs` — exposes existing ADT placement extraction (MCRF, MCAD, MCRE)
- [ ] T058 [US3] Add xUnit tests for `MapExports` at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/MapExportsTests.cs` — placement extraction from `0.5.3-development`
- [ ] T059 [US3] Implement the `WowViewer.LoadMap` console command at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Private/Commands/LoadMapCommand.cpp` — orchestrates terrain, models, liquids, placements
- [ ] T060 [US3] Add UE Automation tests for full `LoadMap` flow at `wow-viewer/external/ue-bridge/Tests/EndToEndTests.cpp` — full `LoadMap` against `0.5.3-development`
- [ ] T061 [US3] End-to-end validation: load `0.5.3-development/development` in UE, verify all ADT objects present, liquids rendered, object transforms correct. Document in `research/us3-validation.md`

**Checkpoint**: US3 acceptance scenarios 1-3 pass. Full map loads with all components. Real-data map loads in < 30s.

---

## Phase 8: User Story 5 — Game Manager (Multi-Root Support)

**Goal**: UE plugin supports registering and switching between multiple staged client roots.
**Independent Test**: Two client roots registered via UE editor panel; switching active root updates all subsequent `LoadMap` calls to use the correct root.

- [ ] T062 [P] [US5] Implement `ArchiveExports` C API (`ClientRootRegister`, `ClientRootList`, `ClientRootSetActive`, `ClientRootGetActive`) at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/ArchiveExports.cs`
- [ ] T063 [P] [US5] Add xUnit tests for `ArchiveExports` at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/ArchiveExportsTests.cs` — register/list/switch flow
- [ ] T064 [US5] Implement `GameManagerSubsystem` (`UEditorSubsystem`) at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Subsystems/GameManagerSubsystem.h/.cpp`
- [ ] T065 [US5] Persist client roots in UE project settings (`.uproject` config) at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Settings/BridgeSettings.h` (extend)
- [ ] T066 [US5] Implement the plugin UI panel (`SCompoundWidget`) at `wow-viewer/external/ue-bridge/Source/WowViewerBridgeEditor/Public/GameManagerPanel.h/.cpp` — list of registered roots, active root indicator, register/switch buttons
- [ ] T067 [US5] Implement root switching logic — active root update propagates to all subsystems (terrain, model, liquid, map loaders)
- [ ] T068 [US5] Add UE Automation tests for `GameManagerSubsystem` at `wow-viewer/external/ue-bridge/Tests/GameManagerTests.cpp`
- [ ] T069 [US5] End-to-end validation: register two client roots (e.g., `0.5.3-development` and `1.12.x-test`), switch between them, load maps from each, verify correct data source. Document in `research/us5-validation.md`

**Checkpoint**: US5 acceptance scenarios 1-2 pass. Roots persist across editor restarts. Switching is atomic.

---

## Phase 9: User Story 6 — DBC/DB2 Data Tables

**Goal**: Expose DBC/DB2 table data as UE DataTable assets.
**Independent Test**: Right-clicking `AreaTable.dbc` in UE content browser creates a `UDataTable` asset with correct column names, types, and row count.

- [ ] T070 [P] [US6] Implement `DbcExports` C API (`DbcLoadTable`, `DbcGetSchema`, `DbcGetRowCount`, `DbcGetRow`, `DbcFreeTable`) at `wow-viewer/src/core/WowViewer.Core.Bridge/Exports/DbcExports.cs`
- [ ] T071 [P] [US6] Define the `DbcSchema` POD struct in `wow-viewer/specs/055-unreal-engine-bridge/contracts/c-api.md` (column name array, column type array, row count)
- [ ] T072 [US6] Implement DBC string table lookup (for `stringref` columns) at `wow-viewer/src/core/WowViewer.Core.Bridge/Dbc/DbcStringTable.cs`
- [ ] T073 [P] [US6] Add xUnit tests for `DbcExports` at `wow-viewer/src/tests/WowViewer.Core.Bridge.Tests/ExportTests/DbcExportsTests.cs` — `AreaTable.dbc`, `Map.dbc`, `LiquidType.dbc` against `0.5.3`
- [ ] T074 [US6] Implement `DbcConverter` at `wow-viewer/external/ue-bridge/Source/WowViewerBridge/Public/Converters/DbcConverter.h/.cpp` — schema + rows → `UDataTable` with `FTableRowBase` derived UStruct
- [ ] T075 [US6] Add the `DbcImport` UE asset action at `wow-viewer/external/ue-bridge/Source/WowViewerBridgeEditor/Private/DbcAssetAction.cpp` — right-click `.dbc` file in content browser → import as DataTable
- [ ] T076 [US6] Add UE Automation tests for `DbcConverter` at `wow-viewer/external/ue-bridge/Tests/DbcConverterTests.cpp` — DataTable round-trip verification
- [ ] T077 [US6] End-to-end validation: import `AreaTable.dbc` from `0.5.3`, verify column count and row count, spot-check 3 rows for correct values. Document in `research/us6-validation.md`

**Checkpoint**: US6 acceptance scenarios 1-2 pass. DBC tables import as DataTables, schema correct, string lookups work.

---

## Phase 10: Polish, Build Pipeline, and Documentation

**Purpose**: Make the bridge shippable. Cross-cutting work that depends on all user stories.

- [ ] T078 [P] [Polish] Create `wow-viewer/scripts/build-bridge.ps1` — produces the AOT DLL and copies it into `wow-viewer/external/ue-bridge/ThirdParty/wowviewer_bridge/`
- [ ] T079 [P] [Polish] Create `wow-viewer/scripts/test-bridge.ps1` — CI script that loads each (client, map) pair from `contracts/test-maps.md` and reports success/failure
- [x] T080 [Polish] `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` — replaced 2026-06-14: viewer-first, libraries bridge to UE. No Vulkan, no WebGL, no Museums.
- [ ] T081 [P] [Polish] Update `wow-viewer/docs/architecture/game-viewer-host-plan-2026-05-13.md` — reflect that `WowViewer.App` is now a CLI/diagnostic host
- [ ] T082 [P] [Polish] Update `wow-viewer/docs/architecture/wow-viewer-library-completeness-plan-2026-05-06.md` — mark renderer gaps as deprecated
- [ ] T083 [P] [Polish] Update memory bank: `wow-viewer/memory-bank/activeContext.md` and `progress.md` with bridge status, phase count, current state
- [ ] T084 [P] [Polish] Write `wow-viewer/external/ue-bridge/README.md` — architecture overview, how to add a new format family, build instructions
- [ ] T085 [P] [Polish] Update `wow-viewer/docs/architecture/speckit-doc-audit-2026-05-18.md` with the new bridge spec/plan/tasks entries
- [ ] T086 [Polish] Final pass: verify every FR-001 through FR-010 in spec.md is satisfied, every SC-001 through SC-006 is measured, every acceptance scenario in US1-US6 has a passing validation log
- [ ] T087 [Polish] Run quickstart-style validation: fresh clone of `wow-viewer` → run `scripts/build-bridge.ps1` → open UE project → run `WowViewer.LoadMap 0.5.3-development/development` → verify world loads. Document in `research/final-validation.md`

**Checkpoint**: Feature complete. Every success criterion met. Every architecture doc aligned. Build pipeline automated.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Pre-Work Spike)**: No dependencies. Must complete before Phase 2.
- **Phase 2 (Bridge Foundation)**: Depends on Phase 1. Blocks all user story work.
- **Phase 3 (BLP Bridge)**: Depends on Phase 2. Blocks all visual user stories (US1, US2, US3, US4).
- **Phase 4 (US1 Terrain)**: Depends on Phase 3. US1 can be implemented in parallel with Phase 5/6/7/8/9.
- **Phase 5 (US2 WMO)**: Depends on Phase 3. US2-WMO can be implemented in parallel with Phase 4.
- **Phase 6 (US2-M2 + US4)**: Depends on Phase 3 and Phase 5 (shares some conversion infrastructure). Can start after Phase 5.
- **Phase 7 (US3 World Composition)**: Depends on Phase 4, 5, 6. Orchestrates everything.
- **Phase 8 (US5 Game Manager)**: Depends on Phase 2 only. Can run in parallel with US1-US4 once Phase 2 is done.
- **Phase 9 (US6 DBC/DB2)**: Depends on Phase 2 only. Can run in parallel with US1-US5 once Phase 2 is done.
- **Phase 10 (Polish)**: Depends on all user story phases (4-9) being complete.

### User Story Dependencies

- **US1 (Terrain)**: Depends on Phase 2 (foundation) + Phase 3 (BLP). No other user story deps.
- **US2 (Model Inspector — WMO + M2)**: Depends on Phase 2 + Phase 3. WMO and M2 share converter infrastructure.
- **US3 (World Composition)**: Depends on US1, US2-WMO, US2-M2. Orchestrates their outputs.
- **US4 (Animation Playback)**: Depends on US2-M2. Extends M2 work.
- **US5 (Game Manager)**: Depends on Phase 2 only. Independent of all other stories.
- **US6 (DBC/DB2)**: Depends on Phase 2 only. Independent of all other stories.

### Within Each Phase

- Tests (where present) should be written and FAIL before implementation
- Models / structs / API definitions before implementations
- C# side (bridge library) before UE C++ side (consumer)
- Story complete and validated before moving to the next priority

### Parallel Opportunities

- **Phase 1**: T002 and T003 can run in parallel (different files).
- **Phase 2**: T006, T007, T008 can run in parallel (different project files). T009 and T010 can run in parallel with T006-T008 (different files). T011-T013 depend on T009. T014-T016 depend on T008.
- **Phase 3**: T018 and T019 can run in parallel. T020 depends on T018/T019. T021-T024 are sequential (converter → cache → test → command).
- **Phase 4-6**: Each user story phase is largely independent. Within each phase, the test task (T028, T036, T047) can run in parallel with API struct definitions (T026, T035, T044).
- **Phase 8 (US5) and Phase 9 (US6)**: Can run fully in parallel with each other and with Phases 4-7 once Phase 2 is done.
- **Phase 10**: T078, T079, T080, T081, T082, T083, T084, T085 can all run in parallel (different files). T086 and T087 are final sequential passes.

---

## Parallel Execution Strategy

With the project being driven primarily by the user (single-developer mode), the realistic execution order is:

1. **Phase 1** (sequential, ~1-2 days)
2. **Phase 2** (sequential, foundation; ~3-5 days)
3. **Phase 3** (sequential, BLP bridge; ~2-3 days)
4. **Phase 4** (US1, sequential; ~5-7 days)
5. **Phase 5** (US2-WMO, sequential; ~5-7 days)
6. **Phase 6** (US2-M2 + US4, sequential; ~7-10 days)
7. **Phase 7** (US3, sequential; ~3-5 days)
8. **Phase 8 and 9 in parallel** (US5 + US6, can be done together since both are independent; ~5-7 days)
9. **Phase 10** (polish, sequential; ~3-5 days)

**Total estimated time**: ~7-10 weeks for a single developer, faster if split.

---

## Implementation Strategy

### MVP First (Phase 1 + 2 + 3 + 4)

1. Complete Phase 1: Pre-Work Spike
2. Complete Phase 2: Bridge Foundation
3. Complete Phase 3: BLP Bridge
4. Complete Phase 4: US1 Terrain Explorer
5. **STOP and VALIDATE**: The MVP is a UE editor where you can load a WoW map's terrain with correct textures. This is the most demonstrable single slice of the bridge.

### Incremental Delivery

1. Phases 1-2 → Foundation ready (DLL loads, plugin skeleton works)
2. Phase 3 → BLP textures work in UE
3. Phase 4 → **MVP: terrain loads in UE** (this is the wow moment)
4. Phase 5 → WMO models spawn in UE
5. Phase 6 → M2 models with animations spawn in UE
6. Phase 7 → Full worlds load in UE
7. Phase 8 → Multi-root switching works
8. Phase 9 → DBC data accessible as DataTables
9. Phase 10 → Polished, documented, shippable

Each step adds value without breaking previous steps.

---

## Notes

- All file paths are relative to `I:\parp\parp-tools\` unless otherwise specified.
- `[P]` tasks can run concurrently only if they touch different files.
- Test tasks (T020, T028, T036, T047, T055, T058, T063, T073, etc.) should be written BEFORE the implementation they verify.
- Commit after each task or logical group (per AGENTS.md review/rework process).
- At every checkpoint, run the spec's "Mandatory Review & Rework Process" (review for bugs, update docs, commit).
- All real-data validation must use `output/tmp/wowarchive-clients/` paths. Never `H:\CLIENTS`.
- Particle/ribbon emitters are explicitly deferred from this feature (analysis recommendation). May be added in a future spec.
- LOD policy for M2 models uses UE's built-in LOD system consuming WoW's 1-4 LOD levels (delivered as part of Phase 6, M2Converter).
- WMO portals are decoded (Phase 5) but not yet used for culling. May be added in a future spec.
