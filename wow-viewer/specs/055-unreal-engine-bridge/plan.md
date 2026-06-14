# Implementation Plan: Unreal Engine Bridge

**Branch**: `055-unreal-engine-bridge` | **Date**: 2026-06-09 | **Spec**: [`../spec.md`](../spec.md)

**Input**: Feature specification from `/specs/055-unreal-engine-bridge/spec.md`

## Summary

Replace the Silk.NET OpenGL viewer in `WowViewer.App` with an Unreal Engine 5.x plugin that consumes the existing C# format/data libraries through a .NET 10 Native AOT-compiled C API. All format parsing stays in `WowViewer.Core` / `WowViewer.Core.IO` / `WowViewer.Core.Runtime`. The UE plugin owns the conversion from flat C data structures to UE-native types (UStaticMesh, USkeletalMesh, UTexture2D, ALandscapeProxy, etc.) and the engine-native workflow surface (Blueprint nodes, console commands, asset import actions, editor settings).

This plan defines the Unreal Engine bridge for `wow-viewer` libraries. The old `wow-engine-modernization-plan-2026-05-14.md` (Vulkan-first) was replaced 2026-06-14 — viewer-first with UE bridge, not a custom engine stack.

## Technical Context

**Language/Version**:
- C# / .NET 10 (bridge library, Native AOT compilation target)
- C++ / UE 5.4 (plugin code, target platform: Windows x64 for initial delivery)

**Primary Dependencies**:
- .NET 10 SDK with Native AOT support
- Unreal Engine 5.4 (LTS-class stability, C++20 support, mature landscape system)
- Visual Studio 2022 (UE C++ toolchain)
- Existing `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.PM4`, `WowViewer.Core.Runtime` (no changes required to these for the bridge itself; possibly minor AOT-friendly adjustments identified in Phase 0)

**Storage**: N/A (bridge is stateless export surface; data lives in the staged client root).

**Testing**:
- C# side: xUnit tests for the new bridge export surface (bridge C API wrapper classes)
- UE C++ side: UE Automation Framework tests for the C API consumers and type conversion logic
- End-to-end: test maps under `output/tmp/wowarchive-clients/` validated via UE editor session

**Target Platform**: Windows x64 first. Linux/Mac not in initial scope; AOT C API is portable but UE plugin must be recompiled per platform.

**Project Type**: library (C# AOT bridge) + UE plugin (C++) + UE test project

**Performance Goals**:
- Real-time framerate (≥30 FPS) for a loaded map with terrain + WMO + M2 placements in UE viewport
- AOT DLL cold-start < 2s
- BLP decode (4096x4096) < 200ms per texture on the C# side
- Memory ownership contract: zero leaks across 1000+ actor spawns

**Constraints**:
- No .NET runtime dependency at runtime (Native AOT)
- No format parser code may move to C++/UE
- C API uses only flat C types (no C++ ABI)
- C# libraries must build cleanly with `dotnet build` for existing tests
- AOT build time < 5 min
- UE plugin must build against installed UE 5.4 with no manual patching

**Scale/Scope**:
- 6 format families (terrain, WMO, M2/MDX, BLP, liquid, DBC/DB2) + archive access
- 2-3 staged client roots as validation surface
- 1 UE test project + 1 UE plugin module

## Constitution Check

*Gate: Must pass before Phase 0. Re-check after each phase.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All C# code stays in `wow-viewer/src/core/`. UE plugin source lives in `wow-viewer/external/ue-bridge/` (or similar). No `gillijimproject_refactor/` references. |
| II. Library-First | PASS | Bridge is a thin export surface. No format parsing moves to UE. New C# wrapper library is added to `wow-viewer/src/core/`. |
| III. Real-Data Validation | PASS | Every phase ends with a test against `output/tmp/wowarchive-clients/`. |
| IV. Residual Model Chain | N/A | No ML changes. |
| V. Streaming-First Dataset Pipeline | N/A | No dataset changes. |
| VI. No Game Client Path Assumptions | PASS | All paths are `output/tmp/wowarchive-clients/...`. |
| Read-Only Reference Codebase | PASS | UE plugin is a new `wow-viewer/external/` subdir; `gillijimproject_refactor/` is untouched. |
| Format Reader/Writer Ownership | PASS | Bridge calls into existing readers; no new format code in the bridge. |
| AlphaWdtWriter Frozen | N/A | Bridge does not write ADTs. |
| One Phase at a Time | PASS | Plan is decomposed into 9 phases, each independently validatable. |
| Bite-Sized Plans | PASS | Each phase has ≤10 small, focused steps (in `tasks.md`). |

## Project Structure

### Documentation (this feature)

```text
specs/055-unreal-engine-bridge/
├── spec.md            # The feature spec
├── analysis.md        # The analyze step output (risks, gaps, revisions)
├── plan.md            # This file
├── tasks.md           # Concrete task breakdown
├── contracts/
│   ├── c-api.md       # C API surface (P/Invoke, structs, function signatures)
│   ├── memory.md      # Memory ownership contract
│   ├── materials.md   # WoW → UE material translation table
│   ├── animation.md   # Animation fidelity contract
│   └── test-maps.md   # Test map matrix
└── research/
    ├── aot-audit.md   # Phase 0: AOT compatibility audit of C# libraries
    ├── ue-spike.md    # Phase 0: UE plugin creation + AOT DLL loading spike
    └── ue-version.md  # Phase 0: UE 5.4 selection rationale
```

### Source Code (new in this feature)

```text
wow-viewer/
├── src/
│   ├── core/
│   │   ├── WowViewer.Core.Bridge/        # NEW: C# wrapper that exports the C API
│   │   │   ├── WowViewer.Core.Bridge.csproj
│   │   │   ├── Exports/
│   │   │   │   ├── BridgeExports.cs     # [UnmanagedCallersOnly] entry points
│   │   │   │   ├── BridgeHandles.cs     # Handle ↔ object mapping
│   │   │   │   ├── BridgeErrors.cs      # Error code + string model
│   │   │   │   ├── BridgeMemory.cs      # Allocator contract
│   │   │   │   ├── ArchiveExports.cs    # Archive + file system exports
│   │   │   │   ├── TerrainExports.cs    # Terrain tile data exports
│   │   │   │   ├── WmoExports.cs        # WMO mesh + material exports
│   │   │   │   ├── M2Exports.cs         # M2/MDX model + animation exports
│   │   │   │   ├── BlpExports.cs        # BLP texture decode exports
│   │   │   │   ├── LiquidExports.cs     # MH2O/MCLQ exports
│   │   │   │   ├── DbcExports.cs        # DBC/DB2 table exports
│   │   │   │   └── Pm4Exports.cs        # PM4 query exports
│   │   │   └── Bridge.cs                # Top-level bridge init/shutdown
│   │   └── ...
│   └── tests/
│       └── WowViewer.Core.Bridge.Tests/  # NEW: xUnit tests for the bridge
│           ├── ExportTests/
│           ├── MemoryContractTests/
│           └── AotCompatibilityTests/
└── external/
    └── ue-bridge/                         # NEW: UE plugin
        ├── WowViewerBridge.uplugin
        ├── Source/
        │   ├── WowViewerBridge/          # C++ module
        │   │   ├── WowViewerBridge.Build.cs
        │   │   ├── Public/
        │   │   │   ├── WowViewerBridge.h
        │   │   │   ├── BridgeLoader.h    # DLL load + symbol resolve
        │   │   │   ├── BridgeHandles.h   # SafeHandle wrappers
        │   │   │   ├── Converters/
        │   │   │   │   ├── TerrainConverter.h
        │   │   │   │   ├── WmoConverter.h
        │   │   │   │   ├── M2Converter.h
        │   │   │   │   ├── BlpConverter.h
        │   │   │   │   ├── LiquidConverter.h
        │   │   │   │   └── DbcConverter.h
        │   │   │   ├── Actors/
        │   │   │   │   ├── WoWTileActor.h
        │   │   │   │   ├── WoWWmoActor.h
        │   │   │   │   ├── WoWM2Actor.h
        │   │   │   │   └── WoWMapActor.h
        │   │   │   ├── Commands/         # Console commands
        │   │   │   │   ├── GenerateTerrainCommand.h
        │   │   │   │   ├── SpawnWmoCommand.h
        │   │   │   │   ├── SpawnM2Command.h
        │   │   │   │   └── LoadMapCommand.h
        │   │   │   ├── Subsystems/
        │   │   │   │   └── GameManagerSubsystem.h
        │   │   │   └── Settings/
        │   │   │       └── BridgeSettings.h
        │   │   └── Private/
        │   │       └── (implementations)
        │   └── WowViewerBridgeEditor/    # Editor module (commands, UI, asset actions)
        └── Tests/                        # UE Automation Framework tests
            ├── BridgeLoaderTests.cpp
            ├── TerrainConverterTests.cpp
            ├── WmoConverterTests.cpp
            ├── M2ConverterTests.cpp
            ├── BlpConverterTests.cpp
            └── EndToEndTests.cpp

# Reference (unchanged, read-only)
gillijimproject_refactor/  # DO NOT TOUCH

# Staged test clients
output/tmp/wowarchive-clients/
├── 0.5.3-development/  # Primary test surface
├── 1.12.x/             # Secondary test surface
└── 3.3.5/              # LK-era test surface (optional)
```

**Structure Decision**: 
- New `WowViewer.Core.Bridge` C# project under `wow-viewer/src/core/`. Library-first: all bridge logic is testable as a normal .NET library before AOT compilation.
- UE plugin lives in `wow-viewer/external/ue-bridge/`. This is external to `wow-viewer/src/` because UE plugins have a strict folder structure that doesn't match the C# library layout, but it remains inside the `wow-viewer/` repo boundary (Rule 4).
- Test client staging under `output/tmp/wowarchive-clients/` follows the existing convention.

## Implementation Phases

The plan is decomposed into 9 phases. Each phase ends with a real-data validation step. The first 4 phases are de-risking / foundation; phases 5-8 deliver the 6 user stories; phase 9 is polish.

### Phase 0 — Pre-Work Spike (de-risk before committing)

**Goal**: Resolve unknowns that could derail the architecture: AOT compatibility of C# libraries, UE plugin development spike, UE version pinning, test map matrix.

**Tasks (see `tasks.md` for details)**:
- 0.1: AOT compatibility audit of `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.PM4`, `WowViewer.Core.Runtime` (no library should require reflection-emit, dynamic loading, or other AOT-incompatible features)
- 0.2: UE plugin spike — create minimal "hello world" UE 5.4 C++ module that loads a small Native AOT-compiled C# DLL and calls a single exported function. Validates the end-to-end toolchain.
- 0.3: Pin UE 5.4 as the target version (5.4 is current LTS-class; matches user-installed engine)
- 0.4: Define test map matrix — which staged clients and which maps serve as the validation surface

**Exit criteria**: AOT audit report (`research/aot-audit.md`), UE spike successful (DLL loads, function called, return value received), test map matrix committed (`contracts/test-maps.md`).

**Validation**:
- `dotnet publish wow-viewer/src/core/WowViewer.Core.Bridge.Spike -c Release` produces a native DLL
- UE test project opens, loads the plugin, executes the spike, logs success

---

### Phase 1 — Bridge Foundation (C API + AOT build + UE plugin skeleton)

**Goal**: Establish the structural skeleton of the bridge: C API contracts, AOT build configuration, UE plugin module, DLL loading, error reporting, memory ownership.

**Tasks**:
- 1.1: Define C API surface (`contracts/c-api.md`) — function signatures, POD structs, handle types, error codes
- 1.2: Define memory ownership contract (`contracts/memory.md`) — who allocates, who frees, when
- 1.3: Create `WowViewer.Core.Bridge` project skeleton with `BridgeExports` class stub
- 1.4: Configure Native AOT build for the bridge project (`<PublishAot>true</PublishAot>`, trim warnings addressed)
- 1.5: Create UE 5.4 plugin skeleton (`WowViewerBridge.uplugin`, module `WowViewerBridge`)
- 1.6: Implement `BridgeLoader` in UE C++ (LoadLibrary, GetProcAddress, call trampolines)
- 1.7: Implement `BridgeHandles` SafeHandle wrappers in UE C++
- 1.8: Implement `BridgeErrors` and `BridgeSettings` subsystems in UE C++
- 1.9: Wire plugin lifecycle to `BridgeInit` / `BridgeShutdown` C API calls
- 1.10: Add an editor module `WowViewerBridgeEditor` with placeholder console command `WowViewer.Status`

**Exit criteria**: UE plugin loads the AOT DLL, calls `BridgeInit`, exposes a console command that prints "bridge version 0.1.0 connected" + supports shutdown.

**Validation**:
- AOT DLL builds in < 5 min
- UE editor opens, plugin enabled, console command works
- Memory: zero leaks reported by Visual Studio diagnostic tools across plugin enable/disable cycles

---

### Phase 2 — BLP Texture Bridge (foundation for all visual assets)

**Goal**: Establish the texture decode pipeline. Terrain, WMO, M2 all consume BLP textures. This phase gets that pipeline right before terrain/model work.

**Tasks**:
- 2.1: Implement `BlpExports` C API (`BlpLoadFile`, `BlpGetInfo`, `BlpDecodeMip`)
- 2.2: Define `BlpTexture` struct in C API (width, height, mip count, format flags, RGBA8 mip data pointer)
- 2.3: Add xUnit tests for `BlpExports` (decode 1x1, 256x256, 1024x1024, DXT1/DXT3/DXT5, palettized)
- 2.4: Implement `BlpConverter` in UE C++ (converts C struct → `UTexture2D`, uploads to GPU)
- 2.5: Add `UTexture2D` upload via `FTexture2DMipMap` and `UpdateResource()`
- 2.6: Implement texture caching subsystem (LRU cache for decoded textures)
- 2.7: Add UE Automation tests for BlpConverter (decode + upload + verify dimensions)

**Exit criteria**: A BLP file from `0.5.3/Textures/Minimap/map0.blp` can be loaded via console command `WowViewer.DecodeBlp` and displayed as a UTexture2D in a UE UI widget.

**Validation**:
- Texture dimensions match original
- GPU upload succeeds (no errors in `r.RHISetGPUCaptureOptions` log)
- Decode 100 BLPs in < 30s (cold path)

---

### Phase 3 — Terrain Bridge (US1: Terrain Explorer)

**Goal**: Load a full WoW map's terrain into UE Landscape with correct heightmaps, alpha layers, and hole masks.

**Tasks**:
- 3.1: Implement `TerrainExports` C API (`TerrainLoadMap`, `TerrainGetTileCount`, `TerrainGetTile`, `TerrainFreeTile`)
- 3.2: Define `TerrainTile` struct (145x145 heightmap, 64x64 alpha layers, hole mask, layer count, texture layer assignments)
- 3.3: Define material translation table (`contracts/materials.md`) — WoW terrain layer flags → UE Material domain/BlendMode settings
- 3.4: Implement `TerrainConverter` in UE C++ (C struct → UE Landscape height/weight data)
- 3.5: Implement `WoWTileActor` (UE actor class wrapping `ALandscapeProxy`)
- 3.6: Implement `GenerateTerrainCommand` console command
- 3.7: Add xUnit tests for `TerrainExports` (terrain tile decode against `output/tmp/wowarchive-clients/0.5.3-development`)
- 3.8: Add UE Automation tests for TerrainConverter (heightmap + alpha layer round-trip)

**Exit criteria**: Console command `WowViewer.GenerateTerrain 0.5.3-development/development` produces a complete UE Landscape for the entire `development` map with correct height values and alpha-blended textures.

**Validation**:
- All tiles generated, no missing chunks
- Height values match the source within ±0.01m (compared via diff)
- Texture blending matches `_tex0.adt` + MCAL alpha output
- Visible terrain: no holes where there shouldn't be, holes where they should be

---

### Phase 4 — WMO Bridge (US2 partial: Model Inspector — static meshes)

**Goal**: Load WMO group files as UE static mesh actors with correct geometry, materials, and portals.

**Tasks**:
- 4.1: Implement `WmoExports` C API (`WmoLoadFile`, `WmoGetGroupCount`, `WmoGetGroup`, `WmoFreeGroup`)
- 4.2: Define `WmoGroup` struct (vertex array, index array, UV array, normal array, material slots, portal data)
- 4.3: Add xUnit tests for `WmoExports` (WMO group decode against test maps)
- 4.4: Implement `WmoConverter` in UE C++ (C struct → `UStaticMesh`)
- 4.5: Implement `WoWWmoActor` (UE actor class with `UStaticMeshComponent` per group)
- 4.6: Implement WMO material-to-UE-material conversion (flags, blends, dual-texture, two-sided)
- 4.7: Implement `SpawnWmoCommand` console command
- 4.8: Add UE Automation tests for WmoConverter (mesh conversion + material round-trip)

**Exit criteria**: Console command `WowViewer.SpawnWmo <path>` loads any WMO from `0.5.3` and produces a static mesh actor with correct geometry and textures.

**Validation**:
- Vertex count matches source WMO
- All material slots assigned, all BLPs decoded
- Visible mesh matches original client rendering
- Portals decoded (even if not yet used for culling in this phase)

---

### Phase 5 — M2/MDX Bridge (US2 remainder + US4: Animation Playback)

**Goal**: Load M2/MDX models as UE skeletal mesh actors with animation playback.

**Tasks**:
- 5.1: Implement `M2Exports` C API (`M2LoadFile`, `M2GetSkinCount`, `M2GetSkin`, `M2GetAnimationCount`, `M2GetAnimation`, `M2SampleAnimation`, `M2FreeModel`)
- 5.2: Define `M2Model`, `M2Skin`, `M2Animation` structs in C API
- 5.3: Implement animation sampling in C# (Bezier interpolation, compressed track expansion) — this is the only logic that lives in the bridge, not in the existing runtime
- 5.4: Define animation fidelity contract (`contracts/animation.md`) — sampling rate, keyframe accuracy, etc.
- 5.5: Add xUnit tests for `M2Exports` and animation sampling (fidelity test: baked C# output vs raw interpolated values from a known reference frame)
- 5.6: Implement `M2Converter` in UE C++ (skin → `USkeletalMesh`, animation → `UAnimSequence`)
- 5.7: Implement `WoWM2Actor` (UE actor with `USkeletalMeshComponent`)
- 5.8: Implement `SpawnM2Command` console command (with `--anim <id>` flag)
- 5.9: Add UE Automation tests for M2Converter (skeletal mesh + animation playback)

**Exit criteria**: Console command `WowViewer.SpawnM2 <path> --anim 0` loads any M2 from `0.5.3` and plays the requested animation at correct bone transforms.

**Validation**:
- Skeletal mesh bones match source M2 hierarchy
- Animation playback at 30 FPS, bone transforms within floating-point tolerance of C#-side reference
- Skin weights, vertex buffers, materials all correct

---

### Phase 6 — Liquid + World Composition (US3: World Composition)

**Goal**: Load a complete WoW map (terrain + WMO + M2 placements + liquids) into UE as a single explorable world.

**Tasks**:
- 6.1: Implement `LiquidExports` C API (`LiquidGetMapLiquids`, `LiquidGetChunk`, `LiquidFreeChunk`)
- 6.2: Define `LiquidChunk` struct (vertex heights, render flags, liquid type ID, geometry type)
- 6.3: Implement `LiquidConverter` in UE C++ (chunk → UE water plane or spline mesh)
- 6.4: Add xUnit tests for `LiquidExports` (MH2O and MCLQ decoding)
- 6.5: Implement ADT placement extraction (MCRF, MCAD, MCRE) on the C# side — this is existing logic, just needs to be exposed via bridge
- 6.6: Implement `MapExports` (`MapLoad`, `MapGetObjectPlacements`) C API
- 6.7: Implement `LoadMapCommand` console command — orchestrates terrain, models, liquids, placements
- 6.8: Add UE Automation tests for full `LoadMap` flow against `0.5.3-development`

**Exit criteria**: Console command `WowViewer.LoadMap 0.5.3-development/development` produces a complete world with terrain + WMO + M2 placements + liquids, all correctly positioned.

**Validation**:
- All ADT objects present in UE world
- Object transforms (position, rotation, scale) match source placements
- Liquids rendered at correct heights and types
- Real-data map loads in < 30s

---

### Phase 7 — Game Manager (US5: Multi-root Support)

**Goal**: UE plugin supports registering and switching between multiple staged client roots.

**Tasks**:
- 7.1: Implement `ArchiveExports` C API (`ClientRootRegister`, `ClientRootList`, `ClientRootSetActive`, `ClientRootGetActive`)
- 7.2: Implement `GameManagerSubsystem` in UE C++ (UEditorSubsystem)
- 7.3: Persist client roots in UE project settings (`.uproject` config)
- 7.4: Implement plugin UI panel (SCompoundWidget) — list of registered roots, active root indicator
- 7.5: Implement root switching (active root update propagates to all subsystems)
- 7.6: Add xUnit tests for `ArchiveExports` (register/list/switch flow)
- 7.7: Add UE Automation tests for GameManagerSubsystem

**Exit criteria**: Two client roots (e.g., 0.5.3 and 1.12.x) registered via UE editor panel, switching active root updates all subsequent `LoadMap` calls to use the correct root.

**Validation**:
- Roots persist across editor restarts
- Switching is atomic (no partially-loaded world)
- UE project settings serialize/deserialize roots correctly

---

### Phase 8 — DBC/DB2 Bridge (US6: Data Tables)

**Goal**: Expose DBC/DB2 table data as UE DataTable assets.

**Tasks**:
- 8.1: Implement `DbcExports` C API (`DbcLoadTable`, `DbcGetSchema`, `DbcGetRowCount`, `DbcGetRow`, `DbcFreeTable`)
- 8.2: Define `DbcSchema` struct (column name array, column type array, row count)
- 8.3: Implement `DbcConverter` in UE C++ (schema + rows → `UDataTable` with `FTableRowBase` derived UStruct)
- 8.4: Implement DBC string table lookup (for `stringref` columns)
- 8.5: Add xUnit tests for `DbcExports` (AreaTable, Map.dbc, LiquidType.dbc)
- 8.6: Add UE Automation tests for DbcConverter (DataTable round-trip)
- 8.7: Add `DbcImport` UE asset action (right-click `.dbc` file in content browser → import as DataTable)

**Exit criteria**: Right-clicking `AreaTable.dbc` in UE content browser creates a `UDataTable` asset with correct column names, types, and row count.

**Validation**:
- Schema columns match expected DBC layout
- Row count matches source DBC record count
- String columns resolve correctly via string table

---

### Phase 9 — Polish, Build Pipeline, and Documentation

**Goal**: Make the bridge shippable: automated build, test map matrix execution, doc hygiene.

**Tasks**:
- 9.1: Build pipeline automation — `scripts/build-bridge.ps1` that produces the AOT DLL and copies it into the UE plugin's `ThirdParty/wowviewer_bridge/`
- 9.2: Test map matrix execution — CI script that loads each (client, map) pair from `contracts/test-maps.md` and reports success/failure
- 9.3: Reference the updated `wow-engine-modernization-plan-2026-05-14.md` (viewer-first + UE bridge, replaced 2026-06-14)
- 9.4: Update `game-viewer-host-plan-2026-05-13.md` to reflect that `WowViewer.App` is now a CLI/diagnostic host
- 9.5: Update `wow-viewer-library-completeness-plan-2026-05-06.md` to mark renderer gaps as deprecated
- 9.6: Update memory bank (`activeContext.md`, `progress.md`) with bridge status
- 9.7: Write a `wow-viewer-bridge` README in the plugin source explaining the architecture and how to add a new format family
- 9.8: Final pass: every phase's success criteria checked; FR-001 through FR-010 verified

**Exit criteria**: A fresh clone of `wow-viewer` can run `scripts/build-bridge.ps1` and produce a working UE plugin that passes the full test map matrix. All affected architecture docs are updated.

**Validation**:
- `dotnet build wow-viewer/WowViewer.slnx -c Debug` succeeds
- AOT DLL build succeeds
- UE plugin builds against installed UE 5.4
- Test map matrix: 100% pass on (0.5.3, development), 90%+ pass on (1.12.x, test_map)

---

## Complexity Tracking

> **No constitution violations requiring justification.**

The bridge adds new code in two new locations (`WowViewer.Core.Bridge`, `external/ue-bridge/`) but introduces no constitution violations. The library-first principle is preserved — the C# bridge wrapper is a normal library that happens to export a C API via Native AOT, and the UE plugin is a thin consumer.

## Risks (forward-referenced from `analysis.md`)

The plan mitigates the major risks identified in `analysis.md`:

| Risk | Mitigation Phase |
|------|------------------|
| AOT compatibility of existing C# libraries | Phase 0 (audit) + Phase 1 (publish trial) |
| UE C++ is a new surface for the project | Phase 0 (spike) + Phase 1 (skeleton) |
| Animation fidelity loss from baking | Phase 5 (fidelity contract + test) |
| Memory leaks across 1000+ actors | Phase 1 (memory contract) + Phase 4/5 (per-phase leak tests) |
| BLP decode performance | Phase 2 (profiling + caching) |
| UE version lock-in | Phase 0 (version pinning) |

## Exit Criteria for the Whole Feature

1. All 6 user stories (US1-US6) have passing acceptance scenarios.
2. SC-001 through SC-006 from the spec are measured and met.
3. AOT DLL builds in < 5 min.
4. UE plugin loads and exposes all 4 primary commands (`GenerateTerrain`, `SpawnWmo`, `SpawnM2`, `LoadMap`).
5. Test map matrix executes with ≥90% pass rate on 2+ clients.
6. All affected architecture docs are updated and reflect the UE-primary direction.
7. No new code in `gillijimproject_refactor/`.
8. No format parser code added to the bridge (all format reading lives in existing C# libraries).
9. Memory ownership contract holds: zero leaks across plugin enable/disable cycles and across 1000+ actor spawns.
