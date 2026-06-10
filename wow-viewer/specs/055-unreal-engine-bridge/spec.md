# Feature Specification: Unreal Engine Bridge

**Feature Branch**: `055-unreal-engine-bridge`

**Created**: 2026-06-09

**Status**: Draft

**Input**: User conversation — "move viewer to Unreal Engine; we have all the parsing bits; we've outgrown the current viewer; build a bridge to Unreal for all exploration"

## Motivation

The current viewer (`WowViewer.App` + Silk.NET OpenGL) has proven the library architecture but the rendering surface is a permanent drag. Every new format, shader, LOD system, animation pipeline, or editor feature requires bespoke GLSL + C# rendering code that will never match what Unreal Engine delivers out of the box. The format-parsing libraries are mature; the rendering tooling is not, and maintaining a competitive renderer alongside the format work doubles the surface area.

Unreal Engine provides: world composition, LOD streaming, skeletal animation, particle systems, physics, material graphs, landscape system, reflection capture, lighting, post-processing, audio engine, editor tooling, and a mature C++/Blueprint plugin architecture.

This spec defines a bridge layer that leaves all data ownership in the existing C# libraries (`WowViewer.Core`, `Core.IO`, `Core.PM4`, `Core.Runtime`) and exposes format-decoded data to Unreal Engine through a native C API produced by .NET Native AOT compilation.

## Bridge Architecture

```
┌────────────────────────────────────────────────┐
│  Unreal Engine 5.x Editor / Runtime             │
│  ┌──────────────────────────────────────────┐   │
│  │  UE Plugin: WowViewerBridge              │   │
│  │  (C++ UE module)                         │   │
│  │  - Loads native AOT library              │   │
│  │  - Converts C bridge data → UE types     │   │
│  │  - Spawns actors / assets                │   │
│  └──────────────────────────────────────────┘   │
└────────────────────────────────────────────────┘
                        │ C FFI
┌────────────────────────────────────────────────┐
│  Native AOT Shared Library (wowviewer_bridge)  │
│  (.NET 10 Native AOT → .dll / .so)             │
│  - Exposes flat C API for:                     │
│    - Archive / file system access              │
│    - Map loading and tile enumeration          │
│    - Terrain heightmap + alpha layer decode    │
│    - WMO mesh + material extraction            │
│    - M2/MDX model + animation extraction       │
│    - BLP texture decode → raw RGBA             │
│    - Liquid data extraction                    │
│    - DBC/DB2 table queries                     │
│    - PM4 data queries                          │
│  - No .NET runtime required at runtime         │
└────────────────────────────────────────────────┘
        calls into
┌────────────────────────────────────────────────┐
│  WowViewer.Core / Core.IO / Core.PM4 / etc.    │
│  (.NET 10 C# Libraries — UNCHANGED)            │
│  - All format readers, writers, converters      │
│  - All runtime data contracts                   │
│  - Repo-independent, library-first              │
└────────────────────────────────────────────────┘
```

### Design decisions

1. **Native AOT, not C++/CLI** — .NET Native AOT produces a native shared library with no managed runtime dependency. This avoids loading the .NET runtime inside the UE process, which is fragile and blocks cross-platform targets. The bridge DLL is a flat C API that any language can call.

2. **C# libraries remain the canonical data layer** — No format parsing moves to C++. The bridge is a thin export surface that calls existing C# code. If a format bug is fixed in `WowViewer.Core.IO`, the bridge automatically gets the fix.

3. **Unreal plugin owns UE-native conversion** — The C++ UE module translates flat C structs (arrays of vertices, indices, UVs, bone weights) into UE-native types (`UStaticMesh`, `USkeletalMesh`, `UTexture2D`, `UWorld`, `ALandscapeProxy`, etc.). This keeps UE-specific code in UE land.

4. **Backend strategy changed** — This spec supersedes the Vulkan-first backend direction in `wow-engine-modernization-plan-2026-05-14.md`. Unreal Engine becomes the primary rendering backend. The OpenGL/Silk.NET path in `WowViewer.App` remains as a headless/diagnostic fallback for validation captures.

## User Scenarios & Testing

### User Story 1 — Terrain Explorer (P1)

A user opens Unreal Engine, activates the WowViewerBridge plugin, selects a staged client root and map name, and the plugin generates a full 3D terrain landscape with correct heightmaps, alpha-blended textures, and hole masks — without writing any GLSL or C++ data-parsing code.

**Why this priority**: Terrain is the foundational world surface. Without terrain rendering, the bridge provides no exploration value. Everything else (models, liquids, sky) layers on top.

**Independent Test**: The bridge plugin exposes a `GenerateTerrain(clientRoot, mapName)` Blueprint/console command that returns the number of tiles generated and renders them in the UE viewport. A test script can validate tile count matches the expected grid.

**Acceptance Scenarios**:

1. **Given** a staged 0.5.3 client root with `development` map, **When** the bridge `GenerateTerrain` command is invoked, **Then** a UE Landscape actor (or tiled `ALandscapeProxy` instances) covers the correct map area with correct height values (±0.01m tolerance).
2. **Given** the generated terrain, **When** viewed in UE viewport, **Then** each tile shows the correct blended textures from `_tex0.adt` + MCAL alpha masks, with no visible seams between tiles.
3. **Given** a tile with active hole mask bits, **When** the `MH2O` / hole mask is decoded, **Then** those cells appear as holes (no geo) in the UE landscape.

---

### User Story 2 — Model Inspector (P1)

A user selects any WMO or M2/MDX file from a client root and the bridge spawns it as a fully-textured static or skeletal mesh in the UE viewport, with correct materials.

**Why this priority**: Models are the second most important asset family. This proves the bridge handles non-terrain format data and the UE conversion pipeline for meshes + textures + materials.

**Independent Test**: The bridge exposes `SpawnWMO(filePath)` and `SpawnM2(filePath)` console commands. A test script can spawn a known model, verify the actor exists in the world, and verify it has the expected number of mesh sections and materials.

**Acceptance Scenarios**:

1. **Given** a path to a WMO group file (e.g., `World\\wmo\\path\\to\\group.wmo`), **When** `SpawnWMO` is called, **Then** a UE actor with static mesh components for each WMO group is created at the origin, with correct vertex positions, UVs, normals, and material assignments.
2. **Given** a path to an M2 model file, **When** `SpawnM2` is called with animation index 0, **Then** a UE skeletal mesh actor is created with correct bind pose, skin weights, and playing the requested animation sequence.
3. **Given** a model with BLP textures, **When** the bridge decodes the BLP and creates UE materials, **Then** the rendered surface matches the original texture appearance (color, alpha, mip levels).

---

### User Story 3 — World Composition (P2)

A user loads an entire map (terrain + WMO placements + M2 placements + liquids + sky) into UE as a single explorable world, with correct positioning and visibility.

**Why this priority**: This is the core exploration use case — walking through a WoW map inside UE. It depends on US1 (terrain) and US2 (models) being functional first.

**Independent Test**: The bridge exposes `LoadMap(clientRoot, mapName)` which spawns all world content. A test can verify that all expected objects from the ADT placement chunks are present as UE actors.

**Acceptance Scenarios**:

1. **Given** a loaded map, **When** world objects (WMO/M2 placements from ADT MCRF/MCAD/MCRE chunks) are processed, **Then** all placed models appear at correct world coordinates with correct rotations and scales.
2. **Given** a loaded map with liquid data (MH2O/MCLQ chunks), **When** the bridge creates UE water/fluid actors, **Then** liquid surfaces are rendered at correct heights with correct type colors/textures.
3. **Given** a loaded map with DBC/DB2 area/liquid type data, **When** the bridge creates materials for terrain and liquids, **Then** material assignments use correct DBC-driven texture paths and liquid type IDs.

---

### User Story 4 — Animation Playback (P2)

A user loads an M2/MDX model with skeletal animations and can play, pause, and blend between animation sequences in the UE viewport.

**Why this priority**: Animation is a major feature of M2/MDX models. Without it, the model inspector is incomplete. This proves the bridge delivers bone pose data and skinning through the C API.

**Independent Test**: Given an M2 file with multiple animation sequences, verify each sequence plays correctly with correct bone transforms and skin deformation.

**Acceptance Scenarios**:

1. **Given** an M2 model with a known animation (e.g., `Stand`), **When** the bridge plays that animation at time t, **Then** the bone transforms at time t match the raw M2 track data within floating-point tolerance.
2. **Given** an M2 model, **When** the bridge extracts skin/profile data, **Then** each skin section has correct vertex buffers, index buffers, bone influence data, and material assignments.

---

### User Story 5 — Game Manager Integration (P3)

The UE plugin can register and switch between multiple staged client roots, cache loaded archives, and display per-root capability summaries in the UE editor UI.

**Why this priority**: Multi-root support is an editor workflow need, not a viewer runtime need. It can follow once the core import pipeline works.

**Independent Test**: The UE plugin's game manager panel lists registered roots, shows detected game version, and allows switching the active root.

**Acceptance Scenarios**:

1. **Given** a UE editor session with the bridge plugin active, **When** a user registers a staged client root via the plugin UI, **Then** the root is persisted in UE project settings and available for subsequent sessions.
2. **Given** two registered client roots (e.g., 0.5.3 and 3.3.5), **When** the user switches between them, **Then** the active root changes and subsequent asset loads use the correct root.

---

### User Story 6 — DBC/DB2 Data Tables (P3)

The bridge exposes DBC/DB2 table data as UE DataTable assets, allowing Blueprint/script access to game configuration data (spell data, creature templates, area data, liquid types, etc.).

**Why this priority**: DBC/DB2 access enables gameplay and logic scripts in UE to reference real game metadata. This is valuable for modding/interop workflows but not needed for basic exploration.

**Independent Test**: Given a selected client root and a DBC table name (e.g., `AreaTable.dbc`), verify the bridge produces a UE DataTable asset with correct column names and row count.

**Acceptance Scenarios**:

1. **Given** a client root with `DBFilesClient\\AreaTable.dbc`, **When** the bridge imports it as a UE DataTable, **Then** the table has the correct number of rows matching the original file's record count and all columns have correct names and types.
2. **Given** a DBC/DB2 table with ID -> string lookups, **When** accessed from Blueprint, **Then** lookups return correct localized strings.

### Edge Cases

- What happens when the bridge encounters a WMO group with missing or corrupted geometry data? → The bridge logs the failure and skips that group; other groups in the same WMO still load.
- How does the bridge handle very large maps (e.g., Kalimdor with thousands of WMO/M2 placements)? → The C API returns paginated results; the UE plugin streams data in manageable batches.
- What if a client path contains Unicode characters? → The C API uses UTF-8 for all string parameters.
- How does the bridge behave when the AE AOT DLL is missing or wrong version? → The UE plugin logs a clear error message and disables itself.

## Requirements

### Functional Requirements

- **FR-001**: The bridge MUST expose a C API, produced by .NET 10 Native AOT compilation, that covers: archive/directory access, map metadata, terrain data, WMO mesh data, M2/MDX model + animation data, BLP texture decode, liquid data, and DBC/DB2 table data.
- **FR-002**: The bridge C API MUST use only flat C types (pointers, length-prefixed arrays, POD structs) with no C++ ABI dependence.
- **FR-003**: The UE plugin MUST call the bridge C API and convert returned data into UE-native types: `UStaticMesh`, `USkeletalMesh`, `UTexture2D`, `UMaterialInstanceDynamic`, `ALandscapeProxy`, `AStaticMeshActor`, `ASkeletalMeshActor`, `UDataTable`, etc.
- **FR-004**: The UE plugin MUST provide Blueprint-accessible functions and console commands for the primary workflows: `GenerateTerrain`, `SpawnWMO`, `SpawnM2`, `LoadMap`.
- **FR-005**: The bridge MUST NOT modify any C# library code beyond what is needed for the AOT export surface — no format parser changes, no converter rewrites.
- **FR-006**: The UE plugin MUST support multiple staged client roots (per Rule 9: `output/tmp/wowarchive-clients/`) and allow switching the active root at runtime.
- **FR-007**: The bridge C API MUST use a memory-mapped or streaming pattern for large data (entire maps, high-resolution textures) to avoid blocking the UE game thread.
- **FR-008**: The bridge MUST preserve the existing `WowViewer.Core` data contracts and validation pipeline — the bridge is an export surface, not a reimplementation.
- **FR-009**: The UE plugin MUST build on Windows for UE 5.x (exact version TBD by installed engine).
- **FR-010**: The C API error model MUST include error codes, human-readable error strings, and per-call success/failure reporting.

### Key Entities

- **ClientRoot**: A registered path to a staged WoW client root (`output/tmp/wowarchive-clients/<build>`), with detected build version and archive/catalog configuration.
- **MapManifest**: Metadata about a loaded map — tile grid extents, area name, liquid type definitions, terrain flags.
- **TerrainTileData**: Per-chunk heightmap (145x145 floats), alpha layers (64x64 bytes per layer), texture layer assignments, hole mask, normal data.
- **WmoPayload**: Mesh sections (vertices, indices, UVs, normals), material slots (BLP texture path, flags), group transforms, portal data.
- **M2Payload**: Skeletal mesh sections (vertices, indices, UVs, bone weights/indices), bone hierarchy, animation tracks (per-bone keyframes), skin profiles, material assignments.
- **BlpTexture**: Decoded texture as RGBA8 mip chain, original BLP format flags.
- **LiquidPayload**: Per-chunk liquid data (vertex heights, render flags, liquid type ID) for MH2O and MCLQ formats.
- **DbcTableSchema**: Column definitions (name, type, size) and row data for a DBC/DB2 file, ready for UE DataTable import.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A user can load a staged 0.5.3 client root, generate the `development` map terrain in UE, walk through it at real-time framerates (≥30 FPS), and every tile has correct height + texture appearance matching the original client.
- **SC-002**: A user can spawn any WMO from `development` map in UE and the mesh matches the original client rendering (same vertex count, same texture appearance) within visual inspection tolerance.
- **SC-003**: A user can load any M2 model with animations and all animation sequences play correctly with smooth bone deformation matching the original client.
- **SC-004**: The bridge C API covers at least 6 format families (terrain, WMO, M2/MDX, BLP, liquid, DBC/DB2) with no format reader code duplicated outside `WowViewer.Core`.
- **SC-005**: Build time for the AOT bridge DLL is under 5 minutes on a standard development machine.
- **SC-006**: The UE plugin loads the bridge DLL, initializes it, and surfaces a "connected" status in the UE editor UI within 10 seconds of project open.

## Non-Goals (Explicitly Out of Scope)

- No game logic porting (spell system, combat, quests, AI, etc.).
- No networking or multiplayer.
- No runtime .NET host inside the UE process.
- No rewriting format parsers in C++.
- No UI tooling porting (existing WowViewer.App remains the CLI/diagnostic host).
- No physics engine replacement (UE's built-in physics is used directly).
- No pixel-perfect parity with original client rendering (close visual match is sufficient).
- No editor import/export shell in UE (that remains in WowViewer.App / future editor host).

## Impact on Existing Plans

1. **`wow-engine-modernization-plan-2026-05-14.md`** — The Vulkan-first backend strategy is superseded by Unreal Engine as the primary rendering backend. OpenGL/Silk.NET remains as headless/diagnostic fallback. The engine-runtime contracts (Pillar B) still apply but now target the UE bridge surface.
2. **`game-viewer-host-plan-2026-05-13.md`** — `WowViewer.App` becomes a diagnostic/compatibility host. The "game-viewer" product identity may merge with or be superseded by the UE editor workflow.
3. **`wow-viewer-library-completeness-plan-2026-05-06.md`** — The c rendering gap column becomes less urgent since UE handles rendering.
4. **Constitution** — The `Silk.NET.OpenGL` line in the Technology Stack section is superseded by Unreal Engine. `.NET AOT` is added as a delivery mechanism.

## Assumptions

- Unreal Engine 5.x is installed on the development machine and accessible via standard UE build tooling.
- The user has access to staged client roots under `output/tmp/wowarchive-clients/`.
- .NET 10 supports Native AOT for the target platform (Windows x64). This is a supported configuration.
- The existing C# libraries build and test cleanly with `dotnet build` / `dotnet test`.
- The UE plugin will be developed as a standard UE C++ `Runtime` module with Editor extensions (console commands, detail customizations, asset actions).
- The user is comfortable with UE C++ development (or will learn it as part of this work).
- BLP texture decoding output is RGBA8 — no GPU-compressed format passthrough in the initial bridge version.
- Animation data is delivered as sampled bone transforms at a configurable frame rate (default 30 fps), not as raw Bezier/interpolation keyframes. The C# side handles the interpolation; the UE side receives baked poses.
