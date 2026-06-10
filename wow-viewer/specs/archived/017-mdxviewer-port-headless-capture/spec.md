# Feature Specification: WowViewer Native GPU Renderer + Headless Capture

**Feature Branch**: `017-native-gpu-renderer`

**Created**: 2026-05-24

**Status**: Draft

## Problem

The existing `WowViewer.App.WorldGpuPreviewRenderer` is a broken, incomplete terrain renderer that produces all-black output for real game maps (Azeroth_30_48). MdxViewer's terrain rendering works correctly but lives in `gillijimproject_refactor` (read-only reference) and can't be directly used from wow-viewer's data pipeline. The automation in MdxViewer is slow because it runs through the full ImGui UI pipeline.

The underlying data infrastructure (`NativeMpqService`, `WorldTerrainTileBuilder`, `WorldTerrainTileData`, `AdtMcalDecoder`, `AdtPlacementCatalog`) is proven and correct — it's the same data that the V16 harvesting pipeline uses. The gap is a **native GPU renderer** that converts this runtime data into OpenGL draw calls and framebuffer output.

## Solution

Build a native GPU renderer library (`WowViewer.Core.Renderer`) in wow-viewer from scratch, designed for headless operation as its primary mode. It consumes the existing runtime data types (`WorldTerrainTileData`, `WorldLiquidTileData`, `AdtPlacementCatalog`) and renders them to an offscreen framebuffer.

Architecture principles:
- **Renderer-first**: The rendering library is the foundation. It has no UI dependencies (no ImGui, no WinForms).
- **Headless as default**: The renderer works without a visible window. GUI mode is a wrapper on top.
- **Thin consumer interface**: Input is the `WorldTerrainTileData` produced by the existing `Core.Runtime.WorldTerrainTileBuilder`. The renderer doesn't do I/O — it just renders what it's given.
- **Process-isolated concurrency**: Each capture runs in its own process with its own GL context. No shared GPU state.

## User Stories

### User Story 1 - Headless Terrain Render (P1)

As a **dataset builder**, I want a CLI tool that renders a single terrain tile from a staged game client to a PNG without any visible window, producing pixel-correct output matching MdxViewer's renders.

**Why this priority**: This is the core value — automated, unattended terrain capture for the V16 ML pipeline. It validates the entire renderer pipeline.

**Independent Test**: Command `capture render --client-root ... --map Azeroth --tile 30 48` produces PNG with >20K unique colors, alpha=255, and visually matches MdxViewer.

**Acceptance Scenarios**:
1. **Given** a staged game client at `output/tmp/wowarchive-clients/0_5_3_3368_wow/`, **When** running `capture render --client-root ... --map Azeroth --tile 30 48`, **Then** the output PNG has >20K unique colors (not all-black)
2. **Given** a staged game client, **When** running capture, **Then** the output alpha channel is 255 on all terrain pixels
3. **Given** the same inputs, **When** running the capture tool and MdxViewer's existing capture, **Then** the pixel output visually matches (verified by comparison)

### User Story 2 - Multi-Variant Batch Capture (P2)

As a **dataset builder**, I want to capture all 4 variant renders (primary, no-liquids, no-objects, objects-only) for a set of tiles in a single command.

**Acceptance Scenarios**:
1. **Given** a game client and a tile list, **When** running `capture batch ... --tile-list tiles.txt --variants all`, **Then** 4 files per tile are produced
2. **Given** a missing tile, **When** the tile has no ADT data, **Then** the tool skips gracefully without crashing

### User Story 3 - Multi-Process Concurrent Capture (P3)

As a **dataset builder**, I want to capture multiple tiles in parallel across CPU cores.

**Acceptance Scenarios**:
1. **Given** 16 tiles to capture, **When** running with `--parallel 4`, **Then** all 16 tiles complete with correct output
2. **Given** a worker crash, **When** one process fails, **Then** other workers continue

### User Story 4 - GUI Debug View (P4)

As a **developer**, I want a visible window showing the rendered terrain with camera controls for debugging.

**Acceptance Scenarios**:
1. **Given** the renderer library, **When** creating a GUI wrapper, **Then** a Silk.NET window opens with terrain rendered
2. **Given** the visible window, **When** using WASD + mouse, **Then** camera moves

## Requirements

### Functional Requirements

- **FR-001**: Renderer library MUST consume `WorldTerrainTileData` (from `WowViewer.Core.Runtime`) and produce rendered framebuffer output
- **FR-002**: Renderer MUST support both Alpha (0.5.3) and Standard (3.3.5) terrain formats via the existing `WorldTerrainTileBuilder` which already handles both
- **FR-003**: Renderer MUST support an offscreen/headless GL context via Silk.NET hidden window
- **FR-004**: Renderer MUST render terrain with correct vertex positions, normals, texture coordinates, alpha blends, and hole masks — matching MdxViewer reference output
- **FR-005**: Renderer MUST support 4 capture variants: primary (full scene), no-liquids, no-objects, objects-only
- **FR-006**: Renderer MUST support 3D WMO rendering for objects-only variant
- **FR-007**: Renderer MUST produce PNG output with correct alpha channel (alpha=255 for rendered pixels)
- **FR-008**: Multi-process concurrent capture MUST use process-level isolation
- **FR-009**: All game file I/O uses existing `IArchiveCatalog` (via `NativeMpqService` or `MpqArchiveCatalog`)
- **FR-010**: Renderer MUST be `wow-viewer` repo-independent (no references outside `wow-viewer/`)
- **FR-011**: Renderer library MUST NOT have any UI dependency (no ImGui, no WinForms in the core library)

### Key Entities

- **RenderContext** — GL context + framebuffer management
- **TerrainMeshCache** — Builds and caches GPU meshes from `WorldTerrainTileData`
- **TextureCache** — Loads and caches BLP textures, binds to GL texture units
- **SceneRenderer** — Orchestrates rendering of all scene elements into the framebuffer
- **CapturePipeline** — Coordinates I/O → mesh building → rendering → readback → PNG write

## Success Criteria

- **SC-001**: Azeroth_30_48 on 0_5_3_3368 produces output visually matching MdxViewer reference (verified by comparison)
- **SC-002**: Azeroth_30_48 on 3_3_5_12340 produces output visually matching MdxViewer reference
- **SC-003**: All 4 capture variants produce distinguishable outputs for the same tile
- **SC-004**: Batch capture of 16 tiles with 4 variants completes without user interaction
- **SC-005**: Parallel capture (4 workers) on 16 tiles is at least 2x faster than sequential
- **SC-006**: Build succeeds with `dotnet build WowViewer.slnx -c Debug` — 0 errors

## Architecture

### Data Flow

```
staged client dirs
    │
    ▼
IArchiveCatalog (NativeMpqService)
    │
    ▼  ReadFile(virtualPath)
byte[] (raw ADT/WDT file)
    │
    ▼  MapFileSummaryReader + WorldTerrainTileBuilder
WorldTerrainTileData + WorldLiquidTileData + AdtPlacementCatalog
    │
    ▼
WowViewer.Core.Renderer  ◄── NEW
    ├── TerrainMeshBuilder (WorldTerrainTileData → GL VAO/VBO)
    ├── TextureCache (BLP → GL texture)
    ├── SceneRenderer (camera, frustum, draw calls)
    └── HeadlessContext (offscreen GL framebuffer)
    │
    ▼
System.Drawing.Bitmap / ImageSharp PNG
```

### Code Layout

```
wow-viewer/src/core/WowViewer.Core.Renderer/        # NEW — GPU rendering library
    WowViewer.Core.Renderer.csproj
    Headless/                                         # Offscreen GL infrastructure
        HeadlessContext.cs
        RenderSurface.cs
    Scene/                                            # Scene management + camera
        SceneCamera.cs
        FrustumCuller.cs
        SceneRenderer.cs
        RenderVariants.cs                             # Variant flags enum
    Terrain/                                          # Terrain rendering
        TerrainMeshBuilder.cs                         # WorldTerrainTileData → GL mesh
        TerrainRenderer.cs                            # GL draw calls for terrain
        TerrainShaders.cs                             # GLSL vertex/fragment source
    Texture/                                          # Texture management
        TextureCache.cs                               # BLP → GL texture binding
        BlpLoader.cs                                  # BLP decode via SereniaBLPLib
    Sky/                                              # Sky rendering (optional for captures)
        SkyRenderer.cs
    Liquid/                                           # Liquid rendering
        LiquidRenderer.cs
    Wmo/                                              # WMO rendering
        WmoRenderer.cs
        WmoMeshBuilder.cs
    Output/                                           # Framebuffer readback + PNG
        FrameCapture.cs
        PngWriter.cs
    Capture/                                          # Capture pipeline orchestration
        CaptureOrchestrator.cs
        TileCaptureJob.cs

wow-viewer/tools/headless-capture/WowViewer.Tool.Capture/  # NEW — CLI capture tool
    WowViewer.Tool.Capture.csproj
    Program.cs
    Commands/
        RenderCommand.cs
        BatchCommand.cs
        ParallelCommand.cs
    Pipeline/
        CapturePipeline.cs                           # Orchestrates end-to-end capture
        ProcessPool.cs                               # Multi-process worker pool
```
