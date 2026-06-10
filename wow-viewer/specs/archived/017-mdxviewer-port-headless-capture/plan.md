# Implementation Plan: WowViewer Native GPU Renderer + Headless Capture

**Branch**: `017-native-gpu-renderer` | **Date**: 2026-05-24 | **Spec**: `specs/017-native-gpu-renderer/spec.md`

## Summary

Build a native GPU renderer library (`WowViewer.Core.Renderer`) in wow-viewer that consumes the existing runtime data types (`WorldTerrainTileData`, `WorldLiquidTileData`, `AdtPlacementCatalog`) via `WorldTerrainTileBuilder` and renders them to an offscreen OpenGL framebuffer. Wrap it in a `WowViewer.Tool.Capture` CLI for headless batch capture with multi-process concurrency.

This is NOT a port of MdxViewer. It's a purpose-built renderer that uses MdxViewer's rendering techniques as reference but is built on top of wow-viewer's existing data pipeline (`NativeMpqService` → `WorldTerrainTileBuilder`).

## Technical Context

**Language/Version**: C# .NET 10.0-windows

**Primary Dependencies**:
- Silk.NET.OpenGL 2.21 + Silk.NET.Windowing 2.21 (GL context)
- SixLabors.ImageSharp 3.1 (PNG output)
- SereniaBLPLib (BLP texture decode)
- Existing: `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`

**Storage**: Staged game clients under `output/tmp/wowarchive-clients/`; output PNGs

**Testing**: `dotnet build WowViewer.slnx -c Debug`; visual comparison against MdxViewer reference

**Target Platform**: Windows x64, OpenGL 4.1+

**Project Type**: Shared library (Core.Renderer) + CLI tool (Tool.Capture)

**Performance Goals**: Single tile capture <5s; batch 16 tiles with 4 variants <60s (4 workers)

**Constraints**:
- Process isolation for parallelism — separate GL context per worker
- References: MdxViewer is REFERENCE for rendering techniques (shader code, vertex layout, terrain UVs) — NOT copied code
- The renderer must handle the same data that MdxViewer handles (Alpha ADT and LK ADT)
- No UI dependencies in Core.Renderer (ImGui, WinForms, input)

## Constitution Check

| Rule | Status |
|------|--------|
| I. Repo Independence | PASS — all code in `wow-viewer/`, no cross-repo references |
| II. Library-First | PASS — rendering is a shared library; CLI tool is thin wrapper |
| III. Real-Data Validation | PASS — validated against staged `0_5_3_3368` and `3_3_5_12340` |
| IV. Residual Model Chain | N/A — this is a renderer, not an ML model |
| V. Streaming-First Dataset | N/A — rendered PNGs are ML supervision targets, not the dataset pipeline |
| VI. No H:\CLIENTS | PASS — uses `output/tmp/wowarchive-clients/` |
| Read-Only Reference | PASS — MdxViewer is REFERENCE only; no code copied from `gillijimproject_refactor` |
| One Phase at a Time | PASS — phases are sequential and independently testable |

## Project Structure

```
wow-viewer/specs/017-native-gpu-renderer/
├── spec.md              # Feature specification
├── plan.md              # This implementation plan
└── tasks.md             # Task breakdown

wow-viewer/src/core/WowViewer.Core.Renderer/     # NEW — GPU rendering library
├── WowViewer.Core.Renderer.csproj
├── Headless/
│   ├── HeadlessContext.cs                       # Offscreen GL context (Silk.NET hidden window)
│   └── RenderSurface.cs                         # FBO + color/depth attachments
├── Scene/
│   ├── SceneCamera.cs                           # Camera math (MdxViewer Camera.cs AS REFERENCE)
│   ├── FrustumCuller.cs                         # View-frustum culling
│   ├── SceneRenderer.cs                         # Orchestrates all render passes
│   └── RenderVariant.cs                         # Enum: Primary, NoLiquids, NoObjects, ObjectsOnly
├── Terrain/
│   ├── TerrainMeshBuilder.cs                    # WorldTerrainTileData → GL mesh
│   ├── TerrainRenderer.cs                       # GL draw calls for terrain tiles
│   ├── TerrainShader.cs                         # GLSL vertex + fragment (MdxViewer AS REFERENCE)
│   └── TerrainConstants.cs                      # Tile/chunk sizes, UV scales
├── Texture/
│   ├── TextureCache.cs                          # BLP → GL texture with caching
│   └── BlpTextureLoader.cs                      # BLP decode via SereniaBLPLib
├── Sky/
│   └── SkyRenderer.cs                           # Simple gradient sky
├── Liquid/
│   └── LiquidRenderer.cs                        # Water/lava/slime rendering
├── Output/
│   ├── FrameCapture.cs                          # Read framebuffer → byte[]
│   └── PngWriter.cs                             # byte[] → PNG via ImageSharp
└── Capture/
    ├── TileCaptureJob.cs                        # End-to-end: I/O → mesh → render → PNG
    └── CaptureOrchestrator.cs                   # Multi-tile + multi-variant coordinator

wow-viewer/tools/headless-capture/WowViewer.Tool.Capture/   # NEW — CLI capture tool
├── WowViewer.Tool.Capture.csproj
├── Program.cs
├── Commands/
│   ├── RenderCommand.cs                         # Single-tile render
│   ├── BatchCommand.cs                          # Multi-tile sequential
│   └── ParallelCommand.cs                       # Multi-process concurrent
└── CapturePipeline.cs                           # Client bootstrap → tile list → orchestrate
```

## Implementation Phases

### Phase 0 — Research: Existing Data Pipeline + MdxViewer Rendering Reference

**Goal**: Fully understand the input contract (what `WorldTerrainTileData` contains) and the rendering techniques MdxViewer uses, so the new renderer is built correctly the first time.

**Approach**:
1. Study `WorldTerrainTileData`, `WorldTerrainChunkData`, `WorldTerrainHeightmapData` — what data is available, what's the coordinate layout
2. Study `AdtTextureFile`, `AdtTextureChunkLayer`, `AdtMcalDecoder` — how texture layers + alpha maps work
3. Study MdxViewer `TerrainTileMeshBuilder.cs` AS REFERENCE — vertex layout, UV math, hole handling, normal encoding
4. Study MdxViewer `TerrainRenderer.cs` AS REFERENCE — shader structure, texture array binding, draw call batching
5. Study MdxViewer terrain shader GLSL source AS REFERENCE — vertex shader, fragment shader, alpha blending
6. Study MdxViewer `LiquidRenderer.cs` — how liquid mesh + transparency works
7. Study MdxViewer `WmoRenderer.cs` — how WMO groups are rendered
8. Study existing `WorldGpuPreviewRenderer.cs` — what was broken, DON'T repeat those mistakes

**Deliverable**: `research.md` with:
- Full schema of `WorldTerrainTileData` (field-by-field)
- Mesh vertex format (position, normal, UV, texture indices)
- Shader approach (texture array, alpha blend, lighting)
- Known pitfalls from the broken `WorldGpuPreviewRenderer`

---

### Phase 1 — Renderer Library: Project + Headless GL Context

**Goal**: Create the shared renderer library project with a working offscreen GL context.

**Tasks**:
1. Create `WowViewer.Core.Renderer.csproj` targeting `net10.0-windows`
   - Dependencies: `Silk.NET.OpenGL`, `Silk.NET.Windowing`, `SixLabors.ImageSharp`, `SereniaBLPLib`
   - Project references: `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`
2. Implement `HeadlessContext.cs`:
   - Create Silk.NET window with `WindowOptions(IsVisible = false)`
   - Initialize OpenGL 4.1+ core profile
   - Expose `GL` instance
   - Expose `Run()` for single-frame, `Close()` for cleanup
3. Implement `RenderSurface.cs`:
   - Create framebuffer object with color texture + depth renderbuffer
   - `Resize(int width, int height)` for configurable resolution
   - `Bind()` / `Unbind()` for render/readback
4. Implement `FrameCapture.cs`:
   - `ReadPixels(GL, framebuffer) → byte[] RGBA`
5. Implement `PngWriter.cs`:
   - `Save(byte[] rgba, int width, int height, string path) → void` via ImageSharp

**Validation**: Create headless GL context, render clear color to framebuffer, read back pixels, save PNG — verify alpha=255 on all pixels.

---

### Phase 2 — Renderer Library: Camera + Terrain Mesh Builder

**Goal**: Build the GPU mesh representation from `WorldTerrainTileData`.

**Tasks**:
1. Implement `SceneCamera.cs`:
   - Position, yaw, pitch → view matrix
   - Configurable FOV, aspect ratio, near/far → projection matrix
   - `LookAtTile(int tileX, int tileY)` — position camera above tile center
2. Implement `TerrainConstants.cs`:
   - Tile size (533.33333f), chunk size, cell sizes
   - UV scale, coordinate origin constants
3. Implement `TerrainMeshBuilder.cs`:
   - Input: `WorldTerrainTileData` (list of `WorldTerrainChunkData`)
   - Output: GL VAO + VBO + EBO per tile (or per-chunk)
   - Vertex format: position (float3), normal (float3), UV (float2), texture indices (uint4), alpha weights (float4 optimized)
   - Hole mask handling: skip indices in hole regions
   - Reference MdxViewer `TerrainTileMeshBuilder.cs` for vertex layout and interleave pattern
4. Implement `TextureCache.cs`:
   - `LoadTexture(string texturePath, IArchiveCatalog) → uint` — load BLP, upload to GL
   - `GetOrCreateTextureArray(List<string> textureNames) → uint` — create GL_TEXTURE_2D_ARRAY from tile's textures
   - Caching by normalized path

**Validation**: Given a `WorldTerrainTileData` from a real tile, build GL mesh and verify vertex count, bounding box, and draw via simple wireframe shader.

---

### Phase 3 — Renderer Library: Terrain Rendering + Shaders

**Goal**: Render terrain tiles with correct texturing and alpha blending.

**Tasks**:
1. Implement `TerrainShader.cs`:
   - Embedded GLSL vertex shader: transforms position, computes UV, passes tex coords
   - Embedded GLSL fragment shader: samples texture array, blends layers by alpha, applies vertex lighting
   - Reference MdxViewer's terrain shaders for the blending math
2. Implement `TerrainRenderer.cs`:
   - `Render(GL, SceneCamera, List<TerrainMesh>, TextureCache) → void`
   - Bind texture array, set uniforms (view/proj/light), draw indexed
   - Per-tile transform (offset by tile coordinates)
   - Vertex lighting from MCCV or direction-based
3. Handle `RenderVariant.NoTerrain`: skip terrain draw calls
4. Implement `SceneRenderer.cs`:
   - Orchestrates terrain → liquid → sky → objects
   - Takes `RenderVariant` flags
   - Calls `TerrainRenderer.Render()` if terrain is enabled

**Validation**: Render a real tile (Azeroth_30_48) with terrain enabled, capture to PNG, inspect visually. Compare with MdxViewer's output for the same tile.

---

### Phase 4 — Renderer Library: Liquid + Sky Rendering

**Goal**: Support the no-liquids variant and basic sky.

**Tasks**:
1. Implement `SkyRenderer.cs`:
   - Simple gradient sky (zenith color → horizon color)
   - Rendered as full-screen quad or far-plane mesh
2. Implement `LiquidRenderer.cs`:
   - Input: `WorldLiquidTileData`
   - Build liquid mesh per chunk from height/UV data
   - Render with transparency (water, ocean, magma, slime colors)
   - Handle `RenderVariant.NoLiquids`: skip liquid draw calls

**Validation**: Liquids variant output differs from primary (no-liquids = water areas show terrain floor instead)

---

### Phase 5 — WMO Rendering (Objects Variant)

**Goal**: Render WMO groups for the objects-only variant.

**Tasks**:
1. Implement `WmoMeshBuilder.cs`:
   - Load WMO model via `IArchiveCatalog.ReadFile()`
   - Parse WMO root + group files using `WowViewer.Core.IO.Wmo` readers
   - Build GL meshes from WMO group geometry
2. Implement `WmoRenderer.cs`:
   - `Render(List<WorldObjectInstance>, Camera, TextureCache) → void`
   - Transform by WMO placement (position + rotation)
   - Handle `RenderVariant.NoObjects`: skip WMO draw calls

**Validation**: Objects-only variant shows WMO buildings on the tile; no-objects variant shows terrain only.

---

### Phase 6 — Capture CLI Tool

**Goal**: Create the CLI tool that uses the renderer library for headless captures.

**Tasks**:
1. Create `WowViewer.Tool.Capture.csproj`
2. Implement `CapturePipeline.cs`:
   - Bootstrap: open `IArchiveCatalog`, read WDT, enumerate tiles
   - For each tile: read ADT → build mesh → render → capture → save
   - Handle both Alpha and LK tile families via `AdtTileFamilyResolver`
3. Implement `RenderCommand`:
   - `capture render --client-root ... --map ... --tile-x ... --tile-y ... --variant ... --output ...`
   - Single tile, single variant
4. Implement `BatchCommand`:
   - `capture batch --client-root ... --map ... --tile-list ... --variants all --output-dir ...`
   - Iterates tiles, produces 4 variants each
5. Implement `ParallelCommand`:
   - `capture parallel --client-root ... --map ... --workers 4 --tile-list ...`
   - Spawns N child `capture render` processes

**Validation**: Both `capture render` and `capture batch` produce correct PNGs for Azeroth_30_48 and 30_49.

---

### Phase 7 — Validation Against MdxViewer Reference

**Goal**: Prove the new renderer produces pixel-correct output matching MdxViewer.

**Tasks**:
1. Generate reference captures from MdxViewer for Azeroth_30_48 on `0_5_3_3368` and `3_3_5_12340`
2. Generate captures from the new tool with the same inputs
3. Compare PNGs pixel-by-pixel, document any differences
4. Fix differences by adjusting shader constants, vertex layout, or texture sampling

**Validation**: Per-pixel comparison shows <2% difference for all 4 variants on both client builds.

---

### Phase 8 — V16 Dataset Pipeline Integration

**Goal**: Wire the new capture tool into the V16 dataset build pipeline.

**Tasks**:
1. Update `build_v16_dataset.py` to call `WowViewer.Tool.Capture batch` instead of launching MdxViewer
2. Wire up `generate-viewer-stubs` to produce tile lists for the new CLI

**Validation**: `build_v16_dataset.py generate-renderer-truth` succeeds and produces correct supervision images.
