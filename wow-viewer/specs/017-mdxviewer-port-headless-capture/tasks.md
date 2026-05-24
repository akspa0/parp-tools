# Tasks: WowViewer Native GPU Renderer + Headless Capture

**Input**: `spec.md`, `plan.md` from `specs/017-native-gpu-renderer/`

**Convention**: `[P]` = parallel with other `[P]` tasks in same phase. `[US1]` = user story.

---

## Phase 0: Research — Data Pipeline + Rendering Reference

**Goal**: Understand input data types and MdxViewer rendering techniques.

- [ ] T001 [P] Study `WorldTerrainTileData`, `WorldTerrainChunkData`, `WorldTerrainHeightmapData` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/` — document every field, coordinate system, and expected range
- [ ] T002 [P] Study `AdtTextureFile`, `AdtTextureChunk`, `AdtTextureChunkLayer`, `AdtMcalDecoder` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/` — document texture layer layout, alpha decode output format
- [ ] T003 [P] Study MdxViewer `TerrainTileMeshBuilder.cs` at `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs` — document vertex format, UV math, hole handling, index generation AS REFERENCE (do NOT copy)
- [ ] T004 [P] Study MdxViewer `TerrainRenderer.cs` at `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs` — document shader structure, draw call pattern, texture array binding AS REFERENCE
- [ ] T005 [P] Study MdxViewer `Terrain/TerrainLighting.cs` — document lighting model (direction, ambient, MCCV handling)
- [ ] T006 [P] Study existing broken `WorldGpuPreviewRenderer.cs` at `wow-viewer/src/viewer/WowViewer.App/WorldGpuPreviewRenderer.cs` — identify specific things that went wrong (all-black output, alpha=0, camera miss)
- [ ] T007 Compile findings into `specs/017-native-gpu-renderer/research.md`

---

## Phase 1: Renderer Library — Project + Headless GL Context

**Goal**: Shared renderer library with offscreen GL context working.

- [ ] T008 Create `wow-viewer/src/core/WowViewer.Core.Renderer/` directory and `WowViewer.Core.Renderer.csproj` targeting `net10.0-windows` with dependencies on Silk.NET.OpenGL, Silk.NET.Windowing, SixLabors.ImageSharp, SereniaBLPLib, and project references to Core, Core.IO, Core.Runtime
- [ ] T009 [P] Implement `HeadlessContext.cs` — hidden Silk.NET window with `IsVisible=false`, `WindowOptions.API = ContextAPI.OpenGL`, OpenGL 4.1 core profile. Exposes `GL` instance, `RenderSingleFrame()`, `Close()`
- [ ] T010 [P] Implement `RenderSurface.cs` — FBO with `GL_COLOR_ATTACHMENT0` texture (RGBA8) + `GL_DEPTH_COMPONENT24` renderbuffer. `Resize(w,h)`, `Bind()`, `Unbind()`, `ReadPixels() → byte[]`
- [ ] T011 [P] Implement `FrameCapture.cs` — `CaptureFrame(GL, RenderSurface) → byte[]` reads framebuffer pixels
- [ ] T012 [P] Implement `PngWriter.cs` — `Save(byte[] rgba, w, h, path)` writes PNG via ImageSharp
- [ ] T013 Add `WowViewer.Core.Renderer` to `WowViewer.slnx`
- [ ] T014 **VALIDATE**: Create headless context, clear to (0.2, 0.3, 0.5, 1.0), read back pixels, save PNG. Verify all pixels have alpha=255 and correct RGB

---

## Phase 2: Renderer Library — Camera + Terrain Mesh Builder

**Goal**: Convert `WorldTerrainTileData` to GPU meshes.

- [ ] T015 [P] Implement `SceneCamera.cs` — position, yaw, pitch → view matrix; perspective FOV → projection matrix; `LookAtTile(tileX, tileY)` for top-down centering
- [ ] T016 [P] Implement `TerrainConstants.cs` — `TileSize = 533.33333f`, `ChunkSize = TileSize / 16f`, `SubCellSize = ChunkSize / 8f`, `AlphaSize = 64`, `VertsPerChunk = 145`
- [ ] T017 Implement `TerrainMeshBuilder.cs` — input `WorldTerrainTileData`, output GL VAO/VBO/EBO. Per-vertex: position (3 floats), normal (3 floats), UV (2 floats), texture layer indices (4 shorts), alpha weights (4 bytes). Handle holes via index omission. Build texture array layers from tile's textures.
- [ ] T018 [P] Implement `TextureCache.cs` — `LoadBlp(path, IArchiveCatalog) → uint` loads BLP, uploads to GL; `CreateTextureArray(List<uint> glTextures) → uint` assembles GL_TEXTURE_2D_ARRAY; cached by normalized path
- [ ] T019 [P] Implement `FrustumCuller.cs` — construct frustum planes from view-projection; `TestAABB(min, max) → bool` for culling terrain tiles
- [ ] T020 **VALIDATE**: Load real ADT, build mesh, compute bounding box. Verify vertex count matches expectation (145 verts × 256 chunks = 37,120 verts per tile)

---

## Phase 3: Renderer Library — Terrain Shaders + Rendering

**Goal**: Render terrain tiles with correct texturing, alpha, and lighting.

- [ ] T021 Implement `TerrainShader.cs` — embedded GLSL vertex + fragment. Vertex: transform position, compute UVs. Fragment: sample GL_TEXTURE_2D_ARRAY at computed UV, blend up to 4 layers by alpha weights. Reference MdxViewer's shader approach.
- [ ] T022 Implement `TerrainRenderer.cs` — `Render(GL, Camera, TerrainMesh[], TextureCache, RenderVariant) → void`. Bind texture array, set uniforms (view, proj, light direction, ambient), draw all visible tiles with indexed triangles. Handle `RenderVariant.NoTerrain` by skipping.
- [ ] T023 Implement `SceneRenderer.cs` — orchestrates terrain → liquid → sky → objects draw calls in order. Manages global GL state (depth test, blend, cull face). Accepts `RenderVariant` to skip passes.
- [ ] T024 [US1] **VALIDATE**: Render Azeroth_30_48 on staged `0_5_3_3368` to offscreen framebuffer, save PNG. Verify >20K unique colors and alpha=255. Compare visually with MdxViewer output.

---

## Phase 4: Renderer Library — Sky + Liquid Rendering

**Goal**: Support no-liquids variant with sky background.

- [ ] T025 [P] Implement `SkyRenderer.cs` — full-screen quad with gradient vertex shader. Zenith color → horizon color gradient. Behind all opaque geometry.
- [ ] T026 [P] Implement `LiquidRenderer.cs` — input `WorldLiquidTileData`. Build liquid mesh per chunk from height/UV data in `WorldLiquidLayerData`. Render with alpha transparency, color-coded by type (blue=water, green=slime, red=magma). Handle `RenderVariant.NoLiquids` by skipping.
- [ ] T027 [US2] **VALIDATE**: Primary and no-liquids variants produce different outputs (water areas differ). Objects-only variant shows terrain + sky but no liquids.

---

## Phase 5: WMO Rendering (Objects Variant)

**Goal**: Render WMO groups for the objects-only variant.

- [ ] T028 Implement `WmoMeshBuilder.cs` — load WMO root file via `IArchiveCatalog.ReadFile()`, parse WMO groups via `WowViewer.Core.IO.Wmo` readers. Build GL meshes per group (vertices, normals, UVs, indices).
- [ ] T029 Implement `WmoRenderer.cs` — `Render(List<WorldObjectInstance>, Camera, TextureCache) → void`. Each instance transformed by placement position + rotation (from `AdtWorldModelPlacement`). Handle `RenderVariant.NoObjects` by skipping.
- [ ] T030 [US2] **VALIDATE**: Objects-only variant shows WMO buildings; no-objects variant shows bare terrain

---

## Phase 6: Capture CLI Tool

**Goal**: CLI tool wrapping the renderer for headless captures.

- [ ] T031 Create `wow-viewer/tools/headless-capture/WowViewer.Tool.Capture/` project. Dependencies: `WowViewer.Core.Renderer`, `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`
- [ ] T032 Implement `CapturePipeline.cs` — bootstrap: `NativeMpqService` → load WDT → enumerate tiles → for each tile: read ADT → `WorldTerrainTileBuilder.Read()` → build mesh → render → capture → save. Handle both Alpha and LK tile families.
- [ ] T033 [US1] Implement `RenderCommand.cs` — `capture render --client-root ... --map ... --tile-x ... --tile-y ... --variant primary --output tile.png`
- [ ] T034 [US2] Implement `BatchCommand.cs` — `capture batch --client-root ... --map ... --tile-list tiles.txt --variants all --output-dir captures/`. Produces 4 PNGs per tile.
- [ ] T035 [US3] Implement `ParallelCommand.cs` — `capture parallel --workers 4 --client-root ... --map ... --all-tiles`. Divides tile list across N child `capture render` processes. Manages child lifecycle, collects results.
- [ ] T036 Add tool to `WowViewer.slnx`
- [ ] T037 [US1] **VALIDATE**: Run `capture render` for Azeroth_30_48 on `0_5_3_3368`. Verify output matches Phase 3 validation (same pixel output).

---

## Phase 7: Validation Against MdxViewer Reference

**Goal**: Prove pixel-correct output matching MdxViewer.

- [ ] T038 Capture reference from MdxViewer for Azeroth_30_48 on `0_5_3_3368` (all 4 variants)
- [ ] T039 Capture from new tool for Azeroth_30_48 on `0_5_3_3368` (all 4 variants)
- [ ] T040 Compare PNGs pixel-by-pixel. Document any differences >2%
- [ ] T041 Fix differences by adjusting shader constants, vertex math, or texture sampling
- [ ] T042 Repeat for Azeroth_30_48 on `3_3_5_12340`
- [ ] T043 [US4] Create quick GUI wrapper `WowViewer.Tool.Capture view` that opens visible window with camera controls for debugging

---

## Phase 8: V16 Dataset Pipeline Integration

- [ ] T044 Update `build_v16_dataset.py` to call `WowViewer.Tool.Capture batch` instead of MdxViewer for renderer-truth generation
- [ ] T045 Wire `generate-viewer-stubs` to produce tile lists for the new CLI

---

## Dependencies & Execution Order

| Phase | Depends On | Duration |
|-------|-----------|----------|
| 0. Research | — | 1 session |
| 1. Project + GL Context | 0 | 1 session |
| 2. Camera + Mesh | 1 | 1 session |
| 3. Terrain Shaders + Render | 2 | 1 session |
| 4. Sky + Liquid | 3 | 1 session |
| 5. WMO | 3 | 1 session |
| 6. Capture CLI | 3, 4, 5 | 1 session |
| 7. Validation | 6 | 1 session |
| 8. V16 Integration | 7 | 1 session |

**Note**: Phases 4 and 5 can proceed in parallel after Phase 3 completes (different subsystems). Phase 6 depends on both 4 and 5 for full variant support.
