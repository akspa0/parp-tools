# Active Context — WowViewer Native GPU Renderer

## Branch
- `v0.5.0-dev` — working on spec `017-native-gpu-renderer`

## Current Focus
Building a native GPU renderer (`WowViewer.Core.Renderer`) from scratch in wow-viewer:
- Consumes existing `WorldTerrainTileData` from `Core.Runtime.WorldTerrainTileBuilder`
- Uses `NativeMpqService` / `MpqArchiveCatalog` for file I/O
- MdxViewer is REFERENCE ONLY for rendering techniques (shader code, vertex layout, UV math)
- Headless rendering is the first-class mode; GUI is a future wrapper

## What Exists (Completed)
- **Spec 017**: spec.md, plan.md, tasks.md, research.md written
- **Phase 1**: `WowViewer.Core.Renderer` project created, builds successfully
  - `Headless/HeadlessContext.cs` — hidden Silk.NET window with offscreen GL context
  - `Headless/RenderSurface.cs` — FBO with color texture + depth renderbuffer
  - `Output/FrameCapture.cs` — framebuffer readback with Y-flip
  - `Output/PngWriter.cs` — RGBA byte[] to PNG via ImageSharp
  - `Scene/SceneCamera.cs` — camera with tile look-at positioning
  - `Scene/FrustumCuller.cs` — view-frustum AABB test
  - `Terrain/TerrainConstants.cs` — game-world constants
  - `Terrain/TerrainMesh.cs` — GPU mesh data model (VAO, textures, bounds)
  - `Terrain/TerrainMeshBuilder.cs` — converts `WorldTerrainTileData` → GL VAO/VBO/EBO with texture arrays

## What's Next
- **Phase 2** (continuing): TextureCache (BLP→GL texture array)
- **Phase 3**: TerrainShader (GLSL from MdxViewer reference), TerrainRenderer (draw calls), SceneRenderer (pass orchestration)
- **Phase 4**: SkyRenderer, LiquidRenderer
- **Phase 5**: WmoRenderer
- **Phase 6**: Capture CLI tool
- **Phase 7**: Validation against MdxViewer reference

## Known Issues
- Normal computation is placeholder (Vector3.UnitZ) — no MCCV vertex color loading yet
- Alpha+shadow texture array upload not yet implemented (TextureCache needs to be built)
- No WMO rendering yet

## Relevant Files
- `wow-viewer/src/core/WowViewer.Core.Renderer/` — all renderer source
- `wow-viewer/specs/017-native-gpu-renderer/` — spec/plan/tasks/research
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/` — input data types
- `wow-viewer/src/core/WowViewer.Core.IO/Files/NativeMpqService.cs` — file I/O
