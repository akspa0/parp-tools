# Progress — WowViewer Native GPU Renderer

## Completed
- 2026-05-24: Spec 017 written (spec, plan, tasks, research)
- 2026-05-24: `WowViewer.Core.Renderer` project created and builds
- 2026-05-24: Phase 1 infrastructure: HeadlessContext, RenderSurface, FrameCapture, PngWriter
- 2026-05-24: Phase 2: SceneCamera, FrustumCuller, TerrainConstants, TerrainMeshBuilder, TerrainMesh

## In Progress
- Phase 2-3: Texture cache, shaders, renderer pipeline

## Next Up
- TerrainShader (GLSL from MdxViewer reference)
- TextureCache (BLP decode + texture array)
- TerrainRenderer (draw calls)
- SceneRenderer (orchestration)
- Capture CLI tool
