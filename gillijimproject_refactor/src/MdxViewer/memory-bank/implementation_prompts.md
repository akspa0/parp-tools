# MdxViewer Implementation Prompts

Paste any of these into a fresh Copilot chat to implement the feature.
Each prompt is self-contained with context, scope, and acceptance criteria.

---

## Cherry-Pick Guide (from main, post-baseline 343dadfa)

Current recovery branch `recovery/terrain-surgical-343dadf` is at `c1e0d29`
(baseline + safe manager/model replay). The dev branch `v0.4.1-dev` has ALL
main commits already. These notes help cherry-pick **individual features** from
main into the recovery branch without pulling in terrain regressions.

### Commits After Baseline (343dadfa → main)

| Commit | Summary | Terrain Risk | Cherry-Pickable? |
|--------|---------|-------------|------------------|
| `177f961` | Alpha masks, 0.5.3 terrain, MH2O, M2, tile batching | **HIGH** — rewrites TerrainRenderer (+1122 lines), adds TerrainTileMesh/Builder | Partial: ModelRenderer (6 lines) safe. Terrain files are the regression source. |
| `d50cfe7` | UI, import/export menus, WorldScene, TerrainImageIo | **MIXED** — new WorldScene.cs + TerrainImageIo.cs safe; touches TerrainRenderer, StandardTerrainAdapter, ViewerApp | Partial: extract new files only |
| `326e6f8` | Heightmap/alpha import/export docs | **LOW** — new TerrainHeightmapIo.cs is standalone; TerrainManager/VlmTerrainManager changes are small | Yes for new files; ViewerApp UI hunks need manual extraction |
| `4e2f681` | Chunk inspector, GLB export | **LOW** — new MapGlbExporter.cs is standalone; small TerrainManager/Renderer touches | Partial: MapGlbExporter.cs safe to take whole |
| `37f669c` | Relaxed alpha decode, patch priority, BZip2 | **HIGH** — 197-line StandardTerrainAdapter rewrite | No whole-commit cherry-pick. BZip2/MPQ changes may be in other dirs (check). |
| `39799bf` | Alpha map + vertex color refactor | **HIGH** — touches all terrain pipeline files | No. This is the core regression commit for alpha handling. |
| `62ecf64` | Terrain audit, WDL preview cache, MCCV, many fixes | **HIGH** — 32 files, 3149 insertions. Mega-commit. | Partial: safe new files only |

### Safe Whole-File Extractions (new files that don't exist at baseline)

These files were **added** in the commits above and have no baseline version to
conflict with. They can be extracted directly:

```bash
# From d50cfe7 — Export tools + WorldScene
git show d50cfe7:gillijimproject_refactor/src/MdxViewer/Export/TerrainImageIo.cs > src/MdxViewer/Export/TerrainImageIo.cs
git show d50cfe7:gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs > src/MdxViewer/Terrain/WorldScene.cs

# From 326e6f8 — Heightmap I/O
git show 326e6f8:gillijimproject_refactor/src/MdxViewer/Export/TerrainHeightmapIo.cs > src/MdxViewer/Export/TerrainHeightmapIo.cs

# From 4e2f681 — GLB exporter
git show 4e2f681:gillijimproject_refactor/src/MdxViewer/Export/MapGlbExporter.cs > src/MdxViewer/Export/MapGlbExporter.cs

# From 62ecf64 — MCCV decoder, WDL preview cache, ViewerApp partial
git show 62ecf64:gillijimproject_refactor/src/MdxViewer/Terrain/MccvColorDecoder.cs > src/MdxViewer/Terrain/MccvColorDecoder.cs
git show 62ecf64:gillijimproject_refactor/src/MdxViewer/Terrain/WdlPreviewCacheService.cs > src/MdxViewer/Terrain/WdlPreviewCacheService.cs
git show 62ecf64:gillijimproject_refactor/src/MdxViewer/ViewerApp_WdlPreviewCache.cs > src/MdxViewer/ViewerApp_WdlPreviewCache.cs

# From 177f961 — Tile mesh (CAUTION: these are the tile-batching system
# that may have terrain regressions — only take if you want the tile pipeline)
git show 177f961:gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMesh.cs > src/MdxViewer/Terrain/TerrainTileMesh.cs
git show 177f961:gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs > src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs
```

### Recommended Cherry-Pick Strategy

**Phase A — Safe new-file extraction (no conflict risk)**
1. Extract `TerrainImageIo.cs`, `TerrainHeightmapIo.cs`, `MapGlbExporter.cs`
2. Extract `MccvColorDecoder.cs`, `WdlPreviewCacheService.cs`
3. These add export/utility features without touching the terrain render pipeline

**Phase B — WorldScene + ViewerApp UI hunks (manual merge)**
1. Extract `WorldScene.cs` from `d50cfe7` (but it references terrain APIs — may need adaptation)
2. Cherry-pick ViewerApp UI hunks for import/export menus, chunk inspector by hand
3. Skip any ViewerApp hunks that touch terrain rendering flow

**Phase C — Terrain pipeline (AFTER regression is fixed)**
1. Only after the terrain alpha/shadow pipeline is verified correct
2. Re-evaluate 177f961 tile batching, 37f669c relaxed decode, 39799bf alpha refactor
3. Apply incrementally with real-data validation between each

### Features NOT in any existing commit (need new implementation)
These features from the prompts below have **zero existing code** to cherry-pick:
- Material/Texture Animation (TXAN, UV scrolling, global sequences)
- Ribbon Emitters (RIBB)
- Detail Doodads (placement + rendering)
- GPU Instancing
- Async MDX texture loading
- Debug overlays (normals, bounds, frustum)

---

## Priority 1 — Core Missing Features

### Prompt 1A: Material & Texture Animation (TXAN / UV Scrolling / Global Sequences)

```
## Task: Implement Material & Texture Animation for MDX models

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Geoset animation (KGAC/KGAL) already works in `Rendering/ModelRenderer.cs`.
Bone animation, keyframe tracks, and the MdxAnimator already work.
What's missing: per-material texture animation — UV scrolling, TXAN tracks,
and global sequence support.

Read these files first for context:
- `gillijimproject_refactor/memory-bank/activeContext.md`
- `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
- `gillijimproject_refactor/src/MdxViewer/Rendering/MdxAnimator.cs`
- `gillijimproject_refactor/src/MdxViewer/Formats/Mdx/` (parser classes)

### Requirements
1. **TXAN track parsing**: Parse TXAN (texture animation) chunks from MDX files.
   Look at how existing keyframe tracks (KGAC, KGAL, bone tracks) are parsed
   in `Formats/Mdx/` and follow the same pattern.

2. **UV scrolling**: In the fragment shader (`ModelRenderer.InitShaders`),
   add a `mat3 uTexTransform` uniform (or per-layer UV offset/scale).
   Apply the animated UV transform before texture sampling.

3. **Global sequences**: Some animations run on global time (not tied to a
   specific animation sequence). The MDX format stores a GLBS chunk with
   global sequence durations. Ensure TXAN tracks that reference a global
   sequence index use elapsed wall-clock time modulo the global sequence
   duration, not the current animation sequence time.

4. **Per-layer material alpha animation**: Wire up any per-material-layer
   alpha/color keyframe tracks so material layers can fade in/out.

### Acceptance Criteria
- Models with scrolling textures (e.g. water effects on MDX, glowing runes)
  show animated UVs
- Global sequences loop independently of the selected animation
- No regressions to existing geoset animation or bone animation
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

### Prompt 1B: Animation Playback UI (Timeline / Speed / MDX Direct)

```
## Task: Complete the Animation Playback UI for MDX models

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Current state:
- `Rendering/MdxAnimator.cs` has SetSequence(), IsPlaying, CurrentFrame
- ViewerApp.cs has a basic sequence dropdown + play/pause for SQL GameObjects
  (see DrawSelectedSqlGameObjectAnimationControls)
- Missing: general-purpose animation panel for any MDX model, timeline
  scrubber, speed control, bone/track info

Read these first:
- `gillijimproject_refactor/src/MdxViewer/Rendering/MdxAnimator.cs`
- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` (search for
  "DrawSelectedSqlGameObjectAnimationControls" and the ImGui rendering sections)

### Requirements
1. **Unified Animation Panel**: Create an ImGui panel (or extend the existing
   properties panel) that shows for ANY loaded MDX model, not just SQL
   GameObjects. Should include:
   - Sequence selector dropdown (list all SEQS from the model)
   - Play / Pause / Stop buttons
   - Speed multiplier slider (0.1x to 3.0x, default 1.0x)
   - Timeline scrubber (horizontal slider showing current frame within
     the sequence's [intervalStart, intervalEnd] range)
   - Display: current frame, total frames, sequence name, bone count

2. **MdxAnimator enhancements**: Add PlaybackSpeed property and a way to
   scrub to a specific frame (seek). Ensure pausing freezes the animation
   at the current frame.

3. **Sequence metadata display**: Show the sequence's moveSpeed, flags
   (looping, etc.), rarity, and blend time if available from the parser.

### Acceptance Criteria
- Loading any MDX model shows the animation panel
- Selecting a sequence plays it; timeline slider tracks progress
- Speed slider works (slow-mo and fast-forward)
- Scrubbing the timeline seeks to that frame
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

### Prompt 1C: MH2O Liquid Rendering (3.3.5 ADT)

```
## Task: Implement MH2O liquid rendering for 3.3.5 terrain

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Current state:
- MCLQ (old per-chunk liquid) works for Alpha-era terrain
- MH2O (3.3.5 split ADT liquid format) is referenced but NOT rendered
- LiquidRenderer.cs exists and handles MCLQ mesh generation
- StandardTerrainAdapter.cs references MH2O but falls back to MCLQ
- The WoWMapConverter.Core library (`src/WoWMapConverter/WoWMapConverter.Core/`)
  has LK format parsers

Read these first:
- `gillijimproject_refactor/memory-bank/activeContext.md`
- `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
- `gillijimproject_refactor/src/MdxViewer/Terrain/LiquidRenderer.cs`
- `gillijimproject_refactor/src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  (search for MH2O / liquid sections)
- `gillijimproject_refactor/src/MdxViewer/Terrain/VlmProjectLoader.cs`
  (search for liquid)
- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/`

### Requirements
1. **MH2O parsing**: Extract liquid headers, instances, and vertex data from
   the MH2O chunk in `_root.adt` or the combined ADT. Each MH2O header has
   instances with liquid type, min/max height, vertex data offset, and a
   visibility mask (8x8 bits).

2. **Mesh generation**: Build liquid surface meshes from MH2O height data.
   Each instance covers a sub-region of the chunk (offsetX/Y, width/height).
   Heights are per-vertex within that sub-region. Use the visibility mask
   to skip hidden quads.

3. **Type detection**: Map MH2O liquidType to visual types (Water, Ocean,
   Magma, Slime) using LiquidType.dbc or hardcoded type ranges (0-3=water,
   4-7=ocean, 8-11=magma, 12-15=slime or similar client mapping).

4. **Render integration**: Feed MH2O meshes into the existing LiquidRenderer
   or create a parallel path. Ensure the liquid shader (already in
   WmoRenderer or LiquidRenderer) handles the geometry.

### Acceptance Criteria
- Loading a 3.3.5 ADT with water/lava shows liquid surfaces
- Liquid heights match the terrain (no floating/submerged planes)
- Different liquid types render with distinct colors
- MCLQ (Alpha) still works — no regression
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

## Priority 2 — Visual Quality

### Prompt 2A: Ribbon Emitters (RIBB)

```
## Task: Implement Ribbon Emitter rendering for MDX models

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Current state:
- Particle emitters (PRE2) fully work: emitter, physics, atlas, billboard
- Ribbon emitters (RIBB) are NOT implemented
- Existing particle code is in `Rendering/` (ParticleEmitter, ParticleSystem,
  ParticleRenderer classes)
- MDX format parsing is in `Formats/Mdx/`

Read these first:
- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
- All files in `gillijimproject_refactor/src/MdxViewer/Rendering/` matching
  Particle*
- `gillijimproject_refactor/src/MdxViewer/Formats/Mdx/` (parser classes)
- The renderer plan: `src/MdxViewer/memory-bank/renderer_plan.md`

### Requirements
1. **RIBB chunk parsing**: Parse ribbon emitter data from MDX files. RIBB
   entries contain: emissionRate, lifeSpan, color, alpha, textureSlot,
   heightAbove/Below keyframe tracks, and attachment to a bone node.

2. **Ribbon trail geometry**: Build a triangle strip from the ribbon's
   history of bone positions. Each frame, sample the attached bone's world
   position and add a new control point. Old points expire after lifeSpan.
   The strip has two edges: heightAbove and heightBelow from the center line.

3. **UV mapping**: U coordinate stretches along the ribbon length (0→1 from
   newest to oldest). V coordinate goes 0→1 from top edge to bottom edge.

4. **Rendering**: Use the particle blend mode (add/blend/alpha) and the
   referenced texture. Render as a triangle strip with per-vertex color
   and alpha that fade over lifetime.

### Acceptance Criteria
- Models with ribbon trails (e.g. weapon enchants, spell effects) show
  the ribbon following the bone
- Ribbons fade out over their lifetime
- Ribbon texture is correctly UV-mapped
- No regression to existing PRE2 particle rendering
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

### Prompt 2B: Detail Doodads (Per-Chunk Foliage)

```
## Task: Implement detail doodad (grass/foliage) rendering for terrain

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Current state:
- Terrain chunks render with texture layers and alpha blending
- WoWConstants.cs defines DetailDoodadDistance=100f, MaxDetailDoodads=64
- ModelRenderer.cs has foliage texture name detection (IsFoliageTexture)
- NO actual detail doodad placement or rendering exists

Read these first:
- `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs`
- `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainManager.cs`
  or `VlmTerrainManager.cs`
- `gillijimproject_refactor/src/MdxViewer/Rendering/WoWConstants.cs`
- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
  (search for IsFoliageTexture)

### Requirements
1. **Placement system**: Create `Terrain/DetailDoodadManager.cs`. For each
   loaded terrain chunk, generate up to 64 doodad positions using a seeded
   PRNG (seed from chunk X/Y coords for determinism). Scatter positions
   within the chunk bounds and sample terrain height for Y placement.

2. **Model selection**: Use the chunk's ground effect textures (MCLY layer
   data) to determine which foliage models/textures to place. Fall back
   to generic grass billboards if no specific doodad MDX is referenced.

3. **Billboard rendering**: Render detail doodads as camera-facing quads
   with an alpha-tested grass/foliage texture. Use a shared VBO for all
   doodads in a chunk, uploaded once.

4. **Distance culling**: Only render doodads within 100 units of the camera.
   Apply alpha fade-out between 80-100 units for smooth pop-in.

5. **Integration**: Hook into the terrain render pass. Render after opaque
   terrain, before transparent objects.

### Acceptance Criteria
- Terrain chunks near the camera show scattered grass/foliage
- Doodads fade out at distance boundaries (no hard pop-in)
- Placement is deterministic (same position every load)
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

## Priority 3 — Performance & Polish

### Prompt 3A: GPU Instancing for Repeated Doodads/Placements

```
## Task: Add GPU instancing for repeated model placements

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Current state:
- World scene renders MDX doodads (MDDF) and WMO objects (MODF)
- Each instance is rendered with a separate draw call (per-model
  MatrixPalette upload + glDrawElements)
- ModelRenderer has BeginBatch/RenderInstance pattern for shared shader state
- No glDrawElementsInstanced or instance buffer usage

Read these first:
- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
  (search for BeginBatch, RenderInstance, RenderGeoset)
- `gillijimproject_refactor/src/MdxViewer/Rendering/MdxRenderer.cs`
  (the batch rendering orchestrator)
- `gillijimproject_refactor/src/MdxViewer/Rendering/WorldScene.cs`
  or the world placement rendering in ViewerApp.cs

### Requirements
1. **Instance buffer**: For static (non-animated) MDX models that appear
   multiple times in the world (e.g. trees, rocks, fences), collect their
   world transform matrices into an instance buffer (VBO with
   divisor=1).

2. **Instanced draw calls**: Use `gl.DrawElementsInstanced()` to render all
   instances of the same static model in a single draw call.

3. **Vertex shader changes**: Add a `mat4 instanceModel` attribute (4x vec4
   columns) read from the instance buffer. Multiply by the standard
   view/projection. Only apply bone transforms for animated models (keep
   the existing non-instanced path for animated MDX).

4. **Model grouping**: In the world scene render pass, group placements by
   model path. Models with >1 static instance use the instanced path.
   Models with animations or a single instance use the existing path.

5. **Frustum culling**: Pre-cull instances on CPU before uploading to the
   instance buffer (only visible instances go into the VBO each frame).

### Acceptance Criteria
- Scenes with many repeated models (forests, fences) show improved FPS
- Visual output is identical to non-instanced rendering
- Animated models still render correctly (not instanced)
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

### Prompt 3B: Async MDX Texture Loading

```
## Task: Implement async BLP texture loading for MDX/WMO models

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Current state:
- Terrain textures load async (background thread parse + render thread
  GPU upload) via TerrainManager/VlmTerrainManager
- MDX model textures load SYNCHRONOUSLY in ModelRenderer.LoadTextures()
  — this blocks the render thread during model loading
- BLP decoding (SereniaBLPLib) is CPU-intensive and causes visible stalls
  when loading new models or entering areas with many doodads

Read these first:
- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
  (search for LoadTextures, _textureHandles)
- `gillijimproject_refactor/src/MdxViewer/Terrain/VlmTerrainManager.cs`
  (for the async pattern used by terrain)

### Requirements
1. **Background BLP decode**: Move BLP file reading and decompression to a
   background thread (Task.Run or a dedicated texture loading thread).
   BLP → RGBA byte array happens off the render thread.

2. **Render-thread GPU upload**: Queue decoded texture data for upload on
   the next render frame. Only `gl.GenTexture`, `gl.TexImage2D`, etc. run
   on the render thread. Use a ConcurrentQueue<PendingTexture> pattern.

3. **Placeholder texture**: While a texture is loading, bind a 1x1 white
   or magenta placeholder texture so the model renders immediately
   (textures pop in when ready rather than blocking).

4. **Texture cache**: If the same BLP path is requested by multiple models,
   share the GPU texture handle. Don't decode the same BLP twice.

5. **Error handling**: If a BLP fails to load, log a warning and keep the
   placeholder. Don't crash or leave a null texture handle.

### Acceptance Criteria
- Loading a model no longer freezes the render loop
- Textures appear within 1-2 frames of decode completion
- Multiple models sharing the same texture decode it only once
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

### Prompt 3C: Debug Overlays (Wireframe / Normals / Bounds)

```
## Task: Add debug visualization overlays to the terrain and model renderers

### Context
MdxViewer is a .NET 10 / Silk.NET OpenGL WoW model viewer at:
  `gillijimproject_refactor/src/MdxViewer/`

Some debug overlays already exist:
- Terrain shader has grid overlay (chunk/tile boundaries) and contour lines
- Terrain shader has alpha debug visualization mode
- Missing: wireframe toggle, normal visualization, bounding box/sphere
  display, frustum visualization

Read these first:
- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` (ImGui debug panels)
- `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs`
- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
- `gillijimproject_refactor/src/MdxViewer/Rendering/WoWConstants.cs`

### Requirements
1. **Wireframe toggle**: Add an ImGui checkbox that calls
   `gl.PolygonMode(GL_FRONT_AND_BACK, GL_LINE)` before terrain/model
   rendering and restores `GL_FILL` after. Apply to terrain, MDX, and WMO.

2. **Normal visualization**: Render vertex normals as short line segments.
   Create a simple line shader. For each vertex, draw a line from position
   to position + normal * scale. Option to show terrain normals, model
   normals, or both.

3. **Bounding box/sphere display**: For each MDX/WMO placement, draw the
   bounding box (wireframe cube) or bounding sphere (wireframe circle
   approximation). Use the frustum culler's bounds data.

4. **Frustum visualization**: Draw the camera frustum as wireframe lines
   (useful when debugging from a second camera or checking culling).

5. **ImGui controls**: Add a "Debug Overlays" collapsible section in the
   UI with toggles for each overlay type.

### Acceptance Criteria
- Each overlay can be toggled independently
- Wireframe mode shows triangle edges for terrain and models
- Normal lines point outward from surfaces
- Bounding volumes match the actual placement bounds
- Build succeeds: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
```

---

## Usage Notes

- **Before terrain work**: The terrain pipeline has active regressions on the
  `recovery/terrain-surgical-343dadf` branch. Read
  `gillijimproject_refactor/memory-bank/activeContext.md` and the terrain
  alpha instructions (`.github/instructions/terrain-alpha.instructions.md`)
  before any terrain-touching prompt.

- **Build command**: Always validate with
  `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`

- **Data paths**: Test data locations are in
  `gillijimproject_refactor/memory-bank/data-paths.md` — never ask the user
  for alternate paths.

- **Copilot instructions**: The workspace `.github/copilot-instructions.md`
  and `.github/instructions/*.instructions.md` files are auto-loaded and
  contain critical rules (FourCC conventions, high-risk file list, terrain
  alpha safety, testing/validation language requirements).
