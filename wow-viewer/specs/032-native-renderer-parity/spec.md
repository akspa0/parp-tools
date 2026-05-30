# Feature Specification: Native Renderer Parity

**Feature Branch**: `032-native-renderer-parity`

**Created**: 2026-05-30

**Status**: Draft

**Input**: Ghidra RE of wowclient.exe build 3368 reveals the complete terrain rendering pipeline, WMO render pass dispatch, water/liquid system, lighting model, LOD strategy, and debug toggle system. The current wow-viewer renderer (via MdxViewer) is a simplified CPU-bound implementation that lacks inner vertices, per-batch material flags, interior/exterior split, interior fog, liquid type dispatch, lightmap pass split, distance-based LOD, per-chunk light selection, animated liquid textures, shadow overlay, and low-detail far terrain. This spec captures every rendering gap and prescribes the work needed for `WowViewer.Core.Runtime` to achieve visual parity with the native client.

## Problem Statement

The current renderer produces visually incorrect output compared to the native client because it:

1. **Uses wrong mesh topology**: Flat 9x9 grid instead of 145-vertex layout with inner vertices and per-cell diagonal splits (spec 031 tracks the data structure fix; this spec tracks the rendering fix)
2. **Has no interior/exterior WMO split**: All WMO groups rendered with same pass, ignoring `flags & 0x48` and per-batch MOMT flags (spec 030 tracks architecture; this spec tracks the rendering implementation)
3. **No interior fog**: Missing entirely
4. **No liquid type dispatch**: Water rendered one way regardless of interior/exterior or magma type
5. **No lightmap pass split**: Interior vs exterior lightmap behavior differs but our renderer doesn't distinguish
6. **No distance-based LOD**: All terrain rendered at full detail regardless of distance
7. **No per-chunk lighting**: Sun light only, no local light sources
8. **No shadow overlay**: Terrain shadow textures not blended
9. **No animated liquid textures**: Water uses static texture instead of 30-frame animation cycle
10. **No low-detail far terrain**: Distant terrain renders at same detail as near terrain
11. **No specular water**: Water is flat-shaded, no specular highlights or pixel shader path
12. **No detail doodad distance fade**: Doodads either visible or not, no distance-based alpha

## Ghidra RE Findings — Rendering Pipeline Details

### Terrain Rendering (`RenderLayers` at 0x006a5d80)

1. **Distance sort**: Chunks placed in 26 distance buckets via `AddMapChunk` (bucket = `ROUND(dist * scale - offset)`, clamped 0..25)
2. **Per-chunk setup**: `SelectLight` picks sun + up to 7 local lights; world transform set to chunk origin
3. **Texture LOD decision**:
   - If `dist > textureLodDist + 256.0`: render only 1 layer (base texture)
   - If `dist > textureLodDist`: alpha-fade extra layers (fade = `(256 - (dist - textureLodDist)) * 128.0 / 256.0`, clamped)
   - Otherwise: render all layers
4. **Per-layer render**: For each texture layer:
   - Set texture on Tex0
   - Set alpha mask on Tex1 (if not using pixel shader)
   - Configure texgen matrices (dmtx for detail, amtx for alpha — both camera-relative)
   - If layer has `props & 0x100`: has alpha mask → blend mode 2
   - If layer has `props & 0x40`: custom texture animation offset (scroll UVs)
   - If layer has `props & 0x80` (sign bit): disable lighting for this layer
   - Render full 145-vertex triangle list via `GxBufRender`
5. **Shadow overlay**: After all layers, if `shadowGxTexture != null && enables & 0x40`:
   - Set `MatDiffuse = CWorld::shadowColor`
   - Set blend mode 2
   - Set `shadowModGxTex` on Tex0, `shadowGxTexture` on Tex1
   - Render the same triangle list again as shadow overlay
6. **Pixel shader paths**:
   - `psTerrain` (if `CMap::enableTerrainShader`): single-pass multi-layer via `psTerrain_LayerMask` uniform
   - `psSpecTerrain` (if `CMap::enableSpecularTerrain`): adds specular highlight, `TERRAIN_SPEC_EXP` exponent
7. **Dynamic path** (`RenderLayersDyn`): Same logic but uses `gxBufDyn` instead of per-chunk static VBO — used when terrain is being edited/modified at runtime

### Terrain LOD (`CreateAreaLowDetailVertices` at 0x0069f440)

- `CMapAreaLow`: 17x17 vertex grid (0x121 = 289 vertices)
- Fog-colored vertices from `DayNightGetInfo()->fogInfo.color`
- Used for far-distance terrain where textures aren't visible
- Index buffer: 16x16 quads, each 4 triangles from 3 vertices (standard grid triangulation)

### WMO Rendering (from spec 030 architecture doc)

- Full dispatch via `DAT_00ec1b98` (interior) / `DAT_00ec1ca0` (exterior)
- 11 render pass functions with per-batch MOMT flag handling
- Interior fog from `DayNightGetInfo()->intFog` + `intFogInfo`
- Lightmap: 256x256 DXT/RGB565, on-demand streaming, 30-second flush
- Liquid dispatch: water (int/ext) vs magma

### Water/Liquid Rendering

**Terrain chunk water** (`CWorldScene::RenderWater` at 0x0066e560):
- Separate render pass after terrain layers
- Uses `riverDiffTexid` (river texture) on Tex0
- Tex1: texgen scrolling (0.14 scale + camera translation) for animated flow
- If `CMap::enableSpecularWater`: pixel shader `psOcean0` + specular
- Otherwise: TexBlend1 = 3 (modulate)
- Per-chunk liquid: `CChunkLiquid::Render` with distance-based lighting via `SelectLight`

**WMO group water** (`RenderLiquid_0` at 0x0069e4b0):
- Interior water: vertex color from material `diffColor`, interior fog
- Exterior water: day/night lighting color (`WaterArray[3]`), normal = (0,0,1), no interior fog
- Magma: separate render path for types 2/3/6/7

**Liquid texture animation** (`GetLiquidTexture` at 0x006736b0):
- 30 texture frames per liquid type (`liquidTex[type][0..29]`)
- Frame index = `(curTimeSec % secsPerLoop) * 30.0 / secsPerLoop`
- Each liquid type has its own `secsPerLoop` cycle time
- Texture filter: `LinearMipNearest`, or `Anisotropic` if `enables < 0` (signed), or `LinearMipLinear` if `enables & 0x800000`

### Lighting System

**Per-chunk light selection** (`CMapChunk::SelectLights` at 0x006a3af0):
- Light 0: Always `CMap::sunLight` (directional, positioned at camera)
- Lights 1-7: Up to 7 local lights from `lightLinkList`
- Disable unused light slots (1-7) explicitly

**Light for WMOs** (`CMap::SelectLight` at 0x00664c80):
- If object within fog distance: call virtual `SelectLight` on the map object
- Otherwise: skip (too far to matter)

**Interior lighting**:
- Interior WMO groups: `GxRsSet(GxRs_Lighting, 0)` — dynamic lighting OFF, MOCV vertex color provides all lighting
- Exterior WMO groups: `GxRsSet(GxRs_Lighting, 1)` — dynamic lighting ON, sun + local lights
- Per-batch override: MOMT flag bit0 can disable lighting per-batch

### `CWorld::enables` Bitfield

| Bit | Name | Effect |
|-----|------|--------|
| 0x02 | Terrain | Toggle terrain rendering on/off |
| 0x20 | TerrainCull | Toggle terrain chunk frustum culling |
| 0x40 | Shadow | Toggle terrain shadow overlay |
| 0x100 | ShowPortals | Render portal wireframes |
| 0x200 | PortalVis | Portal walk visibility debug |
| 0x400 | MapObjLightMode | Toggle lightmaps vs vertex color for WMO |
| 0x800 | MapObjTextures | Toggle WMO textures |
| 0x1000 | ShowPortals | Render portal geometry |
| 0x10000 | DebugBSP | Render WMO BSP polygons |
| 0x20000 | CrappyBatches | Highlight low-quality WMO batches |
| 0x40000 | DebugZones | Zone coloring overlay |
| 0x4000000 | LowDetail | Force low-detail terrain (17x17) |
| 0x800000 | Trilinear | Terrain trilinear filtering |
| 0x1000000 | Water | Toggle water rendering |
| 0x2000000 | Doodads | Toggle doodad rendering |
| 0x8000000 | DetailDoodads | Toggle detail doodads |
| 0x20000000 | ShowTris | Wireframe overlay |
| 0x40000000 | ShowNormals | Vertex normal visualization |

## Scope

### In Scope

- Implementing native-accurate terrain rendering pipeline in `WowViewer.Core.Runtime`
- Implementing native-accurate WMO rendering pipeline in `WowViewer.Core.Runtime`
- Implementing native-accurate water/liquid rendering
- Implementing distance-based LOD for terrain
- Implementing per-chunk light selection (sun + local lights)
- Implementing shadow overlay pass for terrain
- Implementing animated liquid textures
- Implementing interior fog system
- Implementing the `CWorld::enables` debug toggle system for viewer debugging
- Specular terrain and water pixel shader paths

### Out of Scope

- Vulkan/WebGL backend selection (this spec is renderer-agnostic — the same logic applies regardless of backend)
- Terrain data structure changes (spec 031)
- WMO render pass architecture documentation (spec 030 — this spec is the implementation)
- M2 model rendering
- Editor tooling (no evidence of editor capability in the binary)
- Minimap BLP harvesting (spec 029)

## User Scenarios & Testing

### User Story 1 — Terrain renders with correct 145-vertex topology and LOD (Priority: P1)

A viewer user sees terrain rendered with the correct mesh topology (inner vertices, per-cell diagonal splits), with texture layers fading at distance and low-detail far terrain, matching the native client's visual output.

**Why this priority**: Terrain is the most visible surface in the viewer. Wrong topology + no LOD = the "horrible CPU-bound renderer" problem. This is the single biggest visual win.

**Independent Test**: Load a terrain tile and compare against native client screenshot. Verify: (a) mid-cell detail from inner vertices, (b) texture layer fade at distance, (c) low-detail far terrain visible beyond fog distance.

**Acceptance Scenarios**:

1. **Given** a terrain chunk at close range, **When** rendered, **Then** all texture layers are visible with correct alpha blending and the 145-vertex mesh produces mid-cell height detail.
2. **Given** a terrain chunk at `textureLodDist + 128` distance, **When** rendered, **Then** extra texture layers are alpha-fading (not hard-cut).
3. **Given** a terrain chunk at `textureLodDist + 300` distance, **When** rendered, **Then** only 1 base texture layer is rendered.
4. **Given** a terrain chunk beyond fog distance, **When** rendered, **Then** the low-detail 17x17 fog-colored mesh is used instead.
5. **Given** terrain with shadow data, **When** `enables & 0x40` is set, **Then** the shadow overlay is blended on top of all texture layers with the configured `shadowColor`.

---

### User Story 2 — WMO groups render with correct pass selection and per-batch flags (Priority: P1)

A viewer user sees WMO groups rendered with the correct interior/exterior pass, per-batch lighting/fog/culling/emissive/window-lit flags, lightmap UV handling, and interior fog, matching the native client.

**Why this priority**: WMO interiors are the second most visible surface. Wrong pass selection = broken lighting in dungeons.

**Independent Test**: Load a dungeon WMO (e.g., Deadmines) and compare interior lighting, fog, and window brightness against native client screenshots.

**Acceptance Scenarios**:

1. **Given** an interior WMO group (`flags & 0x48 == 0`), **When** rendered, **Then** dynamic lighting is OFF and MOCV vertex color provides illumination.
2. **Given** an exterior WMO group (`flags & 0x48 != 0`), **When** rendered, **Then** dynamic lighting is ON with sun + local lights.
3. **Given** a WMO batch with MOMT flag `bit0=0`, **When** rendered, **Then** lighting is disabled for that batch only.
4. **Given** a WMO batch with MOMT flag `bit0x10` (emissive), **When** rendered, **Then** the batch appears self-illuminated regardless of scene lighting.
5. **Given** a WMO batch with MOMT flag `bit0x20` (window-lit) in an interior group, **When** rendered, **Then** the batch receives exterior sun lighting instead of interior MOCV color.
6. **Given** a WMO group with lightmap data in interior mode, **When** rendered, **Then** `LightmapTex_Int` path is used (lighting off, lightmap on tex1).
7. **Given** a WMO group with lightmap data in exterior mode, **When** rendered, **Then** `LightmapTex_Ext` path is used (lighting on, no lightmap on tex1).
8. **Given** the camera inside a WMO with `intFog != 0`, **When** rendered, **Then** interior fog is applied with correct start/end/color from `DayNightGetInfo`.

---

### User Story 3 — Water/liquid renders with animation and type dispatch (Priority: P2)

A viewer user sees water surfaces with animated scrolling textures, specular highlights, interior vs exterior fog behavior, and correct magma rendering, matching the native client.

**Why this priority**: Water is visually important but secondary to terrain and WMO surfaces. P2 because it depends on the terrain LOD (P1) being in place for distance-based water visibility.

**Independent Test**: View water near a coast (exterior) and inside a dungeon (interior). Verify animation, fog difference, and that magma pools render with the correct path.

**Acceptance Scenarios**:

1. **Given** terrain chunk water at close range, **When** rendered, **Then** the liquid texture animates (30-frame cycle, type-specific `secsPerLoop`).
2. **Given** exterior water, **When** rendered, **Then** day/night lighting color applies to water surface (`WaterArray[3]`).
3. **Given** interior WMO water, **When** the camera is inside the WMO and `intFog != 0`, **Then** interior fog applies to the water surface.
4. **Given** magma liquid (type 2/3/6/7), **When** rendered, **Then** the magma render path is used instead of the water path.
5. **Given** `enableSpecularWater` is true, **When** water is rendered, **Then** the `psOcean0` pixel shader applies specular highlights.
6. **Given** terrain water with river texture, **When** rendered, **Then** the river texture scrolls via texgen on Tex1 (0.14 scale + camera offset).

---

### User Story 4 — Debug toggle system enables visual debugging (Priority: P3)

A developer can toggle any `CWorld::enables` bit at runtime to visualize normals, portals, BSP polygons, wireframes, culling, shadow overlays, and zone boundaries, matching the native client's debug console capabilities.

**Why this priority**: Debug toggles are essential for verifying rendering correctness during development. P3 because the correct rendering (P1/P2) must exist first before toggles are meaningful.

**Independent Test**: Toggle `shownormals`, `showtris`, `showportals`, `showcull`, `showshadow` and verify each produces the expected debug overlay.

**Acceptance Scenarios**:

1. **Given** the `enables` system, **When** `0x40000000` (ShowNormals) is set, **Then** terrain vertex normals are rendered as lines from each vertex.
2. **Given** the `enables` system, **When** `0x20000000` (ShowTris) is set, **Then** a wireframe overlay is rendered over terrain and WMO geometry.
3. **Given** the `enables` system, **When** `0x100` (ShowPortals) is set, **Then** WMO portal polygons are rendered as semi-transparent colored quads.
4. **Given** the `enables` system, **When** `0x20` (ShowCull) is set, **Then** culled chunks are visually distinguished from visible chunks.

---

### User Story 5 — Per-chunk lighting with local lights (Priority: P3)

A viewer user sees terrain and WMO groups lit by local point lights (torches, lamps, spell effects) in addition to the sun, with up to 8 lights per chunk matching the native client's light selection.

**Why this priority**: Local lighting is important for atmosphere but secondary to getting the base rendering correct. P3 because the base lighting (sun-only) from P1/P2 is sufficient for most visual validation.

**Independent Test**: View a WMO interior with torches. Verify that nearby surfaces are lit by the torch point lights in addition to ambient/sun.

**Acceptance Scenarios**:

1. **Given** a terrain chunk near a local light source, **When** `SelectLights` runs, **Then** the light is included (up to 7 local lights beyond the sun).
2. **Given** a WMO group near multiple local lights, **When** rendered, **Then** the surface shows combined illumination from sun + active local lights.
3. **Given** more than 7 local lights near a chunk, **When** `SelectLights` runs, **Then** only the 7 most relevant are selected (distance-priority).

---

### Edge Cases

- Chunks with 0 texture layers should render as `RenderLayersColor` (flat color, no textures)
- WMO groups with `flags & 0x88` should not be rendered at all
- WMO groups with `flags & 0x10000` should use `RenderAlways` (always visible)
- Interior fog color may differ from exterior fog color
- The `window-lit` flag (0x20) is only meaningful for interior groups — exterior groups ignore it
- Liquid tiles may have mixed types within a single group — the client scans for the first non-0xF type
- Shadow overlay color (`CWorld::shadowColor`) is configurable and may be non-gray
- Animated liquid textures may not exist for all liquid types — fallback to first frame

## Requirements

### Functional Requirements

**Terrain Rendering:**
- **FR-001**: The terrain renderer MUST use the 145-vertex mesh topology (9x9 outer + 8x8 inner) with per-cell diagonal splits.
- **FR-002**: The terrain renderer MUST implement texture LOD: render all layers at close range, alpha-fade extra layers at `textureLodDist`, hard-cut to 1 layer at `textureLodDist + 256`.
- **FR-003**: The terrain renderer MUST use low-detail 17x17 fog-colored mesh for far terrain beyond fog distance.
- **FR-004**: The terrain renderer MUST blend the shadow overlay texture (`shadowGxTexture`) when shadow mode is enabled, using `CWorld::shadowColor` as the blend color.
- **FR-005**: The terrain renderer MUST support specular terrain pixel shader path (`psSpecTerrain`) with configurable specular exponent.
- **FR-006**: The terrain renderer MUST support the terrain pixel shader path (`psTerrain`) for single-pass multi-layer rendering when enabled.
- **FR-007**: The terrain renderer MUST handle per-layer properties: `props & 0x40` (animated UV offset), `props & 0x80` (disable lighting), `props & 0x100` (has alpha mask).

**WMO Rendering:**
- **FR-008**: The WMO renderer MUST dispatch interior vs exterior render path based on `group.flags & 0x48`.
- **FR-009**: The WMO renderer MUST test per-batch MOMT flags: bit0 (lighting), bit1 (fog), bit2 (culling), bit0x10 (emissive), bit0x20 (window-lit).
- **FR-010**: The WMO renderer MUST select the correct lightmap pass: interior (lighting off, lightmap on tex1) vs exterior (lighting on, no lightmap on tex1).
- **FR-011**: The WMO renderer MUST apply interior fog when `intFog != 0` and the WMO is the camera's current map object.
- **FR-012**: The WMO renderer MUST skip groups with `flags & 0x88`.
- **FR-013**: The WMO renderer MUST handle `flags & 0x10000` groups via always-render path.

**Liquid Rendering:**
- **FR-014**: The liquid renderer MUST dispatch based on liquid type: water (types 0/4/8) vs magma (types 2/3/6/7).
- **FR-015**: Interior WMO water MUST use material `diffColor` and interior fog; exterior water MUST use `DayNightGetInfo()->light.WaterArray[3]` and no interior fog.
- **FR-016**: The liquid renderer MUST animate water textures using 30-frame cycling with type-specific `secsPerLoop`.
- **FR-017**: Terrain chunk water MUST use river texture with texgen scrolling (0.14 scale + camera offset on Tex1).
- **FR-018**: The liquid renderer MUST support specular water pixel shader (`psOcean0`) when `enableSpecularWater` is true.

**Lighting:**
- **FR-019**: The terrain renderer MUST select up to 8 lights per chunk: sun (light 0) + up to 7 local lights.
- **FR-020**: WMO exterior groups MUST use dynamic lighting; interior groups MUST use MOCV vertex color with lighting disabled.

**Debug Toggles:**
- **FR-021**: The renderer MUST implement the `CWorld::enables` bitfield with toggle support for at least: terrain visibility, shadow, normals, tris, portals, culling, water, low-detail, WMO textures, WMO lightmaps.
- **FR-022**: All code MUST live under `wow-viewer/`.

### Key Entities

- **Texture LOD**: Distance-based texture layer reduction — close range = all layers, medium range = alpha-fade extra layers, far range = 1 layer only.
- **Shadow Overlay**: Post-layer blend pass that applies a shadow texture on top of terrain, tinted by `CWorld::shadowColor`.
- **Interior Fog**: Fog applied only when camera is inside a WMO and `intFog != 0`, with separate start/end/color from exterior fog.
- **Window-Lit**: MOMT flag bit0x20 — interior window polygons receive exterior sun lighting instead of interior vertex color.
- **Liquid Animation Cycle**: 30-frame texture cycling with per-type `secsPerLoop` timing, producing smooth water animation.
- **Low-Detail Terrain**: 17x17 fog-colored vertex grid used for far-distance terrain where textures are not visible.
- **Debug Enables**: A 32-bit bitfield controlling rendering debug overlays (normals, wireframes, portals, BSP, shadows, culling).

## Success Criteria

- **SC-001**: A terrain tile rendered at close range shows mid-cell height detail from inner vertices, matching native client output.
- **SC-002**: A terrain tile rendered at medium distance shows alpha-fading texture layers.
- **SC-003**: Far terrain uses the 17x17 low-detail mesh with fog-colored vertices.
- **SC-004**: A dungeon WMO interior shows correct MOCV lighting (no dynamic lights) with interior fog.
- **SC-005**: A WMO exterior group shows correct dynamic lighting with sun.
- **SC-006**: Window polygons in an interior WMO group show exterior sun lighting (window-lit flag).
- **SC-007**: Water surfaces animate smoothly with type-specific frame cycling.
- **SC-008**: Magma liquid renders with the correct render path (not water).
- **SC-009**: The shadow overlay is visible when toggled on and blends with the configured color.
- **SC-010**: Debug toggles for normals, wireframes, and portals produce visible debug overlays.

## Assumptions

- The rendering backend is Silk.NET.OpenGL (per constitution), but the rendering logic is backend-agnostic — the same state setup, pass selection, and LOD decisions apply regardless of API.
- The `CWorld::enables` system is runtime-toggled, not compile-time. The viewer must expose it as a debug panel or console.
- The `DayNightGetInfo` singleton provides time-of-day lighting state. The viewer must implement an equivalent that can be set to any time of day.
- The terrain and WMO rendering improvements depend on the data structure work from specs 031 and 030 respectively — this spec assumes those data structures are available.
- The low-detail terrain mesh can be generated from the existing 9x9 outer vertex grid (subsample of the 145-vertex layout) rather than requiring a separate 17x17 grid per ADT.
- The native client's `GxBuf` system is a GPU vertex/index buffer abstraction. The wow-viewer equivalent uses Silk.NET buffer objects.

## Relationship to Other Specs

- **Implements rendering for**: `030-wmo-render-pass-architecture` — this spec implements the WMO render passes documented there.
- **Implements rendering for**: `031-terrain-cell-awareness` — this spec implements the 145-vertex mesh rendering documented there.
- **Extends**: `wow-engine-modernization-plan-2026-05-14.md` — renderer parity is a core engine milestone.
- **Depends on**: `030` and `031` data structures being available before rendering implementation begins.
- **Informs**: `020-renderer-culling-and-tile-capture` — correct rendering + LOD + culling enables efficient tile capture.
