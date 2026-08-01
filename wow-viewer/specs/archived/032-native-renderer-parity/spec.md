# Feature Specification: Native Renderer Parity

**Feature Branch**: `032-native-renderer-parity`

**Created**: 2026-05-30

**Status**: Research complete — consumed by spec 056

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

### Sky Rendering (`DayNightRenderSky` at 0x006bd8e0)

The native client renders the sky as a layered composition:

1. **Clear color**: `GxSceneSetClearColor(DAT_010b2460)` — the fog/horizon color from `DayNightGetInfo`
2. **Sky dome** (`DNSky::Render` at DAT_010b223c): Procedural sky hemisphere with 6-band gradient from `CurrentLight.SkyArray[0..5]`:
   - Band 0: Zenith (top of sky)
   - Band 1: Upper sky
   - Band 2: Middle sky
   - Band 3: Lower sky / horizon boundary
   - Band 4: Horizon band
   - Band 5: Below-horizon / fog blend
3. **Stars** (`DNStars::Render`): Star field rendered at night, faded by sun height
4. **Planets** (`DNPlanet::Render` x2): Sun and moon celestial bodies — positioned by `CWorld::CalculateSunPosition` (0x0066a5d0)
5. **Clouds** (`DNClouds::Render` at DAT_010b2170): Cloud layer with procedural noise (`m_nOctaves=4`), color from `CurrentLight.CloudArray[0..4]`, fog blend from `CurrentLight.CloudData[1]`

**Sky rendering order**: Clear → Sky dome → Stars → Planets → Clouds

### Lighting Data Model (`CurrentLight` from `CalcLightColors` at 0x006c4da0)

The native client's complete lighting state per-frame is a `CurrentLight` struct with:

| Field | Track ID | Type | Purpose |
|-------|----------|------|---------|
| DirectColor | 0 | CImVector | Sun/directional light color |
| AmbientColor | 1 | CImVector | Ambient light color |
| SkyArray[0] | 2 | CImVector | Sky zenith color |
| SkyArray[1] | 3 | CImVector | Sky upper band |
| SkyArray[2] | 4 | CImVector | Sky middle band |
| SkyArray[3] | 5 | CImVector | Sky lower band |
| SkyArray[4] | 6 | CImVector | Sky horizon band |
| SkyArray[5] | 7 | CImVector | Sky below-horizon / fog blend |
| ShadowOpacity | 8 | CImVector | Terrain shadow blend factor |
| CloudArray[0] | 9 | CImVector | Cloud top color |
| CloudArray[1] | 10 | CImVector | Cloud mid color |
| CloudArray[2] | 0xB | CImVector | Cloud bottom color |
| CloudArray[3] | 0xC | CImVector | Cloud highlight color |
| CloudArray[4] | 0xD | CImVector | Cloud ambient color |
| WaterArray[0] | 0xE | CImVector | Shallow water color |
| WaterArray[1] | 0xF | CImVector | Deep water color |
| WaterArray[2] | 0x10 | CImVector | Water fog color |
| WaterArray[3] | 0x11 | CImVector | Water surface tint |
| FogEnd | 0x12 | float | Exterior fog end distance |
| FogStartScalar | 0x13 | float | Fog start multiplier |
| CloudData[1] | 0x15 | float | Cloud density / fog blend |
| Darkness | — | float | `LightDataItem.m_highlightSky` (night darkness factor) |

Each track is time-keyframed: `CalcIndividualLightColor` interpolates between adjacent markers at the current `gameTime` (0-2880, wraps at midnight). Color tracks (0-0x11) interpolate per-channel (R/G/B/A byte interpolation). Float tracks (0x12+) interpolate linearly.

**Storm blending**: If a second `LightDataItem` (storm params) exists with `param_5 > 0`, all channels are blended: `result = clear * (100 - stormWeight) / 100 + storm * stormWeight / 100`. Fog end uses a `* 0.01` scale factor in the blend.

### Light Data Loading (`LoadLightsAndFog` at 0x006c4110)

Alpha 0.5.3 loads lighting from `.lit` files (MPQ), NOT from DBC:

- Version: `0x80000004` (0.5.3) or `0x80000005` (0.5.5+)
- Each `LightData` struct is `0x560` bytes containing:
  - `0x40` bytes: header (position, radius, flags, etc.)
  - `0x148` bytes: clear params group (`ReadSingleLightGroup` at offset 0x40)
  - `0x148` bytes: storm params group (offset 0x188)
  - `0x148` bytes: clear underwater params group (offset 0x2D0)
  - `0x148` bytes: storm underwater params group (offset 0x418)
- Each group contains the 18+ color tracks and float bands documented above

The existing `LitLoader.cs` in MdxViewer (761 lines) already reads this format. It is the canonical Alpha-era lighting source and must be ported to `WowViewer.Core.IO`.

### LIT File Purpose and Spatial Application Bug

**What LIT files are for**: LIT files provide the **lowest quality / fallback** zone lighting data. They are not the primary runtime lighting source for the game client — the client uses `DayNightGetInfo` driven by the in-game time cycle with more sophisticated spatial blending. Instead, LIT files serve as:

1. **Minimap rendering fallback**: When rendering minimaps (top-down orthographic), the game needs ambient color under terrain holes where no WMO covers the gap. The global default light in the LIT file provides exactly this — the color visible in minimap holes where terrain is missing but no object covers the hole. This has been verified: the global light color matches what appears in those holes.
2. **Editor preview lighting**: WoWEdit and similar internal tools likely use LIT data for preview rendering where terrain is visible below the surface — providing something other than pure white or black under the terrain mesh. This is still true in 2026 for Blizzard's internal tools.
3. **Fallback when no higher-quality lighting is available**: The LIT data is the floor, not the ceiling. The runtime client layers additional lighting (local zones, WMO interior fog, per-chunk lights) on top of this base.

**Known bug — spatial wrapping**: The existing `LitLoader.cs` in MdxViewer **decodes LIT data correctly** (verified by matching global light color to minimap hole observations). However, the spatial application of the lights wraps the game world incorrectly — it applies lights horizontally (`><` wrapping around the map edges) instead of treating the light positions as vertical/elevation points. This means local lights near the camera may pick up the wrong zone or blend incorrectly in the current MdxViewer implementation. The data decode is correct; the spatial query is wrong.

**Implication for wow-viewer**:
- The LIT reader port must preserve the correct data decode.
- The spatial light selection logic must be **rewritten** — not simply copied from MdxViewer's `LitLoader.EvaluateLighting`. The MdxViewer spatial selection uses a horizontal wrapping that is incorrect; the wow-viewer implementation must use the correct Ghidra-verified spatial model from `CalcLightColors` + `LoadLightsAndFog`: lights have world-space positions with radius/dropoff, the default light has `ChunkX == -1 && ChunkY == -1`, and local lights blend based on distance to their center with falloff — no horizontal wrapping.
- For minimap rendering specifically, only the **global default light** (ChunkX==-1, ChunkY==-1) matters — local zones are not relevant for top-down orthographic minimap captures. This is the validated use case for training data generation.

### Fog System (`QueryCameraFog` at 0x00689bf0, `ComputeFogBlend` at 0x00689b40)

The native client has a multi-fog system:

1. **Exterior fog**: From `DayNightGetInfo()->fogInfo` — linear fog with `FogEnd` and `FogStartScalar`. Blend factor: `1.0 - (dist - fogStart) / (fogEnd - fogStart)` when `dist >= fogStart`, clamped to 1.0.
2. **Interior fog**: From `DayNightGetInfo()->intFogInfo` — applied when camera is inside a WMO (`camMapObj != null`) and `intFog != 0`. WMO groups can also define their own fog per-group via `SMOFog` entries.
3. **WMO area fog** (`QueryMapObjFog` at 0x006896d0): Up to 4 fog zones per WMO group (`fogIds[0..3]`), each referencing an `SMOFog` entry with center, radius, start/end distances. The client finds the closest fog zone to the camera and blends between overlapping zones using a priority queue.
4. **Height-adjusted fog**: `QueryCameraFog` transforms camera position into WMO-local space, computes distance to fog centers, and blends fog parameters. Not a simple world-space linear fog.

**Fog types**: `SMOFog` has an `enable` flag (bit0) and can be disabled. The blend is always linear in 0.5.3 (no exponential fog in this build — exp/exp2 fog was added later).

### Current State — What Exists vs. What's Missing

| Capability | MdxViewer (reference) | wow-viewer (target) | Gap |
|---|---|---|---|
| Procedural sky dome | `SkyDomeRenderer.cs` (time-of-day colored) | `SkyRenderer.cs` (hardcoded gradient) | **Critical**: No time-of-day, no 6-band sky gradient |
| Skybox backdrop (M2 model) | `WorldScene.cs` skybox classification + render | `WorldSkyboxBackdropClassifier.cs` (path only) | **Moderate**: Runtime bridge exists, no render |
| Sun/moon positioning | Missing | Missing | **Moderate**: No celestial body rendering |
| Cloud rendering | Missing | Missing | **Low priority**: Cloud color data exists in LIT, no mesh |
| Time-of-day slider | `ViewerApp_Sidebars` | Missing | **Critical**: No way to set time of day |
| Alpha .LIT file reader | `LitLoader.cs` (761 lines, **data decode correct, spatial app wrong**) | Missing | **Critical**: No lighting data source for Alpha clients. Must port data decode but **rewrite spatial selection** (MdxViewer wraps horizontally, should use distance-based falloff from light centers) |
| DBC Light.dbc reader | `LightService.cs` (partial) | Missing | **Moderate**: Needed for LK clients |
| LightIntBand/FloatBand DBC | Missing | Missing | **Low**: Only needed for 1.12+ DBC-based lighting |
| LightSkybox.dbc | Missing | Missing | **Low**: Cube map skybox paths |
| 6-band sky gradient | Missing (2-band only) | Missing (hardcoded) | **Critical**: Wrong sky appearance |
| Water color from lighting | Missing | Missing | **Moderate**: WaterArray[0..3] unused |
| Shadow opacity from lighting | Missing | Missing | **Moderate**: ShadowOpacity track unused |
| Exterior fog (linear) | Exists (all renderers) | Exists (hardcoded color) | **Moderate**: No LIT-driven fog |
| Interior fog | Missing | Missing | **Covered by US2** |
| WMO area fog blending | Missing | Missing | **Moderate**: Multi-zone fog per WMO |
| Storm/clear param blending | Missing | Missing | **Low**: Weather system not in scope |

**Why this matters for training data**: Every rendered frame used for ML training depends on correct lighting. Hardcoded sky gradients and missing fog produce frames that don't match the native client's appearance. If the viewer renders a terrain tile at "noon" with a blue sky but the LIT data says the fog should be orange and the water should be dark, the synthesized minimap or terrain capture will be **fundamentally wrong** — the ML model will learn incorrect correlations between terrain features and lighting, producing garbage output. The lighting system is not cosmetic; it is a data-integrity requirement. Additionally, the LIT files themselves serve a validated purpose: the global default light provides the exact color visible under terrain holes in minimaps. This is the floor, not the ceiling — but it's a verified floor we must get right before building higher-quality lighting on top.

### MDX Object Lighting (`MdxReadLights` at 0x0044a6a0, `AnimObjectCreateLight` at 0x0074d800)

MDX models (the Alpha-era format, precursor to M2) carry LITE chunks defining per-model animated lights:

1. **LITE chunk**: `MdxReadLights` seeks the `LITE` FourCC in the MDX binary, reads `numLights` entries, creates a `CGxLight` per entry via `CreateGxLight`.
2. **Per-light data** (from `CreateGxLight` at 0x0044a8b0): Each light has:
   - Position/transform (offsets `0x110-0x118`: XYZ position)
   - Ambient color (offsets `0x158-0x160`: RGB ambient, float → byte packed)
   - Ambient intensity (offset `0x160`)
   - Directional color (offsets `0x15c-0x164`: RGB directional, float → byte packed)
   - Directional intensity (implicit from `m_dirIntensity`)
   - Direction vector
   - Attenuation: `attenStart`, `attenEnd`, `attenDenom` (from `CMapLight` struct)
   - Type: directional (bit0=0) or omni/point (bit0=1), enabled (bit1)
   - Dynamic flag: `CMapLight.dynamic` — whether the light can change at runtime
3. **Animated lights** (`CAnimLightObj`): MDX animation data can animate light properties over time (color, intensity, position, attenuation). `AnimObjectCreateLight` allocates animated light objects per model instance.
4. **Light selection** (`CMap::SelectLight` at 0x00686a10): If `obj->camDist < CWorld::farFog`, call the object's virtual `SelectLight` method. Objects beyond fog distance are skipped — no GPU light setup needed.
5. **Lua API** (`CSimpleModel_SetLight`): The UI scripting layer exposes `SetLight(enabled, omni, dirX, dirY, dirZ, ambIntensity, ambColorR/G/B/A, dirIntensity, dirColorR/G/B/A)` for model preview lights.

**What exists in the codebase**: The MDX readers in MdxViewer parse LITE chunks and store light entries, but the renderer does NOT apply them as GPU point/directional lights. The `M2Renderer` and `ModelRenderer` shaders use only a single global directional + ambient light.

**What's missing**: MDX model-local lights are never submitted to the GPU. Torch flames in dungeons, lamp posts in cities, and spell glow effects all produce no lighting contribution on surrounding surfaces.

### WMO Local Lights (MOLT chunk — I/O exists, rendering missing)

WMO root files contain MOLT chunks with per-WMO point/directional light definitions. The wow-viewer I/O layer **already reads these completely**:

- `WmoLightDetailReader.cs` — full MOLT payload reader
- `WmoLightSummaryReader.cs` — statistical summary
- `WmoLightReaderCommon.cs` — supports both legacy (32-byte) and standard (48-byte) entries

Each `WmoLightDetail` contains:
- **LightType** (byte): directional vs omni
- **UsesAttenuation** (bool): whether attenStart/attenEnd are used
- **Color** (RGBA uint32): light color
- **Position** (Vector3): light position in WMO-local space
- **Intensity** (float): light brightness
- **AttenStart/AttenEnd** (float): attenuation range
- **Rotation** (Quaternion, standard entries only): light direction for directional lights
- **HeaderFlagsWord** (ushort, standard entries only): additional flags

**What's missing**: The MOLT data is read but never consumed by the renderer. WMO interiors with torches, braziers, and glowing crystals have their light data available but unapplied. This is why dungeons look flat — only MOCV vertex color provides interior illumination, and local point lights that should create pools of warm light around fixtures are completely absent.

### BLS Shader Format and Effect-Family Architecture

The native client uses a shader effect-family system:

1. **BLS files** (`.bls` in MPQ): Binary shader programs, format changes per client version (v1.1 Alpha, v1.2 Beta, v1.3 WotLK, v1.4+ Cata). Contains compressed HLSL/assembly shader code. Alpha 0.5.3 BLS files confirmed in Ghidra: `Shaders\Pixel\Ocean0.bls`, `Shaders\Pixel\UTerrain.bls`, `Shaders\Pixel\SpecUTerrain.bls`, `Shaders\Pixel\Terrain.bls`, `Shaders\Pixel\SpecTerrain.bls`.
2. **WFX files**: Effect-mapping definitions that bind named effects (e.g., `MapObjDiffuse`, `MapObjSpecular`, `Model2`) to render passes (Opaque, AlphaKey, Alpha, Add, Mod, etc.) with render state configuration.
3. **M2/MDX shader dispatch**: `M2_GetPixelShaderID()` / `M2_BuildCombinerEffectName()` compute the runtime shader from `shader_id` + `op_count`, selecting from 36 pixel shader families and 16 vertex shader families. Alpha 0.5.3 uses D3D9 fixed-function pipeline with pixel shader extensions (`CGxPixelShader`, `PixelShaderCreate @ 0x00594e90`).
4. **Shader capability gating**: `FUN_0078de60` hard-gates specular on pixel shader support: "Specular not enabled. Requires pixel shaders."

**What exists**:
- BLS format documentation (wowdev.wiki copies, comprehensive for all versions)
- WFX effect-mapping documentation
- M2.skin shader dispatch tables (16 vertex + 36 pixel shader names)
- Ghidra RE of shader loading/dispatch paths
- 14+ inline GLSL shader programs across MdxViewer and wow-viewer (all #version 330/410 core)
- BLS parity plan prompt (describes intent but is not implemented)

**What's missing**:
- BLS reader/parser — no code that reads, decompresses, or extracts BLS shader text
- BLS-to-GLSL conversion tool — no tool converts BLS binary to runnable GLSL
- WFX reader — no code parses WFX effect-mapping files
- Effect-family registry — no runtime system maps native effect names to shader programs
- Extracted/converted shader text from game BLS files
- M2/MDX shader_id dispatch — `M2GetPixelShaderID()` not implemented

**Noggit comparison**: Noggit-red (the open-source WoW map editor) uses converted BLS shaders for its rendering. The community reports that noggit's lighting is "too reddish/yellow" compared to the real renderer — likely because the shader conversion loses subtle color-space or gamma corrections that the native D3D9 pipeline applies. This is a known pitfall: even with correct shader text, the wrong uniform bindings or color-space assumptions produce visibly wrong output.

**Approach for wow-viewer**: Rather than attempting direct BLS execution (which would require a D3D9 HLSL assembler + GLSL cross-compiler), the plan is **shader-family reconstruction**: for each named effect family (e.g., `MapObjDiffuse`, `Combiners_Opaque`, `psTerrain`), write a modern GLSL shader that produces visually equivalent output using the same uniform inputs the native client would provide. This is what noggit attempts, but with the additional rigor of Ghidra-verified uniform bindings and color-space behavior.

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
- **Implementing MDX model-local lighting** — per-model LITE chunk lights (directional + omni/point) submitted as GPU lights, with attenuation and animated properties
- **Implementing WMO MOLT local lighting** — per-WMO point/directional lights from MOLT data applied to WMO group rendering
- **Implementing shader-family reconstruction** — writing modern GLSL equivalents of the native effect families (`MapObjDiffuse`, `MapObjSpecular`, `Combiners_Opaque`, `psTerrain`, `psOcean0`, etc.) with Ghidra-verified uniform bindings
- **Implementing the `CurrentLight` data model** — all 18+ color tracks + float bands
- **Porting Alpha .LIT file reader** to `WowViewer.Core.IO` (reference: MdxViewer `LitLoader.cs`)
- **Implementing 6-band sky dome rendering** from `CurrentLight.SkyArray[0..5]`
- **Implementing time-of-day lighting evaluation** — `CalcLightColors`-style track interpolation
- **Implementing exterior fog from lighting data** — `FogEnd`, `FogStartScalar`, fog color from LIT/DBC
- **Implementing water color from lighting data** — `WaterArray[0..3]`
- **Implementing shadow opacity from lighting data** — `ShadowOpacity` track
- **Implementing time-of-day control** — settable `gameTime` (0-2880) exposed to viewer UI
- **Implementing storm/clear param blending** when storm data is present

### Out of Scope

- Vulkan/WebGL backend selection (this spec is renderer-agnostic — the same logic applies regardless of backend)
- Terrain data structure changes (spec 031)
- WMO render pass architecture documentation (spec 030 — this spec is the implementation)
- M2 model rendering
- Editor tooling (no evidence of editor capability in the binary)
- Minimap BLP harvesting (spec 029)
- DBC-based lighting for LK/Beta clients (only Alpha .LIT format is in scope for initial implementation; DBC Light family is deferred)
- Cloud mesh rendering (cloud color data will be available in the lighting model, but the actual cloud mesh/texture rendering is deferred)
- Weather particle system
- Sun/moon 3D mesh rendering (position will be computed for lighting direction; rendering the celestial disc/sprite is low priority)
- Exponential fog types (0.5.3 only uses linear fog; exp/exp2 fog is a later addition)
- Direct BLS execution or BLS-to-GLSL auto-conversion (shader-family reconstruction is the approach, not literal BLS binary execution)
- WFX reader (effect-mapping format; deferred — the effect names and pass types are known from documentation)

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

### User Story 6 — Lighting data drives all rendered output (Priority: P1)

A viewer user sets the time of day and sees correct lighting across all surfaces: the sky dome gradient changes, fog color and distance change, water tint changes, shadow intensity changes, and terrain/WMO lighting matches the native client at that time of day. This is a **data-integrity requirement** for ML training — wrong lighting produces wrong training data.

**Why this priority**: P1 because every other rendering feature depends on correct lighting inputs. Without `CurrentLight` producing correct colors/fog/water values, terrain renders with wrong fog, water has wrong tint, sky has wrong gradient, and any training data captured from the viewer is fundamentally bunk. This is the foundation that all visual output rests on.

**Independent Test**: Set time to midnight vs. noon. Verify: (a) sky gradient changes (dark/blue bands vs. bright bands), (b) fog color changes, (c) water tint changes, (d) shadow opacity changes, (e) WMO interior lighting unaffected (MOCV-only).

**Acceptance Scenarios**:

1. **Given** a `.lit` file loaded for the current map, **When** `gameTime` is set to 1440 (noon), **Then** `CurrentLight.DirectColor` is bright, `CurrentLight.SkyArray[0]` is blue zenith, `CurrentLight.FogEnd` is long (distant fog), `CurrentLight.WaterArray[3]` is bright water tint.
2. **Given** the same `.lit` file, **When** `gameTime` is set to 0 (midnight), **Then** `CurrentLight.DirectColor` is dark, `CurrentLight.SkyArray[0]` is dark zenith, `CurrentLight.FogEnd` is short (close fog), `CurrentLight.WaterArray[3]` is dark water tint.
3. **Given** `.lit` data with storm params and `stormWeight > 0`, **When** evaluated, **Then** clear and storm parameters are blended: `result = clear * (100 - weight) / 100 + storm * weight / 100`.
4. **Given** the lighting system, **When** `gameTime` is set via the viewer UI, **Then** all downstream consumers (sky dome, fog, water, shadows) receive updated values on the next frame.
5. **Given** a default light entry (ChunkX==-1, ChunkY==-1) and a local light entry near the camera, **When** evaluated, **Then** the local light blends with the default using spatial falloff weight.

---

### User Story 7 — Sky dome renders with 6-band gradient from lighting data (Priority: P1)

A viewer user sees a sky dome with correct 6-band gradient matching the native client's `DNSky::Render`, with colors driven by `CurrentLight.SkyArray[0..5]` from the LIT data at the current time of day.

**Why this priority**: P1 because the sky is the most visible surface after terrain. A wrong sky = every screenshot is visually wrong. The current hardcoded 2-band gradient in `SkyRenderer.cs` produces an obviously incorrect sky that contaminates all captured imagery.

**Independent Test**: Set time to dawn. Verify the sky dome shows warm horizon bands (SkyArray[3-5]) transitioning to cool zenith (SkyArray[0]), matching the native client's dawn sky.

**Acceptance Scenarios**:

1. **Given** `CurrentLight.SkyArray[0..5]` from lighting evaluation, **When** the sky dome renders, **Then** the hemisphere shows a 6-band gradient from zenith (band 0) to below-horizon (band 5).
2. **Given** `gameTime` at dawn, **When** the sky dome renders, **Then** horizon bands (3-5) show warm orange/pink tones while zenith stays dark blue.
3. **Given** `gameTime` at noon, **When** the sky dome renders, **Then** the entire dome is bright blue with pale horizon.
4. **Given** `gameTime` at midnight, **When** the sky dome renders, **Then** the dome is dark with minimal gradient.
5. **Given** the camera underwater, **When** the sky dome renders, **Then** it uses `DAT_010b2460` clear color (fog color) instead of the sky gradient.

---

### User Story 8 — Fog drives distance attenuation and culling from lighting data (Priority: P1)

A viewer user sees terrain and objects fade into fog at the correct distance, with fog color and distance driven by `CurrentLight.FogEnd` and `CurrentLight.FogStartScalar` from the LIT data at the current time of day. Objects beyond fog end are culled. Interior fog applies when inside a WMO.

**Why this priority**: P1 because fog distance and color are the primary culling/visibility mechanism and the dominant visual cue at distance. Wrong fog = wrong object visibility = wrong training data for distant features.

**Independent Test**: Set time to noon (long fog) vs. night (short fog). Verify objects appear/disappear at different distances and fog color matches native client.

**Acceptance Scenarios**:

1. **Given** `CurrentLight.FogEnd` from lighting evaluation, **When** terrain renders, **Then** fog blend factor = `1.0 - (dist - fogStart) / (fogEnd - fogStart)` when `dist >= fogStart`, clamped to [0,1].
2. **Given** `CurrentLight.FogStartScalar`, **When** computing fog start, **Then** `fogStart = fogEnd * (1.0 - FogStartScalar)` (or equivalent scaling).
3. **Given** `CurrentLight.SkyArray[5]` (below-horizon/fog color), **When** terrain at maximum fog distance renders, **Then** the terrain color blends to this fog color.
4. **Given** the camera inside a WMO with `intFog != 0`, **When** WMO interior renders, **Then** interior fog (from `DayNightGetInfo()->intFogInfo`) applies instead of exterior fog.
5. **Given** a WMO with multiple `SMOFog` zones, **When** the camera is near a fog zone boundary, **Then** the two fog zones blend based on distance to their centers.

---

### User Story 9 — MDX model-local lights illuminate nearby surfaces (Priority: P2)

A viewer user sees MDX models (torches, lamps, spell effects) casting local light onto nearby terrain, WMO surfaces, and other models, matching the native client's per-model LITE chunk lighting.

**Why this priority**: P2 because model-local lights are the dominant visual cue for atmosphere in dungeons and cities — torch pools, lamp halos, brazier glow. Without them, interiors look flat despite correct MOCV + interior fog. It's P2 not P1 because the base rendering (terrain topology, WMO dispatch, sky/fog) must work first.

**Independent Test**: View a dungeon WMO with torch MDX placements. Verify warm light pools appear on nearby WMO walls and floor around each torch, matching native client.

**Acceptance Scenarios**:

1. **Given** an MDX model with a LITE chunk containing an omni/point light, **When** rendered, **Then** the light contributes illumination to nearby surfaces (terrain, WMO, other MDX) within `attenEnd` distance.
2. **Given** an MDX light with attenuation (`UsesAttenuation`), **When** a surface is at distance `d` from the light, **Then** intensity falls off: full intensity when `d < attenStart`, linear falloff to zero at `d = attenEnd`.
3. **Given** an MDX model beyond fog distance (`camDist >= farFog`), **When** `CMap::SelectLight` evaluates it, **Then** the light is skipped (no GPU light setup).
4. **Given** an MDX light with animated properties (CAnimLightObj), **When** the animation plays, **Then** light color, intensity, and position update per-frame.

---

### User Story 10 — WMO MOLT lights illuminate interior group surfaces (Priority: P2)

A viewer user sees WMO interior groups lit by their MOLT point/directional lights (torches on walls, glowing crystals, braziers), creating pools of warm light around fixtures, matching the native client.

**Why this priority**: P2 because WMO MOLT lights are the primary source of local illumination variety inside dungeons. Combined with MDX model lights (US9), they make interiors look alive rather than flat. P2 because MOLT data is already decoded — the gap is purely in the rendering pipeline consuming it.

**Independent Test**: View a dungeon WMO with MOLT entries. Verify point lights create visible illumination pools on surrounding geometry.

**Acceptance Scenarios**:

1. **Given** a WMO root with MOLT entries, **When** rendering interior groups, **Then** each MOLT light contributes as a GPU point/directional light within its attenuation range.
2. **Given** a MOLT omni light at position P with color C and intensity I, **When** a surface at distance d from P is rendered, **Then** the surface receives illumination from this light proportional to I and attenuation.
3. **Given** a MOLT directional light with rotation quaternion, **When** rendered, **Then** the light illuminates surfaces in the rotated direction, not omni-directionally.
4. **Given** MOLT lights and MDX model lights in the same scene, **When** rendered, **Then** both contribute to the total light count for per-chunk/per-group light selection (up to 7 local + sun = 8 total).

---

### User Story 11 — Modern GLSL shaders produce visually native-equivalent output (Priority: P2)

The viewer's GLSL shaders (terrain, WMO, liquid, MDX/M2, sky) produce output visually equivalent to the native client's effect-family rendering, with correct color-space handling and uniform bindings verified against Ghidra evidence.

**Why this priority**: P2 because shaders are the mechanism that makes all other rendering features visible. Without correct shaders, correct data still produces wrong pixels. P2 not P1 because the data pipeline (lighting model, mesh topology, pass selection) must be correct first — shaders can't fix wrong inputs.

**Independent Test**: Render the same scene in the viewer and the native client at the same time of day. Compare pixel output for terrain, WMO interior, and water surfaces.

**Acceptance Scenarios**:

1. **Given** the terrain shader with `CurrentLight`-driven fog and lighting, **When** terrain renders, **Then** the output matches the native client's `psTerrain` effect family at the same time of day (no "too reddish/yellow" color shift like noggit).
2. **Given** the WMO shader with interior MOCV lighting and local MOLT lights, **When** a WMO interior group renders, **Then** the output shows correct vertex color blending with point light contributions, matching `MapObjDiffuse` effect family.
3. **Given** the liquid shader with animated textures and specular, **When** water renders, **Then** the output matches `psOcean0` effect family (correct specular highlights, no blown-out highlights or missing reflections).
4. **Given** the MDX/M2 model shader, **When** a model renders, **Then** the output matches the native client's `Combiners_Opaque`/`Combiners_Mod` effect families for the model's shader_id, with correct alpha blending and specular gating.

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

**Lighting Data Model and Evaluation:**
- **FR-023**: The runtime MUST implement the `CurrentLight` data model with all 18+ color tracks (DirectColor, AmbientColor, SkyArray[0..5], ShadowOpacity, CloudArray[0..4], WaterArray[0..3]) and float tracks (FogEnd, FogStartScalar, CloudData[1]).
- **FR-024**: The runtime MUST implement time-keyframed interpolation for all tracks, matching `CalcIndividualLightColor`: linear byte interpolation for color tracks, linear float interpolation for float tracks, wrapping at midnight (0-2880 range).
- **FR-025**: The runtime MUST implement storm/clear parameter blending: `result = clear * (100 - weight) / 100 + storm * weight / 100`, with `* 0.01` scale factor for fog float tracks.
- **FR-026**: The runtime MUST implement spatial light blending — local lights blend with the default light using falloff weight based on camera distance to the light zone center.
- **FR-027**: The I/O layer MUST include an Alpha `.lit` file reader ported from MdxViewer's `LitLoader.cs`, supporting versions `0x80000003` through `0x80000005`, all 18+ color tracks, 4 param groups (Clear, Storm, ClearUnderwater, StormUnderwater), and float band data.
- **FR-028**: The runtime MUST expose a settable `gameTime` property (0-2880) that triggers re-evaluation of all lighting tracks.

**Sky Rendering:**
- **FR-029**: The sky dome MUST render a 6-band gradient hemisphere from `CurrentLight.SkyArray[0..5]` — zenith to below-horizon — matching the native client's `DNSky::Render`.
- **FR-030**: The sky dome clear color MUST be set to `CurrentLight.SkyArray[5]` (fog/horizon color) when not underwater.
- **FR-031**: The sky dome MUST be camera-following (translates with camera position, no parallax).

**Fog System:**
- **FR-032**: Exterior fog MUST use `CurrentLight.FogEnd` and `CurrentLight.FogStartScalar` from the lighting evaluation, with fog color from `CurrentLight.SkyArray[5]`.
- **FR-033**: Fog blend MUST follow the linear formula: `blendFactor = 1.0 - (dist - fogStart) / (fogEnd - fogStart)` when `dist >= fogStart`, clamped [0,1], matching `ComputeFogBlend`.
- **FR-034**: Interior fog MUST use `DayNightGetInfo()->intFogInfo` (start, end, color) when camera is inside a WMO and `intFog != 0`.
- **FR-035**: WMO area fog MUST support up to 4 `SMOFog` zones per group, with distance-based blending between overlapping zones, matching `QueryCameraFog`.
- **FR-036**: Before any terrain shader, culling, WDL, or object-visibility consumer receives fog
  values, the active range MUST be finite, positive, and have `FogStart < FogEnd`; a missing or
  degenerate lighting sample MUST fall back to a visible range.
- **FR-037**: A user-selected active fog range MUST remain distinct from the lighting recommendation;
  LIT/DBC evaluation may update colors and recommendations but MUST NOT overwrite that override.

**Water Color from Lighting:**
- **FR-036**: Exterior water surface tint MUST use `CurrentLight.WaterArray[3]`.
- **FR-037**: Water fog color MUST use `CurrentLight.WaterArray[2]`.
- **FR-038**: Shallow/deep water colors MUST use `CurrentLight.WaterArray[0]` and `WaterArray[1]` respectively.

**Shadow from Lighting:**
- **FR-039**: Terrain shadow overlay blend factor MUST use `CurrentLight.ShadowOpacity` to control shadow intensity.

**MDX Model-Local Lighting:**
- **FR-040**: The MDX renderer MUST read LITE chunk data from model definitions and create per-model `CGxLight` entries (directional or omni/point, with attenuation).
- **FR-041**: MDX model lights within fog distance MUST be submitted to the per-chunk/per-group light selection system (up to 7 local lights beyond the sun).
- **FR-042**: MDX lights beyond fog distance (`camDist >= farFog`) MUST be skipped — no GPU light setup.
- **FR-043**: Animated MDX lights (CAnimLightObj) MUST update color, intensity, and position per-frame from animation data.

**WMO Local Lighting:**
- **FR-044**: WMO MOLT lights MUST be applied as GPU point/directional lights when rendering the WMO's interior groups.
- **FR-045**: MOLT omni lights MUST use attenuation (attenStart/attenEnd) for distance falloff.
- **FR-046**: MOLT directional lights MUST use the rotation quaternion to determine light direction.
- **FR-047**: MOLT lights and MDX model lights MUST share the same per-group light budget (up to 7 local + sun = 8 total).

**Shader Family Reconstruction:**
- **FR-048**: The terrain GLSL shader MUST produce output visually matching the native `psTerrain`/`psSpecTerrain` effect families when given the same `CurrentLight` inputs — no "too reddish/yellow" color-space deviation as seen in noggit.
- **FR-049**: The WMO GLSL shader MUST produce output visually matching `MapObjDiffuse`/`MapObjSpecular` effect families for interior/exterior groups.
- **FR-050**: The liquid GLSL shader MUST produce output visually matching `psOcean0` for specular water.
- **FR-051**: The MDX/M2 GLSL shader MUST support the key combiner families (`Combiners_Opaque`, `Combiners_Mod`, `Combiners_Mod2x`) with correct alpha blending and specular gating.
- **FR-052**: All shaders MUST handle the same color-space and gamma behavior as the native D3D9 pipeline — verified by side-by-side comparison with native client at the same time of day.

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
- **CurrentLight**: The complete per-frame lighting state with 18+ color tracks (DirectColor, AmbientColor, 6 sky bands, shadow opacity, 5 cloud bands, 4 water colors) and float tracks (FogEnd, FogStartScalar, CloudData). Evaluated from LIT/DBC data at the current `gameTime`.
- **LIT File**: Alpha-era lighting format (`.lit` files in MPQ) containing time-keyframed color and float tracks for up to 4 parameter groups (Clear, Storm, ClearUnderwater, StormUnderwater) per light zone.
- **6-Band Sky Gradient**: The sky dome's hemisphere coloring using `SkyArray[0..5]`: zenith → upper → middle → lower → horizon → below-horizon. Each band is a separate color from the lighting evaluation.
- **Storm/Clear Blend**: When storm data is present, clear and storm parameters are blended by weight percentage: `result = clear * (100 - stormWeight) / 100 + storm * stormWeight / 100`.
- **WMO Area Fog**: Up to 4 `SMOFog` zones per WMO group, each with center position, radius, start/end distances. The client blends between overlapping zones based on camera distance to each fog center.
- **MDX LITE Chunk**: Per-model animated light definitions in MDX files. Each entry creates a `CGxLight` (directional or omni/point, with attenuation, color, intensity, direction, animation).
- **WMO MOLT Chunk**: Per-WMO point/directional light definitions already decoded by `WmoLightDetailReader`. Contains position, color, intensity, attenuation range, light type, rotation. Currently read but never rendered.
- **Shader-Family Reconstruction**: Writing modern GLSL equivalents of native D3D9 effect families (MapObjDiffuse, psTerrain, psOcean0, Combiners_Opaque, etc.) with Ghidra-verified uniform bindings and color-space behavior — NOT direct BLS execution or auto-conversion.
- **CGxLight**: The native client's GPU light abstraction — supports directional and omni/point types, ambient + directional color/intensity, attenuation, enable/disable, dynamic flag.

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
- **SC-011**: The `CurrentLight` data model produces correct time-of-day lighting values at noon, midnight, and dawn when evaluated against a real `.lit` file, matching values from `LitLoader` + manual inspection.
- **SC-012**: The sky dome renders a 6-band gradient that visually matches the native client at the same time of day.
- **SC-013**: Fog distance and color change when time of day changes, and objects cull at the correct fog distance.
- **SC-014**: Water surface tint changes with time of day (bright at noon, dark at night), matching `WaterArray[3]`.
- **SC-015**: Shadow overlay intensity changes with time of day, matching `ShadowOpacity`.
- **SC-016**: Storm/clear blending produces intermediate lighting when storm weight > 0.
- **SC-017**: Interior fog in WMOs overrides exterior fog when camera is inside the WMO.
- **SC-018**: A training-data capture at a specific time of day produces visually identical output to the native client at that same time — no "bunk" lighting data.
- **SC-019**: MDX models with LITE chunk lights create visible illumination on nearby terrain and WMO surfaces (torch pools, lamp halos).
- **SC-020**: WMO interiors with MOLT lights show correct local illumination pools around light fixtures.
- **SC-021**: Side-by-side comparison of viewer GLSL output vs. native client at the same time of day shows no significant color shift (no "too reddish/yellow" deviation like noggit).
- **SC-022**: Combined MDX + MOLT lights respect the 8-light budget per chunk/group (sun + up to 7 local).

## Assumptions

- The rendering backend is Silk.NET.OpenGL (per constitution), but the rendering logic is backend-agnostic — the same state setup, pass selection, and LOD decisions apply regardless of API.
- The `CWorld::enables` system is runtime-toggled, not compile-time. The viewer must expose it as a debug panel or console.
- The `DayNightGetInfo` singleton provides time-of-day lighting state. The viewer must implement an equivalent that can be set to any time of day.
- The terrain and WMO rendering improvements depend on the data structure work from specs 031 and 030 respectively — this spec assumes those data structures are available.
- The low-detail terrain mesh can be generated from the existing 9x9 outer vertex grid (subsample of the 145-vertex layout) rather than requiring a separate 17x17 grid per ADT.
- The native client's `GxBuf` system is a GPU vertex/index buffer abstraction. The wow-viewer equivalent uses Silk.NET buffer objects.

## Alpha 0.6.0 Terrain Vertex Inversion

Alpha terrain has a critical difference from LK terrain that the renderer must handle:

- **Alpha MCVT layout**: Non-interleaved — 81 outer vertices (9x9) first, then 64 inner vertices (8x8). Heights are **absolute world-space Z** (no base-Z addition).
- **LK MCVT layout**: Already interleaved — row 0 outer(9), row 0 inner(8), ..., row 8 outer(9). Heights are **deltas from MCNK Position.Z**.
- **Reinterleaving on read**: All existing Alpha readers (AlphaWdtReader, AlphaTerrainAdapter, AlphaEmbeddedAdtReader) reinterleave Alpha's non-interleaved data to LK interleaved order at read time. The renderer always receives interleaved data regardless of source format.
- **Inner/outer semantic inversion**: Alpha 0.6.0 uses the 9x9 and 8x8 grids in inverted semantic roles compared to what the LK renderer expects for diagonal splitting. The existing codebase already handles this by dropping/ignoring the incorrectly-positioned vertex values from the decoded Alpha data during reinterleaving — the renderer must NOT attempt to use the raw 145 values as-is from Alpha ADTs.
- **Implication for spec 032**: The terrain render pipeline must consume the already-reinterleaved 145-vertex array from `WorldTerrainChunkData`. It must NOT re-decode from raw MCVT bytes. The reinterleaving and Alpha-vs-LK divergence is owned by the I/O layer (AlphaWdtReader, AlphaTerrainAdapter, LkAdtReader), not the runtime.

## Relationship to Other Specs

- **Implements rendering for**: `030-wmo-render-pass-architecture` — this spec implements the WMO render passes documented there.
- **Implements rendering for**: `031-terrain-cell-awareness` — this spec implements the 145-vertex mesh rendering documented there.
- **Extends**: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (viewer-first + UE bridge; this spec drives OpenGL renderer parity within the viewer)
- **Depends on**: `030` and `031` data structures being available before rendering implementation begins.
- **Informs**: `020-renderer-culling-and-tile-capture` — correct rendering + LOD + culling enables efficient tile capture.
