# WoW 1.0.0 WMO Rendering Pipeline — Ghidra Static Trace (2026-07-15)

## 1. Overview

This document traces the **WMO (World Map Object) rendering pipeline** in WoW.exe 1.0.0 (beta-3) via Ghidra static analysis. The goal is to document everything the 1.0.0 client does when rendering WMOs — lighting, shaders, blending, portals, BSP, fog, liquids, doodads — so the wow-viewer renderer can be upgraded from a "0.1 era" brute-force renderer to a proper "1.0 era" world renderer.

All findings are from static string sweeps and xref analysis via the GhidraMCP HTTP API. Function addresses are entry points in the 1.0.0 binary.

---

## 2. Class Hierarchy

The WMO system in 1.0.0 uses this class hierarchy (recovered from RTTI strings and debug assertions):

| Class | Source file | Role |
|-------|------------|------|
| [`CMapBaseObj`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapBaseObj.cpp` | Base class for all map objects |
| [`CMapBaseObjLink`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | — | Linked-list node for map objects |
| [`CMapObj`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapObj.cpp` | WMO root object — loads .wmo file, manages groups |
| [`CMapObjDef`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapObjDef.cpp` | WMO definition (placement on map, from WDT MODF chunk) |
| [`CMapObjDefGroup`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | — | WMO definition group — frustum culling, light/entity/doodad links |
| [`CMapObjGroup`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapObjGroup.cpp` | WMO group — rendering unit, has VBOs, batches, BSP |
| `VertArray@CMapObjGroup` | — | Vertex array nested class |
| [`CMapDoodadDef`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapDoodadDef.cpp` | Doodad definition (M2 model placed inside WMO) |
| [`CMapEntity`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapEntity.cpp` | Entity (game object placed inside WMO) |
| [`CMapLight`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapLight.cpp` | Light (MOLR lights in WMO groups) |
| [`CMapCacheLight`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | — | Cached light (lighting cache for performance) |
| [`CMapChunk`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapChunk.cpp` / `MapChunkRender.cpp` | Terrain chunk |
| [`CMapArea`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `MapArea.cpp` / `CMapArea.h` | Map area |
| `CMapAreaLow` | — | Low-detail map area |
| [`CWModelFadeout`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | — | WMO model fadeout (distance-based fading) |
| [`CChunkLiquid`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | — | Terrain chunk liquid |
| [`CDetailDoodadData`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) | `DetailDoodad.cpp` | Detail doodad system |
| `CDetailDoodadGeom` | — | Detail doodad geometry |
| `CDetailDoodadInst` | — | Detail doodad instance |
| `CDetailDoodadInstAdd` | — | Detail doodad instance add |
| `CDetailDoodadVertex` | — | Detail doodad vertex |

### Object Type Flags

```
CMapBaseObj::Type_Entity
CMapBaseObj::Type_DoodadDef
CMapBaseObj::Type_MapObjDef
```

---

## 3. WMO File Format (1.0.0)

### 3.1 WMO Root File

The WMO root file is loaded by `CMapObj::Create()` which calls `mapObj->Read()`:

```
CMapObj::Create(): mapObj->Read("%s") failed
```

Source: `MapObj.cpp`, `MapObjRead.cpp`

### 3.2 WMO Group Chunks

Confirmed WMO group chunks in 1.0.0 (from `pIffChunk->token=='XXXX'` assertions):

| Chunk | Description |
|-------|-------------|
| **MOGP** | Group header |
| **MOPY** | Materials / properties |
| **MOVI** | Vertex indices |
| **MOVT** | Texture coordinates |
| **MONR** | Normals |
| **MOBA** | Batches |
| **MORB** | Render batches |
| **MOBR** | BSP references |
| **MOBN** | BSP nodes |
| **MOLR** | Light references |
| **MOCV** | Vertex colors (pre-baked lighting) |
| **MLIQ** | Liquids (WMO-internal) |

### 3.3 WDT WMO References

The WDT file references WMOs via:
- **MWMO** chunk — WMO filename string
- **MODF** chunk — WMO placement data

WDT reader functions:
- `FUN_006976f0` — references `iffChunk.token == 'MWMO'`
- `FUN_006c2d70` — references `mIffChunk->token == 'MWMO'`

### 3.4 WMO Area Table

WMOs use area table lookups for name/flags:
- `WMOAreaTable.dbc` → `WMOAreaTableRec`
- `AreaTable.dbc` → `AreaTableRec`
- `SMAreaInfo::FLAG_EXISTS` — area info existence flag
- `areaInfo[index].flags` — area info flags

---

## 4. WMO Shader System

### 4.1 WMO Pixel Shaders (.bls)

All 6 WMO pixel shaders are loaded from a single function: **`FUN_006abab0`**

| Shader file | Purpose |
|-------------|---------|
| `Shaders\Pixel\MapObjSpecular.bls` | Specular highlight on opaque WMO geometry |
| `Shaders\Pixel\MapObjTransSpecular.bls` | Specular on transparent WMO geometry |
| `Shaders\Pixel\MapObjTransDiffuse.bls` | Diffuse-only on transparent WMO geometry |
| `Shaders\Pixel\MapObjOverbright.bls` | Overbright (HDR-like) lighting on WMO geometry |
| `Shaders\Pixel\MapObjMetal.bls` | Metal effect (environment reflection?) |
| `Shaders\Pixel\MapObjExtWater0.bls` | Exterior water effect on WMO geometry near water |

### 4.2 Shader Target Backends

The CGx abstraction supports multiple shader backends:

**OpenGL:**
- `GL_NV_register_combiners` / `GL_NV_register_combiners2` — NV register combiners (primary)
- `GL_NV_texture_shader` / `GL_NV_texture_shader2` / `GL_NV_texture_shader3` — NV texture shader
- `GL_NV_vertex_program` / `GL_NV_vertex_program2` / `GL_NV_vertex_program3` — NV vertex programs
- `GL_ATI_fragment_shader` — ATI fragment shader
- `GL_ARB_vertex_program` / `GL_ARB_fragment_program` — ARB programs
- `GL_ARB_vertex_buffer_object` — VBO
- `GL_ARB_texture_env_combine` / `GL_ARB_texture_compression` / `GL_ARB_multitexture` — ARB extensions
- `!!ARBfp1.0` — ARB fragment program header

**D3D:**
- `CGxPixelShader::Target_ps_1_1` through `Target_ps_2_0`
- `CGxPixelShader::Target_nvrc` through `Target_arbfp1` (OpenGL targets)

**M2 vertex shader:**
- `shaders\vertex\Model2.bls` — M2 vertex shader

### 4.3 Shader Toggle System

Console commands and debug toggles:
- `pixelShaders` — "Use pixel shaders" toggle
- `specular` — "Specularity" toggle
- `mapObjOverbright` — "Map object overbright" toggle
- `M2UseShaders` — "skin models using vertex shaders" toggle

Status messages:
- `Pixel shaders disabled/enabled/unsupported on current API/HW.`
- `Specular disabled/enabled/unsupported on current API/HW.`
- `MapObjOverbright disabled/enabled/unsupported on current API/HW.`

---

## 5. WMO Batch System

### 5.1 Batch Types

WMO groups have two batch categories:

| Batch type | Field | Description |
|-----------|-------|-------------|
| Interior batches | `group->intBatchCount` | Opaque (interior) batches — rendered first |
| Transparent batches | `group->transBatchCount` | Transparent batches — rendered after opaque |

### 5.2 Vertex / Index Buffers

Each WMO group has its own VBOs:
- `group->vertexVB->buf` — vertex buffer (size checked via `GxBufSize`)
- `group->indexVB->buf` — index buffer (size checked via `GxBufSize`)
- `group->liquidVerts` — liquid vertex grid (x × y dimensions)

### 5.3 Batch Rendering

- `renderList.IsLinked(batch)` — batch is linked to the render list
- `b < nBatches` — batch iteration
- `division->batches.count == 1` — M2 batch check (single batch per division)
- `count <= Gx_MaxBatchCount` — max batch count limit
- `%2u batches, %5u verts, %5u prims, %s` — batch stats format

### 5.4 Debug Toggles

- `Crappy batches enabled/disabled` — low-quality batch rendering mode

---

## 6. WMO Lighting System

### 6.1 Lighting Architecture

The WMO lighting system has three layers:

1. **Pre-baked vertex colors** (MOCV chunk) — static lighting baked into vertex colors
2. **Dynamic lights** (MOLR chunk → CMapLight) — point/spot lights in WMO groups
3. **Cached lights** (CMapCacheLight) — lighting cache for performance

### 6.2 Light Classes

| Class | Role |
|-------|------|
| `CMapLight` | Dynamic light in WMO group (from MOLR) |
| `CMapCacheLight` | Cached light for performance |
| `CGxuLight` | Graphics-layer light |
| `CGxuLightLink` | Light link (connects lights to renderable objects) |

### 6.3 Light Linking

WMO definition groups link lights to renderable objects:
- `mapObjDefGroup->lightLinkList` — light link list per WMO definition group
- `cacheLight` — cached light field
- `CGxuLight::s_lights` — global light list

### 6.4 Lighting Parameters

Directional light parameters (from debug strings):
- `PLightDirIntens` — directional light intensity
- `PLightDirColor` — directional light color
- `PLightDirPos` — directional light position

Ambient light:
- `Ambient intensity: %.2f, RGB: (%.2f, %.2f, %.2f)` — ambient light format

### 6.5 Lighting LOD

- `mapObjLightLOD` — lighting LOD level (0-2)
- `MapObjLightLOD must be 0-2` — LOD validation
- `MapObj lighting enabled/disabled` — lighting toggle

### 6.6 Overbright

- `mapObjOverbright` — overbright toggle
- `MapObjOverbright disabled/enabled/unsupported on current API/HW.`
- Uses `Shaders\Pixel\MapObjOverbright.bls` shader

---

## 7. WMO Fog System

### 7.1 Fog Per Group

Each WMO group can have multiple fog instances:
- `SMOGroup::NUM_FOGS` — max fog count per group
- `index < SMOGroup::NUM_FOGS` — fog index validation
- `index < fogCount` — active fog count check
- `sub < NUM_FOGS` — sub-fog validation

### 7.2 Fog Classes

| Class/struct | Role |
|-------------|------|
| `FogQ` | Fog class |
| `LightDataFog` | Fog data structure (linked to lighting) |

### 7.3 OpenGL Fog

Direct OpenGL fog calls:
- `glFogfv` / `glFogf` / `glFogi` — fog parameter setting
- `OPTION ARB_fog_exp2` — ARB exponential² fog
- `OPTION ARB_fog_exp` — ARB exponential fog
- `OPTION ARB_fog_linear` — ARB linear fog

### 7.4 Console Commands

- `SetFogNear(value)` — set fog near plane
- `SetFogFar(value)` — set fog far plane
- `SetFogColor` — set fog color
- `ClearFog` — clear fog

Fog parameters:
- `fogNear` / `fogFar` / `FogColor`

---

## 8. WMO Portal / Visibility System

### 8.1 Portal Limits

- `portal->count <= 12` — **max 12 portals per WMO group**

### 8.2 Portal Structures

| Struct | Role |
|--------|------|
| `USPortalExt` | Portal extension data |
| `portalExt` | Portal extension field |

### 8.3 Portal Debug Toggles

- `TogglePortals` — toggle portal display
- `Portal display enabled/disabled` — portal debug rendering
- `Portal vis enabled/disabled` — portal visibility toggle

---

## 9. WMO BSP System

### 9.1 BSP Nodes and References

WMO groups use BSP (Binary Space Partitioning) for visibility culling:
- **MOBN** chunk — BSP nodes
- **MOBR** chunk — BSP references (to batches)
- **MORB** chunk — render batches (BSP-ordered)

### 9.2 BSP Source

- `C:\build\buildWoW\WoW\Common\AaBsp.cpp` — BSP implementation source file
- `nbspace` — BSP-related identifier

### 9.3 BSP Node Cache

The BSP system has a node cache for performance:
- `bspcache` — BSP node cache
- `BSP node caching` — cache system
- `BSP node cache already disabled` — cache disabled state
- `Disabling BSP node cache` — cache disable
- `Enabling BSP node cache (first time - starting up)` — initial enable
- `Enabling BSP node cache (already enabled, so clearing content.)` — cache clear

### 9.4 BSP Debug Toggle

- `BSP render enabled/disabled` — BSP debug rendering toggle

---

## 10. WMO Liquid System

### 10.1 WMO-Internal Liquids (MLIQ)

WMO groups can contain liquid geometry via the **MLIQ** chunk:
- `pIffChunk->token == 'MLIQ'` — MLIQ chunk check
- `group->liquidVerts` — liquid vertex grid (x × y)
- `(idxBase[i] - vtxSub) < (uint) (group->liquidVerts.x * group->liquidVerts.y)` — liquid index validation

### 10.2 Terrain Chunk Liquids (MCLQ)

Terrain chunks use a separate **MCLQ** chunk:
- `iffChunk->token=='MCLQ'` — MCLQ chunk check
- `CChunkLiquid` — chunk liquid class
- `WCHUNKLIQUID` — memory pool tag

### 10.3 Liquid Types

12 liquid types (`LIQUID_COUNT = 0xC`):
- `liquid < LIQUID_COUNT` — liquid type validation
- `liquidTexBaseName[liquid]` — liquid texture base name table
- `liquid != LIQUID_NONE` — liquid type check

Known liquid texture paths:
- `XTextures\ocean\ocean_h.%d.blp` — ocean (animated, %d = frame)
- `XTextures\lava\lava.%d.blp` — lava (animated)
- `XTextures\slime\slime.%d.blp` — slime (animated)

### 10.4 Liquid Grid

- `MD_LIQUID_NPOLY` — liquid polygon grid size
- `pos.x >= 0 && pos.x < MD_LIQUID_NPOLY && pos.y >= 0 && pos.y < MD_LIQUID_NPOLY` — grid bounds

### 10.5 Water Effects

| Class | Role |
|-------|------|
| `Water0Ripple` | Water ripple effect |
| `WaterRadWave` | Radial wave effect |

Water settings:
- `waterRipples` — ripple toggle
- `waterWaves` — wave toggle
- `waterParticulates` — particulate toggle
- `waterSpecular` — specular toggle
- `waterMaxLOD` — max LOD
- `waterLOD` — "Water geometry LOD" setting
- `waterLOD fixed to 0` — LOD fixed message

Water debug:
- `Water enabled/disabled` — water toggle
- `MapWater.cpp` — water source file

### 10.6 WMO Exterior Water Shader

- `Shaders\Pixel\MapObjExtWater0.bls` — WMO exterior water shader (when WMO geometry is near/under water)

---

## 11. WMO Doodad System

### 11.1 Doodad Classes

| Class | Role |
|-------|------|
| `CMapDoodadDef` | Doodad definition (M2 model placed in WMO) |
| `CDetailDoodadData` | Detail doodad data |
| `CDetailDoodadGeom` | Detail doodad geometry |
| `CDetailDoodadInst` | Detail doodad instance |
| `CDetailDoodadInstAdd` | Detail doodad instance add |
| `CDetailDoodadVertex` | Detail doodad vertex |

### 11.2 Doodad Properties

- `doodadDef->model` — M2 model reference
- `doodadDef->scale` — scale (must be non-zero: `CMath::fnotequal_(doodadDef->scale,0.0f)`)
- `doodadDef->parentLinkList` — parent link list
- `doodadDefLinkList` — linked list of doodad definitions
- `doodadDefHash` — hash table of doodad definitions
- `doodadDefUpdateList` — update list

### 11.3 Doodad Linking to WMO Groups

- `mapObjDefGroup->doodadDefLinkList` — doodads linked to WMO definition groups
- `chunk->doodadDefLinkList` — doodads linked to terrain chunks
- `chunk->detailDoodadInst` — detail doodad instance per chunk

### 11.4 Doodad Debug Toggles

- `showSimpleDoodads` — simple doodads toggle
- `showDetailDoodads` — detail doodads toggle
- `detailDoodadAlpha` — detail doodad alpha
- `Detail doodads enabled/disabled`
- `Terrain doodads enabled/disabled`
- `Terrain doodads AA Box visuals enabled/disabled`
- `Terrain doodads collision visuals enabled/disabled`
- `Simple doodads enabled/disabled`
- `Full alpha on doodads enabled/disabled`
- `Doodad animation enabled/disabled`
- `Detail doodad debug test enabled/disabled`

---

## 12. Scene Rendering Pipeline

### 12.1 World Scene

The world scene is managed by functions in `WorldScene.cpp`:
- `FUN_0067b6d0` — major WorldScene function (11 xrefs to WorldScene.cpp)
- `FUN_00682c50` — WorldScene function (3 xrefs)
- `FUN_00681a90` — WorldScene function (3 xrefs)
- `FUN_00680480` — WorldScene function (3 xrefs)
- `FUN_0067fff0` — WorldScene function (3 xrefs)
- Plus many more in the 0x0067cxxx-0x00682xxx range

### 12.2 Query Flags

The world query system uses flags to control what gets rendered:
- `CWorld::WQF_doodadMask` — doodad rendering mask
- `CWorld::WQF_gameObjMask` — game object rendering mask
- `CWorld::WQF_terrain` — terrain rendering
- `CWorld::WQF_liquid` — liquid rendering

### 12.3 Visible Lists

The scene maintains visible object lists:
- `visMapObjDefGroupList` — visible WMO definition groups
- `visMapObjDefGroupLiquidList` — visible WMO liquid groups
- `visDoodadList` — visible doodads
- `mapObjDefGroup->frustumList` — frustum-culled WMO groups

### 12.4 Camera-Related WMO

- `camMapObjGroup` — camera WMO group
- `camMapObj` — camera WMO

### 12.5 Render Modes

- `Rendering all visible groups (standard)` — standard render mode
- `Rendering only current group` — single-group render mode

### 12.6 WMO Fadeout

- `CWModelFadeout` — WMO model fadeout class (distance-based fading)

---

## 13. Graphics Abstraction Layer (CGx)

### 13.1 Device Classes

| Class | Role |
|-------|------|
| `CGxDeviceOpenGl` | OpenGL device implementation |
| `CGxDeviceD3d` | Direct3D device implementation |
| `CGxVboBroker` | VBO broker (manages vertex buffer objects) |
| `CGxStateBom` | State bomb (batches render state changes) |

### 13.2 Render State

- `CGxPushedRenderState` — pushed render state (stack-based)
- `CGxAppRenderState` — applied render state
- `EGxRenderState` — render state enum
- `GxRenderStates_Last` — render state count
- `Gx_MaxRsStackDepth` — max render state stack depth
- `Gx_MaxMatrixStackDepth` — max matrix stack depth

### 13.3 Vertex Formats

- `CGxVertexPC` — Position + Color
- `CGxVertexPT0T1` — Position + Tex0 + Tex1 (two texture coordinate sets)

### 13.4 Texture System

- `CGxTex` — texture class
- `CGxTexCache` — texture cache
- `CGxTexFlags` — texture flags (m_renderTarget, m_filter, m_generateMipMaps)
- `GxTex_Anisotropic` — anisotropic filter
- `GxTex_Dxt1` through `GxTex_Dxt5` — DXT compressed formats
- `Gx_MinTexWidth` / `Gx_MaxTexWidth` / `Gx_MinTexHeight` / `Gx_MaxTexHeight` — texture size limits
- `Gx_MaxTexAspect` / `Gx_MinTexAspect` — texture aspect ratio limits

### 13.5 Buffer System

- `CGxBuf` — graphics buffer
- `CGxPool` — buffer pool
- `Gx_MaxBufSize` — max buffer size
- `Gx_MaxVertices` / `Gx_MaxIndices` — vertex/index limits

### 13.6 Shader System

- `CGxPixelShader` — pixel shader class
- `CGxVertexShader` — vertex shader class
- `CGxShaderParam` — shader parameter
- `Gx_MaxLights` — max lights

### 13.7 Primitive Rendering

- `CGxDeviceOpenGl::PrimRender()` — OpenGL primitive render
- `CGxDeviceD3d::PrimRender()` — D3D primitive render

---

## 14. Terrain Shaders (for context)

Terrain uses its own set of shaders (separate from WMO):

| Shader | Description |
|--------|-------------|
| `terrain1.bls` | 1-layer terrain |
| `terrain2.bls` | 2-layer terrain |
| `terrain3.bls` | 3-layer terrain |
| `terrain4.bls` | 4-layer terrain |
| `terrain1_s.bls` ... `terrain4_s.bls` | Specular variants |
| `terrainp.bls` | Projective terrain |
| `terrainp_s.bls` | Projective specular terrain |
| `terrainp_u.bls` | Projective unlit terrain |
| `terrainp_us.bls` | Projective unlit specular terrain |
| `ocean0_s.bls` | Ocean specular |

Other shaders:
- `FFXBlur_2.bls` — full-screen blur
- `FFXGlow_2.bls` — full-screen glow
- `FFXMidtoneMap.bls` — midtone mapping
- `Desaturate.bls` — desaturation effect

---

## 15. Function Address Map

### WMO Rendering Functions (MapObjRender.cpp)

| Function | Xrefs to MapObjRender.cpp | Notes |
|----------|--------------------------|-------|
| `FUN_006ba9d0` | 3 | Major rendering function |
| `FUN_006babc0` | 5 | Major rendering function |
| `FUN_006baf70` | 2 | Rendering function |
| `FUN_006bb9c0` | 1 | Rendering function |
| `FUN_006bb530` | 1 | Rendering function |
| `FUN_006bb660` | 1 | Rendering function |
| `FUN_006ba230` | 3 | Rendering function |
| `FUN_006b9900` | 1 | Rendering function |
| `FUN_006b9ad0` | 1 | Rendering function |
| `FUN_006b9bb0` | 1 | Rendering function |
| `FUN_006b9600` | 2 | Rendering function |
| `FUN_006bcac0` | 1 | Rendering function |
| `FUN_006bc250` | 1 | Rendering function |
| `FUN_006bc520` | 1 | Rendering function |
| `FUN_006bc7a0` | 1 | Rendering function |
| `FUN_006b95e0` | 1 | Rendering function |

### WMO Shader Loading

| Function | Role |
|----------|------|
| `FUN_006abab0` | Loads all 6 WMO .bls shaders |

### WMO Definition Functions (MapObjDef.cpp)

| Function | Notes |
|----------|-------|
| `FUN_006ab0f0` | MapObjDef function |
| `FUN_006ab080` | MapObjDef function |
| `FUN_006ab160` | MapObjDef function |
| `FUN_006aa920` | MapObjDef function |
| `FUN_006aaf40` | MapObjDef function |

### WMO Object Functions (MapObj.cpp)

| Function | Notes |
|----------|-------|
| `FUN_006abc50` | MapObj function |
| `FUN_006ad850` | MapObj function |
| `FUN_006ad220` | MapObj function |
| `FUN_006ad600` | MapObj function |
| `FUN_006ad690` | MapObj function |
| `FUN_006ad740` | MapObj function |
| `FUN_006ad7a0` | MapObj function |
| `FUN_006acd00` | MapObj function |
| `FUN_006acda0` | MapObj function |
| `FUN_006ac180` | MapObj function |
| `FUN_006ac2a0` | MapObj function |

### WorldScene Functions (WorldScene.cpp)

| Function | Xrefs | Notes |
|----------|-------|-------|
| `FUN_0067b6d0` | 11 | Major WorldScene function |
| `FUN_00682c50` | 3 | WorldScene function |
| `FUN_00681a90` | 3 | WorldScene function |
| `FUN_00680480` | 3 | WorldScene function |
| `FUN_0067fff0` | 3 | WorldScene function |
| `FUN_0067bcf0` | 1 | WorldScene function |
| `FUN_0067be10` | 1 | WorldScene function |
| `FUN_00681250` | 1 | WorldScene function |
| `FUN_00681690` | 3 | WorldScene function |
| `FUN_0067f190` | 1 | WorldScene function |
| `FUN_0067f2f0` | 1 | WorldScene function |
| `FUN_0067eb40` | 1 | WorldScene function |
| `FUN_0067e390` | 1 | WorldScene function |
| `FUN_0067e770` | 1 | WorldScene function |
| `FUN_0067d4f0` | 2 | WorldScene function |
| `FUN_0067d760` | 1 | WorldScene function |
| `FUN_0067ca90` | 1 | WorldScene function |
| `FUN_0067cba0` | 1 | WorldScene function |
| `FUN_0067ccb0` | 1 | WorldScene function |
| `FUN_0067cd90` | 2 | WorldScene function |
| `FUN_0067cea0` | 1 | WorldScene function |
| `FUN_0067cfd0` | 2 | WorldScene function |

### Terrain Chunk Render Functions (MapChunkRender.cpp)

| Function | Notes |
|----------|-------|
| `FUN_006c1c50` | MapChunkRender function |
| `FUN_006c0bc0` | MapChunkRender function |

### Terrain Chunk Functions (MapChunk.cpp)

| Function | Notes |
|----------|-------|
| `FUN_006b3890` | MapChunk function |
| `FUN_006b3a10` | MapChunk function |
| `FUN_006b3ba0` | MapChunk function |
| `FUN_006b3090` | MapChunk function |
| `FUN_006b3170` | MapChunk function |
| `FUN_006b33e0` | MapChunk function |
| `FUN_006b3580` | MapChunk function |
| `FUN_006b3700` | MapChunk function |
| `FUN_006b2ef0` | MapChunk function |
| `FUN_006b5820` | MapChunk function |
| `FUN_006b5cf0` | MapChunk function |
| `FUN_006b5c90` | MapChunk function |
| `FUN_006b5d50` | MapChunk function |
| `FUN_006b5db0` | MapChunk function |
| `FUN_006b5120` | MapChunk function |
| `FUN_006b52d0` | MapChunk function |
| `FUN_006b5350` | MapChunk function |
| `FUN_006b5560` | MapChunk function |
| `FUN_006b5760` | MapChunk function |
| `FUN_006b4920` | MapChunk function |
| `FUN_006b4c60` | MapChunk function |
| `FUN_006b4230` | MapChunk function |
| `FUN_006b44b0` | MapChunk function |

### WDT Reader Functions

| Function | Notes |
|----------|-------|
| `FUN_006976f0` | WDT reader (MWMO chunk) |
| `FUN_006c2d70` | WDT reader (MWMO chunk) |

---

## 16. Source File Map (from debug strings)

All confirmed source file paths from the 1.0.0 binary:

| Source file | Component |
|-------------|-----------|
| `WoW\Source\WorldClient\CMapObj.h` | WMO root header |
| `WoW\Source\WorldClient\MapObj.cpp` | WMO root implementation |
| `WoW\Source\WorldClient\MapObjDef.cpp` | WMO definition |
| `WoW\Source\WorldClient\MapObjGroup.cpp` | WMO group |
| `WoW\Source\WorldClient\MapObjRender.cpp` | WMO rendering |
| `WoW\Source\WorldClient\MapObjRead.cpp` | WMO file reading |
| `WoW\Source\WorldClient\MapLight.cpp` | Map lighting |
| `WoW\Source\WorldClient\MapChunk.cpp` | Terrain chunk |
| `WoW\Source\WorldClient\MapChunkRender.cpp` | Terrain chunk rendering |
| `WoW\Source\WorldClient\MapEntity.cpp` | Map entity |
| `WoW\Source\WorldClient\MapDoodadDef.cpp` | Doodad definition |
| `WoW\Source\WorldClient\MapBaseObj.cpp` | Map base object |
| `WoW\Source\WorldClient\MapArea.cpp` | Map area |
| `WoW\Source\WorldClient\MapWater.cpp` | Water/liquid rendering |
| `WoW\Source\WorldClient\WorldScene.cpp` | World scene |
| `WoW\Source\WorldClient\DetailDoodad.cpp` | Detail doodads |
| `WoW\Common\AaBsp.cpp` | BSP implementation |
| `ENGINE\Source\gx\CGxDeviceOpenGL\CGxDeviceOpenGl.cpp` | OpenGL device |
| `ENGINE\Source\gx\CGxDeviceOpenGL\CGxOglState.cpp` | OpenGL state |
| `ENGINE\Source\gx\CGxDeviceOpenGL\CGxOglShader.cpp` | OpenGL shader |
| `ENGINE\Source\gx\CGxDeviceOpenGL\CGxOglTexture.cpp` | OpenGL texture |
| `ENGINE\Source\gx\CGxDeviceOpenGL\GlExtSupport.cpp` | OpenGL extensions |
| `ENGINE\Source\gx\CGxDeviceOpenGL\W32\CGxOglDeviceW32.cpp` | Win32 OpenGL device |
| `ENGINE\Source\gx\CGxDevice\CGxDevice.cpp` | Base graphics device |
| `ENGINE\Source\gx\CGxDeviceD3d\CGxD3dDevice.cpp` | D3D device |
| `ENGINE\Source\gx\CGxDeviceD3d\CGxD3dShader.cpp` | D3D shader |
| `ENGINE\Source\gx\CGxDeviceD3d\CGxD3dTexture.cpp` | D3D texture |
| `ENGINE\Source\gx\CGxDeviceD3d\CGxD3dXform.cpp` | D3D transforms |
| `ENGINE\Source\gx\CGxDeviceD3d\CGxD3dState.cpp` | D3D state |
| `ENGINE\Source\Model2\M2Scene.cpp` | M2 scene |

---

## 17. What This Means for wow-viewer

### 17.1 Current Renderer Gaps

The current wow-viewer renderer is a brute-force approach that:
1. Renders all WMO groups without portal/BSP culling
2. Does not implement WMO-specific shaders (specular, overbright, metal, extwater)
3. Does not implement WMO fog per group
4. Does not implement WMO liquid rendering (MLIQ)
5. Does not implement WMO vertex color lighting (MOCV)
6. Does not implement WMO dynamic lights (MOLR)
7. Does not implement WMO doodad rendering with proper linking
8. Does not implement distance-based WMO fadeout

### 17.2 Implementation Priorities

To reach a "1.0 era" renderer:

1. **WMO vertex colors (MOCV)** — pre-baked lighting, easiest win
2. **WMO batches** — split intBatch (opaque) vs transBatch (transparent), render opaque first
3. **WMO fog** — per-group fog with exp2/exp/linear modes
4. **WMO shaders** — load and route .bls shaders (specular, overbright, metal, trans)
5. **WMO liquids (MLIQ)** — liquid rendering with animated textures
6. **WMO lights (MOLR)** — dynamic point/spot lights
7. **WMO portals** — portal-based visibility culling (max 12 per group)
8. **WMO BSP** — BSP node traversal for batch visibility
9. **WMO doodads** — proper doodad linking and rendering
10. **WMO fadeout** — distance-based model fading

### 17.3 Key Design Decisions

- The 1.0.0 client uses **two batch categories**: opaque (intBatch) and transparent (transBatch). The renderer must render opaque first, then transparent.
- The 1.0.0 client uses **per-group fog** with up to `NUM_FOGS` fog instances per group.
- The 1.0.0 client uses **BSP + portals** for visibility, not just frustum culling.
- The 1.0.0 client uses **pre-baked vertex colors (MOCV)** as the primary lighting, with dynamic lights (MOLR) as a secondary layer.
- The 1.0.0 client uses **CGx abstraction** with OpenGL (NV register combiners, ARB programs) and D3D backends. The wow-viewer should use a modern abstraction (e.g., OpenGL/Vulkan) but follow the same shader routing logic.
- The 1.0.0 client uses **distance-based fadeout** (`CWModelFadeout`) for WMO models.

---

## 18. Decompiled Function Analysis (targeted code path tracing)

### 18.1 WMO Shader Loading Chain

**`FUN_006abab0`** (WMO shader loader):
- Sets global flags (`DAT_00aade18 = 0`, `DAT_00ab5d68 = 0x800`)
- Calls `FUN_0058ee90()` **6 times** (once per WMO .bls shader)
- Sets `DAT_00aade84 = 1` (shaders loaded flag)

**`FUN_0058ee90`** (shader load function):
- Takes `param_1` (shader name) and `param_2` (shader type)
- If `param_2 == 0`: logs error at `ENGINE\Source\...` line 0x7cb (1995)
- Calls virtual method on CGx device: `(*DAT_00a1ce58)->vtable[0xb4](param_1, param_2, 1)`
- `DAT_00a1ce58` = global CGx device pointer

### 18.2 WMO Group VBO Setup — `FUN_006ba9d0`

This function creates vertex and index buffers for a WMO group.

**Parameters:**
- `param_1` = CMapObjGroup pointer
- `param_2` = vertex format mode (3 = standard, 4 = extended)

**Flow:**
1. If vertexVB (`param_1 + 4`) is null: create it via `FUN_0058c490` + `FUN_006a47e0`
2. If VB creation failed: use immediate mode with vertex format from `param_1 + 0xd0`/`0xd4` (mode 3) or `0xf0` (mode 4)
3. If VB exists: set global state, begin pass, set vertex stream, validate size against `param_1 + 0xe8`
4. Repeat for indexVB (`param_1 + 8`): create with format from `param_1 + 0x124` (standard) or `0x128` (alternate, when flag at `param_1 + 0xc8` is set and `DAT_00a93790` is true)
5. Validate index buffer size against `param_1 + 0xec`

### 18.3 WMO Batch Renderer — `FUN_006babc0`

This is the **main WMO batch rendering function**. It iterates BSP-ordered batches, looks up materials, sets render state, and draws.

**Parameters:**
- `param_1` = CMapObjGroup pointer
- `param_2` = mode (0 = reset batch flags before rendering)

**Flow:**
1. Assert group is not null (`MapObjRender.cpp:882`)
2. Begin render pass (`FUN_0058ccb0`)
3. Set up VBOs with mode 3 (`FUN_006ba9d0`)
4. Assert `transBatchCount == 0` and `intBatchCount == 0` (this function handles BSP-ordered batches, not the int/trans split)
5. Iterate `batchCount` batches (each 0x18 = 24 bytes):
   - If mode 0: clear upper nibble of batch flags (`batch + 0x16 &= 0x0f`)
   - If batch not yet processed (upper nibble == 0) and visibility check passes:
     - Mark as processed (`batch + 0x16 |= 0xf0`)
     - Look up material: `materialIndex * 0x40 + materialArray`
     - Check visibility (`FUN_0044ecb0`)
     - Set color based on material flags (bit 4 = use material color vs default)
     - Set blend/alpha state
     - Toggle render state bits 3 and 4 based on material flags
     - If material has passCount == 1 and specular enabled: render specular pass
     - Draw batch (`FUN_006baea0` + `FUN_0058dd90`)
     - Restore state after specular pass
6. End render pass (`FUN_0058ccc0`)

### 18.4 Batch Draw Setup — `FUN_006baea0`

Fills draw parameters from the batch struct.

**Primitive type:**
- Default: 3 (GL_TRIANGLES)
- If `DAT_00a93790` (triangle strips enabled) AND batch flag bit 0: 4 (GL_TRIANGLE_STRIP)

**Draw parameters filled:**
- `drawParams[0]` = primitiveType (3 or 4)
- `drawParams[1]` = baseVertex (from batch + 0x0c)
- `drawParams[2]` = startIndex (from batch + 0x10, uint16)
- `drawParams[2.5]` = count (from batch + 0x12, uint16)
- `drawParams[3]` = primCount (from batch + 0x14, uint16)

### 18.5 Recovered Struct Layouts

#### WMO Batch Struct (0x18 = 24 bytes)

| Offset | Size | Field |
|--------|------|-------|
| 0x00 | int16 | boundingBox min X (frustum cull) |
| 0x02 | int16 | boundingBox min Y |
| 0x04 | int16 | boundingBox min Z |
| 0x06 | int16 | boundingBox max X |
| 0x08 | int16 | boundingBox max Y |
| 0x0a | int16 | boundingBox max Z |
| 0x0c | uint32 | baseVertex (starting vertex in VBO) |
| 0x10 | uint16 | startIndex (starting index in IBO) |
| 0x12 | uint16 | count (number of vertices or indices) |
| 0x14 | uint16 | primCount (number of primitives) |
| 0x16 | byte | flags (bit 0 = use triangle strip, upper nibble = render state) |
| 0x17 | byte | materialIndex (index into material array) |

#### WMO Material Struct (0x40 = 64 bytes)

| Offset | Size | Field |
|--------|------|-------|
| 0x00 | uint32 | flags (bit 3 = render state, bit 4 = 0x10 = use material color, bit 5 = 0x20 = use vertex colors/MOCV) |
| 0x04 | uint32 | passCount (0 = single pass, 1 = two-pass specular, 2 = extended two-pass specular) |
| 0x14 | uint32* | color (when flag 0x10 is set, at material + 0x14) |

#### WMO Group (CMapObjGroup) Fields

| Offset | Size | Field |
|--------|------|-------|
| 0x04 | ptr | vertexVB (vertex buffer object) |
| 0x08 | ptr | indexVB (index buffer object) |
| 0x3c | int16 | transBatchCount (transparent batch count) |
| 0x3e | int16 | intBatchCount (interior/opaque batch count) |
| 0x40 | uint16 | batchCount (total BSP-ordered batch count) |
| 0xc8 | int32 | flag (use alternate index format?) |
| 0xd0 | uint32 | vertexFormat |
| 0xd4 | uint32 | vertexCount |
| 0xd8 | ptr | batchArray (array of 0x18-byte batch structs) |
| 0xe8 | int32 | vertexBufferSize (expected VBO size) |
| 0xec | int32 | indexBufferSize (expected IBO size) |
| 0xf0 | uint32 | extendedVertexFormat (mode 4) |
| 0x124 | uint32 | standardIndexFormat |
| 0x128 | uint32 | alternateIndexFormat (32-bit indices?) |
| 0x12c | uint32 | vertexDataSize/Format |
| 0x1d8 | ptr | materialArray (array of 0x40-byte material structs) |

#### Draw Parameters Struct

| Offset | Size | Field |
|--------|------|-------|
| 0x00 | uint32 | primitiveType (3 = GL_TRIANGLES, 4 = GL_TRIANGLE_STRIP) |
| 0x04 | uint32 | baseVertex |
| 0x08 | uint16 | startIndex |
| 0x0a | uint16 | count |
| 0x0c | uint16 | primCount |

#### Key Global Variables

| Address | Type | Description |
|---------|------|-------------|
| `DAT_00a93790` | bool | Use triangle strips / 32-bit indices |
| `DAT_00aadec1` | bool | Specular enabled |
| `DAT_007dd528` | uint32 | Global render state value |
| `DAT_007dd288` | uint32 | Specular render state |
| `DAT_00a1ce58` | ptr | CGx device pointer (vtable[0xb4] = shader load) |

### 18.6 Evidence

Decompiled code saved to: `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/wmo_render_pipeline.c`

---

## 19. Additional Rendering Pipeline Details (deep sweep)

### 18.1 Map Rendering Entry Points

Three separate rendering source files:
- **`MapRender.cpp`** — main map rendering entry point (`C:\build\buildWoW\WoW\Source\WorldClient\MapRender.cpp`)
- **`WorldScene.cpp`** — world scene management (40+ functions, 0x0067cxxx-0x00682xxx)
- **`MapObjRender.cpp`** — WMO-specific rendering (16+ functions, 0x006b9xxx-0x006bcxxx)

Engine render framework:
- **`CSimpleRender`** — `C:\build\buildWoW\ENGINE\Source\Frame\CSimpleRender.h` / `.cpp`
- **`RENDERCALLBACKNODE`** — render callback node struct (for render callbacks)

### 18.2 Culling System

| Cvar/Setting | Description |
|-------------|-------------|
| `showCull` | Show culling debug visualization |
| `Terrain culling enabled/disabled` | Terrain culling toggle |
| `DistCull` | "Object distance culling" — distance-based culling (range: 1.0 to max) |
| `SmallCull` | "Object size culling" — size-based culling (range: 0.001 to 2.0) |
| `smallCull` | Small cull threshold setting |

### 18.3 Triangle Strip Rendering

- `Use triangle strips to render` — toggle for triangle strip mode
- `Triangle strips disabled/enabled on restart` — requires restart to take effect
- Default: triangle lists (not strips)

### 18.4 Vertex Optimization Regions

The M2 system uses vertex optimization regions:
- `optRegion->vertexStart` / `optRegion->vertexCount` — optimized vertex region
- `regionA->vertexStart` / `regionA->vertexCount` — original vertex region
- Optimization merges adjacent regions: `optRegion->vertexStart <= regionA->vertexStart`

### 18.5 Vertex / Index Limits

| Limit | Value |
|-------|-------|
| Max vertices per draw | `Gx_MaxVertices` |
| Max indices per draw | `Gx_MaxIndices` |
| Max batch count | `Gx_MaxBatchCount` |
| Max buffer size | `Gx_MaxBufSize` |
| Max vertices (WMO) | `0x40000` (262,144) |
| 16-bit index limit | `0x10000` (65,536) |
| Min indices per draw | 3 (one triangle) |
| Max textures per batch (M2) | 2 (`m_batch->textureCount < 2`) |

### 18.6 Sort System

- `sortEntry` — sort entry for render ordering
- Sorting is used for transparent geometry (back-to-front) and state batching

### 18.7 Texture System

| Class/struct | Role |
|-------------|------|
| `CTextureHash` | Texture hash (for texture lookup/caching) |
| `TEXTURECACHEROW` | Texture cache row |
| `HTEXTURECACHE` | Texture cache memory pool tag |

Texture cache:
- `C:\build\buildWoW\WoW\Common\TextureCache.cpp` — texture cache source
- `m_textureCache` — per-face texture cache array
- `m_currentFace->m_textureCache[textureNumber].GetTexturePtr()` — per-face texture lookup
- `GxuFontTextureCache.cpp` — font texture cache

Texture operations:
- `glBindTexture` — OpenGL texture binding
- `SetTexture` — console command
- `TextureGetMips()` / `TextureGetInfo()` — texture info retrieval
- `Ds_TextureArray0` / `Ds_ActiveTexture` — texture array state
- `g_textureMipBits` — texture mipmap bits
- `systemCaps.m_maxTextureSize` — max texture size
- `textureLodDist` — texture LOD distance setting
- `ValidTextureCoords()` — texture coordinate validation

### 18.8 Render State Management

| Class/struct | Role |
|-------------|------|
| `CGxPushedRenderState` | Pushed render state (stack-based) |
| `CGxAppRenderState` | Applied render state |
| `EGxRenderState` | Render state enum |
| `CGxStateBom` | State bomb (batches render state changes) |

Render state stack:
- `Gx_MaxRsStackDepth` — max render state stack depth
- `Gx_MaxMatrixStackDepth` — max matrix stack depth
- `mStackOffsets` — render state stack offsets

### 18.9 Performance Counters

- `GxPerfCounters_Last` — performance counter count
- `counter < GxPerfCounters_Last` — counter iteration

### 18.10 WMO Root File Chunks (not found as strings)

The WMO root file chunks (MOHD, MOGN, MOGI, MOTX, MOMT, MOPV, MOPT, MOPR, MODS, MODD, MODN) do NOT appear as assertion strings in the binary. This means:
1. The root file reader does not validate individual chunk tokens with assertions
2. The root file is likely read with a generic chunk reader that doesn't check each token
3. The group-level chunks (MOGP, MOPY, MOVI, etc.) DO have assertions because they're parsed in a sub-chunk reader

This is consistent with the WMO root file being read by `CMapObj::Read()` in `MapObjRead.cpp`, which likely uses a generic IFF/chunk reader without per-chunk validation.

---

## 19. Open Follow-ups

1. **Decompile WMO rendering functions** — The decompile endpoint was unavailable during this session. The 16+ functions in the 0x006b9xxx-0x006bcxxx range need decompilation to recover exact batch rendering logic. *(Substantially done — see §18 and §20.)*
2. ~~**MOPY material flags**~~ — **RESOLVED (§20.4).** MOPY entry = 2 bytes `{flags:u8, materialId:u8}`, count = size/2. Runtime uses it only as a per-face collision filter mask; render selection is via MOBA.
3. **MOBA/MORB batch format** — Recovered in §18.5 (0x18-byte batch) and §20.1 (MORI/MORB strip override). Light/color detail still partial.
4. **MOLR light format** — Parser recovered (§20.1: `group[0xe0]`, count `size>>1`, u16 light refs); per-light record semantics still open.
5. ~~**MLIQ liquid format**~~ — **RESOLVED (§20.5).** 30-byte header + (xVerts·yVerts)×8-byte vertex grid + (xTiles·yTiles)×1-byte tile-flags.
6. ~~**Portal visibility algorithm**~~ — **RESOLVED (§20.6).** Screen-space-rectangle portal culling with a 32-deep frustum stack.
7. ~~**BSP traversal algorithm**~~ — **RESOLVED (§20.3).** 16-byte `CAaBspNode`; ray + AABB recursion; 8-way node cache.
8. ~~**WorldScene render order**~~ — **RESOLVED (§20.2).** `CWorldScene::Render` = `FUN_0067c460`; camera-in/out branch → portal vis → opaque → fog → transparent/effects.
9. **CGxStateBom** — Need to understand the "state bomb" render state batching system.
10. **WMO fadeout algorithm** — Need to trace the `CWModelFadeout` distance-based fading.

---

## 20. Scene / Portal / BSP / MOPY / MLIQ — decompiled (2026-07-15 follow-up)

Follow-up decompilation pass resolving Open Follow-ups #2, #5, #6, #7, #8. Every
function below was decompiled from WoW.exe 1.0.0.3980. Full annotated source:
[`evidence/1.0.0-ghidra/wmo_scene_portal_bsp.c`](wow-viewer/specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/wmo_scene_portal_bsp.c).

### 20.1 MOGP sub-chunk parsers

`CMapObjGroup::Read` = `FUN_006c5380` (validates `MVER==0x11` + `MOGP`, copies
the 0x58-byte MOGP header into the group, then dispatches sub-chunks). Two parsers:

**`FUN_006c55a0` — mandatory chunks, fixed order** (assert on any token mismatch):

| Chunk | ptr field | count field | element |
|-------|-----------|-------------|---------|
| MOPY | `group[0xc0]` | `group[0x120]` = size>>1 | 2 B / triangle |
| MOVI | `group[0xc4]` | `group[0x124]` = size>>1 | u16 vertex index |
| MOVT | `group[0xcc]` | `group[0x12c]` = size/0xc | C3Vector (12 B) |
| MONR | `group[0xd0]` | `group[0x130]` = size/0xc | C3Vector normal |
| MOTV | `group[0xd4]` | `group[0x134]` = size>>3 | 2×f32 UV (8 B) |
| MOBA | `group[0xd8]` | `group[0x138]` = size/0x18 | 24-B SMOBatch |

Vertex buffer size `group[0xe8]` = `vertCount × (36 if (flags&0x48)==0 else 32)` — the
4-byte delta is the MOCV colour slot. Then calls `FUN_006c5810` for optional chunks.

**`FUN_006c5810` — optional chunks, each gated by an SMOGroup flag (`group[0x10]`):**

| Flag | Chunk(s) | Stored |
|------|----------|--------|
| 0x00200 | MOLR | `group[0xe0]`, count size>>1 (u16 light refs) |
| 0x00800 | MODR | `group[0xe4]`, count size>>1 (u16 doodad refs) |
| 0x00001 | MOBN + MOBR | → `FUN_00695b80` BSP init (§20.3) |
| 0x00400 | MPBV/MPBP/MPBI/MPBG | map-object particle batches (skipped) |
| 0x00004 | MOCV | `group[0xf0]`, count size>>2 (BGRA) |
| 0x01000 | MLIQ | liquid (§20.5) |
| 0x20000 | MORI + MORB | strip indices `group[0xc8]`/`[0x128]`; MORB `group[0xdc]` overrides each `batch[i].baseVertex@0x0c`/`startIndex@0x10` when strips are enabled |

### 20.2 WorldScene render order — `FUN_0067c460` (`CWorldScene::Render`)

Driven by map-render top `FUN_006742e0`. Order:

1. `FUN_0058ccb0` — begin render pass; reset per-frame vis state (frustum index = 0).
2. **`FUN_0067d4f0`** — decide if the camera is inside a WMO. Sets `DAT_00a78e74` =
   camMapObj (0 = outside); resolves camMapObjGroup via `FUN_006ad600`.
3. **Branch:**
   - **Camera outside:** `FUN_0067e3c0` — exterior pass (calls `FUN_00681250`, the
     exterior WMO visibility pass, once per loaded WMO instance).
   - **Camera inside:** `FUN_00681690` — interior visibility (portal walk from the
     camera's groups), **then** drain the **CExtView** list (exterior world seen out
     through portals, `DAT_00ac4030`, max 16) rendering each via `FUN_0067e3c0`.
4. Opaque world passes: `FUN_0067b460`, `FUN_006b3170`, `FUN_00689420`, `FUN_006abed0`, `FUN_006b6cc0`.
5. Fog/environment select — `DAT_00aadec8` chooses interior (`+0xe4`) vs exterior (`+0xf8`) fog params.
6. Transparent / effect passes: `FUN_0067fa70`, `FUN_0067fd40`, `FUN_0067fff0`, `FUN_00681030`, `FUN_0067f870`, `FUN_0067f500`.
7. `FUN_0058ccc0` — end render pass; if `_DAT_00a78b1c & 0x200000` → `FUN_006bd620` (portal debug overlay, `TogglePortals`).

**Frustum stack:** 32 slots × 0xfc bytes at `DAT_00a7a758`, top index `DAT_00a7a428`.
`FUN_0067d760` = push (assert idx<31, copy parent), `FUN_0067e390` = pop (assert idx>0),
`FUN_0067dd30` = build a narrowed frustum from a 2-D portal screen-rect.

### 20.3 BSP — `AaBsp.cpp` (MOBN nodes + MOBR refs)

**`CAaBspNode` = 16 bytes** (recovered from the traversals):

| Off | Type | Field |
|-----|------|-------|
| 0x00 | u16 | `flags` — bit 0x4 = LEAF; low 2 bits = split axis (0=X, 1=Y, 2=Z) |
| 0x02 | i16 | `negChild` (0xFFFF = none) |
| 0x04 | i16 | `posChild` (0xFFFF = none) |
| 0x06 | u16 | `nFaces` (leaf) |
| 0x08 | u32 | `faceStart` — offset into MOBR |
| 0x0c | f32 | `planeDist` |

MOBR = u16 array; each entry indexes a MOVI triangle (== MOPY face index).

- `FUN_00695b80` — `CAaBspTree::Init` (stores prebuilt MOBN/MOBR + the group bbox).
- `FUN_006965f0` — **ray/segment** traversal (split segment at the plane, recurse near/far).
- `FUN_00696820` — **AABB/box** traversal (clamp child box on the split axis, recurse both when straddling).
- `FUN_00696560` — leaf face gather into a global visible-face list with a per-face "already-added" bitmask.
- **BSP node cache** ("BSP node caching" / `bspcache`): `FUN_00696ab0` = 8-way
  set-associative cache (entry stride 0x2460) keyed on node ptr; miss →
  `FUN_00696bf0` builds a compact per-node vertex/index/flags buffer (≤449 verts /
  ≤301 faces, else flagged too-big). **`FUN_00696bf0` is where MOPY flags are read.**

At render time WMO **batch selection is frustum-based, not BSP-based**: `FUN_006babc0`
iterates all `group[0x40]` MOBA batches and per-batch frustum-culls the batch bbox
(`FUN_006ba940`, the 6× i16 bbox at batch+0x00). The BSP is the **collision** tree.

### 20.4 MOPY — per-triangle material/collision flags

On-disk entry = **2 bytes** per triangle: `{flags:u8@0, materialId:u8@1}`; count =
`size/2`. Runtime behaviour recovered:

- **`materialId` is not read by the renderer** — draws come from MOBA (each batch
  carries `materialIndex@0x17`). `materialId==0xFF` (collision-only) triangles simply
  never appear in a MOBA batch.
- **`flags` is a per-face collision filter mask.** During BSP-cache build
  (`FUN_00696bf0`) each face's flags are cached as `MOPY.flags & 0x7f` (bit 0x80
  stripped). Collision queries `FUN_006a2840` (line) / `FUN_006a2c60` (box) skip any
  face where `(cachedFlags & queryMask) == 0`.

Bit *names* are not in the beta strings; cross-referenced to the documented WMO
`SMOPoly` set (stable 1.x→3.x): `0x01 F_UNK_01`, `0x02 F_NOCAMCOLLIDE`, `0x04 F_DETAIL`,
`0x08 F_COLLISION`, `0x10 F_HINT`, `0x20 F_RENDER`, `0x40 F_UNK_40`, `0x80 F_COLLIDE_HIT`
(the 0x80 strip at cache time is consistent with `F_COLLIDE_HIT` being a runtime
result bit, not a static classification).

### 20.5 MLIQ — WMO-internal liquid

Parsed in `FUN_006c5810` under group flag **0x1000**. **30-byte (0x1e) header:**

| Off | Type | Field | Group field |
|-----|------|-------|-------------|
| 0x00 | u32 | xVerts | `group[0xf4]` |
| 0x04 | u32 | yVerts | `group[0xf8]` |
| 0x08 | u32 | xTiles | `group[0xfc]` |
| 0x0c | u32 | yTiles | `group[0x100]` |
| 0x10 | f32 | baseX | `group[0x104]` |
| 0x14 | f32 | baseY | `group[0x108]` |
| 0x18 | f32 | baseZ | `group[0x10c]` |
| 0x1c | u16 | materialId | `group[0x110]` |

- **Vertex grid:** `group[0x114]` = header+0x1e; `xVerts·yVerts` verts × **8 bytes**.
- **Tile-flags grid:** `group[0x118]` = vertexArray + `xVerts·yVerts·8`; `xTiles·yTiles` × **1 byte**.
- `xTiles == xVerts-1`, `yTiles == yVerts-1`. Post-parse: `FUN_006a6070` grows a
  `CMapObjGroup` render VertArray to `xVerts·yVerts` (0xc/vert), then `FUN_006a4cb0`.

This is the classic pre-MH2O WMO liquid layout — single materialId, single base
corner, height/UV vertex grid + tile-flag grid.

### 20.6 Portal visibility propagation — screen-rect culling

Root-object (`CMapObj`) portal arrays: `scene[0x134]` MOPV verts (C3Vector),
`scene[0x138]` MOPT portals (**SMOPortal, 20 B**: `startVtx u16, count u16, plane C4Plane`),
`scene[0x13c]` MOPR refs (**SMOPortalRef, 8 B**: `portalIdx u16, groupIdx u16, side i16, filler`).
Per group: `group[0x2c]`=refStart, `group[0x30]`=refCount.

**SPortalExt** (per portal, 0x1c B, array `&DAT_00ab5d6c`): `flags u16@0` (bit0 = not
visible), screen rect `minX/minY/maxX/maxY f32 @0x04/0x08/0x0c/0x10`, `frameStamp @0x18`.

- **`FUN_006ba230`** — project one portal to a screen rect: transform ≤12 MOPV verts
  (assert `portal->count <= 12`) by view-proj, near-clip (`FUN_006ba6b0`), then take
  min/max screen X/Y of the clipped verts (or set flag bit0 if fully clipped).
- **`FUN_006b9d30`** — the recursion. For each of a group's MOPR refs: skip the entry
  portal (no backtrack); project (cached by frame stamp); back-face cull
  (`dot(camLocal, plane.n)+plane.d`, negated if `side<0`); **intersect** the portal's
  screen rect with the incoming rect; if the result has w,h ≥ 0.001, push+build a
  narrowed sub-frustum and **recurse into the neighbour group** (depth+1, capped by
  `DAT_00ab5d5c`), then pop. Exterior-connected neighbours (flag 0x8) are deferred to
  the CExtView list (max 16) instead.
- **Seeds:** `FUN_006b9600` (camera inside — recurse from the camera's groups, then
  CExtViews, then direct exterior groups flag 0x10000) and `FUN_006b9900` (camera
  outside — flag 0x8 groups recurse, flag 0x10000 render directly). `FUN_006b9cd0`
  marks a group visible; "visible this frame" = the frame stamp `DAT_00aade18`.

**Net:** 1.0.0 uses **2-D screen-rectangle portal culling** (not full frustum-plane
clipping) — the frustum is carried as a shrinking screen rect, narrowed at each
portal, depth-limited, with an explicit deferred list for exterior views seen out
through a WMO's boundary portals.
