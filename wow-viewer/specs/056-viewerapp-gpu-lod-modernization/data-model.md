# Data Model: 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization

**Phase 1 output. Companion to `plan.md` and `research.md`.**
**Date**: 2026-06-10.

This file defines the backend-neutral data model the new shared renderer consumes. It is the **shape** of the contracts in `contracts/*`. The contracts are written against this data model.

---

## 1. Top-Level Inputs

The renderer is **driven** by a `RenderScene` value and **bounded** by a `RenderVariant` value. The `RenderScene` describes the world; the `RenderVariant` describes the per-frame user-controlled knobs (LOD, fog, sky, hide flags).

```text
RenderScene
├── Camera: SceneCamera
├── Tiles: IReadOnlyList<RenderTerrainTile>
├── WorldObjects: IReadOnlyList<RenderWorldObjectRef>
├── SkyState: SkyState
├── FogState: FogState
├── WdlOverride: RenderWdlOverride?     (optional, for far-horizon tiles)
└── AoiBounds: AoiBounds

RenderVariant (extends the existing WowViewer.Core.Renderer.Scene.RenderVariant)
├── HideTerrain / HideLiquids / HideSky / HideObjects / HideParticles / HideMinimap
├── TerrainLod: TerrainLodSettings
├── ObjectLod: ObjectLodSettings
├── WaterLod: WaterLodSettings
├── LightLod: LightLodSettings
├── MipSelection: MipSelectionSettings
└── Quality: RenderQualityLevel
```

## 2. Camera

```text
SceneCamera (existing; extended in Phase 1)
├── Position: Vector3
├── Forward / Right / Up: Vector3
├── ViewMatrix: Matrix4x4
├── ProjectionMatrix: Matrix4x4
├── ViewProjectionMatrix: Matrix4x4
├── InverseViewProjectionMatrix: Matrix4x4
├── Frustum: FrustumPlanes          (6 planes; uses FrustumCuller.ComputePlanes)
├── NearPlane: float
├── FarPlane: float
├── FovY: float
├── AspectRatio: float
└── AoiBounds: AoiBounds
```

## 3. Terrain

```text
RenderTerrainTile
├── TileX: int
├── TileY: int
├── SourcePath: string              (for diagnostics)
├── TileData: WorldTerrainTileData  (from WowViewer.Core.Runtime.World.Terrain)
├── Mesh: RenderTerrainMesh         (retained-mode VBO/IBO + UBO)
├── LodBucket: TerrainLodBucket     (Full / Reduced / WdlOnly / Culled)
├── DistanceFromCamera: float
└── IsVisible: bool                 (frustum-culled)

TerrainLodSettings
├── NearDistance: float             (default: 256.0)
├── MidDistance: float              (default: 1024.0)
├── FarDistance: float              (default: 4096.0)
├── NearMeshResolution: int         (default: 257)
├── MidMeshResolution: int          (default: 33)
├── UseWdlForFar: bool              (default: true)
└── UseMidMeshForReduced: bool       (default: true)

TerrainLodBucket
├── Full                           (near; full 257x257)
├── Reduced                        (mid; reduced mesh)
├── WdlOnly                        (far; use WDL only, no per-chunk ADT mesh)
└── Culled                         (beyond FarDistance + WdlDisable, or frustum-culled)
```

**Reuse**: `WorldTerrainTileData` is the existing type from `WowViewer.Core.Runtime.World.Terrain`. The renderer is a **consumer**.

**Reuse**: `WorldTerrainLodSelector.Select(...)` already returns a `WorldTerrainLodSelection` for per-chunk texture-layer LOD (the "fade to base layer" path). For the per-tile mesh LOD, the renderer is responsible — but the bucket choice is exposed in `RenderTerrainTile.LodBucket` so the runtime can audit it.

## 4. World Objects (M2 / WMO / MDX)

```text
RenderWorldObjectRef
├── InstanceId: int
├── Kind: WorldObjectKind          (M2 / Wmo / Mdx)
├── WorldObjectInstance: WorldObjectInstance   (from WowViewer.Core.Runtime.World)
├── Placement: WorldObjectPlacement
├── ModelPath: string
├── DistanceFromCamera: float
├── IsVisible: bool                (frustum-culled)
├── IsOccluded: bool                (occlusion-culled; optional, gated on quality)
├── LodLevel: ObjectLodLevel       (Near / Far / Culled)
├── DrawDistance: float            (per-instance; computed by the renderer from ObjectLodSettings)
└── RenderSubmission: RenderSubmission   (built by the renderer; M2 / WMO / MDX specific)
```

**Reuse**: `WorldObjectInstance`, `WorldObjectVisibilityCollector`, `WorldObjectVisibilityContext` are the runtime types the renderer consumes.

### 4a. WMO Pass Dispatch (Ghidra-confirmed 3.3.5)

The new WMO renderer conforms to the Ghidra-correctness-oracle in `docs/architecture/wmo-render-pass-architecture-2026-05-30.md`. The data model surfaces the dispatch state:

```text
WmoGroupRenderMode (enum, per group per frame)
├── Interior                    (group.flags & 0x48 == 0; MOCV vertex color, no dynamic lighting)
├── Exterior                    (group.flags & 0x48 != 0; dynamic lighting, sun direction applies)
└── Culled                      (group.flags & 0x80; or no render)

WmoGroupFlags (bitfield, read from CMapObjGroup.flags)
├── HasExteriorVisibility = 0x08
├── ExteriorRenderPath    = 0x40
├── FullExterior          = 0x48    (0x08 | 0x40)
├── NoRender              = 0x80
├── NoRenderNoCollide     = 0x88    (0x80 | 0x08)
├── HasLiquid             = 0x1000
└── AlwaysVisible         = 0x10000  (RenderAlways path; not portal-walked)

WmoBatchMomtFlags (bitfield, per batch from MOMT)
├── Lighting    = 0x01
├── Fog         = 0x02
├── Culling     = 0x04
├── TexAddr     = 0x08
├── WrapClamp   = 0x10
├── Emissive    = 0x20
└── WindowLit   = 0x40  (exterior sun override for interior windows)

WmoBatchRenderPass (enum, per batch per frame)
├── Group_Int                    (no texture, MOCV)
├── Group_Ext                    (no texture, dynamic lighting)
├── GroupColorTex_Int            (texture + MOCV, no dynamic light)
├── GroupColorTex_Ext            (texture + MOCV, dynamic light)
├── GroupColorTex                (dispatcher)
├── GroupLightTex                (per-batch material flags)
├── GroupLightmap                (lightmap only)
├── GroupLightmapTex_Int         (lightmap on tex1, no dynamic light)
├── GroupLightmapTex_Ext         (no lightmap, dynamic light on tex1)
├── GroupLightmapTex             (dispatcher)
├── GroupTex                     (texture only, white vertex color)
├── GroupBsp                     (debug)
├── GroupLiquid                  (interior/exterior water OR magma, dispatched by liquid type)
└── Culled                       (per-batch cull, group cull, or frustum cull)

WmoLiquidType (enum)
├── Water                        (type 0/4/8; interior water vs exterior water dispatched by group.mode)
├── Magma                        (type 2/3/6/7; separate render path)
└── None

WmoPortalWalkState (per interior render)
├── ActiveGroupIdx: int
├── FromGroupIdx: int
├── Depth: int                    (bounded by maxRLevel)
├── ClippedScreenRect: Rect      (post-portal-clip, in screen space)
└── Visited: bool                 (per-portal visited flag for this frame)

WmoInteriorFogState (when WmoGroupRenderMode == Interior)
├── Enabled: bool                 (DayNightGetInfo().intFog != 0 && this == camMapObj)
├── Start: float
├── End: float
└── Color: Vector4

WmoDayNightInfo (read-only consumer; lives in WowViewer.Core.Runtime.World; not a renderer concern)
├── WaterColor: Vector4           (exterior water color from light.WaterArray[3])
├── InteriorFog: WmoInteriorFogState
└── ExteriorFogColor: Vector4
```

The renderer dispatches `WmoBatchRenderPass` from `(WmoGroupRenderMode, WmoBatchMomtFlags, batch has texture?, batch has lightmap?)` per the Ghidra doc's pass table. The dispatch is the single most important conformance check for the new WMO renderer: every batch lands in exactly one pass, and the pass's MOCV/lighting/lightmap behavior matches the doc.

**Reuse**: `WowViewer.Core.Wmo` already exposes `WmoGroupFlags.cs`, `WmoGroupBatchSummary.cs`, `WmoMaterialSummary.cs`, etc. The new renderer is a *consumer* of these surfaces; it does not redefine them.

## 5. Liquid

```text
RenderLiquidTile
├── TileX: int
├── TileY: int
├── LiquidTileData: WorldLiquidTileData   (from WowViewer.Core.Runtime.World.Liquid)
├── DistanceFromCamera: float
├── LodBucket: WaterLodBucket
├── IsVisible: bool
└── RenderSubmission: RenderSubmission

WaterLodSettings
├── NearDistance: float
├── FarDistance: float
├── NearMeshResolution: int
├── FarMeshResolution: int
├── EnableReducedShaderAtFar: bool
└── EnableWindAtFar: bool

WaterLodBucket
├── Full
├── Reduced
└── Culled
```

**Reuse**: `WorldLiquidTileData` and `WorldLiquidLayerData` from `WowViewer.Core.Runtime.World.Liquid`.

## 6. Sky

```text
SkyState
├── SkyboxModelPath: string?
├── TopColor: Vector4
├── BottomColor: Vector4
├── TimeOfDay: float               (0..1)
├── IsOutdoor: bool
└── StarField: StarFieldSettings?
```

## 7. Fog

```text
FogState
├── Enabled: bool
├── StartDistance: float
├── EndDistance: float
├── Color: Vector4
└── Density: float                  (for volumetric variants; future)
```

## 8. Lighting (LOD-aware)

```text
LightLodSettings
├── MapObjLightLod: bool           (per-vertex dynamic light fade with distance)
├── MaxLights: int                 (cap on per-object light count; default: 4 or 8)
├── PerObjectLightSelectionPolicy: PerObjectLightSelectionPolicy

PerObjectLightSelectionPolicy (enum)
├── ClosestN
├── BrightestN
├── BrightestThenClosest
```

**Reuse**: `WorldFramePassCoordinator` is the runtime surface that will be extended with `MaxLights` and `mapObjLightLOD` (Phase 1). The renderer reads from it.

## 9. Texture / Mip

```text
MipSelectionSettings
├── Enabled: bool                   (default: true)
├── NearDistance: float             (full-res mip 0)
├── MidDistance: float              (mip 1..2)
├── FarDistance: float              (mip 3..N)
└── AnisoLevel: int                 (default: 4 or 8)

TextureCacheEntry (extends the existing WowViewer.Core.Renderer.Texture.TextureCache)
├── Handle: TextureHandle
├── SourcePath: string
├── Width: int
├── Height: int
├── MipCount: int
├── BytesPerPixel: int
├── Residency: TextureResidency     (Resident / Streaming / NotResident)
├── LastSelectedMip: int            (per-frame; updated by the renderer)
└── LastSelectionDistance: float
```

**Reuse**: `WowViewer.Core.Renderer.Texture.TextureCache` is extended, not replaced.

## 10. Per-Frame Diagnostic Surface

```text
PerFrameRenderStats (NEW; read by WowViewer.Core.Runtime.World.WorldRenderFrameStats)
├── DrawCallCount: int
├── InstanceCount: int
├── StateChangeCount: int
├── TextureBindCount: int
├── ShaderSwitchCount: int
├── TerrainTileCount: int
├── TerrainTilesByLodBucket: int[4]   (Full, Reduced, WdlOnly, Culled)
├── WorldObjectCount: int
├── WorldObjectsByLodLevel: int[3]    (Near, Far, Culled)
├── LiquidTileCount: int
├── LiquidTilesByLodBucket: int[3]    (Full, Reduced, Culled)
├── ActiveLightCount: int             (capped by MaxLights)
├── MipSelectedDistribution: int[N]    (per mip level)
├── TextureBandwidthBytes: long       (estimated; Phase 5)
├── FrameCpuTimeMs: double
├── FrameGpuTimeMs: double             (if available; Phase 5)
└── Backend: RenderBackendKind         (OpenGL, Vulkan[future])
```

The renderer writes to `PerFrameRenderStats` each frame; the runtime reads it and exposes it to the host (and to the validation harness in Phase 7).

## 11. Render Backend (Backend-Neutral Contract)

```text
RenderBackendKind (enum)
├── OpenGL
└── Vulkan   (future)

RenderBackend (interface)
├── Initialize(viewport: Viewport): void
├── BeginFrame(scene: RenderScene, variant: RenderVariant): PerFrameRenderStats
├── Submit(terrain: IReadOnlyList<RenderTerrainTile>, liquids: ..., objects: ...): void
├── EndFrame(): PerFrameRenderStats
├── Resize(width: int, height: int): void
├── Dispose(): void
└── Stats: PerFrameRenderStats   (read-only after EndFrame)

OpenGLRenderBackend : RenderBackend      (v0.5.0-dev; Phase 1+)
VulkanRenderBackend : RenderBackend      (follow-on spec; not this spec)
```

## 12. Render Resources (Per-Tile / Per-Frame)

```text
RenderResources (interface)
├── Terrain: ITerrainResourceCache
├── Wmo: IWmoResourceCache
├── M2: IM2ResourceCache
├── Mdx: IMdxResourceCache
├── Liquid: ILiquidResourceCache
├── Sky: ISkyResourceCache
├── Texture: TextureCache
└── Shaders: ShaderCache

ITerrainResourceCache
├── GetOrBuildMesh(tile: RenderTerrainTile, lod: TerrainLodBucket): RenderTerrainMesh
├── Evict(tileX: int, tileY: int): void
└── EvictAll(): void

RenderTerrainMesh
├── Vbo: BufferHandle
├── Ibo: BufferHandle
├── VertexCount: int
├── IndexCount: int
├── Ubo: BufferHandle?             (per-mesh UBO if needed)
└── LodBucket: TerrainLodBucket
```

## 13. AOI

```text
AoiBounds
├── MinTileX: int
├── MaxTileX: int
├── MinTileY: int
├── MaxTileY: int
└── WorldSpaceBounds: (Vector3 min, Vector3 max)
```

## 14. Error / Edge-Case States

```text
RendererError (enum)
├── None
├── DeviceLost
├── OutOfMemory
├── ShaderCompilationFailed
├── TextureUploadFailed
├── ResourceNotResident
├── WdlMissingForFarHorizon
└── MapSwitchInProgress
```

The renderer exposes `LastError: RendererError?` and the host must check it after each `EndFrame`.

## 15. Lifecycle States (Map Switch, Loading, etc.)

```text
RendererLifecycleState (enum)
├── Uninitialized
├── Idle                          (no map loaded)
├── Loading                       (between BeginMap and first EndFrame)
├── Rendering                     (steady state)
├── MapSwitching                  (between EndMap and BeginMap)
├── Unloading                     (between EndMap and Idle)
└── Disposed
```

`BeginMap(mapId, profileId)` and `EndMap()` are the explicit transitions. During `Loading` / `MapSwitching` / `Unloading` the renderer is in a no-op or default-scene state and MUST NOT crash.

---

## 16. Summary of Where Each Type Lives

| Type | Lives in | Phase |
|---|---|---|
| `RenderScene` | `WowViewer.Core.Renderer.Contracts.RenderScene` | Phase 1 |
| `RenderVariant` (extended) | `WowViewer.Core.Renderer.Scene.RenderVariant` | Phase 1 |
| `RenderTerrainTile`, `TerrainLodSettings`, `TerrainLodBucket` | `WowViewer.Core.Renderer.Terrain` | Phase 1, 2 |
| `RenderWorldObjectRef`, `ObjectLodLevel` | `WowViewer.Core.Renderer.Contracts` | Phase 1, 3 |
| `RenderLiquidTile`, `WaterLodSettings`, `WaterLodBucket` | `WowViewer.Core.Renderer.Liquid` | Phase 1, 4 |
| `SkyState` | `WowViewer.Core.Renderer.Sky` | Phase 1, 4 |
| `FogState` | `WowViewer.Core.Renderer.Scene` | Phase 1 |
| `LightLodSettings`, `PerObjectLightSelectionPolicy` | `WowViewer.Core.Renderer.Scene` (read by `WorldFramePassCoordinator`) | Phase 1 |
| `MipSelectionSettings`, `TextureCacheEntry` (extended) | `WowViewer.Core.Renderer.Texture` | Phase 1, 5 |
| `PerFrameRenderStats` | `WowViewer.Core.Renderer.Diagnostics` | Phase 1, 5 |
| `RenderBackend`, `RenderBackendKind`, `OpenGLRenderBackend` | `WowViewer.Core.Renderer.Contracts` + `WowViewer.Core.Renderer.OpenGL` | Phase 1 |
| `RenderResources`, `ITerrainResourceCache`, `RenderTerrainMesh` | `WowViewer.Core.Renderer.Contracts` + `WowViewer.Core.Renderer.Terrain` | Phase 1, 2 |
| `AoiBounds` | `WowViewer.Core.Renderer.Scene` | Phase 1 |
| `RendererError`, `RendererLifecycleState` | `WowViewer.Core.Renderer` (top-level) | Phase 1 |

---

*End of data model. Next: `contracts/*` and `quickstart.md`.*
