# System Patterns — wow-viewer

Last verified against code: 2026-09-23 (spec reconciliation fact check).

## FourCC Handling (CRITICAL)

```
READ:  Reverse on disk → Forward in memory (XETM → MTEX)
WRITE: Forward in memory → Reverse on disk (MTEX → XETM)
```

## Terrain Adapter Pattern

`ITerrainAdapter` implementations in `src/viewer/WoWViewer/Terrain/`:

- **AlphaTerrainAdapter** — Alpha 0.5.3 monolithic WDT (all tiles in one file). A separate core-side
  `AlphaTerrainAdapter` also exists in `src/core/WowViewer.Core.IO/Maps/`.
- **StandardTerrainAdapter** — split ADTs from 0.6.0 onward (LK, MoP split files, FileDataID-era MAID
  maps via CASC).
- **AhdrTerrainAdapter** — a folder of DAT v22/v23/v26 (AHDR-family) files; no WDT, each file placed by
  its ALOC chunk or, for v22/v23, the `XX_YY` file-name tile coordinates.
- **RosettaDatastoreTerrainAdapter** — streams multi-version Rosetta maps from the unified Zarr v3 datastore (no loose ADT/WDT files).

All produce `TileLoadResult` with `TerrainChunkData` + placements. Phase/overlay composition is
per-channel (`PhaseChunkMerger`, `PhaseLayers`), and a DAT folder can be a composition layer through
the `dat:<folder>` locator (`DatLayerSource`).

### Liquid Pipeline

- **MCLQ** (per-chunk 0.5.3/0.6.0): Extracted in terrain adapter, per-vertex heights are absolute world Z
- **MH2O** (per-tile 3.3.5+): Parsed via MHDR offset, only when no MCLQ found; vertex format resolved
  through the LiquidObject → LiquidType → LiquidMaterial DBC chain (`LiquidVertexFormatChain`)
- **MLIQ** (WMO groups): Parsed in WmoRenderer

### AOI Streaming

- `TerrainManager` (active viewer project) handles tile load/unload based on camera position
- Background reads throttled by `SemaphoreSlim(4)` (`MaxConcurrentMpqReads`); persistent `_tileCache`
- The legacy MdxViewer-era bootstrap code is quarantined in `src/viewer/WowViewer.App.Defunct/`.

## Data Sources

`IDataSource` (`src/viewer/WoWViewer/DataSources/`) abstracts MPQ archives (`MpqArchiveCatalog`,
`AlphaArchiveReader`; `NativeMpqService` is an alternative MPQ path) and CASC installs
(`CascDataSource` over `Core.IO/Casc/` and vendored TACTSharp), including FileDataID-addressed reads.

## Scene Lighting

`SceneLightManager` collects `SceneLight`s from `ISceneLightEmitter`s (WMO `MOLT`, WMO-internal and
MDX/M2 doodad emitters) plus the frame's ambient/sun (`SceneAmbientLight`, from `TerrainLighting`).
WMO, terrain and unbatched doodad passes query nearby lights per draw. GPU-instanced batches cannot
carry per-placement lights (Epic 249 R-10/R-11).

## PM4 Cache Architecture

Two layers for PM4 overlay data:
1. **In-memory** (`Pm4PerFileCache`, cap 256, LRU-soft): mirrors per-file decoded payloads for the session
2. **On-disk** (`Pm4PerFileCacheService`, magic PM4F, **version 9**): one gzip blob per PM4 file at
   `output/cache/pm4-overlay/{segment}/{map}/files/`, rooted at the app base directory

Read order: in-memory → on-disk → fresh decode. Cam window signatures still use per-window SHA-256 as fast path.

## M2 Pipeline

Model dispatch → `M2ModelReaderDispatcher.ReadDetailed` → era detection (MDLX chunked, MD20 1.x
`0x100`–`0x107` via `M2Era100ModelReader`, era 1121, MD20 3.x classic, MD21 chunked) → reader per era.

Animation: sequence/alias handling in `M2ExternalAnimationRuntime` → `M2TrackSampler` → `M2BonePoseEvaluator`.

The Spec 053 pose-farm export pipeline (BVH / pose-clip builders) was never built; only the
`Core.Anim` loader layer exists.

## Coordinate System (WoW → Renderer)

- WoW: Right-handed, X=North, Y=West, Z=Up, D3D CW winding
- Renderer: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`, `rendererZ = wowZ`
- Reverse triangle winding at upload (CW → CCW for OpenGL)
- 180° Z rotation in all placement transforms

## Spec Kit Workflow

1. `$speckit-specify` → `specs/NNN-name/spec.md`
2. `$speckit-plan` → `specs/NNN-name/plan.md`
3. `$speckit-tasks` → `specs/NNN-name/tasks.md`
4. `$speckit-implement` → execute tasks, one phase at a time
5. Validate each phase against staged game client data before next phase

Since 2026-09-23, open work lives in Epics 248–254; backlog items are scheduled only after operator
triage (`specs/TRIAGE.md`).
