# System Patterns — wow-viewer

## FourCC Handling (CRITICAL)

```
READ:  Reverse on disk → Forward in memory (XETM → MTEX)
WRITE: Forward in memory → Reverse on disk (MTEX → XETM)
```

## Terrain Adapter Pattern

Two implementations for different WDT/ADT formats:
- **AlphaTerrainAdapter** — Alpha 0.5.3 monolithic WDT (all tiles in one file)
- **StandardTerrainAdapter** — 0.6.0 / 3.3.5 split ADTs (per-tile files)

Both produce `TileLoadResult` with `TerrainChunkData` + placements.

### Liquid Pipeline
- **MCLQ** (per-chunk 0.5.3/0.6.0): Extracted in terrain adapter, per-vertex heights are absolute world Z
- **MH2O** (per-tile 3.3.5): Parsed via MHDR offset, only when no MCLQ found
- **MLIQ** (WMO groups): Parsed in WmoRenderer

### AOI Streaming (legacy MdxViewer path)
- `TerrainManager` handles tile load/unload based on camera position
- Background thread pool, throttled by `SemaphoreSlim(4)`
- Persistent `_tileCache`

## PM4 Cache Architecture (spec 054)

Two layers for PM4 overlay data:
1. **In-memory** (`Pm4PerFileCache`, cap 256, LRU-soft): Mirrors per-file decoded payloads for the current session
2. **On-disk** (`Pm4PerFileCacheService`, magic PM4F, version 8): One gzip blob per PM4 file at `output/cache/pm4-overlay/{segment}/{map}/files/`

Read order: in-memory → on-disk → fresh decode. Cam window signatures still use per-window SHA-256 as fast path.

## M2 Pipeline

Model dispatch → `M2ModelReaderDispatcher.ReadDetailed` → era-detection (MDLX chunked, MD20-1x era1121, MD20-3x classic) → reader per era.

Animation: `M2SequenceAliasResolver` → `M2TrackSampler` → `M2BonePoseEvaluator`.

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

## M2 Animation Pose Farm Pipeline (spec 053)

```
M2ModelReaderDispatcher.ReadDetailed
    → M2SequenceAliasResolver (alias chain → terminal sequence)
    → M2BoneTrackStreamExtractor (walk M2 track defs → per-bone TRS keyframes)
    → BvhDocumentBuilder (M2 bone hierarchy → BVH joint tree)
    → BvhDocumentWriter (.bvh file per sequence)
    → PoseClipBuilder (Mixamo-normalized .poseclip.json)
    → PoseLibraryIndexBuilder (batch top-level index)
```
