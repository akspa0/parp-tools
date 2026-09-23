# Phase 1 Data Model — Modern-to-Legacy Map Conversion

Date: 2026-09-18

## Entities

### SourceMap
A modern map selected for conversion.

| Field | Type | Notes |
|---|---|---|
| `MapName` | string | e.g. `Azeroth` |
| `MapId` | uint? | modern map id when known |
| `WdtPath` | string | resolved modern WDT path |
| `OccupiedTiles` | IReadOnlyList<(int X, int Y)> | tiles with terrain |
| `BuildFingerprint` | string | client build identity (Constitution III) |

**Validation**: `WdtPath` must resolve through the modern readers; a map with zero occupied tiles is
reported and skipped, not failed.

### LayerStack
The ordered texture layers + blend weights for one chunk.

| Field | Type | Notes |
|---|---|---|
| `Layers` | IReadOnlyList<LayerEntry> | base first, overlays in file order |
| `AmapWeights` | float[,,]? | per-vertex blend weights (modern) |

`LayerEntry`: `TextureFileDataId` (uint), `TexturePath` (string?), `HeightRange` (float,float),
`Flags` (uint).

**Validation**: at least one layer; `AmapWeights` shape must match the chunk vertex grid when present.

### MergePolicy
The deterministic rule mapping a source `LayerStack` onto a target capacity.

| Field | Type | Notes |
|---|---|---|
| `TargetCapacity` | int | 4 for LK v18 and Alpha 0.5.3 |
| `Ranking` | enum | `CoverageDescending` (default) |
| `TieBreak` | enum | `LayerIndexAscending` |

**State transition**: `SourceStack → Rank → Keep(base + top capacity-1) → Fold(dropped into nearest
kept) → MergedStack + MergeRecord`.

### MergeRecord (per tile)
The FR-003 report for one chunk.

| Field | Type | Notes |
|---|---|---|
| `TileX`, `TileY`, `ChunkX`, `ChunkY` | int | location |
| `KeptLayers` | IReadOnlyList<int> | source layer indices kept |
| `MergedLayers` | IReadOnlyList<int> | folded into a kept layer |
| `DroppedLayers` | IReadOnlyList<int> | dropped outright |
| `UnresolvedTextures` | IReadOnlyList<uint> | FileDataIds that did not resolve |

### ConversionRun
A batch of source maps for one target.

| Field | Type | Notes |
|---|---|---|
| `Target` | MapConversionTargetFormat | `LkAdtV18` or `AlphaWdt053` |
| `OutputRoot` | string | generated project folder |
| `Results` | IReadOnlyList<MapConversionResult> | one per map |

`MapConversionResult`: `MapName`, `Status` (`Succeeded`/`Failed`/`Skipped`), `TilesConverted`,
`TotalTiles`, `ElapsedMs`, `Error` (string?), `MergeRecords` (per tile), `OutputPath`.

**Validation**: one map's failure must not abort the run (FR-004); `OutputRoot` must never be inside a
client root (FR-006).

### AssetManifest
The FR-007 record of referenced assets.

| Field | Type | Notes |
|---|---|---|
| `Included` | IReadOnlyList<AssetEntry> | copied beside the output |
| `Unresolved` | IReadOnlyList<UnresolvedAsset> | with a reason |

`AssetEntry`: `Kind` (`Texture`/`Model`/`Minimap`), `SourcePath`, `OutputRelativePath`, `Sha256`.
`UnresolvedAsset`: `Kind`, `Reference` (FileDataId or path), `Reason`.

## Relationships

```text
ConversionRun 1 ── * MapConversionResult 1 ── * MergeRecord
SourceMap     1 ── * LayerStack (per chunk)
LayerStack    * ── 1 MergePolicy → MergedStack + MergeRecord
ConversionRun 0..1 ── 1 AssetManifest
```

## Determinism rules

- All collections iterated in a stable sort order (map name, tile X/Y, layer index, asset path).
- No dependence on dictionary enumeration order.
- Provenance carries a content hash so reruns are byte-identical (SC-004).
