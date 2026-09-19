# Spec 237 Evidence — DAT GLB Export + Historical Data Record

Date: 2026-09-19

## Operator request

"if we could get GLB export working for these weird DAT files, then we'd be golden. the synthesized
minimap thing does not work with them, either, and i'd like to get proper record of the data in these
files, out, for historical reasons."

## 1. GLB export for DAT folders

**Root cause**: the terrain GLB export was gated on `_dataSource != null`, but a DAT folder can be
opened with no client data source (the load message says "no data source open; textures will be
missing"). The menu item was also gated on `_renderer != null` (a standalone-model flag), so it was
disabled for terrain.

**Fix**:
- [`MapGlbExporter`](../../../src/viewer/WoWViewer/Export/MapGlbExporter.cs): `ExportTile` and its
  helpers now take `IDataSource?`; placements and textures are skipped when it is null, while the
  terrain mesh (built from the tile's own chunks) still exports.
- [`ViewerApp`](../../../src/viewer/WoWViewer/ViewerApp.cs): the GLB menu is enabled for
  `_renderer != null || _terrainManager != null`, and the terrain branch no longer requires a data
  source.

## 2. Historical data record — `dump-dat`

New harvest command [`dump-dat`](../../../tools/harvest/WowViewer.Tool.Harvest/Program.cs:1577) writes
a JSON record of every AHDR-family DAT file in a folder (or a single file): MVER/AHDR version, grid
and chunk dimensions, ALOC, tile X/Y, texture names, model names, per-chunk layer/object counts,
ADST model-file references, and diagnostics.

**Produced record**: [`output/dat-records/kalimdor-lost-isles.json`](../../../output/dat-records/kalimdor-lost-isles.json)
(4 records — the two `.dat` files and their byte-identical `.error` companions).

**What the record shows** — these are **Lost Isles (`expansion03`)** terrain files, confirming the
operator's "deleted region from the early Lost Isles pre-alpha era":

| File | MVER | Grid | Chunks | Layers | Textures |
|---|---|---|---|---|---|
| `area_51_31.dat` | 23 | 129×129 | 256 | 310 | `Tileset\Generic\Red.blp`, `expansion03\lostisles\li_dirtb/li_sandb/li_dirtf.blp` |
| `area_51_32.dat` | 23 | 129×129 | 256 | 614 | `expansion03\lostisles\li_dirtb/li_sandb/li_dirte/li_dirtf/li_grassb/li_grassc.blp`, `Tileset\Generic\Black.blp` |

No objects, no ADST references, no ALOC (hence the filename tile-location fallback).

## Verification

| Command | Exit | Output |
|---|---:|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` | 0 | `Build succeeded. 0 Error(s)` |
| `dotnet build wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug` | 0 | `Build succeeded. 0 Error(s)` |
| `WowViewer.Tool.Harvest.dll dump-dat --input "E:\WC2\wrath\World\Maps\Kalimdor" --output ...` | 0 | `Wrote 4 DAT record(s) to ...kalimdor-lost-isles.json` |

## Proof boundary

Source + build + real-file dump proof. **No runtime GLB-load or visual proof is claimed** — the
operator should export a GLB from the loaded DAT folder and open it in a viewer.

## Still open

**Synthesized minimap for DAT files.** The harvest's `synthetic-minimap` reads a client root through
`AdtTensorPackBuilder` (MCNK/MCNR), which has no AHDR-family support. Making it work needs a new
AHDR→`TerrainTileTensorPack` builder (heights via `AdtAhdrTileSlicer.SliceHeights`, normals via
`SliceStoredNormals`, layers via `AdtAhdrAlpha.WeightsToSequentialAlpha`, textures from
`tile.TextureNames`) plus a DAT-folder input mode. That is the next bounded step.
