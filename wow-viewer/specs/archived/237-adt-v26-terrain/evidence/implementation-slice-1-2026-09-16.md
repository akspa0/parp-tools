# Implementation slice 1: CASC-backed DAT v26 terrain in the viewer (2026-09-16)

## What landed

| Piece | File | Spec task |
|---|---|---|
| TACTSharp submodule (wowdev, `cc5bf85`), central package management opt-out | `libs/wowdev/TACTSharp`, `Directory.Packages.props` | 238 T001/T002 |
| Local CASC storage (per product; offline; `NotPresent` / `NotLocal` / `KeyUnavailable` / `Failed`) | `src/core/WowViewer.Core.IO/Casc/CascStorage.cs` | 238 T007 (as `CascReadStatus`), T013, T014 (standalone class, not yet `IArchiveCatalog`) |
| Community listfile (`id;path`) loader | `src/core/WowViewer.Core.IO/Casc/CommunityListfile.cs` | 238 T011 |
| CLI `casc products`, `casc read`, `casc exists` | `tools/inspect/.../CascCommandSupport.cs` | 238 T016 (partial) |
| DAT v26 model + reader (never throws; diagnostics) | `src/core/WowViewer.Core/Maps/AdtAhdr/AdtAhdrTile.cs`, `src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs` | 237 T010, T045 |
| Slicer (145-vertex chunks, height-derived normals) | `src/core/WowViewer.Core.IO/Maps/AdtAhdrTileSlicer.cs` | 237 T046 (known-answer unit test still open) |
| CLI `adt-ahdr check` | `tools/inspect/.../AdtAhdrCommandSupport.cs` | 237 T047 (as CLI check, not yet an xUnit test) |
| Viewer CASC data source (products newest-version first, fall-through on missing local data) | `src/viewer/WoWViewer/DataSources/CascDataSource.cs` | 238 T034 |
| Viewer DAT v26 terrain adapter (ALOC placement; heights, normals, texture layers, alpha) | `src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs` | 237 T048, T036 (partial), T051 (textures only) |
| File menu: "Open CASC Install (local)...", "Open DAT v26 Terrain Folder..." | `src/viewer/WoWViewer/ViewerApp_CascAhdr.cs`, `ViewerApp.cs` | 237 T049, 238 T035 (local only) |

Not yet: object placements (ACDO frame unverified), vertex colours (ACVT order unverified), shadows (ASHD bit layout
unverified), DB2 tables for CASC builds (Spec 239), the detector change in `WowFileDetector` (237 T004–T009).

## Measurements

Install: a local CASC install with two products in `.build.info`:

| Product | Version | Build config |
|---|---|---|
| `wow_classic_era` | 1.15.9.69722 | `3645f0fe9dc5215ea90eaf7cdb7379ce` |
| `wow_classic_beta` | 1.60.1.69876 | `e7fab7248766e9e7daddb3b6083c9c3c` |

- `wow_classic_beta` opens locally in 759 ms with no network access (`TryCDN = false`).
- **Byte identity**: FileDataID 6893600 read from `wow_classic_beta` is 533,261 bytes, SHA-256 `AC99569C…F199`, identical to `test_data/v22_adts/unknown/6893600`. The DAT v26 files ship in build 1.60.1.69876.
- **Asset availability** for the 318 distinct names in `ATEX`/`ADOO` (`casc exists`, vendored listfile parts):

| Product | `.BLP` (33) | `.M2` (266) | `.WMO` (19) |
|---|---|---|---|
| `wow_classic_beta` 1.60.1 | NotLocal 33 | Ok 266 | Ok 19 |
| `wow_classic_era` 1.15.9 | Ok 30, NotPresent 3 | Ok 231, NotPresent 35 | Ok 17, NotPresent 2 |

  "NotLocal" means the root manifest lists the FileDataID but its data is not in the local install.
- **Reader/slicer on the corpus** (`adt-ahdr check --root test_data\v22_adts\unknown`): 700 files, all revision 26, 699 unique tiles (1 duplicate: the `.adt` copy), 0 files with diagnostics, 0 ACNK index mismatches, slicer seams 673/673 X-neighbour pairs and 670/670 Y-neighbour pairs exact.

## Axis mapping (unverified against the game world)

The renderer's chunk rows run along its tile X (`TerrainMeshBuilder`, `RosettaDatastoreTerrainAdapter`). DAT v26 rows run along
ALOC tile Y. The adapter places renderer TileX = ALOC tile Y and renderer TileY = ALOC tile X, so no tile is transposed and
seams stay exact. Whether the assembled map is mirrored relative to the game world is not yet known.

## Operator check (outstanding)

1. File → Open CASC Install (local)... → the install folder.
2. File → Open DAT v26 Terrain Folder... → `test_data\v22_adts\unknown`.
3. Confirm: no cracks at tile edges; terrain shape plausible (no spikes, which would indicate a wrong inner-grid order); textures blend.
