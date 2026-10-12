# wow_classic_beta 1.60.1.69876 (12.0-based client): assets, maps, DB2 (2026-09-16)

Target build: `wow_classic_beta` 1.60.1.69876, build config `e7fab7248766e9e7daddb3b6083c9c3c`, opened from a local CASC
install. All numbers come from `inspect casc …` commands against that install (scripts under `scripts/`).

## Inputs refreshed this session

| Input | Before | After |
|---|---|---|
| `libs/wowdev/WoWDBDefs` | `456d294` | `2b8d984` "Merge 1.60.1.69893" (definitions list `BUILD 1.60.1.69876, 1.60.1.69893` for Map, AreaTable, LiquidType, Light) |
| `libs/wowdev/wow-listfile` | `8eddaa1` | `50d13bb` |
| `%LOCALAPPDATA%\WoWViewer\community-listfile-withcapitals.csv` | 150,419,239 bytes | 152,651,163 bytes (latest release) |

The viewer's listfile resolution now picks the most recently written candidate. Previously a stale bundled copy
(`gillijimproject_refactor/test_data`, October 2025) took priority over the fresh download.

## Maps

- 58 WDTs from the listfile are present and local in the beta build (1016 listfile WDTs checked).
- `Map.db2` decodes: 73 rows, including ID 0 `Azeroth` ("Eastern Kingdoms") and ID 451 `development` ("Development Land").
- `AreaTable` 1372 rows, `LiquidType` 51, `Light` 625 (`casc db2`).

### development.wdt (FileDataID 857684), `casc map-survey`

| Measurement | Value |
|---|---|
| MPHD flags | `0x248` |
| Top-level chunks | MVER, MPHD(32), MAIN(32768), MAID(131072) |
| MAID tiles | 82; root/obj0/tex0 readable for all 82 |
| tex0 texture tables | MTEX in 0 files; MDID and MHID in 82 |
| MCLY layers per MCNK | 0:9247, 1:2035, 2:2088, 3:3647, 4:3975 (max 4) |
| obj0 | MDDF, MODF, MCNK only (no MMDX/MMID/MWMO/MWID) |
| MDDF flags | `0x40`: 8927, `0x240`: 2268 (all have 0x40 = FileDataID name) |
| MODF flags | `0xC`: 147 (all have 0x8 = FileDataID name) |
| MDID textures | 65 distinct: 63 NotLocal, 2 Ok |
| MDDF models (0x40) | 541 distinct: 541 Ok |
| MODF WMOs (0x8) | 72 distinct: 72 Ok |

### Azeroth.wdt (775971) and tile 31_40

- MPHD `0x3CA`; MAIN, MAID (736 tiles), and an additional `MAI2` chunk (131072 bytes, not yet interpreted).
- Root `Azeroth_31_40.adt`: MCNK sub-chunks MCVT, MCCV, MCNR, MCSE. tex0: MAMP, MDID, MHID, MCNK(MCLY, MCAL), max 4 layers. obj0: MDDF, MODF, MCNK.

## Models and textures

- `DUSKWOODCHAPEL.WMO` (v17): no MOTX/MODN. GFID 2/2 groups, 11/11 MOMT texture FileDataIDs and 35/35 MODI doodads resolve and read.
- `SILVERPINETREE03.M2`: MD21 (M2 version 272) with SFID [508760] and TXID [203980, 399526]. The skin reads locally; both textures are listed but not local.
- Local BLP data is absent for most textures in this install. CDN fill for the same build returned FileDataID 399526 as a 2048×2048 DXT5 BLP2 (5,593,604 bytes). The wow_classic_era 1.15.9 version of the related texture 203980 is 44,876 bytes.

## Viewer changes this slice

- `StandardTerrainAdapter`: registers MAID tile files as data-source aliases; tex0 texture names from MDID when MTEX is absent; MDDF 0x40 / MODF 0x8 names via FileDataID.
- `IDataSource.RegisterFileDataIdAlias` (default no-op); `CascDataSource` resolves `fdid:` → alias → listfile.
- CASC load sets the build to the newest product version and loads AreaTable, replaceable textures and Map.db2 discovery (each guarded).

## Not verified here

Rendering (operator visual check). MAI2 meaning. MHID/MAMP use in the terrain shader. Terrain texture decoding at 2048².
