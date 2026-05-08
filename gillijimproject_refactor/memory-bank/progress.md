# PROGRESS — V14 Branch (V11 Reset)

## POSITION
- V10 pipeline: dead. Two-stage nonsense with broken archive path.
- V11: v9 proven pipeline + ConvNeXt backbone + MCAL/MCLY multi-task + proper training infra.
- V14: wow-viewer library completeness effort, modular terrain model system, and a now-proven harvest/tensor-pack lane from Alpha `0.5.x` through `4.0.0`.

## VALIDATED
| What | Status |
|------|--------|
| MpqArchiveCatalog probe fix | DONE — break→continue, 256 limit |
| MCAL/MCLY in v9 BuildDirectCacheEntry | DONE |
| V11 model forward pass | DONE — 35.5M, 6 output heads |
| V11 training loop (GPU) | DONE — AMP, EMA, cosine, uncertainty loss |
| V11 inference + OBJ export | DONE |
| Channel layout audit | DONE — 26ch, shadow removed, MCCV at 3x dropout |
| Cache memory (LRU 2GB cap) | DONE |
| Zero-samples / empty vocab guards | DONE |
| wow-viewer Phase A: Terrain type system | DONE |
| wow-viewer: NativeMpqService port | DONE — pure C#, no StormLib |
| wow-viewer: AlphaTileData.ToTileLoadResult | DONE |
| wow-viewer: TerrainTileTensorPack.ToTileLoadResult | DONE |
| wow-viewer: Harvest tool extract-unified | DONE |
| wow-viewer: MDDF/MODF model name resolution | DONE |
| wow-viewer: Harvest --export-placements | DONE |
| wow-viewer: Alpha object/precise masks | DONE |
| wow-viewer: Alpha shadow residual mask | DONE |
| **Alpha 0.5.3/0.5.5: all 11 signals** | **DONE** |
| **Retail 3.x: all 11 signals** | **DONE** |
| **Cata 4.0.0: 12 signals** | **DONE** — + MCCV vertex colors |
| **0.7.0 extraction** | **DONE** — AdtProfile0703694 |
| **Alpha object mask projection** | **DONE** |
| **Minimap via Md5TranslateResolver** | **DONE** |
| **All staged clients pass** | **DONE** — 0.5.3, 0.5.5, 0.7.0, 3.0.1, 3.3.5, 4.0.0 |
| **Placement flat arrays in NPZ** | **DONE** |
| **Placement model names resolved** | **DONE** |
| **BuildKey provenance** | **DONE** |
| **WL* loose-file liquid fallback** | **DONE** |
| **Coordinate fixes** | **DONE** |
| **Phase C: AlphaToLk writer infrastructure** | **DONE** — WdlWriter, LkWdtWriter, LkAdtWriter, AlphaToLkConverter |
| **Phase C: AlphaToLk CLI command** | **DONE** — convert-alpha-to-lk in WowViewer.Tool.Converter |
| **Phase C: AlphaToLk real-data validation** | **DONE** — 755/755 Azeroth (0.5.5), 972/972 Kalimdor, 256/256 EmeraldDream, 25/25 PVPZone01, 25/25 Shadowfang |

## IN PROGRESS
| What | Status |
|------|--------|
| Multi-client full shard dataset prep | SWITCHED TO HARVEST PATH — use `WowViewer.Tool.Harvest harvest-map-mpq` on staged clients, not converter `dataset-scan` manifests |
| Phase C: AlphaToLk AreaID crosswalk | NOT YET — `AreaIdMapper` exists in `WowViewer.Core.IO/Dbc/`, not yet wired to converter |
| Phase C: LkToAlpha converter | NOT PORTED |
| Phase C: Mdx↔M2 converters | NOT PORTED |
| Phase C: Wmo v14↔v17 converters | NOT PORTED |

## NOT YET
- Explicit Alpha 0.6.0 split ADT validation via `AdtProfile060070Baseline`
- Full extraction run on 6 staged game clients (800-1500+ shards) via `harvest-map-mpq` into `wow-viewer/output/datasets/`
- Production training run (300 epochs)
- Model evaluation on held-out tiles
- DBC/DB2 metadata enrichment (WorldSafeLocs, AreaTable, GroundEffects, LiquidType)
- MCRF per-chunk reference arrays
- MODF doodadSet/nameSet resolution
- PM4 masks for development map build (4.0.0.12304 loose files)
- Development map extraction pipeline (wow-viewer/test_data/original_development)

## BUGS FIXED IN ALPHAtoLK VALIDATION (2026-05-08)
1. **ChunkedFileReader crash on monolithic WDTs** — replaced with `AlphaWdtReader.ReadExistingTiles()`
2. **MHDR/MCIN empty payload** — wrote declared-size chunk headers with 0 data bytes; fixed by writing pre-allocated zero arrays
3. **MPHD size mismatch** — wrote 9 uint32s (36 bytes) but declared 32 bytes; fixed by removing extra `Write(0u)`
4. **MAIN index formula** — `tileX * 64 + tileY` was wrong; fixed to `tileY * 64 + tileX` (row-major with y as row). This caused 420/755 Azeroth tiles to fail before the fix.

## WORKFLOW CORRECTION (2026-05-08)
- Do not route full multi-client dataset generation through `WowViewer.Tool.Converter dataset-scan` / `dataset-audit` / `dataset-curate` / `dataset-build-cache`.
- Those commands remain useful as legacy manifest/audit helpers, but they are not the canonical full-signal shard builder and can miss newer harvest/tensor-pack coverage and metrics.
- Use `WowViewer.Tool.Harvest harvest-map-mpq` for staged archive-backed clients and `harvest-map` for loose on-disk maps.
- Default real dataset outputs belong under `wow-viewer/output/datasets/`, not repo-root `output/tmp/`.
