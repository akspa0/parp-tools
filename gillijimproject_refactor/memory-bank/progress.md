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
| **Alpha 0.5.3/0.5.5: all 11 signals** | **DONE** — height, normals, MCAL/MCLY, MCSH, holes, objects, shadow_residual, minimap, WL liquid (when MCLQ missing) |
| **Retail 3.x: all 11 signals** | **DONE** — + object/precise masks, texture_names, shadow_residual |
| **Cata 4.0.0: 12 signals** | **DONE** — + MCCV vertex colors |
| **0.7.0 extraction** | **DONE** — AdtProfile0703694 |
| **Alpha object mask projection** | **DONE** — tile-relative, rendererX→col, rendererY→row |
| **Minimap via Md5TranslateResolver** | **DONE** — .trs for retail, .txt for Alpha |
| **All staged clients pass** | **DONE** — 0.5.3, 0.5.5, 0.7.0, 3.0.1, 3.3.5, 4.0.0 |

## NOT YET
- Explicit Alpha 0.6.0 split ADT validation via `AdtProfile060070Baseline`
- Full extraction run on 6 game clients (800-1500 shards)
- Production training run (300 epochs)
- Model evaluation on held-out tiles
- OBJ reconstruction quality check
- WL file support for retail path (already in Alpha, retail already has via disk)
- PM4 masks for development map build (4.0.0.12304 loose files)
