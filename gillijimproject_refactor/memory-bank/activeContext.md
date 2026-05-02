# ACTIVE CONTEXT — V11 CLEAN SLATE

## BRANCH
`v0.4.9` at `ced5899`. V10 pipeline deceased. V11 is the real deal.

## WHAT FIXED
- **MpqArchiveCatalog probe bug** — `FindFileInArchive` + `TryFindBlockByName` now skip past empty hash slots with 256-probe limit. Was `break` on empty. This was THE hang.
- **MCAL/MCLY in v9 `dataset-build-cache`** — v9 pipeline now produces NPZ shards with `mcal_alpha_pack_256`, `mcly_layer_mask`, `mcly_texture_ids`. No temp files. Zero disk writes.

## V11 TRAINER (`train_v11.py`)
- **Backbone:** ConvNeXt V2 Tiny (28.6M) from `timm`. LayerNorm, batch-size agnostic.
- **Total:** 35.5M params, fits batch 32 in 8GB, batch 64+ in 17GB.
- **Inputs:** 26 channels — minimap, MCAL alpha, normals, MCCV (3x dropout), coarse height, liquid, objects, PM4, hole, luma, gradient, range.
- **Outputs:** height_17/65/257 + MCAL alpha (4ch) + MCLY class + hole binary.
- **Loss:** Uncertainty-weighted sigmas per task. Automatic balancing.
- **Extras:** EMA, cosine+warmup, gradient clip, signal dropout, LRU cache (2GB cap).

## WHAT WORKS
- `dataset-build-v10-stage1 --input-dir <dir> --minimap-root <dir>` — filesystem mode, no archives
- `dataset-build-cache --input <curated> --output-dir <dir>` — v9 pipeline, now with MCAL/MCLY
- `train_v11.py <shards> --epochs N` — full training with all signals
- `infer_v11.py <checkpoint> <shards>` — predict heights + MCAL + MCLY + holes, export OBJ

## WHAT BROKE (archive path, DONT USE)
- `--client-root` mode (was already broken, probe bug now fixed but untested)
- `build_v10_2_dataset.py`, `train_v10_2_terrain_synth.py` — dead code
- Shadow masks — never exist on minimap tiles, removed from channel list

## NEXT
1. Extract 800-1500 shards via filesystem mode on staged clients
2. `train_v11.py <shards> --output-dir runs/v11_prod --epochs 300 --batch-size 32`
3. `infer_v11.py runs/v11_prod/best_ema.pt <shards> --export-obj`
