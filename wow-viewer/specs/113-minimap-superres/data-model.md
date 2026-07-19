# Phase 1 Data Model: Minimap Super-Resolution

## Detail-Preserving Render (content upgrade to `minimap_rgb_1024`)

Not a new store signal — the same `minimap_rgb_1024` array (uint8, 1024×1024×3, per-tile), with its
render semantics upgraded from material-average to real-texel sampling. Spec 112's coverage/parity
contract is unchanged: the detail render succeeds/fails on the same texture-decodability condition,
so `minimap_rgb_1024` still equals `minimap_rgb` in row coverage.

| Aspect | Value |
|---|---|
| Signal | `minimap_rgb_1024` (existing) |
| Render mode | new `detail` mode: per-pixel real BLP texel sample (bilinear) at terrain UV, MCAL-blended, MCNR-lit |
| Selection flag | `synthetic-minimap --detail` (1024 pass); 256 pass stays material-average |
| Honesty | tiles with undecodable textures are skipped (unavailable), never flat-filled |
| Provenance | synthesis manifest records `render_mode=detail` and the texel-repeat frequency used |

## Alignment Report (US1 gate artifact)

Output of `minimap_alignment.py` over a sample of tiles carrying both `minimap_rgb_authored` and the
detail `minimap_rgb_1024`.

| Field | Type | Notes |
|---|---|---|
| `sample_tiles` | list[{map, tile_x, tile_y}] | the tiles registered |
| `per_tile` | list[{tile, best_transform, residual_error}] | best dihedral transform + small translation, and its NCC/phase-corr residual |
| `best_transform_global` | str | one of `identity` / `rot90` / `rot180` / `rot270` / `flip_h` / `flip_v` / `transpose` / `anti_transpose` |
| `transform_is_consistent` | bool | true iff one transform wins for every sampled tile within tolerance |
| `residual_p50` / `residual_p95` | float | aggregate registration error under the chosen transform |
| `gate` | `pass_identity` / `pass_with_transform` / `fail_inconsistent` | US1 verdict; `fail_inconsistent` halts the spec |
| `corrective_transform` | str \| null | the transform to apply to the render (or pairing) when `pass_with_transform` |

## SR Pair Set (trainer-facing, `v50-sr-pairset-v1`)

A curriculum-style Zarr store referencing store rows; each entry is one tile's aligned pair.

| Field | Type | Notes |
|---|---|---|
| `lr` | uint8 (N,256,256,3) | authored client minimap (`minimap_rgb_authored`) |
| `hr` | uint8 (N,1024,1024,3) | detail render (`minimap_rgb_1024`), with the US1 corrective transform applied if any |
| `index.parquet` | table | one row per pair |
| index: `build,map,tile_x,tile_y` | | tile identity |
| index: `source_store` | str | the per-build store the pair came from |
| index: `source_group_id` | str | `real:{build}:{map}:{tile}` — leak-safety key |
| index: `split` | `train`/`val` | deterministic, per-`source_group_id`, per-map stratified |
| attrs: `schema` | `v50-sr-pairset-v1` | |
| attrs: `scale` | int | 4 (256→1024); not hardcoded elsewhere (FR-011) |
| attrs: `corrective_transform` | str | carried from the alignment report |
| attrs: `maps` | list[str] | `["Azeroth","Kalimdor"]` only |

Coverage honesty (FR-004): a tile enters the pair set only if BOTH `minimap_rgb_authored` and the
detail `minimap_rgb_1024` are populated for it; excluded tiles are counted in the summary, never
zero-filled.

## SR Model Config

| Field | Type | Notes |
|---|---|---|
| `arch` | str | `rrdbnet_x4` |
| `stage` | `psnr` / `gan` | stage 1 (L1/PSNR) then optional stage 2 (GAN fine-tune) |
| `losses` | dict | stage 1: `{l1: 1.0}`; stage 2 adds `{perceptual_vgg: w, gan: w}` |
| `init_from` | str \| null | optional public RealESRGAN ×4 weights, if license/shape permit |
| `patch_size` | int | training crop (HR) for patch-based training on a 16 GB GPU |
| `degradation` | const `none_real_pairs` | NOT synthetic — real authored LR is the input (research Decision 4) |

## SR Training Run Summary

| Field | Type | Notes |
|---|---|---|
| `pairset_identity` | str (hash) | binds the run to an exact pair-set build |
| `split` | str | carried from the pair set |
| `model_config` | dict | arch/stage/losses/init/patch as above |
| `metrics` | dict | held-out PSNR/SSIM/LPIPS vs detail HR |
| `baseline_comparison` | dict | model vs bicubic(authored LR) and vs material-average 1024, on the SC-004 detail metric |
| `best_step` | int | |
| `eval_maps` | list[str] | Kalimdor/Azeroth only (FR-009) |
