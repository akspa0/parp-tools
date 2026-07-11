# Active Context — wow-viewer

Last updated: 2026-07-10
Keep current contract only. Older notes live in `memory-bank/archive/2026-07-04-pre-2026-06-27.md`.

## Main target

- Spec 089 `089-dav2-height-predictor`.
- No more local training runs unless explicitly reopened. The next proof owner is Pod-side validation with the curated V18/V22 path.
- Source work through Phase 7 is local-complete.
- Current local proof: `uv run python -m pytest tests/v23 -m v23 -q` passed with `35 passed, 14 warnings`.
- Current hardware proof: `train_v23_height.py` T035 run `t035_local_12gb_20260704` on RTX 4070 Ti SUPER, 16 real V22 train tiles, 4 val tiles, zero CUDA OOM, `peak_vram.json` max allocated `0.408541184 GB`. Caveat: HF DA-V2-Small weights were not cached, so this validates the CUDA/trainer envelope, not pretrained quality.
- Current curated Northrend proof: `v23_curated_northrend_labeled_smoke_20260705` used `--curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet` plus `--maps Northrend`, checkpoint config records the curation thresholds, zero CUDA OOM, max allocated `0.3959296 GB`. Validation preview is labeled and selected from high-mismatch curated rows (`bucket=hard`, mismatch score visible). Same HF-cache caveat.
- Trainer observability is now a first-class loss surface: startup config, train/val batch `loss=...`, component breakdowns, epoch `train_loss`/`val_loss`/`best_val_loss`, checkpoint paths, `metrics.json`, `loss_history.jsonl`, and CUDA peak VRAM. Use `--log-interval N` to change batch log cadence; `--log-interval 1` is the local bring-up default. `peak_vram.json` is capacity proof only, not learning proof.
- Step heartbeat is now explicit: with `--log-interval 1`, each logged batch prints `status=start` before work and `status=done` after work, with `step`, `batch`, `samples`, `pct`, `elapsed`, `eta`, `optimizer_step`, loss breakdown, and CUDA memory.
- Startup batch autotune is now available for V23: `--autotune-batch-size`, `--autotune-batch-candidates`, `--autotune-safety-factor`, and `batch_autotune.json`. The Pod default uses builds `0_5_3_3368 3_3_5_12340`, maps `Azeroth Kalimdor Kalidar PVPZone01 PVPZone02 Northrend Expansion01`, packaged V18 curation manifest, 24 GB profile, GPCT-K 2, and batch candidate ladder `1 2 4 8 12 16 24 32 40 48 64 80 96`.
- `--val-interval` is now honored. Validation is forward-only measurement, not training. The no-arg Pod wrapper uses `--val-interval 2 --val-preview-interval 2` so epoch 1 is training-only and the final epoch gets the validation/best-checkpoint pass.
- RunPod `train.sh` now has a curated no-arg default. It requires `config/curation_manifest.parquet` and `config/tileset_prune_table.json`, then runs the curated 2K key-map corpus with visible step logging, SDC/GPCT/bias-free masking, and startup autotune. Explicit args still pass through to `train_v23_height.py`.
- First 2K key-map run learned (`train_loss 16415.93 -> 10314.05`, `val_loss 11482.61 -> 6794.52`) but exposed fixes: batch candidate ladder ended too low at 24 despite only ~6.21 GB reserved, and SDC was dead-zero due sparse valid masks. Recommended candidates now extend to `32 40 48`; SDC uses fractional patch weights.

## Current gate

- Do not treat V23 quality as proven until the HF DA-V2-Small checkpoint is cached/downloaded and rerun through the curated V18 manifest path.
- Remaining external gates: T046 Pod smoke, curated 2K key-map Pod training, T048 full-corpus training, T050-T053 CAI/determinism evidence.
- Do not claim remote proof from Pod creation alone.

## V23 contract

- One signal only: height.
- Input = Spec 088 V22 paths-only store, which is built from the V18 substrate plus enrichment. V23 does not read V18 directly during training, but the V18 arrays are carried into V22.
- Model = DA-V2-Small encoder + LoRA + compact height head + affine anchor.
- Trainer handles memory profiles, grad accumulation, and OOM backoff.
- Inference + CAI stitch path exist and are deterministic in local tests.

## V22 contract

- Spec 088 is active V22 design.
- Canonical stores exist for `0_5_3_3368` and `3_3_5_12340`.
- `3_3_5_12340` V18 and V22 both include `Northrend` with 1,131 tiles; local smoke must use `--maps Northrend` when the proof is about Wrath/Northrend terrain knowledge.
- Store is `paths_only`; no embedded M2/WMO/BLP payload blobs.
- Remaining bounded gate: rerun same proof for `4_0_0_11927`, then close 088.

## UI compatibility lane

- Spec 080 is now re-centered on `wow-viewer/src/viewer/WoWViewer`; the earlier `gillijimproject_refactor/src/MdxViewer` Phase A work remains legacy source-only context, not the active target.
- 2026-07-05 small Phase 1 slice: bottom bar now exposes split world wireframes (`Terrain WF`, `M2/WMO WF`), standalone model/WMO wireframe, standalone WMO group bounding boxes, standalone WMO group names, and a Settings launcher.
- 2026-07-07 correction: world `M2/WMO WF` is overlay-based over normal solid rendering and no longer enables hover-only reveal; non-hovered objects should remain visible. Build proof passed; manual world-map viewport proof remains open.
- 2026-07-05 small Phase 2/3 follow-up: File -> Settings, Tools -> Settings, bottom-bar Settings, utility popup Settings, and workspace Settings all route to `_showSettingsWindow`; Settings now includes persisted Camera Speed and FOV defaults.
- World bottom tabs now include `LOD`, with a first-pass World LOD panel for WDL visibility, bounding boxes, PM4 overlay status, loaded tile/chunk counts, and ADT detail-tile budget. This is a destination for future WDL/world-distance facts, not a completed right-sidebar migration.
- Standalone WMO group names default to visible for all groups; selected/highlighted labels still work.
- Spec 080 now has `plan.md` and `tasks.md` with the right-sidebar audit/migration broken into phases. Right sidebar remains the known messy surface; left sidebar remains in scope only for later wording cleanup.
- Proof level: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with existing warning noise and 0 errors. Manual viewer checks for standalone WMO and world map toggles remain open.

## Viewer memory/performance lane

- Spec 090 `090-viewer-memory-profiler` is active for the 4.0.0 Stormwind RAM blow-up investigation.
- First slice adds Runtime Stats process/GC memory counters plus MPQ/world raw-cache byte totals, and caps `WorldAssetManager` raw file cache at 512 MiB by LRU. Live M2/WMO renderer eviction remains unchanged until counters prove renderer residency is the retained owner.
- Proof level: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with existing warning noise and 0 errors. Manual staged `4_0_0_11927` Stormwind measurement remains open.

## Render performance and WMO liquid lane

- Spec 093 `093-render-performance-liquid-audit` is active for the slow-frame and WMO-liquid audit.
- Source audit finding: WMO placements are still submitted per visible placement and per visible group/material batch; MDX "batched" counts mean shared-shader submission, not true GPU instancing; WMO transparent and WMO MLIQ liquid work were previously hidden inside the MDX transparent timing bucket.
- WMO MLIQ basic GL state already enables alpha blending and disables depth writes, so "opaque liquid" most likely points to the current flat color-only shader/material/order path, not total absence of blending.
- Runtime Stats now reports WMO visibility/opaque/transparent timing plus WMO batch/fallback/liquid/doodad/group submission counts. Next proof owner is a manual dense-map capture before any batching rewrite or liquid visual correction.
- Proof level: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with existing warning noise and 0 errors; focused `WorldRenderOptimizationAdvisorTests` passed `3/3`.

## Raw audio pattern lane

- Spec 091 `091-raw-audio-unswizzle` is active for investigating structured patterns seen when map-derived WAV payloads are viewed as raw image bytes.
- First slice adds `data-harvester/scripts/unswizzle_audio_raw_patterns.py`, which strips WAV payloads when possible, sweeps likely widths/deinterleaves/bitplanes/sample interpretations, writes candidate PNGs, ranks them in `summary.json`, and creates `contact_sheet.png`.
- The Azeroth V18 WAV at `output/azeroth_audio/Azeroth_all_tiles_0_5_3_3368_11025Hz.wav` is exactly 41,082,478 mono 16-bit samples, which equals 622 complete `257x257` height tiles with zero remainder. The tool now writes stream-order and `index.parquet` coordinate-order tile mosaics.
- This is evidence tooling only. Structured output is a layout hypothesis, not proof of steganography or hidden payloads.
- Proof level: py_compile and CLI help passed; a bounded smoke run wrote 60 candidates under `C:\tmp\wow-unswizzle-smoke`; the real Azeroth WAV pass wrote 390 projections plus tile mosaics under `wow-viewer/output/analysis/raw-audio-unswizzle/azeroth_0_5_3_3368`.

## Height pattern miner lane

- Spec 092 `092-heightmap-pattern-miner` is active for V23-adjacent repeated height motif analysis.
- `data-harvester/scripts/mine_heightmap_patterns.py` samples `height_257` patches from V18/V22-style Zarr stores, locally normalizes them, hashes coarse low-resolution signatures, suppresses low-variance and saturated artifacts, and writes `summary.json` plus `pattern_atlas.png`.
- Corrected constraint: motif windows are terrain-cell spans, default `32 64` cells, minimum `32` cells, chunk-aligned to the 16-cell MCNK grid by default. Legacy tiny patch matching is not the intended mode.
- Bounded Azeroth proof after correction: `0_5_3_3368`, 128 tiles, cell spans `32 64`, chunk-aligned, grid `4`, quant `4`, max saturated ratio `0.30`. Output kept 16,390 patches, found 14,745 buckets, and surfaced repeated chunk-scale ramps/ridges/falloffs under `wow-viewer/output/analysis/heightmap-patterns/azeroth_0_5_3_3368_chunkcells_coarse`.
- V23 training remains unchanged. Next useful step is joining motif IDs to V23 validation error maps before using patterns for curriculum or weighting.

## V24.1 DA-V2 pretrained convergence model lane (Spec 101)

- **2026-07-10 — Spec 101 created and Slices 1-5 implemented.** Research report at `docs/architecture/v24-convergence-research-2026-07-10.md` surveyed HuggingFace + GitHub for existing models. Key finding: V24 Stage A's 190 L1 is a **model capacity and pretrained features** problem, not a loss function problem. The 335K-param from-scratch U-Net cannot learn minimap → heightmap from ~2,000 tiles.
- **Solution**: Reuse the V23 DA-V2-Small encoder (24.8M params, pretrained on 62M images, already in `harvester/v23/encoder.py`) with LoRA + a new DPT head for WDL prior output. New `StageADAV2` class in `stage_a.py`.
- **Slice 1**: `StageADAV2` model class — DA-V2-Small encoder + LoRA (rank 16) + DPT head → 33×33 quincunx → 17×17 outer + 16×16 inner. Total ~25M params, ~1-2M trainable. Backbone frozen.
- **Slice 2**: `SiLogLoss` — scale-invariant log loss (standard for metric depth, used by DA-V2). Handles negative heights via shift parameter. `hybrid_loss` = 0.7 SiLogLoss + 0.3 L1.
- **Slice 3**: Scheduler fix — OneCycleLR `total_steps` corrected to `n_batches * epochs` (was `epochs` only). Per-batch stepping for OneCycleLR; per-epoch for CosineAnnealingLR. New `--scheduler` flag.
- **Slice 4**: `--dav2` flag on `train_v24_stage_a.py` — loads pretrained DA-V2-Small, uses hybrid loss, lr=5e-6 (pretrained encoder), batch_size=8. Checkpoint records `model_type`, `loss_type`, `scheduler_type`, `dav2`, `guided`.
- **Slice 5**: 15 new tests in `test_stage_a_dav2.py` — model shape (3ch + 9ch), param count, backbone frozen, offline load, SiLogLoss (positive, negative, gradient, perfect, zero-weight), hybrid loss, `build_dav2_input` (3ch, 9ch, no-normal). All pass.
- **Full v24 suite**: 67 passed, 1 deselected, 0 failed (was 46 before Spec 101). No regressions.
- **All 7 slices implemented.** Slice 6 (`StageBPromptDA` in `stage_b.py` — DA-V2 depth completion, 4ch input: 3 RGB + 1 WDL prior prompt → 257×257 heightmap). Slice 7 (`WDLDiscriminator` PatchGAN in new `discriminator.py` — ~693K params, `gan_step` helper with BCEWithLogits + L1, lambda schedule 0→0.1 over epochs 5-30).
- **Full v24 suite**: 77 passed, 1 deselected, 0 failed (was 46 before Spec 101). Zero regressions.
- **Training results (2026-07-10/11)**:
  - Run 1 (lr=5e-6, 40 epochs, 500 tiles): val_l1=91.11 (vs 190 old U-Net). LR too low.
  - Run 2 (lr=1e-4, 200 epochs, 500 tiles, LoRA 16): best val_l1=48.13 @ epoch 143, overtraining detected (train 17 vs val 49). Model learning but overfitting on 500 tiles.
  - Run 3 (lr=1e-4, 200 epochs, 2,011 tiles, LoRA 32, wd=1e-3, bs=8): crashed at epoch 132 with cuDNN OOM.
  - Run 4 (in progress): lr=1e-4, 200 epochs, 2,011 tiles, LoRA 32, wd=1e-3, bs=8, bf16, 8-bit optimizer, gradient checkpointing. 26.3M total params, 1.74M trainable.
- **Key VRAM optimizations added**: `--8bit-optimizer` (bitsandbytes 8-bit AdamW, 4× less optimizer state), `--gradient-checkpointing` (recompute activations, ~2-3× less VRAM), `--lora-rank` (configurable LoRA capacity), `--weight-decay` (regularization), V18 preload cache freed after tensor extraction.
- **Key insight from research**: PromptDA (CVPR 2025, 1,135 stars) is the most directly relevant model for Stage B — takes RGB + low-res depth prompt → high-res metric depth. Exactly our Stage B structure. HuggingFace: `depth-anything/prompt-depth-anything-vits`.
- **Next steps**: Evaluate Run 4 results. If val_l1 < 10, the DA-V2 Stage A is deployment-ready. If not, try guided (9ch with normals), higher LoRA rank (64), or PatchGAN adversarial loss. Then train Stage B (PromptDA) and run full pipeline.

## WDL prior + lattice detailer lane (V24)

- Spec 094 `094-wdl-prior-v24` implemented end-to-end 2026-07-06: C# `WowViewer.Tool.WdlRead` shim (wraps `WdlSummaryReader` + `WdlWriter.ExtractTileHeightsFromAlpha`, unmodified), Python wrappers, merged-WDL-prior builder, minimap cleaner, Stage A (337,485-param residual U-Net) and Stage B (827,681-param conv-deconv) models, and `validate_v24.py`.
- 2026-07-07: V24 now consumes the V18 curation manifest. `build_wdl_prior.py` gained `--curation-manifest` + `--difficulty-bucket` (join on `(build, tile_id)`, keep only `keep==True`), replacing the naive `--min-height-std` heuristic with the curated, mismatch-omitted corpus the rest of the model stack already uses. Committed (`d6544f1a`).
- 2026-07-07 curated open-world run (`v24_openworld_curated_20260706`): 2,011 kept tiles across Azeroth/Kalimdor/Northrend/Expansion01 (`3_3_5_12340`), 30 epochs/stage. Stage A real-cell L1 0.412 < synth-cell 6.54 < `block_reduce` 1.76; Stage B final L1 **0.649** < upsampled-prior 4.31 < `block_reduce+bilinear` 4.20 (~6.5× better than both baselines). SC-002/003/004/005 PASS; SC-001 confidence bound FAIL (75.9% vs 80%, the documented rough-terrain sampling-phase disagreement). This is the reliable, terrain-generalizable proof; the 50-tile Northrend run remains the bounded pipeline proof. Report: `output/v24_validation/v24_openworld_curated_20260706/report.json`; doc: `docs/architecture/v24-validation-2026-07-06.md`.
- **2026-07-09 full-scale curated training started** (user-directed): Stage A training on `openworld_curated.zarr` (2,011 tiles, 50 epochs). Stage B follows (same store, 50 epochs). Output dir: `output/v24_validation/v24_curated_full_v1_20260709/`. All previously-fixed data-loading bottlenecks are now documented in the bug catalog below.
- **Data-loading speed bug (fixed 2026-07-09):** Two independent bottlenecks caused per-tile load times of ~1 s/tile on first access (~25 min for 1,600 tiles). Both fixed in `tiles.py` + both training scripts:
  1. **`clean_minimap()` median-fill loop** — called on every `TileSource.load()`. The V18 store for `3_3_5_12340` has no `no_object_minimap` array, so every tile went through a Python-median-fill loop (up to 512 passes). **Fix:** replaced with raw-minimap normalization (`rgb / 255.0`). The model loss is confidence-weighted (real cells ~75 % of weight), so objects in raw minimap get naturally de-prioritized. Pre-computed cleaned minimaps should be a dataset-build step, not per-load.
  2. **Per-tile random-access Zarr reads** — `source.load(row)` called `v18["minimap_rgb"][v18_row]` for each tile individually. With large V18 chunks (spanning many tiles), each random-access read pulled the entire chunk from disk. **Fix:** added `TileSource.preload(rows)` that reads all needed V18 rows in a single contiguous Zarr slice `arr[lo:hi]` (sequential I/O, 6 array reads total), then caches in `_v18_cache`. Subsequent `load()` calls are dict lookups.
  - Both `train_v24_stage_a.py` and `train_v24_stage_b.py` now call `source.preload(train_rows + val_rows)` before the per-tile iteration. Loading time dropped from ~25 min to < 10 s.
  - **To prevent recurrence:** Any new training script that uses `TileSource` must call `source.preload(rows)` before iterating. The preload logic is in `tiles.py` and only triggers when `_v18_cache` is populated; random-access mode still works for ad-hoc inspection.
- Audited ground truth (overrides the original spec's guesses, see spec.md "Implementation Amendments" A1-A8): C# WDL grid is 17×17 outer + 16×16 inner int16 (MAHO not read by the C# reader); WDLs live inside MPQs (no loose `.wdl` files), resolved via `NativeMpqService`; V18 `minimap_rgb` is 256² uint8 and `object_precise_mask` is 257² float32, not bool.
- Synthetic-vs-real WDL convergence confirmed the user's "99% match" claim: 100% of sampled cells agree within 1.0 world unit on 8 Azeroth tiles (`docs/architecture/wdl-reader-shape-audit-2026-07-06.md`).
- Validation is honest, not forced: on a 100%-real-coverage flat-terrain 50-tile Northrend set, SC-001's confidence bound passes but SC-003 fails because the trivial `block_reduce+bilinear` baseline is already near-exact on flat terrain (nothing to beat). On a rough-terrain 50-tile set (`--min-height-std 15`), SC-002 and SC-003 both pass (Stage A/B measurably beat their baselines) but SC-001's confidence bound narrowly misses (78.3% vs 80%) due to int16 sampling-phase disagreement on steep terrain. Full detail in `docs/architecture/v24-validation-2026-07-06.md`.
- Found and worked around a real V18/V22 defect: `holes_16` is inverted (all-True on ordinary terrain) at the C# source (`AdtTensorPackBuilder.ReadMcrfAndHoles`'s flags-based hole derivation is wrong for LK-era MCNKs). V24's `harvester/v24/tiles.py::_normalize_holes` flips majority-True masks as a workaround; the real fix needs a separate spec touching `WowViewer.Core.IO`.
- V22 dataset audit (user-directed scope addition) ran C#-grounded (re-extraction via `WowViewer.Tool.Harvest extract-unified`, Python only compares): V24's actual input signals (height, minimap, alpha, normal, mcnr, object_precise_mask) are sound; object-mask-family divergence from naive re-extraction is by design (V22's enriched projection is richer than the reference heuristic); two per-tile coverage gaps found (alpha/shadow missing on one Expansion01 tile, truthfully flagged). See `docs/architecture/v22-dataset-audit-2026-07-06.md`.
- Full v24 pytest suite: `uv run python -m pytest tests/v24 -m v24 -q` → 30 passed. Two bounded real-data builds pass SC-001 coverage: Northrend (`3_3_5_12340`, 100% real) and Azeroth (`0_5_3_3368`, 85%/15% real/synthetic).
- Determinism (SC-004) and hardware envelope (SC-005) both pass cleanly: bit-identical inference across seeds, peak VRAM 0.19 GB, max wall-time 0.05 s/tile (target was < 4 GB / < 3 s).
- **2026-07-09 full 50-epoch curated run** (`v24_curated_full_v1_20260709`): Stage A real_cell L1=0.397 (beats previous 0.412), Stage B final L1=0.857 (beats prior 4.30 & block_reduce 4.20 baselines by 5x). 7/10 SC checks pass (SC-001 confidence bound FAIL is the known terrain-sampling disagreement).
- **Data-loading fixes documented in bug catalog above** — `clean_minimap()` removed from per-load path (replaced by raw-minimap normalization as fallback); `TileSource.preload(rows)` added for contiguous V18 reads.
- **Pre-computed cleaned minimaps** added as dataset-build step: `scripts/precompute_v24_cleaned_minimaps.py` stores `clean_minimap()` output as `cleaned_minimap_256` array in V24 store (no compression, 1-tile chunks). `TileSource.load()` prefers stored array when available. Retrain in progress.
- **2026-07-09 Spec 096 — minimap-only deployment wiring** (closes the deployment gap from FR-013 / Spec 094 US3 scenario 5): trained `StageAMinimapOnly` (3-channel, 334,965 params) on `3_3_5_12340_v24_all_v1.zarr` for 50 epochs (2,241 train / 560 val, autotune batch=512, peak_vram=0.005 GB, best_val_l1=190.31, overtraining detected). Wrote `scripts/infer_v24_stage_a_png.py` (~200 lines, PIL+numpy+torch+harvester.v24) — loads a bare PNG, runs the model, emits `(17,17)+(16,16)` WDL prior NPZ + 4-up preview PNG. No V24 store / no V18 store / no staged client. Strict-checkpoint refusal of 13-channel cheat checkpoints. Patched `train_v24_stage_a.py` to record `minimap_only: bool` + correct `in_channels` in the saved config. Extended `validate_v24.py` with `--minimap-only-checkpoint` and a new `SC-002-MINIMAP` gate. Added 5 new tests (model + script). Full v24 suite: 36/36 green.
- **Honest deployment result (Risk 1 materialised):** On the held-out 560-tile V24 prior validation, the minimap-only regime's val_l1 is **190.31 world units** vs the cheat regime's **1.21** and the `block_reduce` baseline's **1.31** — a 158× gap. The bare RGB minimap does not carry enough signal to predict the WDL prior at the precision the grid requires. The CLI works; the model is not yet accurate enough to be useful. This is the documented honest-failure-mode the spec planned for; `SC-002-MINIMAP` gate is **FAIL** and is recorded as such in the validation report (`output/v24_validation/v24_minimap_only_validation_20260709/report.json`). Architecture doc: [`docs/architecture/v24-minimap-deploy-2026-07-09.md`](../docs/architecture/v24-minimap-deploy-2026-07-09.md). Next step: **Spec 095 (learned minimap cleaner)** is the most likely path to closing the 158× gap.
- **2026-07-09 — one-shot wrapper + OBJ mesh export.** Wrote `v24_run_on_png.py` and `v24_prior_to_obj.py` so the user can drop a PNG minimap into one command and get back a (17,17)+(16,16) prior NPZ + 4-up preview PNG + a 257×257 textured OBJ mesh. `--batch-dir <dir>` runs the wrapper on every PNG in a folder and stitches the tiles into a single grid OBJ with an atlas. X-flip fix: the mesh was opening mirrored along the X axis (image-X runs opposite to world-X); the prior is now `np.fliplr`'d at load time so the mesh opens correctly in any 3D viewer. 40/40 v24 tests pass (was 36 before this slice).
- **2026-07-10 — Spec 097 Slice 1: per-map V18 Zarr → stitched OBJ + baked atlas with edge alignment.** Wrote `v24_export_map.py` (per-map V18 Zarr → single stitched OBJ + atlas, edge-aligned across tile boundaries). Northrend smoke: 29 rows × 39 cols = 1,131 tiles, 7,453 × 10,023 heightmap, 74.7M vertices, world -786.9..409.3, 6.3 min wall time on the 12 GB GPU. Output: `<map>.obj` + `<map>.atlas.png` + `<map>_manifest.json` + `tiles/<tx>_<ty>.prior.npz` per tile. Edge alignment: 16-pixel band on each side of every shared border is averaged; corner cells (4-way) inherit the average. The OBJ opens with continuous height across the seams (no visible hard step at the 256-pixel-tile borders). Honest caveat: the minimap-only prior is the same 190-world-unit L1 model as Spec 096; the seams are continuous in the *predicted* height, not in the *true* V18 height. SC-097-001 is met; Slices 2/3/4 (WDL writer, ADT writer, round-trip smoke) are next-session work.
- **2026-07-10 — atlas row-alignment fix (user-reported).** The first Northrend export had the synthetic colour atlas laid out in the same order as the OBJ's `tile_rows` (north first), but the OBJ vertex order is south-first (Y=0 is the south end). The atlas row for tile at OBJ row 0 was the colour for the **north** end, so the texture was visually misaligned with the mesh. Fixed by placing the colour for `tile_rows[ri]` at atlas row `len(tile_rows) - 1 - ri` (south end first). Re-exported Northrend, all tests still green.
- **2026-07-10 — X-flip reverted in both single-tile and per-map paths (user-reported, second pass).** The earlier "X-flip fix" was wrong: in the single-tile case the texture is the **source PNG**, which is in original (un-flipped) image orientation. Flipping the heightmap while keeping the texture un-flipped misaligns them — the user saw the texture flipped along X relative to the mesh. The correct contract is: heightmap and texture keep the same orientation (both un-flipped). Reverted the `np.fliplr` in `v24_prior_to_obj.py` and `v24_export_map.py`. The OBJ writer's V-flip still handles the Y axis. All tests still green.
- **2026-07-10 — batch wrapper now writes outputs to a sibling folder, not into the input (user-reported, design bug).** `v24_run_on_png.py --batch-dir` previously wrote `<batch-dir>/<stem>.mesh/...` into the input directory. Now it defaults to `<batch-dir>/../<batch-dir>_v24_objs/` (a sibling folder), so the input is never modified. Use `--batch-output-dir` to override.
- **2026-07-10 — quilt mode (single flat folder, world-positioned OBJs).** Wrote `v24_quilt_objs.py`. For each PNG in `--input-dir`, writes one `<stem>.obj` + one `<stem>.png` (the source minimap as the texture) + one `<stem>.mtl` into `--output-dir` (no subdirs). The OBJ vertices are positioned in world space at `(tile_x * 533, tile_y * 533, 0)` parsed from the filename. The user drags the whole output folder into MeshLab / Blender / Windows 3D Viewer and the tiles line up as a quilted map. **Default `--output-dir` is the repo root's `output/v24_quilt/<input-basename>/`** (always outside the input). The script **refuses to run** if the resolved output is inside the input. **User-reported bugs fixed in this slice**: (a) input-folder pollution — outputs no longer default to a child of the input; (b) per-OBJ mtl sharing — each OBJ gets its own `<stem>.mtl` so the viewer's texture lookup does not collide across the quilt; (c) YX-vs-XY filename convention — added `--naming {xy,yx}` flag; (d) spiky-bit boundaries — added `_align_tile_boundaries` that snaps each tile's 16-pixel border to the average with the east/south neighbour. Same safety check added to `v24_run_on_png.py --batch-dir`. 42/42 v24 tests pass.
- **2026-07-10 — image splitter (user's "1:1 plotting" ask).** Wrote `v24_split_image.py`. Takes a single composite image aligned to the 64×64 tile grid (e.g. a hand-rendered pre-alpha minimap quilt), splits it into individual 256×256 `tile_X_Y.png` (or `tile_Y_X.png` with `--naming yx`) PNGs the V24 inference scripts expect. Defaults: `--grid-cols 64 --grid-rows 64 --tile-size 256`. Output goes to `wow-viewer/output/v24_tiles/<image-basename>/` by default. Hard safety: refuses to write inside the image's directory. Refuses undersized composites. 4 new tests (count, naming, safety, undersize); 46/46 v24 tests pass. **The splitter closes the user's "I have one big image, give me a tool to split it" gap** — pipeline is now: composite image -> split into tiles -> quilt mode -> OBJ output.
- **2026-07-10 — Spec 098 written (vision document, not implementation).** `wow-viewer/specs/098-v24-lattice-reconstruction/spec.md`. Breaks the user's "next step" vision into 5 sub-specs (099 Stage A retrain on every V18 build, 100 Stage B border-consistency loss, 101 fractal/hand-painted detail detector, 102 V24 reconstruction lattice model, 103 full-map round-trip integration). Pre-conditions: Spec 095 (minimap cleaner) + Spec 097 Slices 2/3 (WDL/ADT writers) land first. Realistic timeline: 3-6 months of focused work. The order Spec 098 lays out is the recommended path.
- **2026-07-10 — Spec 097 Slices 2/3/4 NOT shipped this session.** WDL and ADT writers are substantial binary format work. The proper round-trip path is a small `write` subcommand on the existing C# `WowViewer.Tool.WdlRead` shim, which is a multi-step C# change. Faking the writers in Python would produce files the C# readers cannot open — worse than no output. Documented in `specs/097-v18-to-wdl-adt/tasks.md` as the next-session handoff. Spec/plan/tasks/checklist are all written under `wow-viewer/specs/097-v18-to-wdl-adt/`.

## Recent background still live

- 2026-07-04: repo doc audit rewrote `AGENTS.md`, root `README.md`, `docs/PLANS-OVERVIEW.md`, `docs/WoWViewer/*`, and `data-harvester/README.md`. Canonical doc routing now starts at `docs/DOCUMENTATION-STATUS.md`.
- 2026-06-30: Spec 088 replaced broken V22 payload plans with `V22Enrich` + paths-only store.
- 2026-06-29: Spec 077 loss-gate fix moved teacher-prior weighting to `object_precise_mask` first.
- Spec 076 and Spec 077 remain paused/background unless user reopens them.

## Boundaries

- Do not move new work back into `gillijimproject_refactor`.
- Do not claim remote proof from Pod creation alone.
- Do not claim UI compile validation from legacy-solution failures outside touched slice.
- Staged clients only under `output/tmp/wowarchive-clients/`.
