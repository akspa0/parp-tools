# Active Context — wow-viewer

Last updated: 2026-07-05
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

## WDL prior + lattice detailer lane (V24)

- Spec 094 `094-wdl-prior-v24` implemented end-to-end 2026-07-06: C# `WowViewer.Tool.WdlRead` shim (wraps `WdlSummaryReader` + `WdlWriter.ExtractTileHeightsFromAlpha`, unmodified), Python wrappers, merged-WDL-prior builder, minimap cleaner, Stage A (337,485-param residual U-Net) and Stage B (827,681-param conv-deconv) models, and `validate_v24.py`.
- 2026-07-07: V24 now consumes the V18 curation manifest. `build_wdl_prior.py` gained `--curation-manifest` + `--difficulty-bucket` (join on `(build, tile_id)`, keep only `keep==True`), replacing the naive `--min-height-std` heuristic with the curated, mismatch-omitted corpus the rest of the model stack already uses. Committed (`d6544f1a`).
- 2026-07-07 curated open-world run (`v24_openworld_curated_20260706`): 2,011 kept tiles across Azeroth/Kalimdor/Northrend/Expansion01 (`3_3_5_12340`), 30 epochs/stage. Stage A real-cell L1 0.412 < synth-cell 6.54 < `block_reduce` 1.76; Stage B final L1 **0.649** < upsampled-prior 4.31 < `block_reduce+bilinear` 4.20 (~6.5× better than both baselines). SC-002/003/004/005 PASS; SC-001 confidence bound FAIL (75.9% vs 80%, the documented rough-terrain sampling-phase disagreement). This is the reliable, terrain-generalizable proof; the 50-tile Northrend run remains the bounded pipeline proof. Report: `output/v24_validation/v24_openworld_curated_20260706/report.json`; doc: `docs/architecture/v24-validation-2026-07-06.md`.
- Audited ground truth (overrides the original spec's guesses, see spec.md "Implementation Amendments" A1-A8): C# WDL grid is 17×17 outer + 16×16 inner int16 (MAHO not read by the C# reader); WDLs live inside MPQs (no loose `.wdl` files), resolved via `NativeMpqService`; V18 `minimap_rgb` is 256² uint8 and `object_precise_mask` is 257² float32, not bool.
- Synthetic-vs-real WDL convergence confirmed the user's "99% match" claim: 100% of sampled cells agree within 1.0 world unit on 8 Azeroth tiles (`docs/architecture/wdl-reader-shape-audit-2026-07-06.md`).
- Validation is honest, not forced: on a 100%-real-coverage flat-terrain 50-tile Northrend set, SC-001's confidence bound passes but SC-003 fails because the trivial `block_reduce+bilinear` baseline is already near-exact on flat terrain (nothing to beat). On a rough-terrain 50-tile set (`--min-height-std 15`), SC-002 and SC-003 both pass (Stage A/B measurably beat their baselines) but SC-001's confidence bound narrowly misses (78.3% vs 80%) due to int16 sampling-phase disagreement on steep terrain. Full detail in `docs/architecture/v24-validation-2026-07-06.md`.
- Found and worked around a real V18/V22 defect: `holes_16` is inverted (all-True on ordinary terrain) at the C# source (`AdtTensorPackBuilder.ReadMcrfAndHoles`'s flags-based hole derivation is wrong for LK-era MCNKs). V24's `harvester/v24/tiles.py::_normalize_holes` flips majority-True masks as a workaround; the real fix needs a separate spec touching `WowViewer.Core.IO`.
- V22 dataset audit (user-directed scope addition) ran C#-grounded (re-extraction via `WowViewer.Tool.Harvest extract-unified`, Python only compares): V24's actual input signals (height, minimap, alpha, normal, mcnr, object_precise_mask) are sound; object-mask-family divergence from naive re-extraction is by design (V22's enriched projection is richer than the reference heuristic); two per-tile coverage gaps found (alpha/shadow missing on one Expansion01 tile, truthfully flagged). See `docs/architecture/v22-dataset-audit-2026-07-06.md`.
- Full v24 pytest suite: `uv run python -m pytest tests/v24 -m v24 -q` → 30 passed. Two bounded real-data builds pass SC-001 coverage: Northrend (`3_3_5_12340`, 100% real) and Azeroth (`0_5_3_3368`, 85%/15% real/synthetic).
- Determinism (SC-004) and hardware envelope (SC-005) both pass cleanly: bit-identical inference across seeds, peak VRAM 0.19 GB, max wall-time 0.05 s/tile (target was < 4 GB / < 3 s).

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
