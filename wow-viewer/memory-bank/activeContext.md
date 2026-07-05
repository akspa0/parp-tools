# Active Context — wow-viewer

Last updated: 2026-07-04
Keep current contract only. Older notes live in `memory-bank/archive/2026-07-04-pre-2026-06-27.md`.

## Main target

- Spec 089 `089-dav2-height-predictor`.
- Local first. The real 12 GB CUDA smoke is now done; next proof owner is cached/pretrained quality plus Pod-side validation.
- Source work through Phase 7 is local-complete.
- Current local proof: `uv run python -m pytest tests/v23 -m v23 -q` passed with `35 passed, 14 warnings`.
- Current hardware proof: `train_v23_height.py` T035 run `t035_local_12gb_20260704` on RTX 4070 Ti SUPER, 16 real V22 train tiles, 4 val tiles, zero CUDA OOM, `peak_vram.json` max allocated `0.408541184 GB`. Caveat: HF DA-V2-Small weights were not cached, so this validates the CUDA/trainer envelope, not pretrained quality.
- Current curated Northrend proof: `v23_curated_northrend_labeled_smoke_20260705` used `--curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet` plus `--maps Northrend`, checkpoint config records the curation thresholds, zero CUDA OOM, max allocated `0.3959296 GB`. Validation preview is labeled and selected from high-mismatch curated rows (`bucket=hard`, mismatch score visible). Same HF-cache caveat.
- Trainer observability is now a first-class loss surface: startup config, train/val batch `loss=...`, component breakdowns, epoch `train_loss`/`val_loss`/`best_val_loss`, checkpoint paths, `metrics.json`, `loss_history.jsonl`, and CUDA peak VRAM. Use `--log-interval N` to change batch log cadence; `--log-interval 1` is the local bring-up default. `peak_vram.json` is capacity proof only, not learning proof.
- Step heartbeat is now explicit: with `--log-interval 1`, each logged batch prints `status=start` before work and `status=done` after work, with `step`, `batch`, `samples`, `pct`, `elapsed`, `eta`, `optimizer_step`, loss breakdown, and CUDA memory.
- Startup batch autotune is now available for V23: `--autotune-batch-size`, `--autotune-batch-candidates`, `--autotune-safety-factor`, and `batch_autotune.json`. The intended local 2K key-map command uses builds `0_5_3_3368 3_3_5_12340`, maps `Azeroth Kalimdor Kalidar PVPZone01 PVPZone02 Northrend Expansion01`, V18 curation manifest, 12 GB profile, GPCT-K 2, and batch candidate ladder `1 2 4 8 12 16 24`.
- `--val-interval` is now honored. Validation is forward-only measurement, not training. For the 2-epoch 2K key-map local command, use `--val-interval 2 --val-preview-interval 2` so epoch 1 is training-only and the final epoch gets the validation/best-checkpoint pass.
- First 2K key-map run learned (`train_loss 16415.93 -> 10314.05`, `val_loss 11482.61 -> 6794.52`) but exposed fixes: batch candidate ladder ended too low at 24 despite only ~6.21 GB reserved, and SDC was dead-zero due sparse valid masks. Recommended candidates now extend to `32 40 48`; SDC uses fractional patch weights.

## Current gate

- Do not treat V23 quality as proven until the HF DA-V2-Small checkpoint is cached/downloaded and rerun through the curated V18 manifest path.
- Remaining external gates: T046 Pod smoke, T048 full-corpus training, T050-T053 CAI/determinism evidence.
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

- Spec 080 Phase A source slice is landed in `gillijimproject_refactor/src/MdxViewer`.
- Bottom display bar owns terrain/world toggles.
- Top toolbar is launcher strip.
- PM4 Object Match and PM4/WMO Correlation now render from `DrawUI()`.
- Proof level = source-only. Legacy `MdxViewer.sln` build still fails on pre-existing missing refs outside this slice.

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
