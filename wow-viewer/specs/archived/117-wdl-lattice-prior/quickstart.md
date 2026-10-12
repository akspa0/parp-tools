# Quickstart: WDL-Lattice Coarse Prior for Terrain Geometry

**Feature**: 117-wdl-lattice-prior | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This is the operator runbook. **The user runs every training and heavy step**; the assistant
prepares the scripts and hands off exact invocations. Every command is dry-run / print-plan by
default.

**Status (2026-07-21): US1–US3(i) are implemented and code-verified.** US1's catalog/config change
was exercised for real against the real frozen catalog doc. US2's dry-run and missing-array
refusal were exercised for real against a fixture store (not just unit-tested). US3(i)'s bridge and
both existing trainers' `--feature-store` acceptance were exercised for real against fixture stores
with zero trainer code changes. What remains is entirely USER-RUN: rebuilding a real store with the
new signals (§1 needs a real rebuild to show real data, not just confirm the array names exist),
real `--confirm-run` training (§2), and the real paired comparison (§4).

All commands run from `wow-viewer/data-harvester/` via `uv run`.

## 0. Prerequisites

- **`v50_pipeline_runner.py --confirm` alone is NOT enough.** It only rebuilds the raw per-map
  stores (`0_5_3_3368-<Map>.zarr`) and their curation manifests — it never touches the derived,
  merged training curriculum (`curriculum-0_5_3_3368-dual_v*.zarr`), which is what every trainer
  actually reads. Real sequence, verified end-to-end on 2026-07-21 against `H:\CLIENTS`:
  ```powershell
  # 1. Rebuild the raw per-map stores (this DOES pick up the new WDL arrays)
  uv run python scripts/v50_pipeline_runner.py --confirm

  # 2. Regenerate the terrain-quality-only curation manifests (min_rgb_std=0, max_object_coverage=1
  #    -- the Spec 115 no-op-threshold policy this project's dual curriculum has always used)
  uv run python scripts/spec103_curate_dataset.py --store "../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr" --output "../output/datasets/v50/v50.1/curation-0_5_3_3368-Kalimdor-terrain-v2" --min-rgb-std 0.0 --max-object-coverage 1.0
  uv run python scripts/spec103_curate_dataset.py --store "../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr" --output "../output/datasets/v50/v50.1/curation-0_5_3_3368-Azeroth-terrain-v2" --min-rgb-std 0.0 --max-object-coverage 1.0

  # 3. Rebuild the merged dual curriculum FROM those fresh per-map stores + manifests.
  #    NOTE: this script has NO --write flag -- unlike every other v50 CLI, it writes
  #    unconditionally as soon as you give it --output. There is no dry-run mode.
  uv run python scripts/v50_build_training_curriculum.py --store "../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr" --store "../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr" --curation-manifest "../output/datasets/v50/v50.1/curation-0_5_3_3368-Kalimdor-terrain-v2" --curation-manifest "../output/datasets/v50/v50.1/curation-0_5_3_3368-Azeroth-terrain-v2" --output "../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr" --val-fraction 0.15
  ```
  Verified real result: 2,959 rows (1,629 authored + 1,330 synthetic), train=2,516/val=443, all
  four `wdl_*` arrays present with shape `(2959,17,17)`/`(2959,16,16)`.
  ```bash
  STORE="../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v3.zarr"
  ```
- The existing Spec 116 spatially-isolated held-out split (reused verbatim, D-03):
  ```bash
  SPLIT="../output/datasets/spec116/spec116-held-out-0_5_3_3368-dual_v2"
  ```
- Output root for this feature:
  ```bash
  OUT="../output/datasets/spec117"
  ```

## 1. US1 — verify the signal export (no model)

No new CLI: re-run whatever store build/finalize path already produces `$STORE` for this build, and
confirm the new arrays exist (real names, discovered already-live in the harvest stream — not the
`wdl_lattice_*` placeholder names this quickstart originally drafted):

```bash
uv run python -c "
import zarr
g = zarr.open_group('$STORE', mode='r')
for name in ('wdl_outer_17','wdl_inner_16','wdl_outer_present','wdl_inner_present'):
    print(name, name in g, g[name].shape if name in g else None)
"
```
- **What it does**: confirms the four new arrays exist with the expected shapes
  `(N,17,17)`/`(N,16,16)` before any Python training code is written against them.
- **Time**: seconds, CPU.
- Regenerating the catalog-derived config after any future catalog edit:
  ```bash
  uv run python scripts/v50_generate_manifest_template.py \
    --catalog-doc ../docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md \
    --build-id 0_5_3_3368 --release v50.1 \
    --output v50_configs/v50-manifest-template-0_5_3_3368.json \
    --signals-output v50_configs/v50-signals-0_5_3_3368.json
  ```

## 2. US2 — standalone lattice predictor (USER-RUN; dry-run first, then confirm)

> **Scheduling note (2026-07-22 fix).** The early-stopper is now **warmup-aware**: with
> `--lr-schedule onecycle` it does not count "stale" epochs until the OneCycleLR warmup phase
> completes. The first real run died at epoch 17 (best epoch 2) because the default 30-epoch
> warmup was longer than `--patience 15` — the run was killed mid-warmup before the LR ever
> reached its peak. Pass `--pct-start 0.1` for this small dataset (43 steps/epoch) so only ~10
> epochs are spent warming up instead of 30.
>
> **Architecture note (2026-07-22 fix).** The first post-fix run (warmup-aware, `lattice-authored-v2`)
> survived warmup but still plateaued at val 0.2307 vs tile-mean 0.1277 — and train MAE was *also*
> above tile-mean, i.e. **underfit**, not overfit. v1's plain 4-conv encoder pooled to a 16×16
> bottleneck with no skip connections, so it could not localize the 17×17 height field. `LatticeNet`
> is now a **U-Net-lite** (v2): the bottleneck is decoded back up with skip connections (e3, e2)
> and each head fuses all four feature levels (16/32/64/128) at the lattice resolution. Capacity
> rose 178K → 675K params at `--base 24`; still constructable from `base` alone so the bridge is
> unchanged. If v2 overfits 679 tiles, lower `--base` (e.g. 16) before raising any regularization.

### 2a. Dry run
```bash
uv run python scripts/spec117_train_lattice.py \
  --store $STORE --held-out-split $SPLIT \
  --output "$OUT/lattice-run1" --run-id lattice-authored-v1 --source authored \
  --epochs 100 --lr-schedule onecycle --pct-start 0.1
```
- Prints the full plan and exits without training. `--run-id`/`--source`/`--held-out-split` are all
  required (unlike the coarse/detailer trainers, this one has no `--val-key`/`--val-value`
  fallback — FR-004 requires refusing an unspecified split, not defaulting away from one).
- Refuses closed with a clear message if `$STORE` predates the Spec 117 catalog amendment (missing
  `wdl_outer_17`/etc.) — exercised for real during implementation, not just unit-tested.

### 2b. Real run (user adds `--confirm-run`)
```bash
uv run python scripts/spec117_train_lattice.py \
  --store $STORE --held-out-split $SPLIT \
  --output "$OUT/lattice-v2-run1" --run-id lattice-v2-authored-v1 --source authored \
  --epochs 100 --lr-schedule onecycle --pct-start 0.1 --gradient-weight 0.1 \
  --confirm-run
```
- **What it writes**: `checkpoint_best.pt` + `model_stage_run.json` (`v50-model-stage-run-v1`,
  `stage="lattice_prior"`).
- **Visual output (so you are not flying blind on a number)**: every time val MAE improves the
  trainer writes `validation/best_previews/epoch_XXXX.png` — a sheet of 8 fixed held-out tiles
  showing [minimap RGB, truth lattice, predicted lattice, tile-mean baseline, signed error, abs
  error], where the lattice is the dense 256×256 bilinear-average field the bridge will actually
  emit. At the end it also writes `validation/final_best/fixed_rows.png` and `worst_cases.png`.
  **Look at these**: if the prediction looks like a blurred/wrong-position version of truth while
  the tile-mean baseline column is sharper-on-average, that is the underfit/localization failure
  the v2 U-Net + `--gradient-weight` are meant to fix; if v2 overfits (train MAE ≪ val), lower
  `--base`.
- **`--gradient-weight 0.1`**: a loss-only 2D finite-difference gradient term (ported from the V7
  height regressor's gradient-consistency stack). 0 = pure masked smooth-L1 (parity). It rewards
  matching the local *slope field*, not just per-point values — directly targets the "right values,
  wrong arrangement" failure that beats pure point loss.
- **Read the result**: `metrics.best_val_mae` against `baselines.tile_mean.val_mae` (D-02). If the
  predictor does not beat the trivial baseline, US3 does not proceed without an explicit override
  (spec US2 acceptance 3) — this is the same honesty gate that made tonight's other results
  trustworthy.

## 3. US3(i) — bridge the frozen predictor into the existing feature-store shape

```bash
uv run python scripts/spec117_lattice_to_feature_map.py \
  --store $STORE \
  --checkpoint "$OUT/lattice-run1/checkpoint_best.pt" \
  --output "$OUT/lattice-feature-map-v1" [--device cpu|cuda] --write
```
- **What it does**: runs the frozen predictor over every tile, writes a `(N,1,256,256)`
  `feature_map` array under `schema="v115-feature-map-v1"`, `class_count=1` — the exact shape the
  existing coarse/detailer trainers already validate (D-01). **No new trainer flags** — proven for
  real during implementation: both `v50_train_direct_geometry.py --feature-store` and
  `v50_train_geometry_detailer.py --feature-store` were dry-run against a bridged fixture store and
  accepted it unmodified.
- **Time**: ~1 min, CPU (`--device cuda` also available, mirroring
  `spec116_structure_to_feature_map.py`).

## 4. US3(ii) — paired chain-integration comparison (USER-RUN, existing trainers)

Run the **existing, unmodified** trainers with and without the bridged store, on the identical held-
out split, and compare against the pre-existing real baseline (this session's structure-augmented
detailer run):

**Coarse, without the lattice prior (baseline — already exists, no need to re-run if reused):**
```bash
# Reuse an existing coarse run's model_stage_run.json rather than retraining if one already
# covers this exact store/split/source combination.
```

**Coarse, with the lattice prior:**
```bash
uv run python scripts/v50_train_direct_geometry.py \
  --store $STORE --architecture mit_b0_regression --source authored \
  --held-out-split $SPLIT \
  --feature-store "$OUT/lattice-feature-map-v1" \
  --output "$OUT/coarse-with-lattice-run1" --run-id coarse-with-lattice-run1 \
  --confirm-run
```

**Materialize the coarse-with-lattice checkpoint's output (the detailer needs this as input).**
`v50_materialize_coarse_relief.py` originally had no way to feed the lattice feature-map channel
back in, so a checkpoint trained with `--feature-store` crashed at `load_state_dict` (channel-count
mismatch) — fixed 2026-07-21 by adding `--feature-store` here too, mirroring exactly how the
trainer already handles it:
```bash
uv run python scripts/v50_materialize_coarse_relief.py \
  --store $STORE --checkpoint "$OUT/coarse-with-lattice-run1/checkpoint_best.pt" \
  --feature-store "$OUT/lattice-feature-map-v1" \
  --source authored --output "$OUT/coarse-relief-with-lattice-v1" --write
```
Real result on this session's actual run: `checkpoint_best.pt` was epoch 28, val_mae 0.2333;
materialization produced all 1,629 authored rows with `input_channels: 4`.

**Detailer, with the lattice prior stacked on top of the already-proven structure prior:**
```bash
uv run python scripts/v50_train_geometry_detailer.py \
  --store $STORE --coarse-store <materialized coarse store matching the run above> \
  --source authored --held-out-split $SPLIT \
  --feature-store "$OUT/lattice-feature-map-v1" \
  --frequency-2d-weight 0.1 --laplacian-weight 0.1 --edge-weight 0.1 \
  --transition-focus-weight 0.5 --band-lf-weight 0.05 --band-hf-weight 0.05 \
  --lr-schedule onecycle --pct-start 0.1 --val-tolerance 0.01 \
  --output "$OUT/detailer-with-lattice-run1" --run-id detailer-with-lattice-run1 \
  --confirm-run
```
- **Why `--pct-start 0.1 --val-tolerance 0.01` here**: the detailer's zero-init residual head
  starts AT the coarse baseline and cannot improve validation until the LR rises. The first run
  (default 30-epoch warmup, strict `--val-tolerance 0.0`, `--patience 15`) early-stopped at
  epoch 17 with best epoch 2 — frozen at the coarse baseline, killed mid-warmup. The warmup-aware
  early-stopper (code fix) now prevents that kill regardless; `--pct-start 0.1` shortens the
  warmup so less of a 100-epoch run is spent under-LR, and `--val-tolerance 0.01` stops
  sub-1%-of-best validation noise from counting as stale once warmup ends.
- **Read the result**: `metrics.best_val_mae` from `model_stage_run.json` is NOT the number to
  trust on its own — it's a raw aggregate over every held-out pixel, and ~39% of this corpus is
  near-flat terrain where a trivial per-tile constant wins easily (this project's own established
  finding). The real comparison is the relief-stratified rescore. `--rescore-checkpoint` only
  understands single-stage geometry checkpoints; **detailer checkpoints need
  `--rescore-detailer-checkpoint`** (added 2026-07-21, verified against a real run), which
  reconstructs the coarse+residual composition (`GeometryDetailerNet(rgb[+features], coarse) ->
  residual; final = coarse + residual`) instead of a single forward pass:
  ```bash
  uv run python scripts/spec116_train_structure.py \
    --store $STORE --split $SPLIT \
    --rescore-detailer-checkpoint "$OUT/detailer-with-lattice-run2/checkpoint_best.pt" \
    --feature-store "$OUT/lattice-feature-map-v1" \
    --rescore-source authored --device cpu --print-only
  ```
  `--coarse-store` defaults to the path the checkpoint itself recorded at train time — pass it
  explicitly only if that path moved. Report `stratified_mae.relief.mae` against
  `trivial_baseline_mae` and `sc007_beats_trivial_on_relief`, and compare that number (not
  `best_val_mae`) against the existing structure-augmented detailer's real relief-stratified
  result. Report which feed point helped, per spec US3 acceptance scenario 3 — a null result is a
  valid, reportable outcome.

## Validation commands (assistant-run, lightweight)

```bash
uv run python -m pytest tests/spec117/ -q
uv run ruff check src/harvester/spec117 scripts/spec117_*.py
uv run python -m py_compile src/harvester/spec117/*.py
```

## What the assistant will never launch

- Any `--confirm-run` training pass.
- Any GPU job or full-corpus rebuild.
- The assistant prepares the script, states what it writes and how long it takes, and hands the
  exact CLI to the user. Only the user presses go.
