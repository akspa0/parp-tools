# Quickstart: Direct Minimap-to-Terrain Reconstruction

## Status — authored baseline ran and failed; the bakeoff tooling is ready

The proven Spec 112 CNN is pinned as Spec 114 architecture `direct_cnn_v112`. Its authored-only
bootstrap (T017) **completed 100 epochs and failed SC-001**: best epoch 92, validation MAE
0.1492665126 vs tile-mean 0.1387469612 — immutable negative evidence, do not rerun that recipe
(research.md T017/T018 record). The next candidates are the same CNN on the corrected dual-view
curriculum and the `mit_b0_regression` architecture, both runnable through the T015 trainer below.

Implemented since the bootstrap (T004-T009, T014-T015):

- `model_stage_contract.py`: dependency-free validator for all three published contract variants,
  with sha256 identity binding and generated-input provenance attachment.
- `reconstruction_curriculum.py` + `v50_build_reconstruction_curriculum.py`: dual-view admission
  policy (grouped-split leak refusal, honest stale-lighting exclusion, no zero-filling) emitting
  the schema-valid `v50-reconstruction-curriculum-v1` summary plus a row-selection parquet.
- `direct_geometry_model.py`: architecture registry — `direct_cnn_v112` (1,561,537 params) and
  `mit_b0_regression` (SegFormer-B0-scale, one continuous 257×257 output; ~3.8M params).
- `direct_geometry_train.py` + `v50_train_direct_geometry.py`: flat AND tile-mean in-run baselines,
  SC-001 (≥5% over both baselines AND the frozen Spec 112 run), SC-002 border-vs-interior-p95,
  per-row quantile/worst-case sheets, and a schema-validated `model_stage_run.json` with
  `promotion_verdict=pending`. Optional `--amp` / `--lr-schedule onecycle` / `--clip` address the
  bootstrap's audit finding; defaults are bootstrap parity.

Real dry-run proof on the frozen curriculum (bootstrap era):

- 1,561,537 trainable parameters;
- 1,629 authored rows: 1,384 train / 245 validation;
- Kalimdor 951 / Azeroth 678;
- batch 16 gives 87 train steps per epoch;
- deployment inputs: `minimap_rgb` only;
- target: numeric `height_257` encoded under offset-invariant contract `v112.1`.

## Gate 0 — finish the corrected source evidence

Before Spec 114 can use synthetic rows or run the final dual-view bakeoff:

1. Complete Spec 113 T010b with the fixed-noon-white synthetic renderer.
2. Visually compare the real authored tile, synthetic 256 tile, and synthetic 1024 detail tile.
3. Refresh only the stale synthetic RGB arrays; numeric height, normals, liquid, material, alpha,
   authored RGB, and other unaffected v50 signals remain valid.
4. Record the new store/manifest hashes and `NoonWhiteGlobal` provenance.

The authored-only run below does not wait on this gate because neither authored RGB nor numeric
height truth changed.

## T017 record — authored bootstrap run (completed, rejected)

The commands below were executed by the user on 2026-07-19. The run completed all 100 epochs and
failed SC-001 (see Status). They are preserved as the reproducible record of that evidence; the
active commands are the curriculum build and bakeoff sections further down.

From `wow-viewer/data-harvester`, first run the exact validated preview. It performs contract/index
checks and does not create the output directory or allocate CUDA training state:

```powershell
uv run python scripts/v50_train_height_relative.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --source authored `
  --output ../output/v50/v50.1/direct_geometry/direct_cnn_v112-authored-v1 `
  --epochs 100 --batch 16 --workers 0 --patience 15 --seed 114
```

If the preview reports exactly 1,384 train / 245 validation authored rows, launch training by adding
the final confirmation flag:

```powershell
uv run python scripts/v50_train_height_relative.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --source authored `
  --output ../output/v50/v50.1/direct_geometry/direct_cnn_v112-authored-v1 `
  --epochs 100 --batch 16 --workers 0 --patience 15 --seed 114 `
  --confirm-run
```

Expected on the local RTX 4070 Ti SUPER: approximately 15–60 minutes with early stopping,
comfortably within 16 GB VRAM, and under 50 MB of checkpoints/summaries. `--workers 0` is
intentional for the current Windows-local dataset implementation. The user launches this command;
the assistant does not.

The console prints `train_loss`, exact validation MAE, tile-mean baseline, best MAE, and early-stop
staleness every epoch. When it finishes, inspect/share the promotion summary with:

```powershell
Get-Content -Raw ../output/v50/v50.1/direct_geometry/direct_cnn_v112-authored-v1/training_summary.json
```

Expected artifacts are `training_plan.json`, `run_identity.json`, `checkpoint_best.pt`,
`checkpoint_last.pt`, and `training_summary.json`. Do not rename or reuse the run directory.

Lightweight proof completed before handoff (bootstrap era): focused model/trainer/curriculum tests
23 passed; real authored-only dry run 1,629 selected / 1,384 train / 245 validation; real
`--source all` dry run correctly refused for missing corrected-light provenance. Current proof
after T004-T009/T014-T015: full v50 suite 242 passed / 4 skipped; Ruff clean on all new modules;
both new CLIs dry-run correctly and write nothing without `--write`/`--confirm-run`.

## Architecture comparison to implement

| Stage | Required baseline | Candidate | Output | Explicitly excluded |
|---|---|---|---|---|
| Direct geometry | Spec 112 lean CNN | MiT-B0/SegFormer-style continuous decoder | `relative_height_257` | WDL prior, DA-V2, GAN/diffusion height |
| Object visibility | Empty/all-object baselines | Compact SegFormer semantic mask | one object mask | authored-minus-synthetic RGB labels |
| Terrain features | Majority family | Compact SegFormer semantic classifier | one feature map | shared geometry weights |
| Texture families | Per-map majority | Small ordered family selector | ordered family IDs | alpha prediction head |
| Alpha stack | Base-only/uniform blend | Lean U-Net/FPN regressor | one ordered alpha stack | texture identity head |
| Visual detail | Spec 113 RRDB floor | Spec 113 RealPLKSR | detailed RGB | numeric terrain truth |

Pretrained weights are optional ablations. Any Hub artifact must be license-recorded and pinned to
an immutable revision/hash; every stage must retain a from-scratch/local baseline on the same split.

## Build the Spec 114 reconstruction curriculum (T010 handoff)

The builder reads the frozen dual-source curriculum store and applies the admission policy. Against
today's store (which predates the corrected compositor) the dry run MUST report 1,629 authored rows
kept (808+ Kalimdor/Azeroth train/validation per the frozen split) and 1,361 synthetic rows
excluded under `synthetic_stale_lighting`. After the Spec 113 rerender freezes
`NoonWhiteGlobal` provenance on a rebuilt store, the same command admits both views unchanged.

```powershell
# Dry run first: prints the schema-valid summary, writes nothing.
uv run python scripts/v50_build_reconstruction_curriculum.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --output ../output/datasets/v50/v50.1/reconstruction-direct-v1 `
  --curriculum-id reconstruction-0_5_3_3368-dual-v1

# Then persist summary.json + selection.parquet (refuses to overwrite a non-empty directory).
uv run python scripts/v50_build_reconstruction_curriculum.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --output ../output/datasets/v50/v50.1/reconstruction-direct-v1 `
  --curriculum-id reconstruction-0_5_3_3368-dual-v1 `
  --write
```

Seconds to run; CPU-only; the user launches it. Verify the printed `input_origins`,
`excluded_counts`, and `group_leak_count: 0` before any training.

## Next geometry candidates — user-owned CUDA training

Dry run (validated plan, no output, no CUDA):

```powershell
uv run python scripts/v50_train_direct_geometry.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --source authored `
  --architecture mit_b0_regression `
  --run-id mit_b0-authored-v1 `
  --output ../output/v50/v50.1/direct_geometry/mit_b0-authored-v1 `
  --epochs 100 --batch 16 --workers 0 --patience 15 --seed 114
```

If the plan reports 1,384 train / 245 validation authored rows and ~3.8M parameters, launch:

```powershell
uv run python scripts/v50_train_direct_geometry.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --source authored `
  --architecture mit_b0_regression `
  --run-id mit_b0-authored-v1 `
  --output ../output/v50/v50.1/direct_geometry/mit_b0-authored-v1 `
  --epochs 100 --batch 16 --workers 0 --patience 15 --seed 114 `
  --amp --lr-schedule onecycle --clip 1.0 `
  --confirm-run
```

Expected on the RTX 4070 Ti SUPER: under 16 GB VRAM at batch 16 with AMP; recalibrate wall time
from the dry-run step count. Artifacts: `training_plan.json`, `run_identity.json`, both
checkpoints, fixed-row best-epoch previews, all-validation per-row metrics, error-quantile and
worst-case sheets, and `model_stage_run.json` (schema-validated, `promotion_verdict=pending`).
Promotion requires SC-001 + SC-002 + the user's visual review of the sheets.

**`mit_b0-authored-v1` outcome (2026-07-19)**: best epoch 93, val MAE 0.187802, SC-001 false
(tile-mean 0.138747), SC-002 true. Visually the strongest geometry to date — correct land/water
layout and mountain placement — but relief is smooth and under-amplituded: classic spectral bias
against the terrain's fractal ridge/drainage structure. The documented next ablation enables the
spectral guidance terms (Spec 068 US1 revived, loss-only, no deployment change):

```powershell
uv run python scripts/v50_train_direct_geometry.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --source authored `
  --architecture mit_b0_regression `
  --run-id mit_b0-authored-v2-spectral `
  --output ../output/v50/v50.1/direct_geometry/mit_b0-authored-v2-spectral `
  --epochs 150 --batch 16 --workers 0 --patience 20 --seed 114 `
  --amp --lr-schedule onecycle --clip 1.0 `
  --spectral-weight 0.1 --multiscale-weight 0.25 `
  --confirm-run
```

The `--source all` dual-view variants of these commands stay fail-closed until Gate 0 completes;
pretrained MiT encoder weights remain an optional FR-013 ablation (`--mit-pretrained` with pinned
`--mit-revision`/`--mit-sha256`), never the default.

## Residual detailer stage — the next lever after the v1/v2 plateau

Both coarse runs plateau at val MAE ≈0.19 while train loss sits at ≈0.016: the single stage has
extracted what 1,384 tiles can teach it. The detailer is a small independent residual refiner
(constitution IV) that takes RGB + the FROZEN coarse model's generated output and predicts only
`truth − coarse`. Final relief is `coarse + residual`. The coarse-only composition is the strong
baseline the detailer must beat by ≥5%.

**Step 1 — materialize the frozen coarse checkpoint's outputs** (CPU, seconds-to-minutes depending
on row count; dry run first):

```powershell
uv run python scripts/v50_materialize_coarse_relief.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --checkpoint ../output/v50/v50.1/direct_geometry/mit_b0-authored-v1/checkpoint_best.pt `
  --output ../output/datasets/v50/v50.1/coarse-mit_b0-authored-v1.zarr `
  --source authored --device cpu

# Then add --write to persist the derived coarse store (refuses to overwrite).
uv run python scripts/v50_materialize_coarse_relief.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --checkpoint ../output/v50/v50.1/direct_geometry/mit_b0-authored-v1/checkpoint_best.pt `
  --output ../output/datasets/v50/v50.1/coarse-mit_b0-authored-v1.zarr `
  --source authored --device cpu --write
```

The dry run prints the plan (selected rows, split counts, checkpoint hash, output shape); verify
`selected_rows` matches the coarse trainer's row count and the checkpoint hash matches
`run_identity.json` before writing.

**Step 2 — train the detailer** (USER runs CUDA; dry run first):

```powershell
uv run python scripts/v50_train_geometry_detailer.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --coarse-store ../output/datasets/v50/v50.1/coarse-mit_b0-authored-v1.zarr `
  --source authored `
  --run-id detailer-mit_b0-authored-v1 `
  --output ../output/v50/v50.1/direct_geometry/detailer-mit_b0-authored-v1 `
  --epochs 100 --batch 16 --workers 0 --patience 15 --seed 114 `
  --amp --lr-schedule onecycle --clip 1.0 `
  --spectral-weight 0.1 --multiscale-weight 0.25

# Then add --confirm-run to launch.
```

The detailer starts AT the coarse baseline (zero-initialized head), so epoch 1 should already
match `coarse_only` and improve from there. Promotion requires ≥5% relative val-MAE improvement
over `coarse_only`, SC-002, and your visual review of the fixed/quantile/worst sheets. The
`model_stage_run.json` records `upstream_models` naming the coarse checkpoint, so the detailer is
independently replaceable — swap the coarse checkpoint, re-materialize, retrain only the detailer.

## Deployment inference — tiles the model never saw (FR-015)

The geometry model has no tile-identity input: any authored 256x256 minimap tile runs through the
same contract. Dry run prints the per-tile manifest and writes nothing:

```powershell
uv run python scripts/v50_infer_direct_geometry.py `
  --checkpoint ../output/v50/v50.1/direct_geometry/mit_b0-authored-v1/checkpoint_best.pt `
  --input path\to\some_tile.png `
  --input path\to\folder_of_tiles `
  --output ../output/v50/v50.1/direct_geometry/inference-review
```

Add `--write` to persist, per tile: a 16-bit grayscale relative-relief PNG (`*_relief16.png`),
plus one fixed-scale side-by-side `review_sheet.png` and an `inference_manifest.json` binding each
input hash to the checkpoint hash and output hash. CPU is the default and is fast enough for
folders; `--device cuda` is optional. Inputs must be exactly 256x256 RGB (PNG/JPEG); anything else
is refused, never silently resampled. Outputs are RELATIVE relief per tile (contract `v112.1`) —
absolute world altitude is not identifiable from one minimap and remains a possible future
independent stage.

## Geometry promotion gate

The first Spec 114 checkpoint is promotable only when all of these hold:

- No WDL, ground-truth normal, height, alpha, material, or mask enters deployment inference.
- Authored and synthetic views share one group/split; leakage count is zero.
- Validation MAE beats flat/tile-mean and the strongest recorded Spec 112 result by at least 5%.
- Best epoch is later than epoch 1.
- Adjacent-border error passes SC-002 and the user accepts the held-out visual sheet.
- The run summary validates against `contracts/model-stage-and-curriculum.schema.json`.

Stop after this gate. Object-mask implementation is the next phase, not concurrent work.
