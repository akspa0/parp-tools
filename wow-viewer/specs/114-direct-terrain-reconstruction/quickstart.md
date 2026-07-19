# Quickstart: Direct Minimap-to-Terrain Reconstruction

## Status — authored-only baseline completed and failed promotion

The proven Spec 112 CNN is now pinned as Spec 114 architecture `direct_cnn_v112`. Its trainer can
select only authored rows, prints a validated no-training plan unless `--confirm-run` is present,
and refuses synthetic rows until the curriculum records corrected `NoonWhiteGlobal` provenance.
This is direct RGB→relative-height training with no WDL prior.

The user-run 100-epoch result is frozen as evidence: best epoch 92 reached validation MAE 0.149267
against the 0.138747 per-tile constant baseline, a 7.59% regression. The checkpoint is retained as
a diagnostic baseline, not a promoted model. It also exposed a trainer handoff defect: there were no
prediction sheets, per-row metrics, gradient/border metrics, or reviewable error cases, and the
trainer omitted the repo's proven bounded optimization stack. Do not rerun the same command;
T017a/T017b repair observability and optimization first.

Real dry-run proof on the frozen curriculum:

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

## Completed bootstrap command — do not rerun unchanged

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

Actual artifacts are `training_plan.json`, `run_identity.json`, `checkpoint_best.pt`,
`checkpoint_last.pt`, and `training_summary.json`. Do not rename, reuse, or overwrite the run
directory. The T017a evaluator writes backfilled evidence to a separate output directory.

## Backfill the missing validation evidence from the failed checkpoint

This is a bounded user-run evaluation over the 245 held-out rows. It performs no training and does
not modify the original run directory. From `wow-viewer/data-harvester`:

```powershell
uv run python scripts/v50_evaluate_height_relative.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --checkpoint ../output/v50/v50.1/direct_geometry/direct_cnn_v112-authored-v1/checkpoint_best.pt `
  --source authored `
  --output ../output/v50/v50.1/direct_geometry/direct_cnn_v112-authored-v1-validation-v1 `
  --batch 16 --workers 0 --device cuda
```

It writes `summary.json`, `per_row_metrics.json`, `error_quantiles.png`, `worst_cases.png`, and
`evaluation_identity.json`. Every height panel uses the same fixed `[0,1]` scale. The sheets include
the real RGB input, truth, prediction, per-tile constant baseline, signed error, and absolute error;
they must be reviewed before selecting the optimized retry.

Lightweight proof completed before handoff:

- focused model/trainer/curriculum tests: 23 passed; full v50 focus: 178 passed / 4 skipped;
- Ruff and `py_compile`: pass;
- real authored-only dry run: 1,629 selected, 1,384 train, 245 validation, 87 steps/epoch;
- real `--source all` dry run: correctly refused for missing corrected-light provenance;
- dry runs created no output directory.

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

## Later planned commands — corrected dual-view bakeoff

From `wow-viewer/data-harvester`:

```powershell
# After T008: build the leak-safe authored + corrected-synthetic geometry curriculum.
uv run python scripts/v50_build_reconstruction_curriculum.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr `
  --output ../output/datasets/v50/v50.1/reconstruction-direct-v1.zarr

# After T014/T015: user-run candidate on the corrected identical split/target.
uv run python scripts/v50_train_direct_geometry.py `
  --store ../output/datasets/v50/v50.1/reconstruction-direct-v1.zarr `
  --architecture mit_b0_regression `
  --output ../output/v50/v50.1/direct_geometry/mit_b0_regression `
  --confirm-run
```

The later runtime estimate must be recalibrated from that trainer's no-training dry run. Object,
feature, texture, and alpha commands are added only when their preceding phase passes.

## Geometry promotion gate

The first Spec 114 checkpoint is promotable only when all of these hold:

- No WDL, ground-truth normal, height, alpha, material, or mask enters deployment inference.
- Authored and synthetic views share one group/split; leakage count is zero.
- Validation MAE beats flat/tile-mean and the strongest recorded Spec 112 result by at least 5%.
- Best epoch is later than epoch 1.
- Adjacent-border error passes SC-002 and the user accepts the held-out visual sheet.
- The run summary validates against `contracts/model-stage-and-curriculum.schema.json`.

Stop after this gate. Object-mask implementation is the next phase, not concurrent work.
