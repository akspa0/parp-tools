# Quickstart: Universal Image-to-Terrain Reconstruction

## Status — narrow authored-only baseline completed, measured, and rejected

The proven Spec 112 CNN is now pinned as Spec 114 architecture `direct_cnn_v112`. Its trainer can
select only authored rows, prints a validated no-training plan unless `--confirm-run` is present,
and refuses synthetic rows until the curriculum records corrected `NoonWhiteGlobal` provenance.
This was direct WoW-minimap RGB→relative-height training with no WDL prior. It is retained as
negative evidence, not as the deployment-domain definition.

The user-run 100-epoch result is frozen as evidence: best epoch 92 reached validation MAE 0.149267
against the 0.138747 per-tile constant baseline, a 7.59% regression. The checkpoint is retained as
a diagnostic baseline, not a promoted model. It also exposed a trainer handoff defect: there were no
prediction sheets, per-row metrics, gradient/border metrics, or reviewable error cases, and the
trainer omitted the repo's proven bounded optimization stack. Do not rerun the same command.

The separate evaluator subsequently measured the best checkpoint over all 245 held-out rows:

- MAE `0.1493349023` versus tile-mean baseline `0.1387469612` (`+0.0105879408`, failed);
- gradient MAE `0.0058671215`;
- border MAE `0.1607286124`;
- checkpoint epoch 92, with review artifacts in the separate validation directory.

The governing correction is larger than optimizer repair: deployment now means **any decodable
raster image → normalized view-axis relief → deterministic terrain mesh**. The next trainer must use
multiple visual/source families and whole-family holdouts. The v50 WoW corpus is exact top-down
supervision, not the complete input domain.

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
directory. The separate evaluator writes backfilled evidence to a separate output directory.

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

## Corrected universal geometry route

| Stage | Evidence/baseline | Candidate | Output | Explicitly excluded |
|---|---|---|---|---|
| Universal geometry | rejected Spec 112 CNN; constant and luminance relief | pinned DINOv2-small visual student + one continuous decoder | normalized `relative_relief` + deterministic mesh/UVs | WoW-only promotion, WDL, DepthAnything family, multi-head geometry |
| Broad pseudo-relief | no teacher | pinned DPT-Hybrid/MiDaS offline teacher | normalized teacher label only | teacher at deployment, unlabeled pseudo-truth |
| Object visibility | Empty/all-object baselines | Compact SegFormer semantic mask | one object mask | authored-minus-synthetic RGB labels |
| Terrain features | Majority family | Compact SegFormer semantic classifier | one feature map | shared geometry weights |
| Texture families | Per-map majority | Small ordered family selector | ordered family IDs | alpha prediction head |
| Alpha stack | Base-only/uniform blend | Lean U-Net/FPN regressor | one ordered alpha stack | texture identity head |
| Visual detail | Spec 113 RRDB floor | Spec 113 RealPLKSR | detailed RGB | numeric terrain truth |

Current Hub candidates are Apache-2.0, but their exact revisions and file hashes must be frozen
before any label build or training. The DPT teacher is offline supervision only. The deployable
student accepts the raster alone.

## Next commands are intentionally gated

Do not substitute the old `v50_train_height_relative.py` command here. The next user-run commands
are added only after T006–T020 land and pass lightweight proof:

1. build pinned broad-image pseudo-relief labels;
2. build the leak-safe multi-family universal curriculum;
3. preview the exact training plan without allocating CUDA training state;
4. train only after the preview reports at least five visual families, a non-empty whole-family
   holdout, and zero group/family leakage;
5. run any-image inference to emit relief preview, OBJ mesh, material/UV metadata, and validation
   sheet.

This gate prevents another expensive run against a contract that cannot meet the product.

Lightweight universal contract proof completed after the reset:

- `universal_relief_contract.py` accepts RGB/RGBA/grayscale and preserves non-square coverage by
  edge-padding only when smaller than a model tile and overlap-tiling at native aspect otherwise;
- stitched relief crops back to the exact source dimensions;
- constant inputs produce stable zero relief and a finite flat mesh;
- deterministic X/Z grid vertices, upward normals, triangles, full `[0,1]` UV coverage, and OBJ/MTL
  export are implemented;
- focused tests: 9 passed; Ruff, `py_compile`, and `git diff --check`: pass.

## Geometry promotion gate

The first universal Spec 114 checkpoint is promotable only when all of these hold:

- Any valid RGB/RGBA/grayscale test raster produces a finite continuous mesh and complete UVs.
- No WDL, teacher, ground-truth normal, height, alpha, material, or mask enters deployment inference.
- Every derived view shares its source group; group and whole-family leakage counts are zero.
- Whole-family paired holdouts beat both constant-relief and direct-luminance baselines by at least 5%.
- Adjacent-border error passes SC-004 and the user accepts at least 80% of the arbitrary-image sheet.
- The run summary validates against `contracts/model-stage-and-curriculum.schema.json`.

Stop after this gate. Object-mask implementation is the next phase, not concurrent work.
