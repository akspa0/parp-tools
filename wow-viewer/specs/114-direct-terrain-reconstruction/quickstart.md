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

The DINOv2 student initialization is Apache-2.0 and pinned by exact revision/file hash. It is model
initialization only; all training rows in tonight's route come from our v50 store. The deployable
student accepts the raster alone.

## Universal build and training commands

Do not substitute the old `v50_train_height_relative.py` command here. The next user-run commands
are separate, fail-closed stages:

1. build an index over our exact v50 RGB/height pairs with one whole map held out;
2. preview the exact training plan without allocating CUDA training state;
3. train only after the preview reports train, validation, and compatibility rows plus zero leakage;
4. run any-image inference to emit relief preview, OBJ mesh, material/UV metadata, and validation
   sheet.

Each CLI is a dry run unless its explicit confirmation flag is present. This prevents another
expensive run against a contract that cannot meet the product.

### Build the exact curriculum from our v50 datastore

The first training run uses our existing v50 corpus only. It reads `minimap_rgb` as input and
`height_257` as exact truth directly from the immutable dual curriculum store; it does not export
PNGs, create arbitrary-image folders, download another dataset, or run a relief teacher.

Preview an exact authored-only curriculum with all Azeroth rows reserved as the whole-map
compatibility set:

```powershell
cd I:\parp\parp-tools\wow-viewer\data-harvester
uv run python scripts/v50_build_universal_relief_curriculum.py `
  --v50-store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --v50-source authored `
  --holdout-map Azeroth `
  --output ../output/datasets/v50/v50.1/universal-relief-exact-authored-v1.zarr
```

The validated dry run reports 1,629 exact rows: 808 Kalimdor train, 143 Kalimdor validation, and 678
Azeroth compatibility; `teacher_pseudo=0`; both leak counts are zero. It writes nothing. If those
values match, rerun the same command with `--confirm-build` appended. This creates only an immutable
index/summary that references our existing arrays; it does not duplicate the datastore.

The authored-only route is deliberate for tonight because that RGB is unaffected by the stale
synthetic-light provenance. After corrected `NoonWhiteGlobal` synthetic RGB is rebuilt into a store
that records that contract, a later run can use `--v50-source all` without changing the model.

### Universal relief training

Once the curriculum preview/build reports 808 train, 143 validation, 678 whole-map compatibility,
1,629 exact rows, zero pseudo rows, and zero leakage, preview the training plan:

```powershell
uv run python scripts/v50_train_universal_relief.py `
  --curriculum ../output/datasets/v50/v50.1/universal-relief-exact-authored-v1.zarr `
  --output ../output/v50/v50.1/universal_relief/dinov2-small-exact-authored-v1 `
  --epochs 50 --batch 8 --workers 0 --patience 10 --seed 114
```

The preview writes nothing and does not download the student. Verify the source identity, 808/143/
678 row split, `wow_authored:Azeroth` holdout, deployment input `rgb`, height/normal/liquid guidance,
AMP/EMA/OneCycle settings, validation-only checkpoint selection, and output path. Then the user
launches the CUDA run by appending:

```powershell
  --device cuda --confirm-run
```

The first confirmed run downloads the pinned 88.2 MB DINOv2-small safetensors, freezes that general
encoder by default, and trains only the compact relief decoder. Budget roughly 30–120 minutes on the
local RTX 4070 Ti SUPER, then replace that estimate with the plan's actual `steps_per_epoch`. The run refuses a
non-empty output and writes `training_plan.json`, `history.json`, `checkpoint_best.pt`,
`checkpoint_last.pt`, per-row metrics, best/final/worst visual sheets, and
`training_summary.json`.

Checkpoint selection uses only Kalimdor validation MAE. Azeroth never selects an epoch; it is used
only for the stricter whole-map promotion check against per-tile constant and direct-luminance
baselines, where MAE and gradient MAE must beat all four comparisons by at least 5%.

### Any raster to textured terrain

After a checkpoint promotes, preview conversion of any decodable RGB, RGBA, or grayscale image:

```powershell
uv run python scripts/v50_image_to_terrain.py `
  --image I:\parp\parp-tools\wow-viewer\output\source-images\inference\any-image.png `
  --checkpoint ../output/v50/v50.1/universal_relief/dinov2-small-relief-v1/checkpoint_best.pt `
  --output ../output/v50/v50.1/universal_relief/inference/any-image-v1 `
  --mesh-max-resolution 257 --extent-x 533.3333333333 --vertical-scale 128
```

The dry run reads and identifies the raster/checkpoint but writes nothing and does not download the
student. Add `--device cuda --confirm-run` to emit `source.png`, 16-bit normalized relief,
`terrain.obj`, `terrain.mtl`, `validation.png`, and `manifest.json`. The source raster is UV-mapped
onto the mesh immediately. For perspective photographs or artwork this is a view-axis bas-relief
terrain interpretation, not a claim of unique metric scene geometry.

Lightweight universal contract proof completed after the reset:

- `universal_relief_contract.py` accepts RGB/RGBA/grayscale and preserves non-square coverage by
  edge-padding only when smaller than a model tile and overlap-tiling at native aspect otherwise;
- stitched relief crops back to the exact source dimensions;
- constant inputs produce stable zero relief and a finite flat mesh;
- deterministic X/Z grid vertices, upward normals, triangles, full `[0,1]` UV coverage, and OBJ/MTL
  export are implemented;
- focused tests: 9 passed; Ruff, `py_compile`, and `git diff --check`: pass.
- teacher-label tests: 7 passed; CLI help, Ruff, `py_compile`, and dry-run/no-output contract: pass.
- universal curriculum tests: 12 passed; exact-v50-only whole-map holdout, optional mixed-teacher,
  source/target drift, content-relabel, and full Parquet-lineage gates; the real authored dry run
  reports 808/143/678 and writes nothing.
- universal student tests: 7 passed; pinned full DINOv2-small revision/safetensors SHA, RGB-only one-
  relief output, finite bounded forward, frozen-backbone discipline, full-resolution RGB detail
  retention, explicit unfreeze ablation, and fail-closed weight/channel/patch validation. No student
  weights were downloaded.
- trainer/inference tests: 16 passed; exact/pseudo weighting, normal/liquid masks, multiscale and
  structural losses, D4/style augmentation, EMA, validation-only selection, whole-map promotion, fixed-scale named
  review sheets, arbitrary-aspect inference planning, checkpoint pin checks, and no-write dry runs.
- combined Spec 114 universal focus: 51 passed; Ruff and `py_compile` pass. No Hub download, real
  curriculum build, CUDA training, or inference run was launched.
- broader `tests/v50`: 224 passed / 4 skipped; only expected Zarr sidecar warnings remain.

## Geometry promotion gate

The first universal Spec 114 checkpoint is promotable only when all of these hold:

- Any valid RGB/RGBA/grayscale test raster produces a finite continuous mesh and complete UVs.
- No WDL, teacher, ground-truth normal, height, alpha, material, or mask enters deployment inference.
- Every derived view shares its source group; group and whole-family leakage counts are zero.
- Whole-family paired holdouts beat both constant-relief and direct-luminance baselines by at least 5%.
- Adjacent-border error passes SC-004 and the user accepts at least 80% of the arbitrary-image sheet.
- The run summary validates against `contracts/model-stage-and-curriculum.schema.json`.

Stop after this gate. Object-mask implementation is the next phase, not concurrent work.
