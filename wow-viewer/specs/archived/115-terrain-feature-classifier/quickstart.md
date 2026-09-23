# Quickstart: Terrain Feature Classification for Geometry Deconfounding

## Status — US1 implemented and validated on real data; training is the next user-run step

Everything below Phase 1 and 2 is built, linted, and verified against the real 0.5.3.3368 corpus.
The label store exists on disk. The next action is a user-owned CUDA training run (Phase 3).

Proof already recorded:

- Full v50 suite **316 passed / 4 skipped**; 33 new Spec 115 tests; Ruff clean on all new modules.
- `dump-texture-names` verified against ground truth: Kalimdor tile 24,40 reproduces its true MTEX
  table (`DarkshoreSandRocks | DarkshoreSand | DarkshoreRockLighter | DarkshoreGrass`) exactly.
- Label coverage over the real curriculum: **2729 / 2990 rows labelled**, 261 excluded (no dump
  entry), **0 invalid pixels**, unknown driven from 10.02% down to **0.05%**.

## The number that shapes every training decision

| Family | Pixels | Share |
|---|---:|---:|
| terrain | 169,382,676 | 94.71% |
| structure | 7,343,974 | 4.11% |
| water | 1,557,054 | 0.87% |
| **road** | **471,556** | **0.26%** |
| unknown | 92,484 | 0.05% |

Road — the class this whole feature exists to isolate — is **0.26% of pixels**. A model that predicts
"terrain" everywhere scores ~95% pixel accuracy and finds zero roads. So the trainer weights the loss
by capped inverse frequency (road 50×, terrain 0.21×) and **gates on road IoU, never accuracy**, with
the majority-class baseline computed in-run so the degenerate solution is impossible to mistake for
success. 666 rows (24%) contain road pixels, which is ample row-level signal.

## Phase 1 — texture-name dumps (done; re-run only if the client changes)

```powershell
# From wow-viewer/data-harvester
dotnet ../tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll dump-texture-names `
  --client-root "H:\CLIENTS\0_5_3_3368" --map Kalimdor `
  --output ../output/v50/v50.1/texture-names/Kalimdor.json

dotnet ../tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll dump-texture-names `
  --client-root "H:\CLIENTS\0_5_3_3368" --map Azeroth `
  --output ../output/v50/v50.1/texture-names/Azeroth.json
```

Seconds to a couple of minutes each, CPU-only. Recovers the ordered MTEX table per tile — necessary
because the v50 store keeps only a *local* texture index and the global tileset→name list is not
persisted anywhere (see research.md Decision 1; the obvious `asset_inventory.parquet` substitute was
tested against the real client and falsified).

## Phase 2 — label store (done; already written to disk)

```powershell
uv run python scripts/v50_build_terrain_feature_labels.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --texture-names ../output/v50/v50.1/texture-names/Kalimdor.json `
  --texture-names ../output/v50/v50.1/texture-names/Azeroth.json `
  --output ../output/datasets/v50/v50.1/terrain-feature-labels-v1.zarr
# then re-run with --write to persist (refuses to overwrite a non-empty directory)
```

Dry run by default; prints full coverage and exclusion counts and fails closed if pixel or row
counts do not reconcile. Minutes on CPU.

## Phase 3 — train the classifier (USER RUNS THIS; next action)

Dry run first — validates contracts, prints the plan, allocates no CUDA state:

```powershell
uv run python scripts/v50_train_terrain_features.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --labels ../output/datasets/v50/v50.1/terrain-feature-labels-v1.zarr `
  --source authored `
  --run-id terrain_features-authored-v1 `
  --output ../output/v50/v50.1/terrain_features/terrain_features-authored-v1 `
  --epochs 60 --batch 16 --workers 0 --patience 12 --seed 115
```

Expect the plan to report **1368 selected rows / 1166 train / 202 val**, 73 steps per epoch, and
class weights `road 50.0, terrain 0.211, water 22.97, structure 4.87`. If it does, launch:

```powershell
# ...same command plus:
  --amp --confirm-run
```

Expected on the RTX 4070 Ti SUPER: comfortably inside 16 GB at batch 16; the model is 1,562,693
parameters — the same capacity class as the existing geometry models — so wall time should be in the
same envelope as their runs. Recalibrate from the first epochs' console output.

The console prints `road_iou`, `macro_iou`, and `pixel_accuracy` per epoch. **Watch road_iou; ignore
accuracy.** Artifacts: `checkpoint_best.pt` (selected on best road IoU) and a schema-validated
`model_stage_run.json` with `stage: "terrain_features"` and `promotion_verdict: pending`.

## Phase 4 — the out-of-distribution gate (the one that actually matters)

This is the test the geometry model failed and the reason this feature exists. It runs on the
`ek.jpg` tiles, which have **no client-derived ground truth of any kind**:

```powershell
uv run python scripts/v50_infer_terrain_features.py `
  --checkpoint ../output/v50/v50.1/terrain_features/terrain_features-authored-v1/checkpoint_best.pt `
  --input output/ek_tiles_256 `
  --output ../output/v50/v50.1/terrain_features/ood-review `
  --write
```

CPU is fine. Writes one `*_features.npy` per tile (5×256×256 class probabilities), a
`review_sheet.png` showing `[input | predicted classes | road probability]`, and an
`inference_manifest.json` binding every input hash to the checkpoint hash and output hash.

**Promotion requires**: road IoU beats the in-run majority-class baseline on the held-out split, AND
your visual review of the OOD sheet shows the visibly road-like regions flagged as road (orange in
the class panel, bright in the road-probability panel). If the OOD sheet fails, the model memorised
per-map texture statistics rather than learning road appearance — that is a real failure, and the
held-out numeric win alone does not override it.

## Phase 5 — geometry retrain (US2) — NOT YET IMPLEMENTED

Deliberately not built, because it consumes a promoted classifier checkpoint that does not exist
until Phase 3 runs. Scoped in plan.md Phase C:

- `direct_geometry_model.py` gains `in_channels` (default 3, folded into the architecture config
  hash so existing RGB-only checkpoints stay bit-identical and cannot be confused with a
  deconfounded one);
- `v50_materialize_feature_maps.py` runs the promoted classifier across curriculum rows;
- `direct_geometry_train.py` gains `--feature-store` plus the road-region error metric FR-008
  requires.

US3 (re-pairing the detailer against the new coarse checkpoint) reuses the existing
`v50_materialize_coarse_relief.py` / `v50_train_geometry_detailer.py` scripts unchanged.
