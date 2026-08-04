# Progress — wow-viewer

Last updated: 2026-08-03

## Session: Spec 130 PM4 decode (2026-08-03)

### Done

1. **Spec 130 planned** — `plan.md`, `research.md` (10 findings), `data-model.md`, `quickstart.md`,
   6 contracts. Nine implementation phases; Phase 1 is a prior-art harvest.
2. **Prior art located**: ~60 files of PM4 object-assembly work sit unported on `main` in
   `parpToolbox/src/parpToolbox/Services/PM4/` (incl. `Pm4CrossTileObjectAssembler.cs` and a 56 KB
   `Pm4GroupingTester.cs`) plus `PM4Tool/docs/pm4/`. `AGENTS.md` line 345 already designates these
   extraction inputs. The public `akspa0/parp` repo is a dead end for geometry.
3. **The April 2025 "lost geometry" was never lost** — 15,096 verts = MSPV + MSVT, 7,382 faces =
   sum(MSUR.IndexCount - 2), i.e. MSUR as triangle fans, which the viewer already does. MSPI
   produced zero faces then, so connective geometry was never decoded in any era.
4. **MSPV/MSPI decoded** as a vertical planar quad mesh — the walls. `pm4 connective-geometry`,
   6 detector-power tests, corpus-wide evidence.
5. **Walls render in the viewer**; two viewer bugs fixed (PM4 tab unreachable, walls invisible in
   wireframe).
6. **`pm4 bounds-audit`** built; the raw-band measurement it produced is correct but was
   misinterpreted — see item 8.
7. **`pm4 mprr`** built; structural hypothesis eliminated, run lengths found to be 94% length-3 or
   length-7.
8. **Coordinate frames solved against ADT ground truth.** MSVT is ADT placement space —
   `placement = (17066.666 - MSVT.X, 17066.666 - MSVT.Y, MSVT.Z)`, no axis swap, because
   `MSVT.X == MDDF.rawY` and `MSVT.Y == MDDF.rawX`. 55,978/60,560 (92.4%) of real MDDF/MODF
   positions land inside their paired PM4's footprint; the unswapped reading scores 0.7%.
   `Pm4PlacementMath.ConvertPm4VertexToWorld` turns out to be correct — its swap is cancelled by
   the viewer's world-to-renderer step — so the 7 `PlacementMath_*` tests were right to defend it.
9. **`pm4 bounds-audit --by-region`** built; **region-scoped frames refuted** (1,877/1,895 objects
   on one frame, 1,892/1,895 with zero tile displacement, 61 of 62 multi-file regions homogeneous).
10. **`pm4 yaw-evidence`** built; the yaw correction is **disproven**. Scored against MODF world
    bounding boxes over 127 objects whose box can see a rotation (each proven by a 45° control):
    canonical 93.3% inside, fitted yaw 88.2%, control 79.0% — yaw hurts 96, helps 3, ties 28. It
    moves geometry the same direction as a known-wrong rotation. `WG_GATE01.WMO` drops 100% → 50%.

11. **The fitter is out of the render path.** `WorldScene.ResolveCk24CoordinateModeResolution` now
    returns a constant canonical resolution and `WorldScene.ResolvePlacementSolution` uses the
    identity planar transform with zero yaw, keeping the real world centroid as the pivot.
    `Pm4PlacementMath` is untouched, so all 16 `PlacementMath_*` tests still pass. The clincher was
    that **MSCN and MSPV already rendered correctly** via `(MapOrigin - p.X, MapOrigin - p.Y, p.Z)`
    with no fitter, while the MSUR mesh — the one path that used it — did not.

### Open

**Not visually confirmed yet.** The viewer held a file lock during the build, so the exe was not
replaced. Close it, rebuild, and check the tents on tile 1_0.

### Test state

102 passed / 1 failed (`Pm4RegionObjectGrouperTests`, pre-existing, confirmed at baseline).

### Not started

Spec 130 Phases 1-6 and 8-9 proper (prior-art harvest, evidence register, grouping-rule harness,
candidate rules, object identity, reconstruction). `/speckit.tasks` not yet run.

## Session: Spec 125/126 training — residual extractor + stacked height model

### What was done

1. **Residual extractor (Spec 125 US7)** trained and validated across multiple runs:
   - `tmp-x`: best_epoch=100, val_mae=0.08815, rolling -25.2%, steep -18.4%, beats_baseline=True
   - `residual-extractor-v2`: best_epoch=92, val_mae=0.08941, beats_baseline=True
   - `residual-extractor-v5-guided` (with spectral/sobel/multiscale/laplacian guidance): best_epoch=54, val_mae=0.08929, beats_baseline=True — guidance showed marginal improvement
   - Verdict: extractor converges at ~0.089 MAE, shading information fully extracted.

2. **Residual→height feed-forward (Spec 125 US4)** failed conclusively:
   - `spec125-height-01` (uncurated, all 685 rows): best_val_mae=0.1333, baseline=0.0982, beats_baseline=FALSE
   - `spec125-height-02-curated` (rolling+steep only, 459 rows): best_val_mae=0.2095, baseline=0.1733, beats_baseline=FALSE — "learns then unlearns" oscillation
   - Verdict: residual→height proven not learnable from single-view shading. The residual is shading (normals), not height.

3. **Forward-model-as-referee (Spec 126 US7)** built and tested:
   - `terrain_lighting_torch.py`: differentiable forward model (height→normals→Lambert shading)
   - `v50_refine_height_from_residual.py`: optimization loop, Adam optimizer, shape_loss + affine-fit L1
   - Result: shading fits to 0.0103 MAE (92.9% better than flat), but recovered height correlates at r=0.0024 with ground truth
   - Verdict: single-view shading does not carry enough information to recover height. This is a provable information limit, not a training problem.

4. **Direct minimap→height (Spec 114)** audit:
   - `direct_cnn_v112-authored-v1`: best_val_mae=0.1878, beats_baseline=FALSE
   - `mit_b0-authored-v3-deconfounded`: best_val_mae=0.1723, beats_baseline=FALSE
   - Verdict: existing checkpoints don't beat baseline. Needs retraining.

5. **Stacked height model** (new architecture):
   - `HeightRelativeNet` extended with `in_channels` parameter
   - `direct_cnn_v112` accepts 4 input channels (RGB + frozen residual extractor)
   - `direct_geometry_train.py` has `--residual-checkpoint` flag
   - All spectral/fractal guidance, OneCycle, AMP, gradient clipping preserved
   - NOT YET TRAINED — the `--residual-checkpoint` flag was just implemented

6. **Supporting tools built**:
   - `residual_extractor_infer.py` — visual-review contact sheets
   - `v50_deploy_height_to_mesh.py` — minimap→MiT-B0→height→OBJ mesh deploy
   - `v50_refine_height_from_residual.py` — forward-model-as-referee refinement

### Test suite state

- Full data-harvester suite: ~1150 passed / ~45 skipped / 3 pre-existing unrelated failures — unchanged.

### User-run gates still open

- Stacked height model (direct_cnn_v112 + residual channel) needs training
- Existing direct minimap→height checkpoints need retraining
- Forward-model-as-referee proved single-view shading insufficient for height

### Historical summary

- Spec 125 residual extractor: WORKS (albedo-stripping, shading extraction)
- Spec 125 residual→height: DEAD (beat_baseline=FALSE, r=0.0024, three independent approaches agree)
- Spec 126 US7 forward-model-as-referee: BUILT but proves the information limit
- Spec 114 stacked height model: IMPLEMENTED but not yet trained
---

## Training plan (drafted 2026-08-03, nothing run)

No GPU work executed. This is the ordered plan for the next training session.

### Blocker cleared this session

`direct_geometry_train.py --residual-checkpoint` crashed at the first best-epoch checkpoint
(`expected input[1, 3, 256, 256] to have 4 channels`). Root cause: the residual channel was built
only in the trainer's own `RowDataset`; four other inference paths rebuilt inputs independently.
Fixed via one shared `build_model_input_channels` used by all five sites. The stacked model is now
runnable — it was never a model problem.

Existing `output/runs/stacked-height-v1/` is the crashed attempt (reached epoch 1, val_mae 0.28126,
`input_channels` absent = pre-fix checkpoint). `require_new_output` refuses non-empty dirs, so the
next run needs a fresh path (`stacked-height-v2`). Do not reuse or delete v1 — it is the negative
record of the crash.

### Step 1 — decide the curation change BEFORE spending GPU time

`surviving_height_levels` (distinct height count per tile, in `tile_inventory.py`) should gate the
curriculum in both directions. Measured on the 0.5.3 corpus:

- **Exclude**: 127 tiles currently classified usable/terrain_no_minimap hold <=64 distinct heights.
  Four Azeroth tiles (29_24, 30_24, 31_24, 32_24) hold exactly **2** across a 516-unit range. Under
  the v112.1 per-tile min-max target these become perfect binary step functions — actively teaching
  the model that a texture edge is a 500-unit vertical wall.
- **Admit**: 26 compressed-rich tiles are excluded by curation today but their target is ALREADY
  correct. Amplification above `RANGE_FLOOR` is provably a no-op on the target (verified: x1 and
  x1,000,000 give bit-identical targets, because min-max normalization cancels affine scale). They
  need un-excluding, nothing more.
- 7 further rich tiles sit below `RANGE_FLOOR = 1.0`; those need the floor gated on level count.
  Do NOT lower the floor globally — it would amplify the 144 two-to-eight-level tiles into
  full-range targets.

This is cheap, CPU-only, and changes what the model sees. Do it first.

### Step 2 — stacked height run

`--architecture direct_cnn_v112 --source authored --residual-checkpoint <extractor>`, fresh output
dir. Extractor candidates by best val_mae: **v4 = 0.08840**, v5-guided = 0.08929, v3 = 0.08941,
v2 = 0.09119. v5-guided carries the multiscale/sobel/spectral/laplacian guidance.

Prior split for reference: authored source, 1384 train / 245 val, onecycle + AMP, 100 epochs.

**Confound to record in the run identity, not discover afterwards**: every extractor on disk trained
on `minimap_rgb_dxt1`; `curriculum-0_5_3_3368-dual_v3.zarr` has only `minimap_rgb`. The frozen
extractor runs out-of-distribution. The trainer warns and records
`input_array_matches_training: false`. Any result is a LOWER BOUND on the residual channel's value.
If the stacked run underperforms, this is the first thing to rule out — not evidence the channel is
useless.

### Comparison targets

- tile-mean and flat baselines (computed in-run)
- `SPEC112_FROZEN_BEST_VAL_MAE = 0.1492665126`
- SC-001 requires beating all three by 5% relative
- prior direct runs: v1/cnn 0.1878, v3-deconfounded 0.1723 — neither beat baseline

### Explicitly not on the path

- Full-corpus multi-client harvest — dropped, see `weak-signal-tile-archaeology.md`
- Residual->height feed-forward — dead (r=0.0024, three approaches agree)
- Spec 127 viewer explorer — drafted, unimplemented, not a training dependency
