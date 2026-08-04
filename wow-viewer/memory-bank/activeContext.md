# Active Context — wow-viewer

Last updated: 2026-08-03

## CURRENT: Spec 130 PM4 decode — walls decoded, coordinate frames still wrong

Branch `130-pm4-remaining-decode`. Spec + plan + research + contracts committed; implementation
partially started. **Pick up here.**

### Landed and verified

- **MSPV/MSPI decoded: it is a vertical planar QUAD MESH — the walls between the MSUR floors.**
  Corpus-wide over 616 files: 98.05% of MSLK path windows hold exactly 4 indices, 99.6% coplanar,
  and **zero of 598,790 faces have Z as their dominant normal** (mean |normal.Z| = 0.000) against
  MSUR's 91.7% Z-dominant. Polyline and triangle-list readings eliminated with evidence.
  `pm4 connective-geometry`, with 6 detector-power tests that prove the discriminator separates a
  quad from a collinear run from a triangle list *before* any corpus claim.
- **Walls now render in the viewer**, toggle at Tools > PM4 > Overlay, emitting outlines as well as
  triangles so they appear in the default wireframe view.
- **Two viewer bugs fixed**: `Tools > PM4` switched on `_activeBottomTabIndex` (the Tools selector),
  so it always rendered Correlation and five panels were unreachable; and walls emitted triangles
  only, invisible unless Solid Fill was on.
- **`pm4 bounds-audit`** — new, and it found the coordinate bug below.
- **`pm4 mprr`** — the structural hypothesis is ELIMINATED (no chunk's entry count matches the run
  count; best is MPRL at 5/502 files). New finding: **94% of the 3,171,410 sentinel-delimited runs
  are exactly length 3 (75.5%) or length 7 (18.5%)** — small fixed-shape records, not a bulk index
  stream. No domain explains Value1 by bounds; MPRL is the *worst* of nine at 48.5%.

### THE OPEN BUG — read this first

**MSVT is stored in ABSOLUTE WORLD coordinates on both horizontal axes; Z is height.** Measured over
all 309 non-empty files: X lies inside the world band of the SECOND filename number 309/309 times,
Y inside the band of the FIRST 309/309, each spanning exactly one tile. `development_22_18` is the
clean case — 126,596 vertices, X = 9600.0..10133.3 (exactly column 18), Y = 11733.3..12266.7
(exactly row 22). Only `development_00_00` is also consistent with
tile-local storage, because there both indices are 0 and world equals local — which is why that one
tile always looked right and everything else piled up around it.

**CORRECTED LATER IN THE SESSION — read this part carefully.** That bounds measurement is
consistent with TWO readings and cannot separate them: (a) filename is ROW_COL with axes mapping
straight through, or (b) filename is ordinary COL_ROW and **MSVT's own X and Y are swapped relative
to world**. User ground truth settles it as **(b)**: the human tents in `development_01_00.pm4`
belong on the tile to the right of 0_0 in the same row — world (x=1, y=0) — and raw MSVT has X in
band 0 (41.0..52.9), Y in band 1 (778.9..790.6). Only `world = (MSVT.Y, MSVT.X)` places them right.

So **`Pm4PlacementMath` was CORRECT ALL ALONG** — its `XYPlaneZUp` case already applies that swap,
and the 7 `PlacementMath_*` tests that blocked the proposed "fix" were defending real behaviour.
**Do not invert those two lines.** `Pm4CoordinateService.Pm4LocalToAdtPlacement` now applies the
swap too. Lesson: a bounds fit proves which BAND a value is in, not which AXIS it means.

**Placement is still wrong and the cause is NOT settled.** Two objects sharing `region=146` are
wrong together and identically (correct tile, polar opposite) while a `region=6` object is wrong
differently (one tile off) — which looks like per-region frames keyed on `MSHD.Field04`. But two
things weaken that:

- **Phasing does not explain it.** The region-6 tents live in `development_01_00.pm4`, which has
  **no phased or destructible chunks at all**. Phasing may still matter for 00_00 specifically, but
  at least one misplaced population has nothing to do with it.
- **Region 0 is a shared bucket.** M2 content is generally local region PLUS region 0, spread across
  the whole map, so region 0 is not a located area and a pure region-keyed frame has to explain what
  frame it carries. The `--by-region` audit must treat region 0 separately.

Regions ARE confirmed real: region 245 is a whole ~2006 prototype zone for what became Sholazar
Basin, built from Feralas/vanilla plus Wrath assets. See `pm4-restoration-epic.md`.

### Next step, cheapest path

Add `--by-region` to `pm4 bounds-audit`: group MSVT bounds by `MSHD.Field04` and compare each region
against the tile band its filename implies. A small number of families (identity, negated,
axis-swapped, 180 out) means the frame is region-scoped and that family table is the fix. Uniform
behaviour kills the hypothesis. Then read the 7 PlacementMath tests and settle whether they encode
the bug or real intent.

### Test state

`WowViewer.Core.PM4.Tests`: 86 passed, 1 failed —
`Pm4RegionObjectGrouperTests.AnalyzeDirectory_DevelopmentCorpus_NonEmptyRegionsHaveObjects`,
**pre-existing**, confirmed failing at baseline with this session's changes stashed.

### Incidental

`pm4 inspect` and `pm4 audit` accept `--output` and silently ignore it; the other pm4 report
commands honour it.

---

## Earlier session: Spec 125/126 training — residual extractor + stacked height model

- **Spec 125 US7 residual extractor** trained and validated. Best epoch 54, val_mae=0.0893, beats_baseline=True. Guidance losses (multiscale/sobel/spectral/laplacian) added to trainer but showed marginal improvement vs baseline. Extractor converges at ~0.089 MAE on the curated rolling+steep regimes.
- **Residual→height feed-forward** (Spec 125 US4) failed conclusively: two runs (uncurated and curated) never beat the tile-mean baseline. The "learns then unlearns" oscillation confirmed the target is not learnable from single-view shading.
- **Forward-model-as-referee refinement** (Spec 126 US7) built and tested: differentiable hillshade forward model (height→normals→Lambert shading), optimization loop (Adam, shape_loss + affine-fit L1 + TV). Shading fits to 0.0103 MAE (92.9% better than flat), but recovered height correlates at r=0.0024 with ground truth — proving single-view shading does not carry enough information to recover height.
- **Direct minimap→height** (Spec 114): existing `direct_cnn_v112` (U-Net-lite, 1.56M params) and `mit_b0_regression` (HF SegFormer, 3.7M params) checkpoints exist but none beat the tile-mean baseline on their own validation splits. The v1/cnn model: best_val_mae=0.1878, beats_baseline=false. The v3-deconfounded: best_val_mae=0.1723, beats_baseline=false.
- **Stacked height model** (new): `direct_cnn_v112` extended to accept 4 input channels (RGB + frozen residual extractor output). The `--residual-checkpoint` flag loads the trained extractor as a frozen preprocessing step. The `HeightRelativeNet` now accepts `in_channels` parameter. The `build_geometry_model` allows 4-channel `direct_cnn_v112`. All spectral/fractal guidance, OneCycle, AMP, gradient clipping preserved.
- **Residual extractor inference** visual-review tool built: `residual_extractor_infer.py` produces 4-panel contact sheets (minimap, predicted residual, ground-truth residual, stripped albedo).
- **Deploy pipeline** built: `v50_deploy_height_to_mesh.py` chains minimap→MiT-B0→height→OBJ mesh export.

## Tile archaeology (2026-08-03) — parked, see `weak-signal-tile-archaeology.md`

Built and validated a weak-signal/white-plate tile pipeline (inventory, per-tile synthesis, 4-mode
whole-map composites incl. liquid, cross-build version diff). Ran on 0.5.3 (4 maps) and 4.0.0.11927
(Azeroth, Kalimdor, Deephome, Gilneas2, LostIsles). Full-corpus campaign across all clients was
dropped: ~237 GB needed vs 148.5 GB free, and level counting is biased on non-alpha clients until
the MCVT-delta issue is fixed.

**The one training-relevant result**: `surviving_height_levels` (count of distinct heights per tile)
should gate curation in BOTH directions — it excludes 127 tiles currently in the corpus that hold
<=64 distinct heights (four Azeroth tiles hold **2** across a 516-unit range, which under a per-tile
min-max target becomes a perfect binary step function), and admits 26 compressed-rich tiles whose
target is already correct and which are only excluded by curation. Not yet implemented.

## Active specs

| Spec | State |
|------|-------|
| 125 minimap-dxt1-inversion | US7 (residual extractor) trained and validated. US4 (residual→height) proven to not work. |
| 126 minimap-terrain-reconstruction | US7 (forward-model-as-referee) built and tested; single-view shading proven insufficient for height recovery. |
| 114 direct-terrain-reconstruction | Stacked height model (direct_cnn_v112 + residual channel) implemented; needs training. |

## Key files created/modified this session

- `src/harvester/v50/terrain_lighting_torch.py` — differentiable forward model (height→normals→Lambert shading)
- `src/harvester/v50/residual_extractor_infer.py` — visual-review contact sheets for extractor
- `scripts/v50_refine_height_from_residual.py` — forward-model-as-referee refinement
- `scripts/v50_deploy_height_to_mesh.py` — minimap→MiT-B0→height→OBJ deploy
- `src/harvester/v50/direct_geometry_model.py` — modified: direct_cnn_v112 accepts 4 input channels
- `src/harvester/v50/height_relative_model.py` — modified: HeightRelativeNet accepts `in_channels` parameter
- `src/harvester/v50/direct_geometry_train.py` — modified: `--residual-checkpoint` flag, frozen extractor preprocessing
- `src/harvester/v50/residual_extractor_train.py` — modified: added `--multiscale-weight`, `--sobel-weight`, `--spectral-weight`, `--laplacian-weight` guidance flags

## Stacked-height trainer: crash FIXED, ready to run (2026-08-03)

`--residual-checkpoint` crashed at the first best-epoch checkpoint: `expected input[1, 3, 256, 256]
to have 4 channels`. The residual channel was built ONLY in the trainer's own `RowDataset`; every
evaluation path (preview, final eval, road-region, object-region) rebuilt inputs independently with
RGB+features and handed 3 channels to a 4-channel model. Fixed with one shared builder,
`height_relative_evaluate.build_model_input_channels`, now the single source of truth for channel
order (`RGB -> residual -> features`), used by all five call sites. Also: extractor now loads AFTER
the dry-run gate (was allocating CUDA on plan-only runs); `direct_cnn_v112` 4-channel now hashes to a
distinct `config_sha256` (was colliding with the RGB-only baseline, 3-channel hash unchanged);
checkpoints record `residual_extractor` + `input_channels`.

**Known mismatch before running**: every extractor on disk (v2-v5) trained on `minimap_rgb_dxt1`;
`curriculum-0_5_3_3368-dual_v3.zarr` only has `minimap_rgb`. The trainer warns and records
`input_array_matches_training: false`. Treat the residual channel's contribution as a lower bound.

## Known open issues

- Stacked height model has not been trained yet — the crash above blocked it; it is now runnable.
- Existing direct minimap→height checkpoints (v1, v3) don't beat baseline.
- The residual→height feed-forward path is proven dead (beat_baseline=false, r=0.0024).
- The residual extractor is good for albedo-stripping, not height recovery.

## Durable constraints

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- User runs training, capture, client-backed proof, and heavy work.
- No DepthAnything/multi-head/shared-weight model paths.
- Constitution IV: per-signal evidence required.