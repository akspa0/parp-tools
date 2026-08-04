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

### COORDINATE FRAMES — SOLVED against ADT ground truth

**MSVT is stored in ADT placement space**: a distance-from-origin coordinate, exactly like a raw
MDDF position, with its two horizontal fields in the **opposite order** to MDDF's
(`MSVT.X == MDDF.rawY`, `MSVT.Y == MDDF.rawX`). The conversion is a per-axis subtraction with **no
axis swap**:

```text
placement = (17066.666 - MSVT.X, 17066.666 - MSVT.Y, MSVT.Z)
```

**Evidence**: over the 179 development tiles holding both a PM4 and a correctly named `_obj0.adt`,
**55,978 of 60,560 (92.4%)** MDDF/MODF positions fall inside their paired PM4's MSVT footprint. The
unswapped alternative scores **412/60,560 (0.7%)** and is eliminated. Worked case —
`development_01_00.pm4` has MSVT X = 41.0..52.9, Y = 778.9..790.6; the three MDDF entries in
`development_1_0_obj0.adt` have rawY = 42.3..51.1 and rawX = 780.3..789.0, nested inside it.

Every earlier reading was a **bounds fit**, which proves only which BAND a value lies in — it cannot
see a reflection about the map centre, because reflecting a band yields a band. The raw band
measurement still reproduces (309/309, X in the band of the filename's SECOND number, Y in the
FIRST); what was wrong was reading a **distance-from-origin band as a map tile index**. The map tile
index is `31 - band`.

**`Pm4PlacementMath.ConvertPm4VertexToWorld` is correct — do not touch it.** It emits an
*intermediate* space the viewer finishes with `renderer = (MapOrigin - world.Y, MapOrigin - world.X,
world.Z)`. Composing the two reproduces the transform above, so its axis swap is cancelled by the
renderer's. The 7 `PlacementMath_*` tests are right, for that reason rather than the recorded one.

### Region-scoped frames — REFUTED

`pm4 bounds-audit --by-region` over **1,895 CK24 objects**, 309 files, 207 regions:

| measurement | result |
|---|---|
| objects on the canonical frame | 1,877 / 1,895 |
| regions spanning >1 file | 62 |
| ...of those with mixed frames | 1 |
| objects with zero whole-tile displacement | 1,892 / 1,895 |

No frame family table exists. And `MSHD.Field04` is a **per-file** header, so "objects in one region
fail identically" and "objects in one file fail identically" were always the same observation — the
hypothesis was unfalsifiable on the evidence that motivated it. Regions ARE still real authored
areas (region 245 is the ~2006 Sholazar prototype zone); they are just not a coordinate frame key.

### THE REMAINING BUG — the per-object MPRL fitter

The viewer runs `Pm4PlacementMath`'s fitter per CK24 group and it overrides the canonical frame:

- `ResolveCoordinateMode` picks `TileLocal` for data that is never tile-local, then adds tile offsets
  to already-absolute coordinates. 18 objects. **The human tents are one of them** —
  `development_01_00.pm4` (region 6) resolves `TileLocal/.UV.` and moves from canonical tile (0,1) to
  (1,-1). All 3 ADT placements for that tile sit inside its canonical footprint, so ground truth says
  canonical is right and the fitter is wrong.
- `TryComputeWorldYawCorrectionRadians` rotates **974 of 1,895 objects (51%)** by 15–45°, fitted
  against MPRL packed angles — and MPRL is the one chunk in a permuted frame. **Now disproven.**

### The yaw correction is wrong — `pm4 yaw-evidence`

The ADT containment test could not judge the yaw: it compares a *point* to a *box*, and a centroid
rotation moves neither. MODF carries a world bounding *box*, so rotating a non-square object inside
it ejects vertices. Objects are matched by **centroid containment**, which a centroid rotation cannot
change — matching on best fit would have selected for the answer it then reported.

1,066 objects matched to a WMO box; **127 have a box able to see a rotation**, proven per object by
a deliberate 45° control:

| geometry | mean vertices inside its WMO box |
|---|---|
| canonical, no yaw | **93.3%** |
| canonical + fitted yaw | 88.2% |
| full resolved solution | 89.5% |
| 45° control (known wrong) | 79.0% |

**yaw hurts 96, helps 3, tie 28** — the fitted yaw moves geometry the same way the known-wrong
control does. Worst cases are real WMOs that fit perfectly without it: `WG_GATE01.WMO` 100% → 50%,
`WG_WALL01.WMO` 100% → 82%, `WALLPIECE01.WMO` 66% → 47%. The 401 objects whose box cannot see a
rotation are excluded from that headline and counted separately.

Reports are in `output/pm4-decode/`: `region-frame-audit.json`, `yaw-evidence.json`.

### The fix — LANDED

The clincher was the user's own observation: **MSCN nodes render correctly at the tents while the
MSUR mesh does not**, in the same file. `EnsurePm4MscnData` and `EnsurePm4MspvData` place points with
`(MapOrigin - p.X, MapOrigin - p.Y, p.Z)` — the canonical transform, applied raw, no fitter. The mesh
was the only path going through `ResolvePlacementSolution`. MSPV/MSVT/MSCN share one chunk frame, so
the mesh had no business using a different one.

`WorldScene.ResolveCk24CoordinateModeResolution` now returns a constant canonical resolution, and
`WorldScene.ResolvePlacementSolution` builds the solution with the identity planar transform and
**zero yaw**, keeping only the real world centroid as the pivot (selection and connector merging need
it). `Pm4PlacementMath` is untouched, so all 16 `PlacementMath_*` tests still pass — the render path
simply no longer asks it to fit anything.

### The cache trap — read this before debugging any placement change

The code fix was correct and compiled, and the viewer **still drew the old positions**. Two disk
caches store geometry *after* the placement transform — triangles, lines, `PlacementAnchor`, bounds
and the resolved planar flags are all post-transform — so a cached tile replays whatever placement
was in effect when it was written:

| cache | magic | version constant |
|---|---|---|
| tile overlay | `PM4C` | `Pm4OverlayCacheService.CacheVersion` |
| per-file | `PM4F` | `Pm4PerFileCacheService.EntryVersion` |

959 MB across 402 blobs were being replayed. **Both versions must move together** for any change to
placement semantics, even when the byte layout is unchanged — the version check is the only
invalidation. Both are now at **9**. Cache root: `<app bin>/output/cache/pm4-overlay/<id>/`.

**Not yet visually confirmed.** The viewer builds clean with the app closed; launch it and the stale
blobs will be rejected and rebuilt (first load is slow).

**The arithmetic that predicts success**: `development_01_00`'s MSVT centroid is ≈ (46, 784), so the
canonical transform puts it at `(17066.666 - 46, 17066.666 - 784)` = **(17020.7, 16282.7)**. The
viewer was drawing it at (16578.4, 16780.2) — a displacement of **≈666**, and the hover tooltip
reported gaps of **657.4 / 658.3 / 664.2** to the three real `HU_TENT02.M2` instances. That match to
within the tents' own ~12-unit spread is what proved the old code path was still executing rather
than the new transform being wrong. After the cache clears, that gap should fall to near zero.

### Also fixed this session

`Pm4CoordinateService.GetObj0PathForPm4` built a zero-padded name (`development_01_00_obj0.adt`) that
exists for **none** of the 616 corpus files, though 411 have a companion under the unpadded ADT
spelling (`development_1_0_obj0.adt`). It is now `TryGetObj0PathForPm4` and returns null instead.
**`Pm4CollisionDumper` still falls back to the first `*_obj0.adt` in the folder** — an arbitrary
unrelated tile — and should be fixed.

### Test state

`WowViewer.Core.PM4.Tests`: **96 passed, 1 failed** —
`Pm4RegionObjectGrouperTests.AnalyzeDirectory_DevelopmentCorpus_NonEmptyRegionsHaveObjects`,
**pre-existing**, confirmed failing at baseline. The 10 new `Pm4PlacementSpaceTests` all pass.

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