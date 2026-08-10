# Workstream — terrain / minimap ML

Owner specs: 114 (direct terrain reconstruction), 125 (minimap DXT1 inversion),
126 (minimap terrain reconstruction), 111 (minimap lighting calibration), 139 (clean-signal
reconstruction), 140 (paste/fractal/tileset evidence).
Last updated: 2026-08-10. **Nothing is training right now.** Spec 139 is the active implementation
route after the terrain-only Spec 134 control-v1 bakeoff; object-sieve and object-marker work is
parked.
No authored v60 real-data corpus has passed the albedo gate. A separate real-terrain synthetic
bridge now exists for diagnostic testing on actual client-terrain geometry.

This file is the durable home for the terrain-ML workstream. `activeContext.md` links here and
stays short; put detail here, not there.

## v50 construction correction — stale synthesized minimaps

The active failure is stale synthesis in the old v50 datastore. The synthesized minimap arrays
were not regenerated after the renderer's lighting fixes. The raw harvested terrain is not the
thing to re-diagnose here, and the 0.5.3 renderer remains the known-good control.

The old builder coupled synthesis to a fresh client-backed build, so there was no narrow refresh
operation. The existing `0_5_3_3368-Azeroth.zarr` has 43 all-zero rows in both synthesized
resolutions, which is direct evidence that its synthetic output needs regeneration.

`scripts/refresh_v50_synthetic_minimaps.py` now provides the bounded repair: use the existing tile
index, render fresh 256/1024 synthetic tiles, require every tile to be written and non-black, copy
the old store to a new path, and replace only `minimap_rgb` and `minimap_rgb_1024`. Raw terrain,
object masks, and other harvested signals remain untouched. The refreshed signal content identities
and coverage metadata are recomputed. The copied arrays use one tile per Zarr chunk because the
historical multi-row chunks caused Windows atomic-replacement collisions during row patching. The
0.5.3 Ghidra findings are separate future archaeology gates, not the explanation for this
stale-synthesis failure.

## 0.5.3 audit gate — no accepted real transfer corpus yet

The 2026-08-09 Ghidra audit of `WoWClient.exe` 0.5.3.3368 does not validate the existing v50/v60
real harvest as a training corpus. Native minimap BLP loading (`BuildPathName`/`SetupTextureHandles`)
is separate from terrain MCSH/LIT rendering and from WMO/dynamic object overlays. The current
pipeline therefore has these required repairs before a 0.5.3 transfer sample can be accepted:

- `AlphaWdtReader` must honor each `MCLY.offsAlpha`; sequential MCAL consumption can shift layers.
- Height extraction needs a written/presence mask; `FillHeightmapGaps` currently treats valid
  absolute zero heights as missing.
- Raw MCSH must stay a separately named, per-chunk packed-mask diagnostic. It is not native
  minimap shading and must not silently become `terrain_shadow_256`.
- Harvested `terrain_shadow_256` is currently synthetic and target-derived: the helper forces
  analytic cast shadows, uses the default tuning rather than the Alpha profile, and does not use
  `lights.lit`. It is not a real observed 0.5.3 label.
- Alpha MDDF/MODF masks are heuristic placement labels without MCRF, geometry visibility, or
  overlay semantics. Keep them auxiliary and separate from the requested screen-space sieve mask.

The direct Alpha harvester tile index, MCVT absolute-height mapping, MCNR transform, and minimap
BLP path are binary-consistent. The shared `AlphaTerrainAdapter` still needs its transposed `MAIN`
index fixed before viewer validation. External WDL and exact MCLQ vertex fields remain separate
proof tasks. Until these gates close, v60 remains a control corpus plus validation tooling, not a
working 0.5.3 real-data corpus.

## Active route — Spec 139 v7 clean-signal reconstruction

## Parallel route — Spec 140 paste/fractal/tileset evidence pipeline

The reconstruction architecture is now explicitly staged rather than monolithic. Spec 140 owns
the evidence and guidance surfaces around Spec 139: albedo-normalized observation confidence,
tileset/biome profiles, alpha and texture-layer descriptors, multiscale FBM/fractal descriptors,
cross-tile paste retrieval, and optional normalized object slots. It does not replace the clean
terrain model and does not make object identity a terrain prerequisite.

The first proof is classical/descriptive retrieval on deterministic synthetic controls: flat,
smooth, hilly/mountainous, island, sheer-dropoff, FBM/ridged, lightning/burn, and patterns that
cross tile boundaries with arbitrary offsets. The atlas must show height, alpha, texture-layer
identity, albedo-normalized observation, auxiliary tileset channels when present, and object
evidence as separate rows. Real 0.x/1.x samples are a small transfer/validation slice, not a
reason to resurrect the broad v50 harvest as training truth.

Guidance is accepted only when it carries a family ID or explicit unconfirmed state, transform,
source provenance, confidence, and content hash. The downstream comparison is parity versus
motif-guided versus tileset-guided Spec 139, with per-signal and seam metrics. The first user-run
gate is the visual atlas and retrieval report; no training command is yet valid.

### New hypothesis: alpha-painted intent precedes sculpted relief

The developer-map evidence now suggests the ordering to test is opaque layer-0 base/“brain”
texture, layer-1 pasted rock/mountain motifs, later alpha painting of intended regions, then height
sculpting and surface refinement. This changes alpha from a passive correlated channel into a
possible upstream latent scaffold for geometry reconstruction.

The first implementation must recover an evidence-bearing paint order, not claim literal editor
history. Preserve MCLY layer order, MCAL offsets, texture IDs, and the layer-0/layer-1 boundary.
Layer 0 is opaque base; layer 1 is the first paste/paint candidate. Derive cumulative and
incremental occupancy hypotheses, then compare them to height curvature/slope and paste families.
A real tile is classified `intact`, `retextured`, `resculpted`, `unknown`, or
`insufficient_data`; a retextured zone is useful evidence of a broken relationship, not a reason
to force current alpha to explain current relief.

The brush evidence has three linked resolutions. The early Python alpha-component path supplies
localized `atomic_brush` candidates. The later C# full-map path supplies `paste_block` children and
`macro_prefab_context` parents that preserve cross-tile and mesh relationships. Both remain valid;
the C# path is not a failed atomic extractor. Parent/child links, provenance, and per-scale metrics
are required before any record is used as guidance.

Alpha is the fan-out boundary for these analyses: retain every source alpha layer and provenance
first, then derive raw occupancy, transition/stroke, atomic, paste-block, macro-context,
ordered-layer, and cross-tile views independently. The pipeline must preserve disagreement between
views because different terrain families may expose different parts of the authored structure.
Missing or opaque alpha is an explicit availability state, never a fabricated empty mask.

The deployment distinction is strict: source-side alpha is supervision/archaeology, while a
minimap-only system must predict a confidence-bearing paint/sculpt-intent scaffold before the
Spec 139 geometry stage. Opaque layer 0 must never become a fabricated `alpha_0` tensor; layer 1
is the first paste/paint candidate. The first new proof is synthetic known-order controls plus a small real
0.x/1.x analysis slice; no GPU run is authorized by this hypothesis alone.

The existing synthetic validation workflow can later provide curriculum difficulty guidance from a
frozen reference checkpoint. It may prioritize `easy`, `learnable_hard`, and `pathological`
synthetics using per-signal and seam/boundary error, confidence, and coverage. It must not be used
to declare data stale, rewrite labels, or stand in for provenance.

### Terrain-only reset and v7 pivot (2026-08-10)

The former active experiment was `terrain_shadow_256` → `height_257` from the validated project-owned
`control-v1` NPZ corpus. Its four-model bakeoff rejected all candidates against the tile-mean
baseline (`0.191047`); `pyramid_cnn` reached `0.236665`, and cross-tile lightning/burn were the
dominant failures. Keep that report as negative evidence.

Spec 139 now owns the next lane. It carries forward v7's coarse/detail decomposition and full
frequency/curvature/edge/transition guidance, but the only deployment input is a four-channel
albedo-normalized observation package: luma, x/y gradients, and albedo confidence. WDL, height
hints, normals, liquid, object, alpha, and all other target-derived arrays are forbidden at
inference. The first architecture candidates are `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2`.

The object sieve, real-library silhouette compositor, and footprint-guided marker are deferred.
They are not terrain inputs, not dependencies, and not evidence for this run. The marker user run
was a failed identity diagnostic: final held-out retrieval top-1 was 0, while 668/708 negatives
were predicted known. Do not reuse its checkpoint.

The first terrain-only model is also not promoted: best held-out MAE `0.228693` versus the
`0.191047` tile-mean baseline. The architecture bakeoff is now implemented with nested control
subsets and one shared evaluator: U-Net control, hierarchical CNN/pyramid, compact DPT-style
multi-scale decoder, and SegFormer comparison. The dry run on `control-v1` reports 1.56M, 15.47M,
3.51M, and 3.71M parameters respectively. All are random-initialized; Depth Anything code and
weights are explicitly excluded after the prior non-repeatable failed attempt.

- Start with project-owned deterministic controls, not a broad v50-derived harvest. The default
  control run is 27 families × 4 variants = 108 rows with family-level holdouts and the four
  `easy`/`medium`/`hard`/`pathological` complexity buckets.
- The generator also emits a sibling `object-sieve-v1` derivative with 540 rows. The control taxonomy
  includes mountainous relief, arbitrary-angle sheer drop-offs, zone-style blends, fBm, ridged
  fractal, dendritic lightning-burn terrain proxies, and
  `cross_tile_lightning`/`cross_tile_burn`. Each cross-tile family is one global 2×2 pattern whose
  four tile rows share one `pattern_id`; the validator rejects missing, duplicate, or mixed-ID
  quartets, and the visualizer emits a stitched atlas. Non-grid terrain records deterministic
  sub-cell offsets; only `chunk_grid` is exactly chunk-aligned.
- First signal contract remains `terrain_shadow_256` → `height_257`. Fractal and lightning-burn
  controls are shape probes, not claims about a literal client semantic. The goal is to expose
  partial patterns and high-complexity signals before attempting a tiny albedo-normalized 0.x/1.x
  transfer sample.
- The C# generator, Python validators, object-sieve model/loss variants, and visual reviews are
  implemented. User runs generation,
  client-backed transfer, and GPU training; Codex does not launch them.
- The object-sieve lane is emitted with the terrain controls. Its synthetic input is
  `objectified_terrain_shadow_256`; targets are clean `terrain_shadow_256` plus the distinct
  `object_contamination_mask_256`. Compare clean-only, auxiliary-mask-loss, and predicted-mask-
  guided variants. Do not conflate this screen-space contamination mask with the existing
  `object_geometry_visible_mask_257` numeric geometry target.
- The promoted derived lane is `v60-object-library-sieve-v1`. It reads the 5,349-entry
  `object_mask_library_0_5_3_3368.zarr` read-only, composites real `capture_rgb`/`capture_mask`
  silhouettes onto clean controls, and emits a union mask plus `object_instance_id_256` and
  per-object library provenance. Library families are isolated between train and validation.
- The prior `real-object-masks-v1` run used tile-level curriculum placement projections and produced
  dot-like targets. It is rejected as precision-object evidence; do not train or promote from its
  `experiment_report.json`.
- The object lane remains split into sieve and marker concerns for later, but neither is active in
  the current terrain experiment. The failed marker result is preserved only as a negative record.
- The old v7 13-channel contract is historical reference only. Its structural loss stack is
  transferable guidance; its WDL trestle and answer-side channels are explicitly rejected.
- Spec 139 contract/model/synthetic/visual slices plus the loss contract are implemented:
  four-channel image-only observation gates, deterministic range-floor plus box9 coarse/detail
  targets, NPZ/hash/split/recomposition validation, atomic builder, family/variant/cross-tile
  review, and local random-initialized `pyramid_cnn`/`segformer_b0`/`unet_lite_v2` adapters. The
  loss module provides independently ablatable parity and v7 structural point/gradient/frequency/
  curvature/edge/transition/border/LF/HF terms. The shared trainer now fixes deterministic split
  identities, target-free four-channel loading, independent final/coarse/detail reports,
  family/bucket metrics, and checkpoint binding. The PowerShell-ready CLI is dry-run by default and
  refuses nonempty output roots. Focused loss/trainer/CLI proof is 9 tests; full `tests/v60` passes
  76 tests. The user has completed the six-cell within-family CUDA matrix and the full-profile
  `pyramid_cnn/v7_structural_v1` complete-family run. The full-profile best epoch 37 reached MAE
  `0.173904` versus tile-mean `0.191047` (`8.97%` overall improvement) across 76 train and 32
  held-out rows. `cross_tile_burn` regressed `15.52%`, `cross_tile_lightning` regressed `229.79%`,
  and the pathological bucket regressed `2.81%`, so the explicit cross-tile acceptance scenario
  holds promotion. The checkpoint is diagnostic only and real transfer remains blocked. A new
  prediction-only consumer now exports exact held-out per-row errors and full/cross-tile atlases;
  the user-run atlas review identified a constant-field padding artifact: `flat-v00` and
  `cross_tile_lightning-v01` have nearly identical inputs while the legacy model emits the same
  ramp for both near-zero targets. New model identities use `reflect-3x3-v1`. The user completed
  the `v2-reflect-padding` full-profile run at best epoch 80 with MAE `0.137891` versus `0.191047`
  baseline (`27.82%` improvement); the flat ramp is fixed, but cross-tile lightning and burn still
  regress `61.17%` and `30.15%`. The next bounded user run is full-profile within-family training
  with all 81 training rows to test family coverage versus missing clean-signal information. A
  separate `real_terrain_synthetic` bridge builder/evaluator now handles harvested
  `terrain_shadow_256` plus `height_257` without treating authored RGB as normalized. The first
  16-row Alpha/Azeroth bridge scored MAE `0.323879` versus `0.157124` baseline (`-106.13%`) with
  zero forbidden reads; it is diagnostic only. The user-run bridge probe trained on 15 rows with
  one validation row and best epoch 4 scored `0.313952` versus `0.109902` baseline; all-16 CPU
  evaluation scored `0.293371` versus `0.157124` (`-86.71%`). Two rows are effectively flat and
  height/shadow dynamics vary widely, so source-integrity bands and more maps/builds are required
  before another real-bridge run.
- Overlap handling is explicit: `object_instance_id_256` stores only the visible winner per pixel,
  so fully occluded instances are skipped and recorded rather than treated as positives. Marker
  corpus publication is atomic through `<output>.partial`; a failed build cannot be validated as a
  manifest-bearing corpus. The user's existing failed `object-marker-v1` directory is partial and
  should be left untouched; rerun with a fresh `object-marker-v2` output.
- The existing v50.1 mixed curriculum still has 1,325 complete authored/legacy-flat same-tile pairs
  out of 1,330 groups. `v60_validate_real_synthetic_pairs.py` remains a validation-only JSON/atlas;
  the first 16-tile Azeroth slice measured mean RGB MAE 0.1812 and RMSE 0.2120. The absolute
  difference is a flat-maptexture diagnostic, not terrain-shadow ground truth. A fresh post-fix C#
  NPZ with `terrain_shadow_256` is required for shadow comparison.

## Settled — including the dead ends, which are the expensive part

- **Residual→height feed-forward is dead.** Two runs (uncurated and curated) never beat the
  tile-mean baseline. The "learns then unlearns" oscillation confirmed the target is not learnable
  from single-view shading.
- **Single-view shading cannot recover height.** The forward-model-as-referee (Spec 126 US7) fits
  shading to 0.0103 MAE — 92.9% better than flat — while the recovered height correlates with
  ground truth at **r = 0.0024**. That is the cleanest statement of the limit and should stop this
  family of ideas being re-proposed.
- **The residual extractor works, for albedo-stripping only.** Spec 125 US7, best epoch 54,
  val_mae 0.0893, beats_baseline true, on the curated rolling+steep regimes. Guidance losses
  (multiscale/sobel/spectral/laplacian) were added but only marginally helped.
- **No existing direct minimap→height checkpoint beats the tile-mean baseline.**
  `direct_cnn_v112` (U-Net-lite, 1.56M) v1 best_val_mae 0.1878; v3-deconfounded 0.1723. Both
  `beats_baseline: false`. `mit_b0_regression` (SegFormer, 3.7M) likewise.
- **MCSH is not in minimaps** (measured r = −0.006) and must never be a target or input. General
  rule: the target has to be visible in RGB.

## Ready to run, not yet run

**Stacked height model** — `direct_cnn_v112` extended to 4 input channels (RGB + frozen residual
extractor output) via `--residual-checkpoint`. `HeightRelativeNet` takes `in_channels`.

The crash that blocked it is fixed: the residual channel was built only in the trainer's own
`RowDataset`, while every evaluation path (preview, final eval, road-region, object-region) rebuilt
inputs with RGB+features and handed 3 channels to a 4-channel model. Now one shared builder,
`height_relative_evaluate.build_model_input_channels`, is the single source of truth for channel
order (`RGB -> residual -> features`) across all five call sites. Also: the extractor loads *after*
the dry-run gate (it was allocating CUDA on plan-only runs), and 4-channel `direct_cnn_v112` hashes
to a distinct `config_sha256` (it was colliding with the RGB-only baseline; the 3-channel hash is
unchanged).

**Known mismatch before running**: every extractor on disk (v2–v5) trained on `minimap_rgb_dxt1`,
but `curriculum-0_5_3_3368-dual_v3.zarr` only has `minimap_rgb`. The trainer warns and records
`input_array_matches_training: false`. **Treat the residual channel's contribution as a lower
bound.**

`output/runs/stacked-height-v1/` is the **crashed** attempt (reached epoch 1, val_mae 0.28126,
`input_channels` absent = pre-fix checkpoint). `require_new_output` refuses non-empty dirs, so the
next run needs a fresh path (`stacked-height-v2`). **Do not reuse or delete v1** — it is the negative
record of the crash.

## Ordered plan for the next training session

### Step 1 — decide the curation change BEFORE spending GPU time

Cheap, CPU-only, and it changes what the model sees. Details below.

### Step 2 — stacked height run

`--architecture direct_cnn_v112 --source authored --residual-checkpoint <extractor>`, fresh output
dir. Extractor candidates by best val_mae: **v4 = 0.08840**, v5-guided = 0.08929, v3 = 0.08941,
v2 = 0.09119. v5-guided carries the multiscale/sobel/spectral/laplacian guidance.

Prior split for reference: authored source, 1384 train / 245 val, onecycle + AMP, 100 epochs.

**Record the confound in the run identity rather than discovering it afterwards** — see the
`minimap_rgb_dxt1` vs `minimap_rgb` mismatch above. If the stacked run underperforms, that is the
first thing to rule out, **not** evidence the channel is useless.

### Comparison targets

- tile-mean and flat baselines (computed in-run)
- `SPEC112_FROZEN_BEST_VAL_MAE = 0.1492665126`
- SC-001 requires beating all three by 5% relative
- prior direct runs: v1/cnn 0.1878, v3-deconfounded 0.1723 — neither beat baseline

### Explicitly not on the path

- Full-corpus multi-client harvest — dropped, see `weak-signal-tile-archaeology.md`
- Residual→height feed-forward — dead (r = 0.0024, three approaches agree)
- Spec 127 viewer explorer — drafted, unimplemented, not a training dependency

## Curation change to decide before spending GPU time

`surviving_height_levels` (distinct heights per tile) should gate curation in **both** directions:

`surviving_height_levels` lives in `tile_inventory.py`. Measured on the 0.5.3 corpus:

- **Exclude**: 127 tiles currently classified usable/terrain_no_minimap hold ≤64 distinct heights.
  Four Azeroth tiles (29_24, 30_24, 31_24, 32_24) hold exactly **2** across a 516-unit range. Under
  the v112.1 per-tile min-max target these become perfect binary step functions — actively teaching
  the model that a texture edge is a 500-unit vertical wall.
- **Admit**: 26 compressed-rich tiles are excluded by curation today but their target is **already
  correct**. Amplification above `RANGE_FLOOR` is provably a no-op on the target (verified: ×1 and
  ×1,000,000 give bit-identical targets, because min-max normalisation cancels affine scale). They
  need un-excluding, nothing more.
- 7 further rich tiles sit below `RANGE_FLOOR = 1.0`; those need the floor gated on level count.
  **Do NOT lower the floor globally** — it would amplify the 144 two-to-eight-level tiles into
  full-range targets.

Not yet implemented. It came out of the tile-archaeology work (see `weak-signal-tile-archaeology.md`,
parked) and is the one training-relevant result from it.

## Key files

- `src/harvester/v50/terrain_lighting_torch.py` — differentiable forward model
  (height→normals→Lambert shading)
- `src/harvester/v50/residual_extractor_infer.py` — 4-panel visual-review contact sheets
- `src/harvester/v50/direct_geometry_model.py` — `direct_cnn_v112` accepts 4 input channels
- `src/harvester/v50/height_relative_model.py` — `HeightRelativeNet` accepts `in_channels`
- `src/harvester/v50/direct_geometry_train.py` — `--residual-checkpoint`, frozen extractor preprocessing
- `src/harvester/v50/residual_extractor_train.py` — guidance-loss flags
- `scripts/v50_refine_height_from_residual.py` — forward-model-as-referee refinement
- `scripts/v50_deploy_height_to_mesh.py` — minimap→MiT-B0→height→OBJ deploy

## Constraints specific to this workstream

- **The user runs all training, capture, and GPU work.** Hand over the exact command; never launch
  it.
- No DepthAnything or shared weights across terrain and object stages. The marker specialist's
  knownness and retrieval outputs are independently reported; they do not become a hidden joint
  height/sieve model.
- Never validate on PVPZone02 or Kalidar; use Kalimdor and Azeroth.
- Constitution IV: per-signal evidence. A strong signal must never mask a dead one, so every signal
  is reported against its own baseline, never rolled into an aggregate score.

## Dataset-backed viewer boundary (2026-08-10)

The v50.1 stores are signal-bearing map evidence, including liquid mask/height arrays, variable
liquid type coverage, MCLY layer/tileset metadata, object placement/mask evidence, and texture/path
inventories. The current v60 output tree is not yet a unified renderable map store; its control NPZ
and experiment folders must not be presented as viewer datasets.

Spec 134 now owns a shared C# dataset-version catalog and viewer Settings selector. Renderable VLM
projects can be switched in-session with camera preservation. Zarr roots expose recognized signal
names, including current liquid aliases, but remain summary-only until the C# decoder and tensor-pack
rehydration are implemented and validated.
