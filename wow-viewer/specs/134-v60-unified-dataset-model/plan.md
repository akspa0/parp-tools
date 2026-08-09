# Implementation Plan: V60 Controlled Terrain Reconstruction Experiment

**Branch**: `134-v60-unified-dataset-model` | **Date**: 2026-08-08 | **Spec**: [spec.md](./spec.md)

## Summary

Four bounded phases, in dependency order:

1. **Synthetic controls** — generate a small deterministic terrain corpus with exact input/target
   pairs and family holdouts.
2. **Object sieve and model experiments** — validate synthetic decomposition, then train a real-mask
   object detector from the existing v50.1 authored rows before attempting height reconstruction.
3. **Albedo normalization and gate** — process a tiny explicit 0.x/1.x real sample, measure how well
   texture/albedo can be removed, and admit only accepted textureless outputs.
4. **Transfer and expansion decision** — compare the accepted real sample to controls, then hold,
   diagnose, or authorize broader processing.

Later signals and later clients are extension phases only after the transfer route is understood.
This plan deliberately removes full-client harvesting, v50-store consolidation, base/delta
historical storage, and a 4,000-sample minimum from v60-control-v1.

## Technical Context

**Languages**: C#/.NET for terrain decode/synthesis and ADT ownership; Python/uv for normalization
orchestration, validation, experiment indexing, and reports.
**Synthetic authority**: existing `TerrainMinimapCompositor.ComposeShadowArray` and the established
synthetic-minimap path. New code must reuse those seams rather than implement a second lighting
equation.
**Initial source policy**: procedural families require no client; client-backed real transfer seeds
are limited to approved and explicitly classified `0.x`/`1.x` roots.
**First input/target**: `terrain_shadow_256` (256x256) → `height_257` (257x257).
**Synthetic object sieve input/targets**: `objectified_terrain_shadow_256` (256x256) → clean
`terrain_shadow_256` plus `object_contamination_mask_256`; the mask is an auxiliary output and
loss-side target, never a ground-truth inference channel.
**Real object-mask input/targets**: configured v50.1 authored `minimap_rgb` (256x256x3) → separate
`object_precise_mask` and `object_mask` projections (256x256); no clean-minimap target is claimed.
**Initial control size**: approximately 32–128 rows, with limited model runs at manifest-selected
training sizes such as 8, 16, and 32.
**Real-data route**: authored minimap → versioned albedo normalization → textureless gate → tiny
transfer sample. The existing metadata-derived texture identity albedo is not treated as this
inverse operation.
**Testing**: focused Python tests, focused C# synthesis/writer tests, JSON contract checks, and
lightweight offline validation. Codex does not launch full harvests, real-client processing, GPU
training, or long-running operations.

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| Repo Independence | PASS | All new implementation remains under `wow-viewer/`. |
| Library First | PASS | Synthesis stays owned by core terrain/compositor code. |
| Per-Signal Evidence | PASS | Control and transfer reports separate input/height evidence from gate/domain evidence. |
| No Hardcoded Paths | PASS | Source manifests and output roots are runtime inputs. |
| User Runs Heavy Work | PASS | Full synthesis, real-client processing, and training commands are handed to the user. |
| One Phase at a Time | PASS | Each gate blocks the next phase until its evidence exists. |
| Fail Closed | PASS | Missing arrays, bad normalization, and uncalibrated textureless outputs are rejected. |

## Project Structure

```text
wow-viewer/
├── src/core/WowViewer.Core.IO/Maps/
│   └── existing terrain synthesis/compositor seams       # reuse; extend only if required
├── tools/harvest/WowViewer.Tool.Harvest/
│   └── Program.cs                                        # control-corpus command
├── data-harvester/src/harvester/v60/
│   ├── control_corpus.py                                 # manifest/hash/split validation
│   ├── object_sieve.py                                    # object-overlay corpus validation
│   ├── object_sieve_model.py                              # clean/mask/guided model variants
│   ├── real_object_mask_model.py                           # real v50 footprint/precise mask model
│   ├── real_synthetic_pairs.py                             # same-tile pair selection/domain report
│   └── albedo_normalization.py                           # planned versioned real-input operation
├── data-harvester/scripts/
│   ├── v60_validate_control_corpus.py                    # offline validator
│   ├── v60_visualize_control_corpus.py                   # family/variant visual atlas
│   ├── v60_validate_object_sieve.py                       # object-sieve validator
│   ├── v60_visualize_object_sieve.py                     # object input/mask atlases
│   ├── v60_validate_real_synthetic_pairs.py               # small real/synthetic validation atlas
│   ├── v60_train_real_object_masks.py                      # user-run v50 mask trainer
│   ├── v60_normalize_albedo.py                           # planned gate runner
│   └── v60_run_experiment.py                             # planned bounded evaluator/trainer wrapper
├── data-harvester/tests/v60/
│   ├── test_control_corpus.py                             # existing focused fixture tests
│   ├── test_object_sieve.py                               # object-overlay contract tests
│   ├── test_object_sieve_model.py                         # model/loss smoke tests
│   ├── test_real_object_mask_model.py                      # real-mask model/target tests
│   └── test_albedo_normalization.py                       # planned fail-closed gate tests
└── specs/134-v60-unified-dataset-model/
    ├── contracts/                                        # manifest, gate, and experiment JSON contracts
    ├── data-model.md
    ├── research.md
    └── quickstart.md
```

## Phase 1: Synthetic control corpus (P1)

**Goal**: Produce a small deterministic corpus without a complete client harvest.

1. Keep the explicit family/variant manifest and initial 0.x/1.x source policy visible.
2. Reuse the existing decoded-terrain and compositor/writer seams to generate exact control rows.
3. Emit the full default taxonomy: smooth/monotonic/plateau, ridged/valley, terraced/cliff,
   mountainous relief, arbitrary-angle sheer drop-offs, zone-style blends, chunk-grid,
   island/archipelago, crater/canyon, fBm/ridged-fractal, lightning-burn, cross-tile lightning/burn,
   mixed, and pathological families.
4. Generate cross-tile families in one global 2x2 coordinate system and persist pattern ID, tile
   coordinates, span, and continuity metadata.
5. Assign every family to the established `easy`, `medium`, `hard`, or `pathological` bucket and
   summarize bucket counts in the manifest.
6. Keep lighting/albedo controls independent from height controls in row metadata.
7. Write `terrain_shadow_256`, `height_257`, row metadata, split membership, and hashes.
8. Hold out complete source families, not random rows.
9. Render family, variant, and stitched cross-tile atlases from the emitted NPZ signals and report
   missing coverage.
10. Validate repeatability, shape/range/finite constraints, cross-tile completeness, and absence of
    missing signals.
11. Record deterministic sub-cell field offsets for every non-grid row; reserve exact chunk alignment
    for the explicit `chunk_grid` diagnostic family.

**Gate**: 32–128 deterministic terrain control rows exist; every row has an exact pair and complexity
bucket; the visual atlas shows the intended family coverage; no v50 store or full-client harvest was
required.

## Phase 2: Object sieve and limited control-data model experiments (P1)

**Goal**: Determine whether the canonical textureless input can first be cleaned of object
contamination, then determine whether it contains enough information for useful height reconstruction.

### Stage A — object sieve

1. Add a deterministic synthetic object overlay authority on top of the canonical terrain shadow;
   do not duplicate the terrain lighting equation.
2. Emit no-object, sparse, dense, overlapping, and boundary-crossing object controls with placement
   metadata and exact `object_contamination_mask_256` targets.
3. Train/evaluate three bounded variants: clean-output-only, auxiliary mask loss, and predicted-mask
   guidance into the clean-output head.
4. Use the predicted mask for guidance during both training and inference. Ground-truth masks are
   loss-side supervision only.
5. Report clean-terrain metrics and mask metrics independently by density, placement regime, and
   held-out object family.

**Stage A gate**: The mask head must beat its trivial baseline and the guided clean output must be
compared against the clean-only and auxiliary-loss variants. A good mask score does not hide a
failed clean-terrain score.

### Stage B — real v50 object-mask model

1. Read the configured `curriculum-0_5_3_3368-obj_v1.zarr` through its existing `index.parquet`;
   do not copy or rewrite the canonical store.
2. Filter to authored rows by default and enforce either the existing manifest split or an explicit
   map holdout. Validate that `source_group_id` never crosses the split.
3. Train a compact RGB mask model with independently selectable `object_precise_mask` and
   `object_mask` heads. An optional RGB-edge input is an ablation, not a hidden default.
4. Select the best checkpoint using the minimum IoU across requested targets and report each target
   independently by map, source type, coverage, and threshold.
5. Render validation previews showing RGB, truth, prediction, and error for both mask targets.
6. Record the empty geometry-visible mask as an audit finding; do not use it as the target.
7. Select a small held-out authored/flat-synthetic pair slice by `source_group_id`, verify same-tile
   identity and split membership, and write an absolute-difference visual/domain report before GPU
   work.
8. Treat the legacy synthetic image as a flat fake-maptexture diagnostic only. Compare it against a
   fresh post-fix C# `terrain_shadow_256` NPZ when available; missing or stale shadow provenance
   fails closed.

**Stage B gate**: A user-run GPU experiment has provenance, no source-group leakage, and per-target
metrics. The pair report separately establishes the authored-vs-flat absolute-difference signal and,
when supplied, its comparison with the post-fix terrain shadow. This proves object-mask detectability
and calibration evidence only; it does not prove clean terrain reconstruction or authorize real
height transfer.

### Stage C — height reconstruction

1. Add a control-v1 loader/evaluator without changing historical training contracts.
2. Select limited training sizes from the manifest, keeping the held-out families fixed.
3. Run the first model on `terrain_shadow_256` only and target `height_257`.
4. Compare against a tile-mean baseline.
5. Report metrics by source family, variant family, and training size.
6. Run retexturing/relighting controls with unchanged terrain targets.
7. Mark flat/weakly informative terrain as ambiguity rather than confident success.

**Stage C gate**: The result says whether the clean signal/height relationship is learnable on
controls. A failed result is a bounded diagnostic and does not trigger a return to broad harvest.

## Phase 3: Albedo normalization and textureless gate (P1)

**Goal**: Build and calibrate the first real-input boundary before transfer.

1. Define a versioned `authored minimap → normalized textureless input` operation contract.
2. Keep the synthetic textureless compositor output as calibration/reference data, not as a hidden
   substitute for real input.
3. Implement an explicit albedo estimate/removal output with method/version and residual metrics.
4. Build positive synthetic controls and deliberately textured/failed negative controls.
5. Calibrate and persist textureless thresholds from those controls.
6. Fail closed for missing, non-finite, uncalibrated, or residual-textured outputs.
7. Run only a tiny explicit 0.x/1.x sample and write accepted/rejected/quarantined decisions.

**Gate**: The report proves which real rows are admitted and why; no rejected or quarantined row
enters the model directory; thresholds and artifacts are reproducible.

## Phase 4: Tiny transfer and expansion decision (P2)

**Goal**: Determine whether the control result transfers to normalized real inputs.

1. Evaluate the accepted tiny real sample with the same input contract and evaluator.
2. Compare input distributions, failure signatures, and baseline-relative metrics with controls.
3. Record a `TransferGate` decision of `hold`, `diagnose`, or `expand`.
4. If held, diagnose normalization/domain shift before changing row count.
5. If expanded, process the next bounded batch through the same gate and preserve client provenance.

**Gate**: Broader processing is allowed only by an explicit `expand` decision; a strong synthetic
score alone never authorizes it.

## Deferred extensions (P3)

Only after Phase 4:

1. Add one exact signal at a time: normals, texture identity, holes, or liquid.
2. Add optional object presence/instance or object-family heads only after the binary contamination
   sieve is proven useful.
3. Add later client source adapters behind the same manifest and gate contracts.
4. Expand control families or variants only in response to measured failure modes.

## User-run commands (prepared, not executed here)

Build the control tool:

```powershell
dotnet build "I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj" -c Debug --no-restore
```

Generate the initial 108-row full-taxonomy procedural control run and its 540-row object-sieve
derivative corpus:

```powershell
dotnet "I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll" control-corpus --output-dir "I:/parp/parp-tools/wow-viewer/output/datasets/v60/control-v1" --variants 4 --holdout-families chunk_grid,island_sea,sheer_dropoff,zone_style_blend,cross_tile_lightning,cross_tile_burn,noise,pathological
```

Validate it before model work:

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run python scripts/v60_validate_control_corpus.py --corpus "../output/datasets/v60/control-v1" --write-report
```

Render the visual family/variant atlases:

```powershell
uv run python scripts/v60_visualize_control_corpus.py --corpus "../output/datasets/v60/control-v1" --output-dir "../output/datasets/v60/control-v1/visual-review" --variants-per-family 4
```

Review `control-family-atlas.png`, `control-variant-atlas.png`, and
`control-visual-review.json`. If `coverage_complete` is false or a family/bucket is visually
uninteresting, adjust the control generator before training.

Albedo normalization, model training, and real transfer commands are intentionally withheld until
their implementation tasks land. They will remain PowerShell-ready and user-run.
