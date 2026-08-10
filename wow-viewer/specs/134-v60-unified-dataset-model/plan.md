# Implementation Plan: V60 Controlled Terrain Reconstruction Experiment

**Branch**: `134-v60-unified-dataset-model` | **Date**: 2026-08-08 | **Spec**: [spec.md](./spec.md)

## Summary

Current execution reset (2026-08-10): terrain learning is the only active model lane. The object
sieve, object-library compositor, and footprint-guided marker are parked experiments and are not
dependencies for the terrain result.

Four bounded phases, in dependency order:

1. **Synthetic controls** — generate a small deterministic terrain corpus with exact input/target
   pairs and family holdouts.
2. **Terrain-only model experiment** — evaluate `terrain_shadow_256` → `height_257` on the NPZ
   control corpus with fixed family holdouts, limited training sizes, and a tile-mean baseline.
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
│   ├── object_library_sieve.py                             # real v50 object-library compositor/validator
│   ├── object_sieve_model.py                              # clean/mask/guided model variants
│   ├── object_marker.py                                   # footprint-guided known-object identity/marker contract
│   ├── real_object_mask_model.py                           # real v50 footprint/precise mask model
│   ├── real_synthetic_pairs.py                             # same-tile pair selection/domain report
│   └── albedo_normalization.py                           # planned versioned real-input operation
├── data-harvester/scripts/
│   ├── v60_validate_control_corpus.py                    # offline validator
│   ├── v60_visualize_control_corpus.py                   # family/variant visual atlas
│   ├── v60_validate_object_sieve.py                       # object-sieve validator
│   ├── v60_visualize_object_sieve.py                     # object input/mask atlases
│   ├── v60_build_object_library_sieve.py                  # real-library derived corpus builder
│   ├── v60_validate_object_library_sieve.py               # real-library corpus validator
│   ├── v60_visualize_object_library_sieve.py               # silhouette/instance visual review
│   ├── v60_train_object_sieve.py                          # user-run object-sieve ablations
│   ├── v60_build_object_marker.py                         # marker corpus from library composites
│   ├── v60_validate_object_marker.py                      # marker corpus contract validation
│   ├── v60_train_object_marker.py                         # user-run marker specialist training
│   ├── v60_mark_known_objects.py                          # image+footprint marker export/inference
│   ├── v60_validate_real_synthetic_pairs.py               # small real/synthetic validation atlas
│   ├── v60_train_real_object_masks.py                      # user-run v50 mask trainer
│   ├── v60_normalize_albedo.py                           # planned gate runner
│   └── v60_run_experiment.py                             # bounded architecture bakeoff wrapper
├── data-harvester/tests/v60/
│   ├── test_control_corpus.py                             # existing focused fixture tests
│   ├── test_object_sieve.py                               # object-overlay contract tests
│   ├── test_object_library_sieve.py                        # real-library compositor tests
│   ├── test_object_sieve_model.py                         # model/loss smoke tests
│   ├── test_object_marker.py                               # marker contract/model/retrieval tests
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

## Phase 2: Terrain-only limited control-data model experiment (P1)

**Current goal**: Determine whether the clean synthetic terrain signal contains enough information
for useful height reconstruction. Object identification and object removal are explicitly out of
scope for this gate.

### Active Stage — height reconstruction

1. Load the validated `control-v1` NPZ manifest without changing historical Zarr training contracts.
2. Keep the manifest's complete-family validation holdout fixed.
3. Evaluate limited training sizes such as 8, 16, and 32 rows.
4. Train only on `terrain_shadow_256` and target only `height_257`.
5. Compare against the per-tile mean baseline and report family/variant metrics.
6. Mark flat or weakly informative controls as ambiguous rather than confident success.

**Active gate**: The report says whether the clean control relationship is learnable. A failure is a
bounded diagnostic and does not authorize object work or broad real-data processing.

### Architecture bakeoff (active next slice)

The current U-Net-lite is retained as the low-capacity control, not the assumed solution. Add a
shared architecture registry and compare four candidates on exactly the same nested training rows:
`unet_lite_v2`, `pyramid_cnn`, locally implemented `dpt_small`, and `segformer_b0`. Every candidate
must produce the same `height_257` tensor and report median/worst-family MAE against the tile-mean
baseline. The first bakeoff uses project-owned random initialization. Depth Anything code and
weights are explicitly excluded; external weights are not part of this lane.

**Architecture gate**: promote only a candidate that beats the tile-mean baseline on the fixed
held-out families and does not hide a failed family behind an aggregate score. If none wins, record
the result as an information/target limitation rather than selecting the largest model.

### Parked Stages — object supervision

The procedural object sieve, real-library silhouette compositor, and footprint-guided marker remain
implemented as isolated experiments for later. Their code and artifacts are preserved, but they are
not part of the current terrain input contract and must not block T019–T025.

### Historical Stage A — object sieve (parked)

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

### Stage B — corrected v50 object-library compositor

1. Read the existing v50 `object_mask_library_0_5_3_3368.zarr` as a read-only source. Its
   `capture_rgb` and `capture_mask` arrays are the precision object-image/mask contract.
2. Place real library captures over the project-owned clean `terrain_shadow_256` control rows with
   deterministic scale, rotation, overlap, and boundary-crossing transforms. Do not use the v50
   curriculum's tile-level placement projections as these targets.
3. Emit `object_contamination_mask_256` as the exact union of the transformed library masks and
   `object_instance_id_256` as a deterministic per-pixel instance map. Preserve each library ID,
   asset path, family, and transform in row metadata.
4. Isolate library families between train and validation, in addition to the terrain-family holdout.
   Refuse a corpus with missing source provenance, blank captures, or a family crossing the split.
5. Render a four-panel visual review: objectified input, clean terrain target, exact union mask, and
   instance-ID map. A dot-like or empty overlay fails before training.
6. Train the three object-sieve variants on the derived corpus and report clean-terrain and mask
   signals independently by placement regime and held-out family. The clean head must use an
   identity-preserving residual so it is compared against, and cannot silently lose to, the
   contaminated-input baseline.
7. Keep the old curriculum `real-object-masks-v1` run as a failed diagnostic. It may inform why the
   tile-level labels were inadequate, but it cannot be promoted as a model result.

**Stage B gate**: A user-run GPU experiment has source-library provenance, no terrain-family or
library-family leakage, a visual review with non-dot silhouettes, and per-signal metrics. This proves
library-derived object contamination decomposition only; it does not prove clean terrain
reconstruction or authorize real height transfer.

### Historical Stage B2 — footprint-guided object identification and marking (parked)

1. Derive one-candidate rows from the corrected library composites. Each row contains a minimap
   image, one candidate footprint, a known/unknown target, and the positive library ID in metadata.
2. Train a small, independently checkpointed marker specialist with a knownness head and a fixed
   embedding head. Do not train a 5,349-way pixel classifier; the library is the retrieval gallery.
3. Build the gallery from the same read-only v50 captures and resolve exact identity by nearest
   embedding match with a persisted threshold.
4. Export `known_object_marker_256` plus an identity table for accepted candidates. Zero is
   background/unaccepted, and table rows—not pixel values—carry library IDs.
5. Include shifted/empty/unknown candidates and report known precision/recall, top-1/top-k
   retrieval, coverage, and family-isolated results independently.
6. Require explicit candidate footprints at inference. Proposal discovery is a later, separate
   stage and is not hidden in marker metrics.

**Stage B2 gate**: A held-out candidate report demonstrates that the marker can reject negative
candidate footprints and retrieve known library identities, and a visual marker export resolves
every nonzero marker instance through its sidecar table. The marker checkpoint may feed the sieve,
but neither stage may consume the other's ground-truth targets.

### Historical Stage C — height reconstruction (superseded by the active stage above)

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

Print the one-of-each terrain architecture bakeoff plan:

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run --no-cache python scripts/v60_run_experiment.py --corpus "../output/datasets/v60/control-v1" --output "../output/datasets/v60/terrain-architecture-runs/control-v1" --architectures "unet_lite_v2,pyramid_cnn,dpt_small,segformer_b0" --train-sizes 32 --epochs 40 --batch-size 8 --lr 1e-3 --seed 6001
```

After reviewing the dry run, the user may add `--confirm-run` to launch the CUDA experiment. The
object lanes and real transfer remain parked until this terrain result is recorded.
