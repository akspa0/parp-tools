# Implementation Plan: Universal Image-to-Terrain Reconstruction

**Branch**: `114-direct-terrain-reconstruction` | **Date**: 2026-07-19 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/114-direct-terrain-reconstruction/spec.md`

## Summary

Replace the old mandatory WDL-prior route with one universal image-to-relative-relief stage. The
accepted deployment input is any decodable raster; corrected v50 authored/synthetic minimaps provide
exact top-down supervision but are only one curriculum family. A compact student combines a pinned
general visual initialization with a single continuous relief decoder. A pinned general monocular-
depth teacher may generate normalized view-axis-relief supervision for broad licensed/BYOD imagery;
exact v50 numeric heights remain authoritative for top-down terrain rows. Whole visual/source
families—not random WoW tiles alone—are held out for promotion.

Keep the model stack modular: universal relief geometry, optional WoW object visibility,
terrain-feature classification, texture-family selection, and alpha-stack reconstruction each own
one output, checkpoint, trainer, and gate. The source raster can be UV-projected onto the generated
mesh immediately; editable material reconstruction remains separate. Spec 113 retains SR/detail
ownership for the WoW-specific branch.

The narrow authored-only run completed on 2026-07-19 and did not promote: best epoch 92 reached
0.149267 validation MAE against the 0.138747 per-tile constant baseline. Before any second training
run, Phase 1 must gain a universal input/mesh contract, broad relief supervision, whole-domain
evaluation, reviewable artifacts, and the repo's proven bounded optimization stack. Optimizing the
same WoW-only corpus is explicitly not the next run.

## Technical Context

**Language/Version**: Python 3.11+ through the existing uv environment; no new C# format reader

**Primary Dependencies**: existing PyTorch, timm, Transformers/Hugging Face Hub, Zarr, NumPy,
PyArrow, Pillow; existing `TerrainMinimapCompositor` for read-only recomposition diagnostics

**Storage**: existing per-build v50 Zarr stores; derived trainer-facing Zarr curricula; Parquet
indexes; JSON schema-conformant run/evaluation summaries

**Testing**: pytest CPU contract/property tests; tiny forward/backward fixtures; schema validation;
user-run CUDA training and real-client visual gates

**Target Platform**: Windows desktop, local RTX 4070 Ti SUPER (16 GB); backend seams must not make
the persisted data/model contract CUDA-specific

**Project Type**: Python model/data library plus thin CLIs under `wow-viewer/data-harvester/`

**Performance Goals**: the deployable student fits 16 GB VRAM for training and supports same-day
iteration; large teacher inference is an offline, user-run label-build step and is never required at
deployment; no 100M+ parameter deployment model

**Constraints**: user launches all training/heavy rebuilds; arbitrary-raster-only deployment
contract; no WDL prior; one output per model; no shared weights; no DepthAnything-family model; no
MCAL parser or AlphaWdtWriter changes; licensed public data or private BYOD only

**Scale/Scope**: at least five visual/source families for universal evaluation; Kalimdor and Azeroth
provide the first exact top-down family; every derived view stays grouped by underlying source

## Constitution Check

*GATE: Must pass before implementation and after every stage design.*

| Principle | Plan response | Status |
|---|---|---|
| I. Repo Independence | All files stay under `wow-viewer/data-harvester/` and `wow-viewer/specs/114-*`; Hub IDs are runtime inputs, never external path references | PASS |
| II. Library-First | Model/data logic lives in `src/harvester/v50/`; scripts are thin wrappers; existing readers/compositor are reused | PASS |
| III. Real-Data Validation | The WoW family uses configured `H:\CLIENTS` evidence/build hashes; universal promotion additionally holds out entire licensed/BYOD visual families and requires user review | PASS |
| IV. Residual Model Chain | Each model predicts one signal with its own weights. Direct geometry predicts relative height as the residual from the fixed zero/mean baseline; no WDL or multi-task head | PASS |
| V. Streaming/Zarr | Source and derived corpora remain Zarr/Parquet; no NPZ side channel | PASS |
| VI. No Client Path Assumptions | Client root remains a CLI/config value; docs show the current operator path only in user-run commands | PASS |
| One Phase at a Time | Geometry completes before mask-guided geometry; masks before semantics; semantics before texture/alpha | PASS |
| Bite-Sized Plans | Each phase below has at most ten independently verifiable tasks | PASS |

## Architecture Decision

```text
any source raster ──> universal relief model ──> normalized relief ──> deterministic mesh + UVs
       │
       ├── direct source-image projection (immediate visual texture)
       │
       ├── optional WoW object-mask model ──> generated cleanup signal ──> relief ablation
       │
       └── land-feature model ──> generated feature classes
                                          │
                                          └── texture-family selector ──> ordered family IDs
                                                                               │
source raster + generated feature/family signals ──────────────────────────────┴──> alpha model
                                                                                      │
                                                                                      └── alpha stack
```

This is a dependency graph, not a shared network. Each arrow carries a persisted/generated signal
with a checkpoint identity. During downstream training, generated upstream outputs—including their
errors—must be present. Spec 113's RealPLKSR output is a separate visualization/detail branch and
does not become ground-truth geometry or alpha.

### Frozen negative baseline

| Item | Recorded value |
|---|---|
| Architecture | `direct_cnn_v112`, U-Net-lite, base width 32, 1,561,537 parameters |
| Input | authored `minimap_rgb`, uint8 RGB normalized to `[0,1]`, shape `3×256×256` |
| Output | one sigmoid-bounded `relative_height_257`, shape `257×257` |
| Target | `v112.1`: `(height - tile_min) / max(tile_range, 1.0)` |
| Loss | Smooth-L1 point loss + `0.25 ×` first-derivative L1 loss |
| Optimizer | AdamW, learning rate `2e-4`, weight decay `1e-4` |
| Schedule | batch 16, max 100 epochs, patience 15, seed 114, no scheduler, Windows workers 0 |
| Split | frozen source-group split: 1,384 authored train / 245 authored validation |
| Augmentation | none in the bootstrap; later paired D4 augmentation requires its own proof |
| Baseline | per-tile mean normalized height, computed from validation truth |
| Checkpoints | immutable best and last; non-empty output directories are refused |
| Failure gate | best epoch 1 is structural failure even if it numerically beats the baseline |

This checkpoint is retained only to prove that a from-scratch WoW-domain CNN and an in-domain random
split are insufficient. Its separately backfilled validation sheets remain useful diagnostics, but
no universal claim or optimized rerun may be based on this table.

## Phase Design

### Phase 0 — Corpus and contract audit

1. Freeze the universal raster normalization, normalized-relief, deterministic mesh, UV, extent,
   and vertical-scale contracts.
2. Freeze at least five visual/source families and group every crop/render/style by its underlying
   source; designate whole-family train, validation, and compatibility partitions.
3. Bind the exact v50 top-down family to corrected fixed-noon store revisions from Spec 113.
4. Pin license, immutable revision/hash, preprocessing, and output orientation for the general
   visual student initialization and optional non-DepthAnything teacher.
5. Freeze model-stage/run-summary schemas, constant/luminance baselines, and arbitrary-image sheets.

### Phase 1 — Direct geometry MVP

1. Implement and fixture-test arbitrary RGB/RGBA/grayscale loading, aspect-preserving tiling/padding,
   relief stitching, deterministic grid-mesh export, UVs, and blank-image stability.
2. Build an immutable universal curriculum index spanning exact v50 top-down rows, procedural/style-
   randomized terrain views, and broad licensed/BYOD image-relief pairs or pinned teacher labels.
3. Keep every view of one source grouped; reject random row splits and report whole-domain holdouts.
4. Implement one compact general-visual student with one continuous normalized-relief output. Start
   from the pinned general initialization; retain the failed CNN only as negative evidence.
5. Train with exact v50 height/normal/liquid guidance where available and normalized teacher relief
   elsewhere. Missing clean signals mask their own loss only and never become deployment inputs.
6. Port AMP, EMA deploy weights, warmup/cosine decay, clipping, multiscale/gradient/normal guidance,
   detached hard-error weighting, history/VRAM evidence, and geometry-consistent spatial plus broad
   photometric/style augmentation as one documented recipe.
7. Emit fixed arbitrary-image previews, paired per-row metrics, constant/luminance comparisons,
   quantile/worst sheets, and exported mesh previews from the best EMA checkpoint.
8. Prove universal input compatibility, finite meshes, source-group isolation, no-WDL/no-teacher
   deployment, and reproducible model/source identities with CPU fixtures.
9. Hand the user separate dry-run, optional teacher-label build, and bounded CUDA training commands.
10. Promote only if SC-001 through SC-004 and user visual review pass.

### Phase 2 — Trusted objects and mask-guided geometry

1. Produce/audit renderer-aligned object-visibility labels from verified placement geometry.
2. Build an honest partial-coverage object-mask curriculum.
3. Train one compact semantic object-mask model; evaluate empty/all-object baselines.
4. Persist generated masks for the frozen geometry split.
5. Retrain/evaluate geometry with generated mask/cleaning input and compare to the raw-RGB baseline.
6. Promote mask guidance only if it improves object-region error without harming clean terrain.

### Phase 3 — Terrain-feature library and classifier

1. Derive deterministic feature labels from height/slope/curvature, liquid, alpha/material, and
   Spec 076/103 pattern evidence.
2. Version the library and freeze family-safe groups plus unknown/unavailable states.
3. Build a compact semantic classifier with its own checkpoint and no shared geometry weights.
4. Run family-majority, per-class coverage, macro-F1, and visual overlay gates.

### Phase 4 — Texture-family selection

1. Map raw per-build texture/tileset IDs to versioned canonical families.
2. Freeze ordered family-tuple targets and unknown/fallback behavior.
3. Train one family selector from RGB plus generated feature context.
4. Gate against per-map majority and family leakage; visually inspect recomposition choices.

### Phase 5 — Alpha-stack reconstruction

1. Freeze the ordered four-layer alpha target and compositor-compatible validation rules.
2. Train one lean spatial alpha-stack regressor using generated family selections.
3. Compare against base-only and uniform-blend baselines in numeric alpha space.
4. Recompose through the existing compositor and run image/visual checks on Alpha and LK fixtures.
5. Promote without changing MCAL decode, renderer blending, or `AlphaWdtWriter`.

## Project Structure

### Documentation (this feature)

```text
specs/114-direct-terrain-reconstruction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── tasks.md
├── checklists/requirements.md
└── contracts/model-stage-and-curriculum.schema.json
```

### Source Code (planned)

```text
data-harvester/
├── src/harvester/v50/
│   ├── universal_relief_contract.py
│   ├── relief_teacher_labels.py
│   ├── universal_relief_curriculum.py
│   ├── universal_relief_model.py
│   ├── universal_relief_train.py
│   ├── object_visibility_labels.py
│   ├── object_mask_model.py
│   ├── object_mask_train.py
│   ├── terrain_feature_library.py
│   ├── terrain_feature_model.py
│   ├── texture_family_library.py
│   ├── texture_family_model.py
│   ├── alpha_stack_model.py
│   ├── model_stage_contract.py
│   └── model_stage_contract.py
├── scripts/
│   ├── v50_build_relief_teacher_labels.py
│   ├── v50_build_universal_relief_curriculum.py
│   ├── v50_train_universal_relief.py
│   ├── v50_image_to_terrain.py
│   ├── v50_build_object_visibility.py
│   ├── v50_train_object_mask.py
│   ├── v50_build_terrain_feature_library.py
│   ├── v50_train_terrain_features.py
│   ├── v50_build_texture_family_library.py
│   ├── v50_train_texture_families.py
│   └── v50_train_alpha_stack.py
└── tests/v50/
    ├── test_universal_relief_contract.py
    ├── test_relief_teacher_labels.py
    ├── test_universal_relief_curriculum.py
    ├── test_universal_relief_model.py
    ├── test_universal_relief_train.py
    ├── test_object_visibility_labels.py
    ├── test_object_mask_model.py
    ├── test_terrain_feature_library.py
    ├── test_terrain_feature_model.py
    ├── test_texture_family_library.py
    ├── test_texture_family_model.py
    ├── test_alpha_stack_model.py
    └── test_model_stage_contract.py
```

**Structure Decision**: extend the existing v50 package because all inputs and lineage already live
there. Keep separate modules and trainers per signal; only immutable schema/identity utilities are
shared. No new C# dataset reader is planned. If trusted object visibility needs a renderer export,
that bounded surface must reuse existing Core.IO/runtime geometry and receive its own proof before
any Python model task begins.

## Post-Design Constitution Recheck

- No monolithic or shared-weight model appears in the design.
- Universal geometry remains one normalized view-axis-relief residual. The general encoder and
  relief decoder form one independently trained stage; no WDL or downstream head is added.
- Texture identity and alpha fields are explicitly separate.
- The only proposed renderer addition is a trusted label export, not a format/parser rewrite.
- Training and heavy data generation remain user-run.

All gates pass. Phase 0 research/design is ready for dependency-ordered tasks.
