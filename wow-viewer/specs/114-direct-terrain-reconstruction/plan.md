# Implementation Plan: Direct Minimap-to-Terrain Reconstruction

**Branch**: `114-direct-terrain-reconstruction` | **Date**: 2026-07-19 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/114-direct-terrain-reconstruction/spec.md`

## Summary

Replace the old mandatory WDL-prior route with one direct image-to-relative-height stage trained on
the corrected dual-view v50 corpus. Keep the model stack modular: trusted object visibility,
relative geometry, terrain-feature classification, texture-family selection, and alpha-stack
reconstruction each own one output, checkpoint, trainer, and gate. Use the existing Spec 112 lean
CNN as the mandatory geometry baseline; evaluate a compact MiT-B0/SegFormer-style dense regression
variant. Use semantic segmentation architectures for object/landform maps, and a lean U-Net/FPN
regressor for the ordered alpha stack. Spec 113 retains all SR/detail ownership.

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

**Performance Goals**: every first-pass model fits 16 GB VRAM at its documented patch/batch size;
geometry and semantic baselines remain small enough for same-day iteration; no 100M+ parameter
default model

**Constraints**: user launches all training/heavy rebuilds; image-only deployment contract; no WDL
prior; one output per model; no shared weights; no DA-V2; no MCAL parser or AlphaWdtWriter changes;
private BYOD only

**Scale/Scope**: Kalimdor and Azeroth first; authored and corrected synthetic views grouped per tile;
expand to later builds only after the first full stack is proven

## Constitution Check

*GATE: Must pass before implementation and after every stage design.*

| Principle | Plan response | Status |
|---|---|---|
| I. Repo Independence | All files stay under `wow-viewer/data-harvester/` and `wow-viewer/specs/114-*`; Hub IDs are runtime inputs, never external path references | PASS |
| II. Library-First | Model/data logic lives in `src/harvester/v50/`; scripts are thin wrappers; existing readers/compositor are reused | PASS |
| III. Real-Data Validation | Every promotion uses configured `H:\CLIENTS` evidence, build identity, store hash, held-out metrics, and user visual review | PASS |
| IV. Residual Model Chain | Each model predicts one signal with its own weights. Direct geometry predicts relative height as the residual from the fixed zero/mean baseline; no WDL or multi-task head | PASS |
| V. Streaming/Zarr | Source and derived corpora remain Zarr/Parquet; no NPZ side channel | PASS |
| VI. No Client Path Assumptions | Client root remains a CLI/config value; docs show the current operator path only in user-run commands | PASS |
| One Phase at a Time | Geometry completes before mask-guided geometry; masks before semantics; semantics before texture/alpha | PASS |
| Bite-Sized Plans | Each phase below has at most ten independently verifiable tasks | PASS |

## Architecture Decision

```text
authored minimap RGB
        │
        ├── object-mask model ──> generated object visibility/cleanup signal
        │                               │
        ├───────────────────────────────┴──> direct geometry model ──> relative_height_257
        │
        └── land-feature model ──> generated feature classes
                                         │
                                         └── texture-family selector ──> ordered family IDs
                                                                              │
authored RGB + generated feature/family signals ──────────────────────────────┴──> alpha model
                                                                                     │
                                                                                     └── alpha_256x4
```

This is a dependency graph, not a shared network. Each arrow carries a persisted/generated signal
with a checkpoint identity. During downstream training, generated upstream outputs—including their
errors—must be present. Spec 113's RealPLKSR output is a separate visualization/detail branch and
does not become ground-truth geometry or alpha.

## Phase Design

### Phase 0 — Corpus and contract audit

1. Bind Spec 114 to corrected fixed-noon synthetic store revisions from Spec 113.
2. Freeze a dual-view geometry curriculum contract with grouped authored/synthetic rows.
3. Audit current object evidence and define a trusted top-down object-visibility target; do not
   promote old dropped masks.
4. Freeze model-stage/run-summary schemas and Hub source/license/hash evidence.

### Phase 1 — Direct geometry MVP

1. Adapt the Spec 112 trainer/curriculum to accept authored and corrected synthetic RGB with no WDL.
2. Retain the existing lean CNN as `direct_cnn_v112` baseline.
3. Add one MiT-B0/SegFormer-style continuous regression candidate with the identical output/target.
4. Prove shape, offset invariance, group leakage refusal, no-WDL input audit, and generated-signal
   provenance using CPU fixtures.
5. Hand the user two bounded training commands on the same frozen split.
6. Promote only if SC-001/SC-002 and user visual review pass.

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
│   ├── direct_geometry_model.py
│   ├── direct_geometry_train.py
│   ├── object_visibility_labels.py
│   ├── object_mask_model.py
│   ├── object_mask_train.py
│   ├── terrain_feature_library.py
│   ├── terrain_feature_model.py
│   ├── texture_family_library.py
│   ├── texture_family_model.py
│   ├── alpha_stack_model.py
│   ├── model_stage_contract.py
│   └── reconstruction_curriculum.py
├── scripts/
│   ├── v50_build_reconstruction_curriculum.py
│   ├── v50_train_direct_geometry.py
│   ├── v50_build_object_visibility.py
│   ├── v50_train_object_mask.py
│   ├── v50_build_terrain_feature_library.py
│   ├── v50_train_terrain_features.py
│   ├── v50_build_texture_family_library.py
│   ├── v50_train_texture_families.py
│   └── v50_train_alpha_stack.py
└── tests/v50/
    ├── test_reconstruction_curriculum.py
    ├── test_direct_geometry_model.py
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
- Direct geometry remains a single relative-height residual, so removing WDL does not violate the
  one-residual rule.
- Texture identity and alpha fields are explicitly separate.
- The only proposed renderer addition is a trusted label export, not a format/parser rewrite.
- Training and heavy data generation remain user-run.

All gates pass. Phase 0 research/design is ready for dependency-ordered tasks.
