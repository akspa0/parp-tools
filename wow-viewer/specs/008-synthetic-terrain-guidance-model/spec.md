# Feature Specification: Synthetic Terrain Guidance Model

**Feature Branch**: `008-synthetic-terrain-guidance-model`

**Created**: 2026-05-22

**Status**: Draft

**Input**: User description: "Build a model that mimics the dataset harvesting process and generates its own synthetic terrain signals that look visually realistic, so we can decouple training from game client harvesting and provide a joint-distribution critic for the V16.1 per-signal trainers."

## Problem Statement

The V16.1 model family treats each terrain signal independently:

```
minimap_rgb_256 → height
minimap_rgb_256 → normal
minimap_rgb_256 → holes
minimap_rgb_256 → liquid (footprint + type)
minimap_rgb_256 → texcomp (MCLY/MCAL decomposition)
```

Each model is trained separately, each has its own checkpoint, each validates independently. This is by design — it avoids task interference and allows per-target improvements without retraining everything.

However, independence creates a blind spot: **there is no mechanism that knows whether the joint configuration of all five predictions describes a physically plausible terrain.** The predicted normals might imply a different height field than the height model predicted. The predicted alpha might paint textures in places the height says are underwater. No existing loss term catches these cross-signal inconsistencies.

Additionally, the entire pipeline depends on harvested game client data. Every new client build, every dataset fix, every signal repair requires a full harvest cycle — reading proprietary files from disk, processing through the C# harvester, streaming into Zarr. This is slow, repetitive, and fundamentally tied to copyrighted data.

A guidance model that learns the joint distribution of `(minimap, height, normal, alpha, liquid, holes)` addresses both problems:

1. **Consistency critic**: score how plausible a joint configuration is, catching cross-model failure modes no per-signal loss can see
2. **Synthetic pair generation**: produce unlimited `(minimap, height, normal, ...)` tuples from the learned distribution, decoupling V16.1 training from client harvesting
3. **Signal infilling**: given any subset of signals, predict the missing ones

## Relationship to V16.1

This model is additive, not replacement. It does not change the V16.1 one-target-per-trainer contract. It sits alongside the per-signal trainers:

```
                ┌──────────────────┐
                │  Guidance Model  │
                │ (joint manifold) │
                └────────┬─────────┘
                         │ consistency score, synthetic pairs, infilled signals
                         ▼
   ┌─────────┬──────────┬──────────┬──────────┬──────────┐
   │ height  │  normal  │  holes   │  liquid  │  texcomp  │
   │ trainer │ trainer  │ trainer  │ trainer  │ trainer   │
   └─────────┴──────────┴──────────┴──────────┴──────────┘
        ▲         ▲          ▲          ▲          ▲
        └─────────┴──────────┴──────────┴──────────┘
                          │ minimap_rgb_256
                          ▼
                    V16 Zarr Dataset
```

The guidance model:
- Is **not** a V14+ residual terrain model and is not subject to Constitution Article IV
- Does **not** share weights with any V16.1 trainer
- Does **not** replace the minimap→signal mapping
- Does **not** require changes to the V16 Zarr dataset contract

## User Scenarios & Testing

### User Story 1 — Synthetic Training Pairs for V16.1 Normal Training (Priority: P1)

A terrain researcher wants to train `v16_1_normal` but does not want to run the full harvest pipeline for every new experiment. They sample synthetic `(minimap_rgb_256, normal_xyz)` pairs from the guidance model and train the normal model on synthetic data. The resulting checkpoint achieves comparable validation metrics to a model trained on harvested data.

**Why this priority**: Data independence is the primary motivation. If synthetic training data works, the entire harvest pipeline becomes optional for model development.

**Independent Test**: Generate a synthetic dataset of 1000 `(minimap, normal)` pairs. Train a V16.1 normal model on synthetic data for 50 epochs. Compare held-out validation metrics against a model trained on real harvested data.

**Acceptance Scenarios**:

1. **Given** a trained guidance model, **When** it generates 100 synthetic `(minimap, normal)` pairs, **Then** the minimaps display visible terrain structure with plausible colors, edges, and shading — no obvious artifacts, garbage pixels, or repeated identical outputs.
2. **Given** the synthetic pairs, **When** a V16.1 normal model trains on them, **Then** the model produces normals with nonzero angular agreement on held-out real validation tiles.
3. **Given** the same V16.1 normal model, **When** evaluated on 100 held-out real tiles, **Then** its `val_normal_angle` is within 15% of a model trained exclusively on real data.

---

### User Story 2 — Consistency Critic for Joint Prediction Validation (Priority: P1)

After running stitched inference with all five V16.1 checkpoints, the researcher scores each tile's joint `(minimap, height_pred, normal_pred, alpha_pred, liquid_pred, holes_pred)` configuration through the guidance model. Tiles with low joint-plausibility scores are flagged for manual review, revealing cross-model inconsistencies that no individual validation metric caught.

**Why this priority**: The independence of the five trainers creates unseen failure modes. A critic that catches these is a direct improvement to the V16.1 inference pipeline without retraining anything.

**Independent Test**: Run stitched inference on 100 tiles through all five V16.1 trainers. Score each tile's joint output with the guidance model. Manually inspect the lowest- and highest-scoring tiles. Verify that low scores correlate with visible inconsistencies (e.g., normals that imply a different height field than the height prediction).

**Acceptance Scenarios**:

1. **Given** a trained guidance model and a set of 50 V16.1 predicted tiles, **When** the model computes a joint-plausibility score for each tile, **Then** the scores produce a visible distribution (not all identical or all NaN).
2. **Given** the lowest-scoring tile from that distribution, **When** visually inspected, **Then** at least one cross-signal inconsistency is visible (e.g., sharp normal variation on a flat height plate, or alpha painting in a liquid region).
3. **Given** the highest-scoring tile, **When** visually inspected, **Then** the five predictions appear mutually consistent.

---

### User Story 3 — Hard-Region Synthetic Oversampling (Priority: P2)

A terrain researcher trains the guidance model. They then condition generation on "hard" terrain parameters (high deformation, steep gradients, complex alpha blends) to create a synthetic training set dominated by the kind of tiles the V16.1.1 curation system would classify as `hard` or `pathological`. The V16.1 normal model trains on this synthetic hard-only set and improves its performance on real hard tiles.

**Why this priority**: The V16.1.1 curation system already identifies hard tiles, but the real corpus has a limited number of them. Synthetic generation can produce arbitrarily many hard examples.

**Independent Test**: Generate 500 synthetic `hard` tiles from the guidance model. Train a V16.1 normal model on these only. Compare its validation performance on the real hard-tile held-out set against a baseline model that never saw extra hard examples.

**Acceptance Scenarios**:

1. **Given** a conditioned generation request for `hard` terrain, **When** the guidance model produces 50 samples, **Then** the samples show above-median deformation richness and height gradients compared to unconditioned samples.
2. **Given** the hard-conditioned synthetic set, **When** used to train a V16.1 normal model, **Then** that model's angular error on held-out real hard tiles is lower than a baseline trained only on real data.

---

### User Story 4 — Signal Infilling for Partial Dataset Repair (Priority: P3)

A harvested V16 store has a corrupted `normal_xyz` array for 200 tiles (all zeros). Instead of re-harvesting the entire build, the researcher infills the missing normals by conditioning the guidance model on the tile's `(minimap_rgb, height_257, alpha_256)` and generating the most likely normal field.

**Why this priority**: Dataset repairs currently require full harvest reruns. Infilling lets the researcher fix partial corruption without touching the C# pipeline.

**Independent Test**: Take 50 real tiles, zero out their normal arrays, infill through the guidance model conditioned on the remaining signals. Compare infilled normals against the original ground truth.

**Acceptance Scenarios**:

1. **Given** a tile with missing normals, **When** infilled from `(minimap, height, alpha)`, **Then** the infilled normals have a mean angular error within 25% of the model's standard prediction error on that tile.
2. **Given** a tile with missing minimap, **When** infilled from `(height, normal, alpha, liquid)`, **Then** the generated minimap shows recognizable terrain structure consistent with the input signals.

### Edge Cases

- What happens when the guidance model generates a tile that is outside the support of the training distribution? The downstream V16.1 trainer should not degrade — synthetic outliers should be detectable and filterable.
- How does the guidance model handle signal shapes that differ from the V16 contract (e.g., different resolution, missing channels)? The model should be strictly bound to the V16 array shapes and dtypes.
- What if the synthetic distribution is narrower than the real distribution (mode collapse)? The downstream trainer should still benefit, but the researcher needs visibility into distributional coverage.

## Requirements

### Functional Requirements

- **FR-001**: The guidance model MUST learn the joint distribution of at minimum: `minimap_rgb_256`, `height_257`, `normal_xyz`, `alpha_256`, `holes_16`, `liquid_mask_256`, `liquid_type_16`, `mcly_texture_ids_16x4`, `mcly_layer_mask_16x4`.
- **FR-002**: The guidance model MUST be able to generate complete synthetic `(minimap, height, normal, alpha, holes, liquid)` tuples that pass a basic visual plausibility test (no obvious garbage).
- **FR-003**: The guidance model MUST be able to score the joint plausibility of an arbitrary set of input signals, producing a scalar or vector score that ranks tiles by cross-signal consistency.
- **FR-004**: The guidance model MUST support conditional generation: given any non-empty subset of signals, generate the remaining signals.
- **FR-005**: The generated signals MUST match the V16 Zarr contract shapes and dtypes exactly.
- **FR-006**: The guidance model MUST train on the existing V16 Zarr corpus without requiring new harvest runs.
- **FR-007**: The guidance model MUST NOT share trainable weights with any V16.1 per-signal trainer.
- **FR-008**: The guidance model MUST export generated synthetic pairs in a format consumable by `V161Dataset` (i.e., as a V16-compatible Zarr store or a dataset wrapper).
- **FR-009**: The guidance model SHOULD support conditioning on difficulty parameters (deformation level, gradient strength, etc.) to steer generation toward hard/pathological terrain.
- **FR-010**: The guidance model SHOULD produce a diversity metric per generation batch so the operator can detect mode collapse.
- **FR-011**: The first implementation target MUST be the joint-distribution model itself, trained on the six-build V16 corpus. The consistency critic and infilling pipelines are downstream consumers of the same learned distribution.
- **FR-012**: The synthetic generation pipeline MUST NOT depend on private tooling, comfyui, or any code outside `wow-viewer/data-harvester/`.

### Key Entities

- **Guidance Model**: A generative model of the joint terrain signal distribution. Input is a complete or partial V16 signal tuple. Output is either a complete generated tuple or a plausibility score.
- **Synthetic Zarr Store**: A V16-compatible Zarr store containing only generated signals, consumable by `V161Dataset` as if it were a harvested build.
- **Consistency Score**: A scalar output of the guidance model indicating how plausible a given joint signal configuration is. Higher = more consistent with the training distribution.
- **Conditioning Vector**: A parameter vector (e.g., `[deformation_level, gradient_strength, alpha_complexity]`) that steers generation toward specific terrain characteristics.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A trained guidance model can generate synthetic `(minimap, normal)` pairs where the minimap passes visual inspection (no obvious artifacts, plausible terrain colors and structure).
- **SC-002**: A V16.1 normal model trained on synthetic-only data achieves a `val_normal_angle` within 15% of the same model trained on real harvested data from the same six-build corpus.
- **SC-003**: The guidance model's consistency score correlates with visible cross-signal inconsistencies in V16.1 stitched inference outputs (verified by manual inspection of top/bottom decile tiles).
- **SC-004**: The training pipeline for the guidance model runs on the existing six-build V16 corpus without requiring new harvests or data outside the Zarr stores.
- **SC-005**: The guidance model can be conditioned on difficulty parameters, and the resulting conditioned samples show measurably higher deformation metrics than unconditioned samples.

## Assumptions

- The existing V16 Zarr corpus (six builds) provides sufficient distributional coverage to train a useful joint-distribution model.
- The V16 array contract (shapes, dtypes, signal names) is stable and will not change during guidance model development.
- Joint-distribution modeling is a tractable research direction that does not require a foundation-model-scale investment.
- The primary use case is synthetic pair generation for V16.1 training, not replacing the C# harvester for production dataset builds.
- Decoupling from client harvesting is a gradual process — the guidance model initially needs the harvested data to train, and only later becomes self-sufficient for downstream training.
