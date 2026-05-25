# Feature Specification: V17 Unified Normal Trainer (V16.1.2 + Height Channel + Curation)

**Feature Branch**: `022-v17-unified-normal-height-refiner`

**Created**: 2026-05-24

**Status**: Draft

## Problem Statement

Current V16.1 normal work is split across multiple variants and specs (V16.1.1, V16.1.2, V16.1.3, V16.1.4). This creates repeated confusion about which trainer/model path is actually active in a run.

The immediate need is a single, unambiguous trainer mode that combines:

1. V16.1.2 refiner + distillation behavior
2. V16.1.3 height input channel for the main normal model
3. Small curated training pool defaults for fast, high-signal experiments
4. Dataset normal-target checkerboard interpolation fix active by default

This unified mode is named **V17** for workflow clarity.

## Goals

- Eliminate run-mode ambiguity.
- Make the intended hybrid behavior explicit and fail-fast.
- Keep experiments small and curated first (quality over quantity).
- Ensure normal ground truth is dense/smooth (no checkerboard halftone supervision artifact).

## Non-Goals

- Replacing the full V16.2 7-channel model family.
- Rewriting all previous trainers.
- Launching long 1000-epoch runs before the unified mode is validated.

## Architecture

V17 uses the V16.1 normal trainer path with explicit hybrid semantics:

```
input = cat(minimap_rgb, height_norm)  # 4ch
pred  = main_normal_model(input)       # 3ch normals

teacher = refiner(cat(pred.detach(), height_norm))

L_total = L_main(pred, gt, mask) + w_distill * L_distill(pred, teacher, mask)
```

Where:

- `L_main` is the existing terrain-aware normal loss
- `L_distill` is cosine distillation from V16.1.2
- refiner activation/evaluation behavior follows V16.1.2, but is allowed with height-channel enabled

## User Scenarios & Testing

### User Story 1 — One Explicit Variant, No Silent Fallback (Priority: P1)

Researcher launches V17 and can verify from startup logs + config that the hybrid behavior is active.

**Independent Test**: Start a 1-epoch run and validate config/log markers.

**Acceptance Scenarios**:

1. **Given** `--normal-variant v17_hybrid`, **When** training starts, **Then** startup logs print resolved variant and all enabled components (`height_channel=true`, `refiner_enabled=true`, distill weight).
2. **Given** conflicting flags, **When** command parsing runs, **Then** the process fails with a clear error (no fallback to another variant).

---

### User Story 2 — Checkerboard GT Artifact Removed From Supervision (Priority: P1)

Researcher validates that normal targets are dense and not half-tone checkerboard before training quality decisions.

**Independent Test**: Inspect one sampled batch and validation panel.

**Acceptance Scenarios**:

1. **Given** V17 dataset loading, **When** reading normal targets, **Then** `normal_mask` coverage for normal-present tiles is near-full (dense supervision, not ~0.5 checkerboard).
2. **Given** validation previews, **When** viewing `normal_gt`, **Then** the obvious checkerboard halftone pattern is absent.

---

### User Story 3 — Small Curated Run Default (Priority: P1)

Researcher runs fast, high-signal experiments by default using curated subsets.

**Independent Test**: Launch with defaults and inspect pool summaries.

**Acceptance Scenarios**:

1. **Given** V17 default run, **When** config is written, **Then** defaults are `epochs=50`, `train_max_tiles=80`, and a curated manifest path is required/provided.
2. **Given** pool selection evidence, **When** summaries are inspected, **Then** selected tile counts match configured small-pool limits.

## Requirements

### Functional Requirements

- **FR-001**: Trainer MUST expose an explicit normal variant selector including `v17_hybrid`.
- **FR-002**: `v17_hybrid` MUST enable height-channel main model input and V16.1.2 refiner/distill logic together.
- **FR-003**: Trainer MUST reject conflicting/ambiguous CLI combinations with clear errors.
- **FR-004**: Run config MUST persist resolved variant and resolved feature toggles.
- **FR-005**: V17 normal data path MUST apply checkerboard-gap interpolation for MCNR-derived targets before loss computation.
- **FR-006**: V17 defaults MUST target small curated scouting runs: 50 epochs, 80 train tiles, bounded validation pool.
- **FR-007**: Curation manifest usage MUST be first-class and visible in run config/evidence.
- **FR-008**: Existing legacy variants (V16.1.1/1.2/1.3/1.4) MUST remain callable for controlled A/B comparisons.

### Key Entities

- **V17 Hybrid Variant**: Explicit trainer mode combining V16.1.2 refiner/distill and V16.1.3 height-channel input.
- **Resolved Variant Contract**: Logged and persisted run metadata proving which behavior executed.
- **Curated Small-Pool Recipe**: Default experiment recipe focused on signal quality and iteration speed.

## Success Criteria

- **SC-001**: A 1-epoch sanity run proves resolved `v17_hybrid` settings in logs/config with no ambiguity.
- **SC-002**: A 50-epoch curated run (`train_max_tiles=80`) completes and writes expected evidence files.
- **SC-003**: Validation `normal_gt` panels no longer show checkerboard halftone artifact.
- **SC-004**: At least one A/B comparison (`v16_1_3_height` vs `v17_hybrid`) can be executed without changing code, only variant flag.

## Assumptions

- Existing V16 dataset artifacts and curation manifests remain available under `wow-viewer/output/datasets/v16`.
- The checkerboard interpolation fix is acceptable as training supervision repair for MCNR-derived targets.
- Small curated pools are sufficient to detect directional improvements before long-run scaling.
