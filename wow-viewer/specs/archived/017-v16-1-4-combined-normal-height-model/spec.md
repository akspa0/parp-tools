# Feature Specification: V16.1.4 Combined Normal + Height Model

**Feature Branch**: `017-v16-1-4-combined-normal-height-model`

**Created**: 2026-05-24

**Status**: Superseded — combined normal+height head was attempted but the V18 distill corpus lane (spec 047) follows the per-signal V16.1 architecture instead. Reroute follow-up to `047-v18-distill-corpus-open-source-loop`.

**Input**: Normal model plateaued at epoch 123 best despite 200+ epochs. Height-from-normals numerical integration produces garbage. Combined normal+height prediction in one model solves both.

## Problem Statement

V16.1.3 trained a normal model with height as input channel. Two issues:

1. **Normal plateau**: val_loss stopped improving at epoch 123 despite continued training. The model may benefit from a multi-task signal (height) to regularize and prevent overfitting.
2. **No height output**: to export OBJ terrain, we need height. Deriving height from normals by numerical integration is numerically unstable and produces garbage meshes.

The fix: train one model that predicts both normals AND height from the same backbone. Two output heads share feature extraction, height supervision regularizes the normal head, and the height output goes directly to OBJ export without integration.

## Architecture

```
V16.1.3: cat(minimap, height)(4ch) → backbone → normal_head → Tanh(3ch normals)
V16.1.4: cat(minimap, height)(4ch) → backbone → normal_head → Tanh(3ch normals)
                                             → height_head → Conv(1ch height)
```

Same backbone, two heads. Loss = `w_normal * L_normal + w_height * L_height`.

## User Scenarios & Testing

### User Story 1 — Combined Model Trains and Converges (Priority: P1)

A terrain researcher trains the combined model. Both normal and height losses decrease. The height output produces a reasonable mesh when exported to OBJ.

**Why this priority**: Core value — does the combined model work?

**Independent Test**: A bounded smoke run (400 train, 48 val, 20 epochs) completes with both losses decreasing. Export a minimap to OBJ using the height output.

**Acceptance Scenarios**:

1. **Given** a V16.1.4 smoke run, **When** 20 epochs complete, **Then** both `train_normal` and `train_height` losses decrease.
2. **Given** the best checkpoint, **When** a minimap is exported to OBJ, **Then** the mesh shows visible terrain relief (not flat).

---

### User Story 2 — Export Works from Single Checkpoint (Priority: P1)

The export script produces a correct OBJ + MTL + texture from the combined model checkpoint, using the height head output directly.

**Why this priority**: The export is broken because height-from-normals integration doesn't work.

**Independent Test**: Run `export_terrain_obj.py` with `--height-checkpoint` pointing to the V16.1.4 best checkpoint. Open in MeshLab — terrain is visible.

**Acceptance Scenarios**:

1. **Given** a V16.1.4 checkpoint, **When** export runs, **Then** the OBJ has 257x257 vertices with visible height variation.
2. **Given** the exported OBJ, **When** opened in MeshLab, **Then** the texture maps correctly and terrain features are recognizable.

---

## Requirements

### Functional Requirements

- **FR-001**: A new model class `V161NormalHeightCombinedModel` MUST have two output heads: normals(3ch Tanh) and height(1ch).
- **FR-002**: The backbone MUST be shared between both heads.
- **FR-003**: The training loss MUST be `w_normal * L_normal + w_height * L_height` with configurable weights.
- **FR-004**: The dataset MUST provide both `normals` and `height_norm` as targets.
- **FR-005**: The export script MUST support loading the combined model and using the height head output directly.
- **FR-006**: The CLI MUST expose `--normal-weight` and `--height-weight` flags for loss balancing.

### Key Entities

- **Combined Normal+Height Model**: Single backbone with two heads predicting both signals.
- **Height Head**: Conv head producing 1ch height output from the shared backbone features.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A bounded V16.1.4 smoke run completes with both losses decreasing over 20 epochs.
- **SC-002**: Export from the V16.1.4 checkpoint produces a mesh with visible terrain relief in MeshLab.

## Assumptions

- The existing V16.1 dataset already carries both `normal_xyz` and `height_257` — no dataset changes needed.
- Multi-task training (normals + height) will regularize the normal head and break the epoch 123 plateau.
- The height head adds negligible parameters (~500 extra params).
