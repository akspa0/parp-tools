# Feature Specification: Image-Only WDL Prior

**Feature Branch**: `108-image-wdl-prior`  
**Created**: 2026-07-15  
**Status**: Implementing

## User Scenarios & Testing

### User Story 1 - Predict terrain prior from minimap RGB (Priority: P1)

A user supplies only a 256×256 minimap RGB tile and receives the paired 17×17 outer plus 16×16 inner WDL height prior, with no height, normal, object, or other game signal read at inference.

**Independent Test**: Run prediction against a held-out complete-map split from a paired store and confirm the output has exactly 545 finite samples.

### User Story 1a - Verify against a real tile and standalone image (Priority: P1)

A user can select one real paired-store row, run RGB-only inference, and inspect the predicted and
ground-truth WDL lattices plus world-unit error metrics. The exported minimap PNG can then be run
through the standalone image path and produces the same kind of paired lattice without a store.

### User Story 2 - Feed generated prior into terrain refinement (Priority: P1)

A user can provide the predicted outer prior to the existing V8 terrain refiner; the refiner must not silently replace it with ground truth.

**Independent Test**: Build V8 input using a generated prior and verify it differs from a ground-truth-prior input while retaining valid shape/range.

### User Story 3 - Recover reusable terrain-art motifs, not zones or rectangles (Priority: P1)

An analyst can inspect repeated, irregular terrain-art candidates from the 0.5.3 corpus. A continuous
alpha-painted zone is never itself called a prefab. A candidate is a connected graph of actual terrain
cells/chunks whose alpha and relative-height topology recurs under translation, mirror, or rotation;
its irregular mask, rather than its bounding rectangle, is the payload.

**Independent Test**: Two equal irregular multi-chunk motifs embedded in larger continuous alpha zones
produce one repeated family with the same cell topology and paired relative-height payload. A whole
zone and a rectangular bounding crop alone cannot qualify as a candidate.

### Requirements

- **FR-001**: Training targets MUST be WDL outer `height_257[::16,::16]` and inner `height_257[8::16,8::16]`, never `height_257[::8]`.
- **FR-002**: Inference MUST read minimap RGB only.
- **FR-003**: Train/validation partitions MUST hold out complete maps or source groups.
- **FR-004**: The predictor MUST serialize model/input normalization/target contract in its checkpoint.
- **FR-005**: Generated-prior integration MUST remain distinct from V8 ground-truth-prior training. Deployment-shaped V8 training MUST consume a hash-bound generated-prior archive and fail closed on wrong-store or missing selected rows.
- **FR-006**: Real-tile evaluation MUST read `height_257` only after RGB-only prediction, solely to
  calculate ground-truth lattice errors.
- **FR-007**: Standalone image inference MUST accept a PNG/JPEG and checkpoint without a paired store,
  WDL file, height grid, or other auxiliary signal.
- **FR-008**: Prefab discovery MUST start from chunk-aligned real alpha/relative-height cell signatures,
  grow transform-equivalent adjacent cells into an irregular motif graph, and preserve its mask. It
  MUST NOT treat macro/blocky connected components, an entire zone, or a fixed rectangular crop as a
  prefab candidate.
- **FR-009**: A motif family MUST be formed only from repeated cell-topology signatures, with source
  tiles, world-cell coordinates, irregular mask, transform, alpha payload, and relative-height payload
  retained for review.

### Success Criteria

- **SC-001**: CPU tests prove output shapes, finite values, RGB-only inference, and exact WDL lattice mapping.
- **SC-002**: A user-run training command writes resumable checkpoints and held-out metrics.
- **SC-003**: A user-run inference command writes one 17×17+16×16 prior pair per synthetic input row, and V8 training records that exact archive path and SHA-256 in `history.json`.
- **SC-004**: A real-tile report records separate outer/inner MAE and RMSE in world units and exports
  the input minimap, predicted lattice, and truth lattice for direct inspection.
- **SC-005**: Motif-review contact sheets show each repeated irregular mask over its local alpha and
  height context, plus transformed placements. A bounding rectangle alone is never presented as the
  motif payload.

## Assumptions

- Spec 103 paired stores contain `minimap_rgb` and `height_257`.
- The user runs training; the agent implements, validates CPU behavior, and prepares commands.
