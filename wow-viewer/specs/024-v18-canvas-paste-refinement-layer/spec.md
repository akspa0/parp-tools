# Feature Specification: V18 Canvas Paste Refinement Layer

**Feature Branch**: `024-v18-canvas-paste-refinement-layer`

**Created**: 2026-05-25

**Status**: Draft

## Problem Statement

Current curation and paste mining are tile-local. That fragments authored structures that span multiple ADT tiles and over-counts duplicate data across builds where the same base content is copy-pasted and lightly retouched.

For V18, we need a dataset refinement layer that models how maps are authored on a large canvas:

- detect large multi-tile pastes from stitched map signals
- dedupe repeated motifs across builds/maps
- represent canonical paste families plus meaningful variants
- produce training manifests that reduce duplicate supervision and improve signal diversity

The goal is smarter, smaller per-signal models by improving training data quality rather than adding model complexity.

## Scope

This feature defines and implements a V18 refinement layer in `wow-viewer/data-harvester` that:

1. Mines paste candidates on stitched map canvases (not tile boundaries)
2. Builds cross-build/cross-map deduped paste libraries
3. Produces normal-aware refined train/val manifests from canonical paste families
4. Integrates with existing V16.1/V17.1 trainer inputs through manifest consumption

5. Defines a two-layer V18 training data contract:
   - **Prefab Library Layer** (micro artwork primitives)
   - **Composition Layer** (macro zone assembly grammar)

Out of scope:

- new terrain model architecture changes
- runtime renderer changes
- shipping model checkpoints

## User Scenarios & Testing

### User Story 1 — Canvas Mining Captures Multi-Tile Patches (Priority: P1)

A researcher mines paste candidates and sees bounding boxes that span multiple tile coordinates, matching visible authored paste boundaries.

**Independent Test**: run canvas miner on a bounded map/build set and verify at least one candidate covers >1 tile.

**Acceptance Scenarios**:

1. **Given** stitched map inputs, **When** mining runs, **Then** candidate outputs include `canvas_bbox` and `tile_coverage` fields.
2. **Given** large paste-like regions, **When** candidates are exported, **Then** multi-tile coverage is preserved as one candidate instead of fragmented tile-local pieces.

---

### User Story 2 — Dedupe Collapses Shared Base Content Across Builds (Priority: P1)

A researcher can remove repeated legacy content and keep one canonical exemplar per paste family plus curated variants.

**Independent Test**: run dedupe on six-build corpus and confirm candidate count drops with reproducible cluster IDs.

**Acceptance Scenarios**:

1. **Given** candidates from multiple builds/maps, **When** dedupe runs, **Then** output includes stable `cluster_id` assignments and duplicate counts.
2. **Given** near-identical candidates from different builds, **When** canonicalization runs, **Then** one exemplar is marked canonical and others are linked as variants.

---

### User Story 3 — Normal-Aware Refinement Produces Cleaner Training Manifests (Priority: P1)

A trainer consumes refined manifests where duplicates are reduced and normal-rich transitions are balanced, improving convergence stability.

**Independent Test**: generate a refined manifest and confirm it includes cluster balancing metadata and normal/transition quality gates.

**Acceptance Scenarios**:

1. **Given** deduped library output, **When** refined train/val manifests are generated, **Then** manifests include cluster weights and selection provenance.
2. **Given** refined manifests, **When** V16.1 normal training starts, **Then** trainer evidence reports reduced duplicate ratio and expected motif diversity.

---

### User Story 4 — Workflow Surfaces Are Reproducible and Inspectable (Priority: P1)

A developer can rerun mining/dedupe/refine with fixed seeds and compare evidence artifacts deterministically.

**Independent Test**: run the same commands twice with identical inputs and compare summary hashes/row counts.

**Acceptance Scenarios**:

1. **Given** fixed seeds/config, **When** workflow reruns, **Then** summary counts and deterministic keys match.
2. **Given** output artifacts, **When** inspecting evidence files, **Then** command/config/source-manifest lineage is explicit.

## Requirements

### Functional Requirements

- **FR-001**: The refinement layer MUST support stitched map-canvas mining with signals aggregated across tile boundaries.
- **FR-002**: Canvas candidates MUST include both canvas-space geometry (`canvas_bbox`) and tile-space coverage (`tile_coverage`).
- **FR-003**: Candidate extraction MUST support alpha/transition/hard-region driven detection and configurable thresholds.
- **FR-004**: Dedupe MUST operate across builds/maps and emit stable cluster identifiers for reproducibility.
- **FR-005**: Dedupe output MUST preserve canonical exemplar + variant lineage metadata.
- **FR-006**: Refinement outputs MUST generate manifest files consumable by existing V16.1/V17.1 trainer manifest input seams.
- **FR-007**: Manifest generation MUST include normal-aware quality gates and cluster-balancing metadata.
- **FR-008**: Workflow MUST write machine-readable evidence (`summary.json`, jsonl manifests, config snapshot, duplicate/cluster stats).
- **FR-009**: Commands MUST support bounded runs (`--builds`, `--maps`, tile/map caps, seed controls) for fast iteration.
- **FR-010**: The refinement layer MUST remain inside `wow-viewer/` and MUST NOT introduce external path dependencies.

### Key Entities

- **Canvas Candidate**: Multi-tile detected paste region with canvas bbox and tile coverage.
- **Paste Cluster**: Dedupe family of near-identical candidates across builds/maps.
- **Canonical Exemplar**: Representative candidate for a cluster used as primary training source.
- **Variant**: Member of a cluster retained for diversity/era-specific retouch differences.
- **Refined Manifest**: Trainer-consumable selection of tiles/regions weighted by cluster diversity and quality gates.
- **Artwork Corpus**: Canonical prefab/paste library treated as authored art assets, not raw tile rows.
- **Alpha Brush Source**: Alpha-layer evidence interpreted as artist brush-work encoding (opacity/pressure-style signal proxy).
- **Composition Graph**: Spatial relationship graph of prefab usage and adjacency over stitched map canvases.
- **Macro Composition Object**: Zone-scale style structure assembled from repeated prefab motifs across multiple tiles.

## Success Criteria

- **SC-001**: On bounded six-build mining, at least 25% of selected top candidates are multi-tile (tile coverage > 1).
- **SC-002**: Cross-build dedupe reduces raw candidate count by at least 35% on the same bounded corpus.
- **SC-003**: Refined manifest generation exposes cluster distribution stats and duplicate ratio metrics in evidence outputs.
- **SC-004**: V16.1 normal training launched from refined manifest shows reproducible command lineage and stable selection counts across reruns.
- **SC-005**: End-to-end workflow (`mine -> dedupe -> refine-manifest`) runs without modifying parser ownership or external repo dependencies.
- **SC-006**: Refined corpus compression achieves at least 40% reduction in raw candidate rows while preserving at least 90% of top transition/hard-region motif coverage.
- **SC-007**: Composition graph outputs include reproducible adjacency/co-occurrence stats for multi-tile zone structures.
- **SC-008**: At least one bounded V18 baseline run consumes refined manifests that include both prefab-family and composition-family balancing metadata.

## Assumptions

- Existing V16 dataset stores and curation manifests remain available under `wow-viewer/output/datasets/v16`.
- Existing normal training seams continue to consume manifest-based tile selection.
- Initial dedupe can use deterministic perceptual/feature hashing before optional richer embedding clustering.
- Refinement layer is a data pipeline enhancement, not a model-architecture change in this feature scope.
- Azeroth/Kalimdor historical continuity implies substantial cross-build motif reuse, so cross-build dedupe is expected to remove many rows without major motif loss.
