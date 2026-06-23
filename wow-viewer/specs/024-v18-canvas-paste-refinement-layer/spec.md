# Feature Specification: V18 Terrain System

**Feature Branch**: `024-v18-canvas-paste-refinement-layer`

**Created**: 2026-05-25

**Status**: Implemented historical surface; superseded for current brush-library direction by `076-full-map-fractal-brush-library`

> Supersession note (2026-06-23): V18 paste mining remains useful prior art and comparison input. It is not the final brush-library truth because it does not fully solve full-map alpha fractal segmentation, 074 component linkage, tileset variant evidence, and curated rejection of chonkers/one-off details.

## Problem Statement

Current curation and paste mining are tile-local. That fragments authored structures that span multiple ADT tiles and over-counts duplicate data across builds where the same base content is copy-pasted and lightly retouched.

**V18 is the whole terrain system.** It unifies the dataset refinement pipeline (canvas mining → dedupe → composition → library → manifest) with the model family (per-signal CNNs) under one namespace.

For the dataset:
- detect large multi-tile pastes from stitched map signals
- dedupe repeated motifs across builds/maps
- represent canonical paste families plus meaningful variants
- produce training manifests that reduce duplicate supervision and improve signal diversity

For the model:
- `train_v18.py` — unified entrypoint for all tasks (normal, height, holes, liquid, texcomp)
- `v18_models.py` — re-exports V16.1 model architectures under V18 names
- `v18_dataset.py` — re-exports V161Dataset as V18Dataset

The goal is smarter, smaller per-signal models by improving training data quality rather than adding model complexity.

## Scope

V18 owns the entire terrain training pipeline in `wow-viewer/data-harvester`:

1. Mines paste candidates on stitched map canvases (not tile boundaries)
2. Builds cross-build/cross-map deduped paste libraries
3. Produces normal-aware refined train/val manifests from canonical paste families
4. Defines a two-layer training data contract:
   - **Prefab Library Layer** (micro artwork primitives)
   - **Composition Layer** (macro zone assembly grammar)
5. Unified training entrypoint (`train_v18.py`) for all per-signal tasks
6. V18-named model and dataset classes (`v18_models.py`, `v18_dataset.py`)

Out of scope:

- new model architecture changes (V18 uses V16.1 model architectures)
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
- **FR-011**: Candidate and dedupe metadata MUST include alpha-layer-aware signatures so visually similar RGB patches with different MCAL-layer composition remain distinguishable.
- **FR-012**: Canvas candidates MUST include MCNK AreaID overlap metadata (`area_id_coverage`, dominant AreaID set) to anchor macro-zone structure.
- **FR-013**: Composition graph construction MUST support AreaID-aware grouping/weighting and treat AreaID as soft labels (tolerant of missing/noisy assignments).
- **FR-014**: Refinement evidence MUST include per-layer variant statistics and AreaID distribution summaries.
- **FR-015**: The refinement layer MUST emit a paste-library metadata catalog with stable IDs, canonical names, aliases, and role/shape tags for each paste family.
- **FR-016**: The paste library MUST support orientation/role descriptors (`start`, `end`, `left`, `right`, `corner`, `connector`, `fill`, `transition`) when inferable from spatial relationships.
- **FR-017**: The library MUST track alpha-layer profiles and normal-relief profiles per family to distinguish 2D/3D variant behavior.
- **FR-018**: The library MUST preserve canonical exemplar + variant linkage and expose family-balanced sampling metadata for downstream manifest generation.
- **FR-019**: Auto-generated names MUST be deterministic with confidence metadata and support human review/lock workflows.
- **FR-020**: Refinement outputs MUST include build-span and AreaID distribution metadata per family for macro-zone lineage analysis.

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
- **Layer Signature**: Per-alpha-layer fingerprint/descriptor for a candidate region.
- **AreaID Coverage**: Fractional overlap of candidate region with one or more MCNK AreaIDs.
- **Paste Family**: Cluster of related paste candidates with one canonical exemplar and controlled variants.
- **Paste Library Catalog**: Metadata store describing paste families, names, tags, lineage, and sampling controls.
- **Role Tags**: Orientation/composition descriptors such as `start`, `end`, `left`, `right`, `corner`, `connector`, `fill`, `transition`.
- **Name Confidence**: Deterministic confidence score for auto-generated canonical naming.

## Success Criteria

- **SC-001**: On bounded six-build mining, at least 25% of selected top candidates are multi-tile (tile coverage > 1).
- **SC-002**: Cross-build dedupe reduces raw candidate count by at least 35% on the same bounded corpus.
- **SC-003**: Refined manifest generation exposes cluster distribution stats and duplicate ratio metrics in evidence outputs.
- **SC-004**: V16.1 normal training launched from refined manifest shows reproducible command lineage and stable selection counts across reruns.
- **SC-005**: End-to-end workflow (`mine -> dedupe -> refine-manifest`) runs without modifying parser ownership or external repo dependencies.
- **SC-006**: Refined corpus compression achieves at least 40% reduction in raw candidate rows while preserving at least 90% of top transition/hard-region motif coverage.
- **SC-007**: Composition graph outputs include reproducible adjacency/co-occurrence stats for multi-tile zone structures.
- **SC-008**: At least one bounded V18 baseline run consumes refined manifests that include both prefab-family and composition-family balancing metadata.
- **SC-009**: Dedupe reports show layer-signature-aware separation where RGB-similar candidates differ materially by alpha-layer composition.
- **SC-010**: Composition outputs include AreaID coverage summaries and dominant AreaID labeling for macro groups.
- **SC-011**: Paste library generation produces stable family IDs and canonical names across reruns with fixed seed/config.
- **SC-012**: Family-balanced refined manifests report reduced duplicate dominance versus raw-frequency sampling.
- **SC-013**: Library metadata includes role/shape tagging coverage and explicit review status for low-confidence names.

## Assumptions

- Existing V16 dataset stores and curation manifests remain available under `wow-viewer/output/datasets/v16`.
- Existing normal training seams continue to consume manifest-based tile selection.
- Initial dedupe can use deterministic perceptual/feature hashing before optional richer embedding clustering.
- Refinement layer is a data pipeline enhancement, not a model-architecture change in this feature scope.
- Azeroth/Kalimdor historical continuity implies substantial cross-build motif reuse, so cross-build dedupe is expected to remove many rows without major motif loss.
- V18 namespace (`v18_models.py`, `v18_dataset.py`, `train_v18.py`) is thin re-export of V16.1 implementations. V16.1 files remain the canonical implementation layer.
- Training output paths use `models/v18/<task>/runs/<run-name>/` for V18 runs.

## Namespace Map

| V18 Name | Implementation | File |
|----------|---------------|------|
| `V18NormalModel` | `V161NormalModel` | `v18_models.py` → `v16_1_models.py` |
| `V18HeightModel` | `V161HeightModel` | same |
| `V18HolesModel` | `V161HolesModel` | same |
| `V18LiquidModel` | `V161LiquidModel` | same |
| `V18TexcompModel` | `V161TexcompModel` | same |
| `V18NormalHeightModel` | `V161NormalHeightModel` | same |
| `V18NormalRefiner` | `V161NormalRefiner` | same |
| `V18Dataset` | `V161Dataset` | `v18_dataset.py` → `v16_1_dataset.py` |
| `train_v18.py` | calls `run_task()` | scripts entrypoint |

## Implemented Surface

All V18 pipeline scripts exist and have been verified on the 6-build corpus:

| Script | Phase | Status |
|--------|-------|--------|
| `mine_v18_pastes_canvas.py` | 1+2: Canvas mining + dedupe | Done |
| `build_v18_refined_manifest.py` | 3: Refined manifest | Done |
| `build_v18_composition_graph.py` | 4: Composition graph | Done |
| `build_v18_paste_library_catalog.py` | 5: Paste library | Done |
| `run_v18_baseline_contract.py` | 6: Baseline comparison | Done |
| `v18_models.py` | Namespace re-export | Done |
| `v18_dataset.py` | Namespace re-export | Done |
| `train_v18.py` | Unified training | Done |

### Key Results (6-Build Full Pipeline)

- 600 tiles → 144 paste candidates → 140 clusters → 65 composition families → 47 refined tiles → 140 paste library families
- Cross-build dedupe: ~141 candidates reduced to ~140 clusters (~0.7% reduction — conservative dedupe at Hamming threshold 12)
- Refined vs non-refined val_loss comparison: 0.5623 (refined) vs 0.6505 (nonref) on small profile (~13.5% improvement)
- GPU: CUDA-supported on RTX 4070 Ti SUPER (16 GB VRAM)
