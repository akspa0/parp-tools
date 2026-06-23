# Implementation Plan: V18 Canvas Paste Refinement Layer

> Historical plan note (2026-06-23): This plan remains useful prior art for paste mining, dedupe, composition graphs, and refined manifests. It is not the active brush-library plan. Current work must follow `wow-viewer/specs/076-full-map-fractal-brush-library/plan.md`, which adds full-map alpha fractal segmentation, 074 evidence linkage, chonker/one-off rejection, and tileset-variant provenance before model training.

**Feature**: `024-v18-canvas-paste-refinement-layer`  
**Spec**: `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/spec.md`

## Constitution Check

- Repo independence: pass (all work under `wow-viewer/`)
- Library/tool ownership: pass (data-harvester scripts as workflow surface)
- Real-data validation: required for signoff (staged multi-build corpus)
- One phase at a time: enforced via phased plan below
- Training-script mutation discipline: this feature targets dataset refinement first; trainer integration only after manifest contract is proven

## Phase 1 — Canvas Mining Surface

Goal: detect paste candidates on stitched map canvases, not per-tile fragmentation.

1. Add stitched canvas assembly utility for bounded build/map subsets.
2. Implement canvas signal composition (`alpha`, `transition`, `hard_region`, masks) over stitched regions.
3. Implement connected-region and rectangle candidate extraction in canvas space.
4. Emit candidate artifacts with `canvas_bbox`, `tile_coverage`, score metrics, and debug overlays.
5. Write deterministic evidence outputs (`summary.json`, json/jsonl manifests, config snapshot).

Validation:

- Bounded run outputs at least one multi-tile candidate on known maps.
- Evidence includes candidate count and multi-tile ratio.

## Phase 2 — Cross-Build Dedupe + Library Construction

Goal: collapse repeated base content across builds/maps into canonical families.

1. Implement deterministic candidate fingerprinting for cluster seeding.
2. Add alpha-layer-aware signatures (MCAL-layer descriptors) to dedupe keys.
3. Implement cluster assignment with canonical exemplar selection.
4. Preserve variant lineage metadata (`cluster_id`, `canonical_id`, variant rank).
5. Emit deduped manifests (`*_deduped.jsonl`) and cluster summary reports.
6. Emit visual atlas per cluster bucket for manual QA.
7. Emit paste-library catalog with stable `paste_id`, canonical names, aliases, build span, and AreaID distribution.

Validation:

- Duplicate reduction metrics present and reproducible.
- Cluster IDs stable across reruns with fixed seed/config.

## Phase 3 — Refined Manifest Generation for V18 Training

Goal: convert library output into trainer-consumable manifests with diversity-aware sampling.

1. Implement refined manifest builder consuming deduped cluster outputs.
2. Add normal-aware gates (transition richness, hard-region quality, mask validity).
3. Add cluster-balancing policy knobs and selection evidence.
4. Add family-balanced sampling metadata so manifests are balanced by paste family, not raw frequency.
5. Output manifests that existing V16.1/V17.1 trainer seams can consume directly.
6. Record command/config lineage and selection hashes.

Validation:

- Refined manifest loads in existing normal trainer path.
- Evidence reports duplicate ratio and cluster distribution.

## Phase 4 — Composition Graph Layer

Goal: model macro zone assembly from prefab relationships across stitched canvases.

1. Build spatial adjacency/co-occurrence graph from deduped canvas candidates.
2. Add MCNK AreaID overlap extraction and dominant AreaID labeling per candidate/group.
3. Emit composition-family IDs (macro style groups) with reproducible keys.
4. Add AreaID-aware weighting/grouping and soft-label tolerance rules.
5. Add graph evidence outputs (node/edge counts, top motifs, adjacency strength, AreaID distributions).
6. Integrate composition-family balancing metadata into refined manifest generation.

Validation:

- Composition graph outputs are deterministic with fixed seed/config.
- Refined manifest carries both prefab-family and composition-family balancing metadata.

## Phase 5 — V18 Baseline Launch Contract

Goal: assign deterministic descriptive names and review state for paste families.

1. Generate deterministic candidate names from role/shape/layer/AreaID descriptors.
2. Emit naming confidence scores and review status (`auto`, `reviewed`, `locked`).
3. Add alias support for observed terminology (`start`, `end`, `left`, `right`, etc.).
4. Persist naming metadata in paste-library catalog outputs.

Validation:

- Reruns with fixed seed/config preserve stable family IDs and candidate names.
- Low-confidence names are explicitly flagged for review.

## Phase 6 — V18 Baseline Launch Contract

Goal: define and execute first V18 baseline runs from refined manifests.

1. Define baseline run profiles (small/medium/large pool) using refined manifests.
2. Capture throughput and convergence evidence compared to prior non-refined curation.
3. Document initial V18 operating profile recommendations.

Validation:

- At least one bounded V18 normal run completes with full evidence package.
- Comparison table produced versus prior run family.
