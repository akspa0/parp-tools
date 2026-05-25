# Implementation Plan: V18 Canvas Paste Refinement Layer

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
2. Implement cluster assignment with canonical exemplar selection.
3. Preserve variant lineage metadata (`cluster_id`, `canonical_id`, variant rank).
4. Emit deduped manifests (`*_deduped.jsonl`) and cluster summary reports.
5. Emit visual atlas per cluster bucket for manual QA.

Validation:

- Duplicate reduction metrics present and reproducible.
- Cluster IDs stable across reruns with fixed seed/config.

## Phase 3 — Refined Manifest Generation for V18 Training

Goal: convert library output into trainer-consumable manifests with diversity-aware sampling.

1. Implement refined manifest builder consuming deduped cluster outputs.
2. Add normal-aware gates (transition richness, hard-region quality, mask validity).
3. Add cluster-balancing policy knobs and selection evidence.
4. Output manifests that existing V16.1/V17.1 trainer seams can consume directly.
5. Record command/config lineage and selection hashes.

Validation:

- Refined manifest loads in existing normal trainer path.
- Evidence reports duplicate ratio and cluster distribution.

## Phase 4 — Composition Graph Layer

Goal: model macro zone assembly from prefab relationships across stitched canvases.

1. Build spatial adjacency/co-occurrence graph from deduped canvas candidates.
2. Emit composition-family IDs (macro style groups) with reproducible keys.
3. Add graph evidence outputs (node/edge counts, top motifs, adjacency strength).
4. Integrate composition-family balancing metadata into refined manifest generation.

Validation:

- Composition graph outputs are deterministic with fixed seed/config.
- Refined manifest carries both prefab-family and composition-family balancing metadata.

## Phase 5 — V18 Baseline Launch Contract

Goal: define and execute first V18 baseline runs from refined manifests.

1. Define baseline run profiles (small/medium/large pool) using refined manifests.
2. Capture throughput and convergence evidence compared to prior non-refined curation.
3. Document initial V18 operating profile recommendations.

Validation:

- At least one bounded V18 normal run completes with full evidence package.
- Comparison table produced versus prior run family.
