# Implementation Plan: Terrain Paste and Fractal Motif Archaeology

## Technical Context

The implementation belongs under `wow-viewer/data-harvester/` and consumes existing NPZ/control-corpus contracts. It must not rewrite ADT/MCAL/MCLY readers or the renderer. The first slice is deterministic corpus indexing, descriptors, transformed-motif retrieval, and visual review. Model training is a later ablation owned by the user.

The pipeline is deliberately staged:

```text
source arrays
  -> observation/albedo normalization
  -> lossless alpha/provenance preservation
  -> parallel alpha views (occupancy, transitions, atomic, block, macro, ordered, cross-tile)
  -> tileset and ordered alpha descriptors
  -> multiscale fractal/motif descriptors
  -> transformed cross-boundary retrieval
  -> paint/sculpt-intent scaffold
  -> confidence-bearing guidance bundle
  -> Spec 139 clean-signal geometry model
  -> optional object-slot and renderer/referee stages
```

## Architecture decisions

- Keep `height_257` as a reference/target, never as a deployment input.
- Treat source-side alpha/MCAL and layer order as supervision/evidence for an inferred paint
  scaffold, never as a hidden deployment input.
- Preserve the distinction between opaque layer 0 and alpha-bearing layer 1. Layer 0 is the base
  tileset, layer 1 is the first paste/paint candidate, and the pipeline must never invent an
  `alpha_0` tensor.
- Classify paint/relief relationships as intact, retextured, resculpted, unknown, or insufficient.
- Treat the first motif index as a deterministic retrieval system; neural motif learning is conditional on retrieval proof.
- Preserve three linked spatial scales: Python-derived atomic brush components, C#-derived paste
  blocks, and C# full-map macro-prefab context. Do not collapse parent context into atomic labels.
- Make complete alpha preservation the fan-out boundary: never discard a source layer because one
  derived interpretation is unhelpful. Keep raw, transition, atomic, block, macro, ordered-layer,
  and cross-tile views independently available with their own validity and confidence.
- Make every signal optional at ingestion but explicit in the manifest.
- Split by `source_group_id` and `paste_family_id`, not by individual overlapping windows.
- Preserve arbitrary spatial offsets and cross-tile context.
- Emit one guidance bundle with independent evidence heads so Spec 139 can run parity and ablations.
- Allow a frozen synthetic reference model to emit curriculum-only difficulty guidance. It may
  change sampling weights, never labels, provenance, split ownership, or staleness status.

## Phase 1: Contract and corpus inventory

1. Define the manifest and availability states.
2. Add deterministic window extraction with arbitrary origins and boundary metadata.
3. Add signal summaries for observation, height reference, alpha, texture IDs, auxiliary channels, and object slots.
4. Add source-group and motif-family split validation.
5. Add corpus validation and deterministic hashes.

## Phase 2: Atlas and descriptors

1. Render the multi-row visual review atlas.
2. Implement multiscale height/alpha/observation descriptors.
3. Implement ordered/cumulative alpha descriptors and paint-addition hypotheses.
4. Implement fractal and transition descriptors without declaring historical brush identities.
5. Implement tileset profiles and auxiliary-channel correlation reports.
6. Implement paint/relief relationship classification.
7. Join atomic brush, paste-block, and macro-prefab-context records by spatial and provenance links.
8. Implement optional normalized object-slot descriptors.

## Phase 3: Retrieval proof

1. Generate transformed synthetic query/reference pairs.
2. Implement deterministic nearest-neighbor and correlation baselines.
3. Add cross-tile matching and transform estimation.
4. Calibrate recurring/unconfirmed/rejected status.
5. Produce separate atomic, block, and macro retrieval metrics and visual match sheets.

## Phase 4: Spec 139 guidance ablation

1. Convert validated matches and paint-order evidence into confidence-bearing scaffolds.
2. Add parity, motif-guided, tileset-guided, and combined input adapters.
3. Evaluate within-family learnability and held-out-family transfer separately.
4. Add the small real 0.x/1.x validation slice only after synthetic gates pass.
5. Record whether guidance improves reconstruction and seam metrics.

## Phase 4A: Curriculum difficulty guidance

1. Freeze a versioned synthetic reference checkpoint and scoring configuration.
2. Score candidate controls with per-signal, seam/boundary, confidence, and coverage metrics.
3. Assign reproducible `easy`, `learnable_hard`, or `pathological` bands and write sampling weights.
4. Verify that difficulty guidance cannot alter labels, provenance, split ownership, or staleness state.

## Phase 5: Deferred object and iterative refinement lane

1. Add object-slot evidence as an optional auxiliary head.
2. Add render-and-compare refinement only after static guidance is validated.
3. Keep exact object identity and broad later-client coverage behind separate evidence gates.

## Proof gates

- **G0**: all required synthetic regimes and cross-tile cases appear in the atlas.
- **G1**: deterministic transformed retrieval reaches the specified benchmark without source leakage.
- **G2**: real recurrence is either proven with independent source groups or explicitly rejected as unproven.
- **G3**: motif-guided and tileset-guided Spec 139 ablations report separate per-signal and seam metrics.
- **G4**: only after G3 may a user-owned GPU run or broader real-data transfer be recommended.

## Risks

- Similar fractal statistics may create false paste matches; spatial alignment and cross-channel evidence are required.
- Repainted alpha masks may break an otherwise real geometry relationship; the system must report broken relationships instead of forcing a match.
- Auxiliary tileset channels may be client-specific; profile capability is part of provenance.
- A guidance scaffold can become a hidden prior if confidence and absence are not passed through. The fail-closed contract prevents that.
