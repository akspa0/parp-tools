# wow-viewer Training Prompt Set

Ordered prompt set for getting the current `v7.5.1` and `v7.6` model lines back onto a trustworthy, structured `datasets/` workflow instead of drifting across legacy export roots, stale caches, and one-off training assumptions.

Current assumptions:

- `gillijimproject_refactor/docs/VLM_Training_Guide.md` is the main continuity surface for the training lanes.
- `v7.5.1` is the active grounded terrain-regression line and should be treated as the first retrain lane to trust again.
- `v7.6` is a separate paired reconstruction branch with its own cache, output, and inference semantics.
- `wow-viewer` converter commands already cover important corpus and audit work, but the downstream training scripts are not yet cleanly aligned with that structured export surface.
- dataset truth comes before model metrics; a fast run on a stale or ambiguous corpus does not count as recovery.

Run these in order unless the user explicitly asks for one named later slice:

1. `01-shared-dataset-truth-gates.md`
2. `02-v75-grounded-training-recovery.md`
3. `03-v76-paired-training-alignment.md`

Validation rule:

- default proof is a reproducible corpus audit plus the minimum real build or script proof needed for the slice
- prefer `wow-viewer` converter commands and structured manifests as the source of truth for corpus state
- treat legacy scripts as consumers or wrappers until they are explicitly realigned
- do not present a single training launch as trustworthy recovery unless the corpus contract, run configuration, and expected outputs were all verified
