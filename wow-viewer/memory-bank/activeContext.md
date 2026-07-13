# Active Context — wow-viewer

Last updated: 2026-07-13

## Current path — Spec 102 M0, simple route (Route A)

- Decision: train M0 (RGB minimap → one object mask) directly on the existing
  `object_precise_mask_257` numeric store. Accept that precise masks include some
  under-terrain object pixels (extra loss noise) and move on — enough samples to
  converge. No reharvest, no strict fragment-trace target, no coverage gate.
- Trainer: `data-harvester/scripts/train_spec102_m0_simple.py` (`M0ObjectMask`,
  3,043,041 params, complete-map holdout). **The user runs training** (AGENTS
  RULE 0); the agent only hands off the CLI command.
- Data: `output/datasets/spec102/numeric_3_3_5_full_raw_v1.zarr` (5,134 tiles,
  `object_precise_mask_257`), copied from the May-2026 V18 raw store.
- 3-epoch smoke checkpoint: `output/spec102_m0_precise_simple_v1/` (val IoU ~0.15,
  still climbing). The full run is the user's to launch.

## On hold — strict fragment-trace target

- The strict `object_geometry_visible_mask` + `strict-geometry-terrain-liquid-
  fragment-trace-v3` sidecar pipeline (C# + Python) is committed (`4f44c7f7`) and
  green (42/42 spec102 tests) but parked as over-engineering. It fixes under-terrain
  masking properly (clips below-terrain fragments) and needs a reharvest. Detail:
  `memory-bank/archive/2026-07-13-spec102-strict-target-detail.md`. Revisit only if
  the simple masks prove too noisy.

## Boundaries

- New work in `wow-viewer/`; `gillijimproject_refactor` is read-only reference.
- Staged clients only: `output/tmp/wowarchive-clients/`. Never `H:\CLIENTS`.
- Spec 080 owns the UI release lane. Spec 089 / V23 / V24 / V25 are historical/paused.
