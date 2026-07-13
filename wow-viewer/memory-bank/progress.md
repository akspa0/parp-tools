# Progress — wow-viewer

Last updated: 2026-07-13

## 2026-07-13 — Spec 102 M0 simple route chosen; strict target committed but parked

- Chose Route A: train M0 on the existing `object_precise_mask` 5,134-tile store,
  accept under-terrain masking noise, and move on. Added a simple trainer
  (`scripts/train_spec102_m0_simple.py`); the user runs training (RULE 0).
- Committed (`4f44c7f7`): strict `object_geometry_visible_mask` + v3 fragment-trace
  reharvest pipeline (C# + Python), spec102 tests green (42/42), fixed a coverage
  `NameError` and a trace-validator field-name bug. Parked as over-engineering.
- Added AGENTS RULE 0: user runs training/heavy/billed runs; respectful, direct
  communication regardless of tone.

## Durable boundaries

- Older strict-target detail:
  `memory-bank/archive/2026-07-13-spec102-strict-target-detail.md`.
- `ef99e715` = trainer control-flow reference only. H0/H1/V23/V24/V25 = historical.
- Viewer/UI (Spec 080), PM4, and legacy data lanes unchanged.
