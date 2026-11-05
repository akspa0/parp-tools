# Active Context — wow-viewer

**Updated:** 2026-11-05. The current registry is [`specs/STATUS.md`](../specs/STATUS.md); the active task contract is always the owning spec's `spec.md` → `plan.md` → `tasks.md`.

## Current handoff

- **Spec 293 — Restoration Workbench, Phase 1:** zoom/pan, tile-range/hover readout, drag selection, and selection Z offset/scale are implemented. Build receipt is in [`tasks.md`](../specs/293-layer-driven-adt-restoration/tasks.md). Operator proof remains: exercise picker, place tiles, adjust Z, save/reload. The full workbench is still draft and requires operator review.
- **Spec 289 — PM4 duplicate-on-match:** occupancy now uses world-space bounds, checks both tile-key orientations, and prevents duplicate same-model spawns by default. Receipt: [`2026-11-05-duplicate-on-match-occupancy.md`](../specs/289-pm4-3d-footprint-and-direct-matching/evidence/2026-11-05-duplicate-on-match-occupancy.md). Operator proof remains on an already-paired real-client tile.
- **Spec 292 — UI frame stability:** NewFrame/EndFrame are paired in `OnRender`; temporary `UiOscilloscope` diagnostics are present. Operator must run the viewer, exercise menus/panels, and confirm stable input/rendering before the diagnostic is removed.
- **Epic 249 / 251 P1 handoff:** R-10 performance capture and U-01 moved-surface smoke remain operator-owned. Spec 255 W0–W7 smoke is also owed. See the status router for current gating and ordering.

## Next bounded action

Operator runs the current viewer against the configured test client: first verify Spec 293 picker/save-load and Spec 289 no-duplicate behavior; in the same session exercise the Spec 292 menu/panel stability probe. Record build, client fingerprint, observed result, and artifacts in each owning spec. Do not infer these runtime gates from build/tests.

## Plan and history routing

- Backlog decisions live in [`specs/TRIAGE.md`](../specs/TRIAGE.md); untriaged epic work is not scheduled.
- Keep Spec Kit plans/tasks in their owning spec directories. Spec 285 is the data contract consumed by Spec 284; they remain separately scoped. Do not combine active plans or change acceptance scope without operator-approved wording.
- Durable research is linked from the owning active spec. The progress ledger is current-month history; archive entries and `specs/archived/` are on-demand only.

## Constraints

- Preserve working format readers and `AlphaWdtWriter`; keep core libraries UI-free and Alpha/Standard terrain adapters separate.
- No new members in `WorldScene` or `ViewerApp`; no source file above ~2,000 lines.
- Operator owns runtime/visual/FPS/audio/client proof, heavy harvests, GPU training, and long captures.
- Client roots are runtime configuration. Keep generated output inside repository/project-managed roots; preserve unrelated worktree changes and stage named files only.
- `[x]` requires the full §9.2 receipt: paths changed, exact commands + exits, and criterion-to-real-output evidence.
