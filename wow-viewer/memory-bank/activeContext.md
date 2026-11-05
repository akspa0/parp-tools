# Active Context — wow-viewer

**Updated:** 2026-11-05. The current registry is [`specs/STATUS.md`](../specs/STATUS.md); the active task contract is always the owning spec's `spec.md` → `plan.md` → `tasks.md`.

## Current handoff

- **Spec 294 — Spatial donor transforms and manual PM4 curation:** implementation Phases 1–4 complete. Receipts: [`Phase 1`](../specs/294-spatial-curation/evidence/phase1-core-transform-2026-11-05.md), [`Phase 2`](../specs/294-spatial-curation/evidence/phase2-group-transform-ui-2026-11-05.md), [`Phases 3–4`](../specs/294-spatial-curation/evidence/phase3-4-manual-curation-persistence-2026-11-05.md). Next is operator real-client verification of group preview/undo, manual catalog placement/pairing, curation save/reopen, and unchanged source-map hashes.
- **Spec 293 — Restoration Workbench, Phase 1:** zoom/pan, tile-range/hover readout, drag selection, and selection Z offset/scale are implemented. Build receipt is in [`tasks.md`](../specs/293-layer-driven-adt-restoration/tasks.md). Operator proof remains: exercise picker, place tiles, adjust Z, save/reload. The full workbench is still draft and requires operator review.
- **Spec 289 — PM4 duplicate-on-match:** occupancy now uses world-space bounds, checks both tile-key orientations, and prevents duplicate same-model spawns by default. Receipt: [`2026-11-05-duplicate-on-match-occupancy.md`](../specs/289-pm4-3d-footprint-and-direct-matching/evidence/2026-11-05-duplicate-on-match-occupancy.md). Operator proof remains on an already-paired real-client tile.
- **Spec 292 — UI frame stability:** NewFrame/EndFrame are paired in `OnRender`; temporary `UiOscilloscope` diagnostics are present. Operator must run the viewer, exercise menus/panels, and confirm stable input/rendering before the diagnostic is removed.
- **Spec 292 — corrected bottom toolbar:** Sky/Fog are excluded; Water/Ocean/Magma/Slime, Doodads/WMOs/Terrain/PM4/Grid/Mute, resident-terrain L0–Ln, one A1–An selector per non-base layer, and both opacity sliders are in source. The status-bar Mute duplicate is gone and audio starts muted. Alpha/liquid state is saved/restored. UI registry is clean, footer tests are 30/30, and viewer compile has 0 errors. Receipt: [`bottom-toolbar-reversion-2026-11-05.md`](../specs/292-ui-bottom-bar-toggles-audio-sky/evidence/bottom-toolbar-reversion-2026-11-05.md). Operator visual/restart proof remains owed.
- **Spec 293 — donor tile picker:** the 64×64 picker surface now draws client minimap tiles instead of the WDL heightmap preview. T005/T006 add per-donor-tile 16×16 chunk masks, whole-map offset placement, and a `Viewport chunk edit` mode that toggles donor chunks by clicking terrain in the 3D view. Receipt is in [`tasks.md`](../specs/293-layer-driven-adt-restoration/tasks.md). Operator visual proof remains owed.
- **Spec 295 — visual PM4 shape library:** spec and quality checklist authored 2026-11-05 for a visual shape catalog, automatic hull-to-object proposals with a calibrated -X centroid correction, operator accept/reject/replace review, and batch analysis/audit. Planning not started; see [`spec.md`](../specs/295-pm4-visual-shape-library/spec.md).
- **Epic 249 / 251 P1 handoff:** R-10 performance capture and U-01 moved-surface smoke remain operator-owned. Spec 255 W0–W7 smoke is also owed. See the status router for current gating and ordering.

## Next bounded action

Operator's next bounded check: run the viewer against the configured test client and verify the Spec 292 bottom toolbar, A1–A7 channel switching/toggle-off behavior, and alpha state across restart; then verify Spec 294 group transforms/manual curation/save-reopen and unchanged source-map hashes, Spec 293 picker/save-load, Spec 289 no-duplicate behavior, and the Spec 292 menu/panel stability probe. Record build, client fingerprint, observed result, and artifacts in each owning spec. Do not infer runtime gates from build/tests.

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
