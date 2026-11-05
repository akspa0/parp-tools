# Spec 294 — Tasks (Spatial Donor Transforms and Manual PM4 Curation)

**Status:** Operator-approved 2026-11-05; implementation phases 1–4 are complete; operator runtime/visual proof remains owed.
**Spec:** [`spec.md`](spec.md) · **Plan:** [`plan.md`](plan.md)
**Rules:** every `[x]` requires a §9.2 receipt (files changed, exact commands + exits, criterion→real-output evidence). Build/tests never prove runtime, visual, or real-client behavior. No new `WorldScene`/`ViewerApp` members; no file past ~2,000 lines (AGENTS.md §10).

## Phase 1 — Donor group transform core (FR-001, FR-002, FR-003, FR-007)

- [x] T001 — Add UI-free `DonorGroupTransformService` in `WowViewer.Core.Maps`: group centroid pivot, angle normalization (negative, >360, fractional), translation, relative-arrangement preservation over [`PhaseTilePlacement`](wow-viewer/src/core/WowViewer.Core/Maps/PhaseComposition.cs:772) lists. Receipt: [`phase1-core-transform-2026-11-05.md`](evidence/phase1-core-transform-2026-11-05.md).
- [x] T002 — Add bounds/collision validation: out-of-map members, duplicate target claims, gap reporting; return structured issues, never throw for user input. Receipt: [`phase1-core-transform-2026-11-05.md`](evidence/phase1-core-transform-2026-11-05.md).
- [x] T003 — Add clone-based preview: apply candidate transform to a [`PhaseLayerSettings.Clone`](wow-viewer/src/core/WowViewer.Core/Maps/PhaseComposition.cs:198) copy; prove the source layer is byte-identical after preview and cancel. Receipt: [`phase1-core-transform-2026-11-05.md`](evidence/phase1-core-transform-2026-11-05.md).
- [x] T004 — Unit tests: multi-tile group moves as one unit; 17° and fractional angles preserve relative arrangement; single-tile group; off-group pivot defaults to centroid. Receipt: [`phase1-core-transform-2026-11-05.md`](evidence/phase1-core-transform-2026-11-05.md).
- [x] T005 — Unit tests: out-of-bounds and colliding transforms are rejected with issues; zero angle is a no-op. Receipt: [`phase1-core-transform-2026-11-05.md`](evidence/phase1-core-transform-2026-11-05.md).
- [x] T006 — Phase 1 receipt: focused `dotnet test` run + criterion→evidence table for FR-001/FR-002/FR-003/FR-007. Receipt: [`phase1-core-transform-2026-11-05.md`](evidence/phase1-core-transform-2026-11-05.md).

## Phase 2 — Group transform UI with preview and undo/redo (FR-004, FR-005, FR-006)

- [x] T007 — Treat the donor layer's multi-tile placements as one group; show member count and centroid pivot. Receipt: [`phase2-group-transform-ui-2026-11-05.md`](evidence/phase2-group-transform-ui-2026-11-05.md).
- [x] T008 — Add arbitrary-angle input beside the existing quarter-turn workflow; preserve the 4-choice combo and report the accepted value. Receipt: [`phase2-group-transform-ui-2026-11-05.md`](evidence/phase2-group-transform-ui-2026-11-05.md).
- [x] T009 — Show cloned before/after tile-coordinate preview with origin and angle; Accept commits, Cancel discards. Receipt: [`phase2-group-transform-ui-2026-11-05.md`](evidence/phase2-group-transform-ui-2026-11-05.md).
- [x] T010 — Add bounded undo/redo snapshots for accepted group transforms. Runtime interaction proof remains owed. Receipt: [`phase2-group-transform-ui-2026-11-05.md`](evidence/phase2-group-transform-ui-2026-11-05.md).
- [x] T011 — Register new controls and regenerate UI reference; `Verify-UiReference.ps1` passed with 0 errors/warnings and viewer build passed. Receipt: [`phase2-group-transform-ui-2026-11-05.md`](evidence/phase2-group-transform-ui-2026-11-05.md).
- [x] T012 — Phase 2 receipt created; operator runtime gate (preview/accept/cancel/undo/redo on a real map) recorded as owed in [`phase2-group-transform-ui-2026-11-05.md`](evidence/phase2-group-transform-ui-2026-11-05.md).

## Phase 3 — Manual object placement and hull pairing (FR-008, FR-009, FR-010, FR-012)

- [x] T013 — Add owned `ManualCurationWorkbenchService` (viewer layer) for hand position/yaw edits without source-file writes. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T014 — Add explicit object-instance + PM4-hull association with exact-duplicate guard. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T015 — Place a selected client Asset Catalog entry as a curation-only placement on the selected hull tile; reuse the existing catalog and viewer asset loader. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T016 — Add SharedUiWidgets curation section with pair members/state and explicit remove/replace; matching/spawn never silently creates or replaces a manual association. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T017 — Add focused tests for duplicate rejection, shared-hull distinct objects, remove/replace, placement update retaining association, and project round-trip. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T018 — Phase 3 receipt created; real-tile hand placement/pairing/inspection runtime gate recorded as owed. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).

## Phase 4 — Persistence and round-trip (FR-011, SC-002, SC-004)

- [x] T019 — Persist exact group transform metadata in [`PhaseLayerProjectFile`](wow-viewer/src/core/WowViewer.Core/Maps/PhaseLayerProjectFile.cs:11); old project files load with absent metadata. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T020 — Persist manual placements and hull associations to a separate curation project file; source maps are not written. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T021 — Add round-trip tests for group angle/placement, hand placements, and pairs; a source-map sentinel remains byte-identical after curation save. Real client hashes remain operator-owned. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).
- [x] T022 — Phase 4 receipt with criterion→evidence for FR-011/SC-002/SC-004. Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](evidence/phase3-4-manual-curation-persistence-2026-11-05.md).

## Phase 5 — Final verification and handoff (SC-001…SC-004)

- [x] T023 — Full solution build + focused test suites; record exact commands and exits in the phase receipts.
- [x] T024 — Update [`specs/STATUS.md`](wow-viewer/specs/STATUS.md) Spec 294 row and [`memory-bank/activeContext.md`](wow-viewer/memory-bank/activeContext.md) handoff in the same pass.
- [ ] T025 — Operator proof checklist: non-quarter-turn group placement with preview/accept/cancel/undo/redo; hand placement + pairing + save/reopen; source files unchanged. Record build, client fingerprint, observed results, and artifacts per owning spec.
