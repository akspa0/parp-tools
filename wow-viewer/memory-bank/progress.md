# Progress — wow-viewer

## 2026-11-05 — Spec 292 bottom-toolbar reversion

- Restored the nine scene toggles (Sky/Fog/Water/Doodads/WMOs/Terrain/PM4/Grid/Mute), added persisted M2/MDX and WMO opacity sliders, retained L0–L7/Alpha quick checks, and removed the Texture Masks bottom-bar button.
- Persisted the nine quick-toggle values in viewer settings and apply them when a scene/map activates. UI registry/reference clean; 18/18 footer tests pass; full solution build succeeds with 0 errors (206 warnings).
- Receipt: [`bottom-toolbar-reversion-2026-11-05.md`](../specs/292-ui-bottom-bar-toggles-audio-sky/evidence/bottom-toolbar-reversion-2026-11-05.md). Operator real-client visual/restart verification remains owed.

## 2026-11-05 — Spec 294 Phases 3–4 manual curation and persistence

- Added curation-only placement snapshots and explicit PM4 hull relations, manual position/yaw controls, client Asset Catalog placement, duplicate-safe pair creation, explicit replace/remove actions, and save/reload to a separate per-map curation JSON.
- Persisted authored donor-group transforms through `PhaseLayerProjectFile`; round-trip tests also verify manual placement/association JSON and that a source-map sentinel remains unchanged.
- Verification: focused Spec 294 tests 22 passed; `Verify-UiReference.ps1` 0 errors/0 warnings; full `WowViewer.slnx` build exit 0 (266 warnings, 0 errors). Runtime/visual/save-reopen proof remains operator-owned.
- Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](../specs/294-spatial-curation/evidence/phase3-4-manual-curation-persistence-2026-11-05.md).

## 2026-11-05 — Spec 294 Phase 2 group transform UI

- Added an owned WorkbenchPanels service for arbitrary-angle/translation entry, cloned coordinate preview, validation display, explicit accept/cancel, stale-preview protection, and bounded per-layer undo/redo. Existing quarter-turn rotation remains available.
- Registered the group-transform section/actions and regenerated the UI reference. Focused tests: 13 passed; `Verify-UiReference.ps1`: 0 errors/0 warnings; CrossPlatform viewer build: 0 errors (2 NU1902 package warnings).
- Receipt: [`phase2-group-transform-ui-2026-11-05.md`](../specs/294-spatial-curation/evidence/phase2-group-transform-ui-2026-11-05.md). Real-map preview/accept/cancel/undo/redo proof remains operator-owned. Next: Phase 3 manual object placement and explicit PM4 hull pairing.

## 2026-11-05 — Spec 294 Phase 1 core transform

- Implemented the UI-free donor-group transform preview, angle normalization, centroid/custom-pivot fallback, integer-grid snapping, and structured validation for bounds, collisions, invalid placements, and gaps.
- Added 13 focused unit cases; `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DonorGroupTransformServiceTests" --nologo` exited 0 (13 passed, 0 failed, 0 skipped).
- Receipt with changed-file list and requirement→evidence table: [`phase1-core-transform-2026-11-05.md`](../specs/294-spatial-curation/evidence/phase1-core-transform-2026-11-05.md). Next: Spec 294 Phase 2 UI; operator runtime proof remains owed.

## 2026-11-05 — Monthly spec and memory-bank consolidation

- Audited every registered active epic/spec task ledger against its receipt references and current tree. Reset 37 unsupported checks across Specs 259, 265, 266, 277, 280, and 288; implementation history is retained and reasons are recorded in the cleanup report.
- Archived the pre-consolidation active context and October progress ledger; current handoff now points to the status router, owning specs, and operator-owned proof only.
- Reconciled the cleanup ledger and missing Spec 293 registry entry; indexed dated memory-bank archives.
- Verification: cleanup inventory/path/receipt audit commands and results are recorded in [`cleanup-2026-11-05.md`](../specs/251-epic-viewer-ux-and-code-health/evidence/cleanup-2026-11-05.md).
