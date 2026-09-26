# Tasks — Epic 251 Viewer UX, Shell & Code Health

Implementation tasks are generated **after** triage. Nothing is checked without a §9.2 receipt.

## Phase 0 — Triage (operator)

- [ ] T001 Operator marks every U-xx item Want / Drop / Later in [TRIAGE.md](../TRIAGE.md).
- [ ] T002 Record decisions here as a dated amendment; Drop items move to a dropped table.
- [ ] T003 Generate phased tasks for Want items; pick the approach per plan.md.
- [ ] T004 Governance Gate 1: operator confirms AGENTS.md §9 rules as binding (224 Gate 1).

## Operator verification sweep (shipped code — proof only)

- [ ] V001 223 T609/T610 UI acceptance retest, recorded separately from build evidence.
- [ ] V002 231 T050 navigation smoke (Editor 4-page IA, Archaeology de-hosting).
- [ ] V003 v0.5.2.3 runtime checks (slider order, overlay balance, hover occlusion, About credits).
- [ ] V004 Click every new export menu / sidebar button shipped in v0.6.0-alpha2.

## U-01 — God-class decomposition (operator P1, 2026-09-23)

Re-ordered plan approved 2026-09-23. E1 and E3 code landed 2026-09-25, then the whole ViewerApp campaign (operator direction 2026-09-25); smoke tasks T003, T007, T013 are operator-owned. E2 still waits for R-10.

- [x] U01-T001 Operator approves the E1→E4 order and dropping the 227-T004 gate. — Approved 2026-09-23 ("Approve re-order"). Receipt: operator answer in session.
- [x] U01-T002 E1 PM4 overlay extraction: verbatim move, delegation, build + tests, line-count receipt. — 2026-09-25: `WorldScene.cs` 17,175 → 8,326; build 0 errors, test failure set unchanged (26 pre-existing, environmental). Runtime not claimed. Receipt: [evidence/u01-e1-pm4-extraction-2026-09-25.md](evidence/u01-e1-pm4-extraction-2026-09-25.md).
- [ ] U01-T003 E1 operator smoke (PM4 overlay, colours, selection, OBJ export).
- [ ] U01-T004 E2 `Render()` pass split (after R-10 lands).
- [ ] U01-T005 E2 operator smoke on a legacy and a modern map.
- [x] U01-T006 E3 `ViewerApp` menu bar + converter dialogs. — 2026-09-25: `MainMenuBarService`, `ConverterDialogsService`, `DatasetExportDialogsService`, `WorldObjectsPanelService`; runtime not claimed. Receipt: [evidence/u01-viewerapp-extraction-2026-09-25.md](evidence/u01-viewerapp-extraction-2026-09-25.md).
- [ ] U01-T007 E3 operator smoke.
- [x] U01-T008 E4 selection/hover service (228 T004–T011). — 2026-09-26: 228 T004–T010 done (pure `WorldSceneSelectionService` in Core.Runtime, 20 tests incl. randomized equivalence, viewer adapter, `WorldScene` seam); 228 T011 smoke is U01-T009. Runtime not claimed. Receipt: [evidence/u01-e4-selection-2026-09-26.md](evidence/u01-e4-selection-2026-09-26.md).
- [ ] U01-T009 E4 operator smoke; update AGENTS.md §10 measured line counts.

### U-01 ViewerApp extraction campaign (operator direction 2026-09-25, spec.md amendment)

- [x] U01-T010 Extraction tooling + `IViewerAppHost` + composition root; converter dialogs
  (`ConverterDialogsService`) and dataset-export dialogs (`DatasetExportDialogsService`) out of `ViewerApp.cs`. — 2026-09-25. Receipt: [evidence/u01-viewerapp-extraction-2026-09-25.md](evidence/u01-viewerapp-extraction-2026-09-25.md).
- [x] U01-T011 Remaining `ViewerApp.cs` clusters (terrain tile I/O, chunk clipboard, weak-signal restore,
  world-objects panel, placement editing, shell layout, data sources, model loading, settings, menu bar). — 2026-09-25: `ViewerApp.cs` 14,777 → 1,725. Receipt: [evidence/u01-viewerapp-extraction-2026-09-25.md](evidence/u01-viewerapp-extraction-2026-09-25.md).
- [x] U01-T012 Over-budget partial files (`ViewerApp_Sidebars`, `ViewerApp_Pm4Utilities`,
  `ViewerApp_CaptureAutomation`) and the remaining feature partials. — 2026-09-25: whole `ViewerApp` class 42,973 lines / 29 files → 2,512 / 6; 51 services + 3 static helpers, every file < 2,000; build 0 errors, test failure set unchanged (26 environmental). Runtime not claimed. Receipt: [evidence/u01-viewerapp-extraction-2026-09-25.md](evidence/u01-viewerapp-extraction-2026-09-25.md).
- [ ] U01-T013 Operator smoke of every moved surface (menus, dialogs, panels, exports, capture).
