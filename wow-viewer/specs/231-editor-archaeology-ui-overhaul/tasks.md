# Tasks — Spec 231 Editor & Archaeology Workspace UI Overhaul

Phases match [plan.md](plan.md). A checkbox may be set `[x]` only with a receipt in this
directory (`evidence/`) per AGENTS.md §9.2. Implementation session: fresh chat (operator
directive, 2026-09-07).

## Phase 0 — Inventory v3 baseline (Spec 227 gate folded in)

- [x] T001 Capture Editor + Archaeology tab screenshots as they exist today into
      `evidence/inventory-v3-baseline.md` (operator-assisted per Spec 227 T003).
      Done 2026-09-07 via the viewer's built-in capture automation (operator directive);
      10-page matrix + run fingerprint in
      [evidence/inventory-v3-baseline.md](evidence/inventory-v3-baseline.md)
- [x] T002 Record the 4-page IA scope freeze: operator acknowledges Placement & Objects /
      Terrain Tools / Data I/O / Converters and the Archaeology analysis-only page list.
      Receipt: [evidence/t002-scope-freeze-ack.md](evidence/t002-scope-freeze-ack.md)
- [x] T003 Write the dated supersession note in `227-ui-reaudit/tasks.md` for the Editor-tab
      item ("sane Editor tabs" → Spec 231). Note written 2026-09-07 in that file's header

## Phase 1 — Page skeletons + migration map

- [x] T010 Create `Workbench/Pages/` service classes: `EditorDataIoPage`,
      `EditorPlacementObjectsPage`, `EditorTerrainToolsPage`, `EditorConvertersPage` with a
      narrow `ViewerAppContext` (constructor-injected; no god-class reach-back)
- [x] T011 `WorkbenchNavigator.GetEditorWorkbenchLabels()` → the 4 new labels; update
      `DrawEditorWorkbenchSubTabContent` to delegate; update
      `NormalizeWorkbenchStateAfterLoad` for old-index remap (0→Placement, 1→Converters,
      2→Placement, 3→Data I/O, 4→Terrain Tools, 5→Placement) and verify with a pre-change
      settings file. Remap implemented at the settings-load boundary gated on
      `WorkbenchNavigationVersion` (see receipt note 1); pre-change settings runtime check
      remains operator-owned
- [x] T012 Gate: `dotnet build wow-viewer/WowViewer.slnx -c Debug` 0 errors; old content
      reachable from new pages; zero behavior change receipt.
      Receipt: [evidence/t010-t012-phase1-receipt.md](evidence/t010-t012-phase1-receipt.md)

## Phase 2 — Data I/O consolidation (D1, D6)

- [x] T020 Move all export/import content into `EditorDataIoPage` (PM4 JSON/OBJ, correlation
      JSON, GLB scene/collision/map tiles, terrain layer export/import, synthesized minimap,
      VLM/ML harvest launchers) — groups collapsed by default. PM4 export command set moved to a
      collapsed "PM4 Exports" section; VLM/ML harvest launchers do not exist (retired workflow —
      see receipt note 1)
- [x] T021 Delete the 5 duplicate export-button sites (ViewerApp_Pm4Utilities ~378/448/1201/
      1782/3900) and the Archaeology-hosted imports page; replace with `Open Data I/O` links
      carrying current selection (selection lives on WorldScene, preserved across navigation).
      Archaeology-tab de-hosting of the imports route is T042 (receipt note 2)
- [x] T022 Background-execution audit on the new page: every potentially >1s command uses the
      `ExportPm4ObjectsObjSet` pattern (minimum: correlation JSON build, object-match report
      build, PM4 JSON dump). Both JSON dumps backgrounded; object-match report refresh is an
      uncalled dead path (receipt D6 table)
- [x] T023 Gate: SC-1 grep receipt for D1 (each label = 1 draw site); build/test clean; export
      smoke on a small map with the UI responsive.
      Receipt: [evidence/t020-t023-phase2-receipt.md](evidence/t020-t023-phase2-receipt.md).
      Interactive export execution (path picker) remains operator-owned

## Phase 3 — Placement & Objects (D2, D4, D5a)

- [ ] T030 Diff the two PM4/WMO correlation panels (Sidebars ~801–901 vs Pm4Utilities
      ~1863–2007); fold unique behavior into the survivor
- [ ] T031 Consolidate transform/match/reconcile/collection/graph tools + surviving correlation
      panel into `EditorPlacementObjectsPage`; delete both original correlation panels and the
      5 duplicate button clusters' transform siblings
- [ ] T032 Doodad-set: object page owns the full combo; toolbar keeps the hovered-WMO quick
      combo (FR-6); delete the other two combos
- [ ] T033 Gate: SC-1 receipts D2/D4; SC-5 proof that the duplicated correlation code block is
      gone; build/test clean

## Phase 4 — Terrain Tools, Converters, Archaeology de-hosting (D3, D5b)

- [ ] T040 Terrain Tools page: template brushes, single "Clipboard + Save" copy (delete both
      originals), stratigraphy/weak-signal editing entries, terrain lab
- [ ] T041 Converters page: map/WMO/M2-MDX converters, ADT utilities, round-trip validation,
      Spec 221 harness
- [ ] T042 Remove Archaeology editor hosting: delete `DrawArchaeologyEditorTasksSubTab` /
      `DrawArchaeologyEditorImportsSubTab` routing; Archaeology = analysis pages only
- [ ] T043 Gate: SC-1 receipts D3/D5; build/test clean; `ViewerApp_Sidebars.cs` and
      `ViewerApp_Pm4Utilities.cs` line counts recorded (must shrink)

## Phase 5 — Navigation test, inventory, closure

- [ ] T050 Operator navigation smoke: US1 (PM4 OBJ export), US2 (placement edit + reconcile),
      US3 (Archaeology analysis-only) each ≤2 navigations; SC-3 1600×900 no-scroll check
- [ ] T051 Inventory v3 rows for every moved/removed surface (Spec 223 FR-9); STATUS.md +
      epic updates
- [ ] T052 Final `dotnet build` + `dotnet test` receipts; spec status → Implemented with user
      gates (→ Complete on operator acceptance)
