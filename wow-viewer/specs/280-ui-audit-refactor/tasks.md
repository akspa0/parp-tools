# Tasks: UI Audit and Pre-Alpha-Style Refactor

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md) · **Audit**: [audit.md](audit.md)

**Branch**: `280-ui-rework`

**Rule**: a task is checked `[x]` only with a receipt (AGENTS §9.2): files changed, exact commands
and exit status, and a criterion→evidence table. Build/test output alone never proves runtime,
visual, or file behavior.

## P0 — Baseline and guardrails

- [ ] T001 Record the baseline build and test result on `280-ui-rework`.
  - Command: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
  - Command: `dotnet test wow-viewer/WowViewer.slnx -c Debug`
  - Receipt: `evidence/p0-baseline.md`
- [ ] T002 Add a UI-string inventory script that lists every visible label and menu path.
  - Output: `evidence/ui-strings-baseline.txt`
  - Purpose: make every later deletion measurable.
- [ ] T003 Confirm the audit's duplicate table is complete against the running viewer; add any
  missing row to [audit.md](audit.md).
  - Receipt: `evidence/p0-audit-closure.md`

**Gate P0**: build and tests pass; the string inventory exists; the audit has no `unknown` rows.

## P1 — File operations repair

- [ ] T010 Replace the "Open File..." text popup with `ImGuiPathPicker`.
  - File: `Workbench/Services/MainMenu/MainMenuBarService.cs`
  - Acceptance: the action opens the shared picker; no text field is the only path input.
- [ ] T011 Audit every Export action: run it, confirm a file is written, add the status-line path
  report.
  - Files: `MainMenuBarService.cs`, `Pm4WorkbenchService.Alignment.cs`,
    `TerrainControlsPanelService.Sidebars.cs`, `ArchaeologyPanelService.PhaseLayers.cs`
  - Acceptance: each export writes a real file and reports the path.
- [ ] T012 Audit every Import action: run it, confirm it applies, or delete it.
  - Files: `MainMenuBarService.cs`, `EditorPanelsService.cs`
  - Acceptance: each import reads and applies a real file, or is absent.
- [ ] T013 Add explicit failure reporting to every export/import action.
  - Acceptance: a failed action reports the failure; no silent no-op remains.
- [ ] T014 Record the operator-reported functional state of each File-menu action in
  [audit.md](audit.md) §2.
  - Values: `works`, `unreliable`, `dead`.

**Gate P1**: the operator opens a file through the picker, exports one artifact of each kind, and
imports one artifact of each kind; every action produces or consumes a real file. Receipt:
`evidence/p1-file-operations.md`.

## P2 — Bottom bar and Options sub-menu

- [ ] T020 Add the destination buttons to the bottom bar: Viewer, Editor, Archaeology.
  - File: `Workbench/Services/ViewerChrome/ViewerChromeService.cs`
- [ ] T021 Add the Options button and the sub-menu anchored next to the chat.
  - Files: `ViewerChromeService.cs`, `FakeChatWindowService.cs`
- [ ] T022 Move the render toggles into the Options sub-menu.
- [ ] T023 Move the overlay, layer, and transparency popups into the Options sub-menu; delete the
  three bottom-bar popups.
- [ ] T024 Keep the FPS/stats readout in the bottom bar in every destination.

**Gate P2**: the operator reaches Viewer, Editor, Archaeology, and Options from the bottom bar in
at most two clicks; the FPS/stats readout is visible; the Options sub-menu does not cover the chat
or the status bars. Receipt: `evidence/p2-bottom-bar.md`.

## P3 — Top menu reduction

- [ ] T030 Reduce the top menu to File / View / Help.
  - File: `Workbench/Services/MainMenu/MainMenuBarService.cs`
- [ ] T031 Delete Tools > Panels; move each surviving action to its owning destination.
- [ ] T032 Delete the Tools > Export and Tools > Import trees; move each action to the owning
  destination.
- [ ] T033 Delete the legacy shell toggles ("Tab System (069)", "Dockable Shell Panels").
- [ ] T034 Delete the View menu entries that duplicate a destination surface.

**Gate P3**: the top menu has three menus; no export/import action lives in the top menu; the
string inventory shows the deleted paths are gone. Receipt: `evidence/p3-top-menu.md`.

## P4 — Destination separation

- [ ] T040 Make the left navigator destination-aware.
  - File: `Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs`
- [ ] T041 Make the right inspector destination-aware.
  - File: `Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs`
- [ ] T042 Delete cross-destination controls from both sidebars.
- [ ] T043 Delete the compatibility enums (`ModelBottomTab`, `WorldBottomTab`, `ToolsBottomTab`,
  `UtilitiesBottomTab`, `TerrainBottomTab`, `Pm4BottomTab`, `ArcheologyBottomTab`).
  - File: `Workbench/WorkbenchNavigator.cs`
- [ ] T044 Delete the legacy tab redirects and `Canonicalize`.
- [ ] T045 Reset or migrate persisted layouts that point at deleted pages.

**Gate P4**: switching destination changes the visible sidebar content; no editor-only control
appears in Viewer or Archaeology; the build has no references to the deleted enums. Receipt:
`evidence/p4-destinations.md`.

## P5 — Archaeology dedupe

- [ ] T050 Delete the legacy 5-tab strip (`DrawArcheologySubTabContent`).
- [ ] T051 Delete the floating UniqueId Archaeology window (`DrawUniqueIdArchaeologyWindow`).
- [ ] T052 Delete the Quick-section archaeology launchers (`DrawArchaeologyQuickSection`).
- [ ] T053 Split the Cartography "Layers" page into named sections: layer stack, DBC phase
  discovery, add phase map, donor tile grid picker.
- [ ] T054 Remove the host bridges for the deleted members.
  - File: `ArchaeologyPanelService.Host.cs`

**Gate P5**: the 6-page destination strip is the only archaeology navigation; the string inventory
shows the deleted entry points are gone. Receipt: `evidence/p5-archaeology.md`.

## P6 — Editor section rename

- [ ] T060 Rename "Tasks & Workspace" to its real owner.
- [ ] T061 Rename "3D Object Library" to its real owner.
- [ ] T062 Rename "Imports & Exports" to its real owner.
- [ ] T063 Delete the one-button "PM4 Placement Tools" section.
- [ ] T064 Remove the second PM4 export home from Data I/O.
- [ ] T065 Delete `MigrateLegacyEditorPageIndex` after the layout reset.
- [ ] T066 Move PM4 matching offline: remove the Correlation, Reconcile, and Alignment tabs, the
  match-curation section, and `Pm4ObjectMatchStore.cs`; the offline tool is
  [Spec 281](../281-museum-adt-patcher/spec.md).

**Gate P6**: the Editor pages name their real content; PM4 has one export home; the viewer has no
matching/alignment/reconciliation tabs; the build has no reference to the deleted migration.
Receipt: `evidence/p6-editor.md`.

## P7 — Floating-window dedupe

- [ ] T070 Delete the 12 Tools > Panels floating windows that duplicate destination pages.
- [ ] T071 Keep only Log and Perf as global floating windows.
- [ ] T072 Delete the PM4 Actions popup; keep one PM4 entry point.

**Gate P7**: no floating window duplicates a destination page; the string inventory shows the
deleted windows are gone. Receipt: `evidence/p7-floating-windows.md`.

## P8 — Styling and naming

- [ ] T080 Add any missing primitives to `SharedUiWidgets` (section header, body text, button row,
  empty state).
- [ ] T081 Replace local colors, fonts, and headers in every sidebar and page with the shared
  primitives.
- [ ] T082 Rename WC3 data support to "WC3 Beta" in the UI.
- [ ] T083 Rename WC3 data support to "WC3 Beta" in the documentation.
- [ ] T084 Final dead-control sweep: remove or wire every remaining `dead` row.
- [ ] T085 Final string search: zero `WC3` labels without `Beta`.

**Gate P8**: every sidebar uses the shared primitives; zero bespoke section styles; zero `WC3`
labels without `Beta`; zero dead controls. Receipt: `evidence/p8-styling-naming.md`.

## Final verification

- [ ] T090 `dotnet build wow-viewer/WowViewer.slnx -c Debug` — 0 errors.
- [ ] T091 `dotnet test wow-viewer/WowViewer.slnx -c Debug` — all pass.
- [ ] T092 Operator visual confirmation of the bottom bar, Options sub-menu, and destination
  sidebars on a real client.
- [ ] T093 Operator functional confirmation of open, export, and import on a real client.
- [ ] T094 Update [audit.md](audit.md) §0.11 status to `implemented` and record the final receipt.

**Gate Final**: T090–T094 pass; the receipt is `evidence/final-receipt.md`.
