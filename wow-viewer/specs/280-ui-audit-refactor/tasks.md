# Tasks: UI Audit and Pre-Alpha-Style Refactor

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md) · **Audit**: [audit.md](audit.md)

**Branch**: `280-ui-rework`

**Rule**: a task is checked `[x]` only with a receipt (AGENTS §9.2): files changed, exact commands
and exit status, and a criterion→evidence table. Build/test output alone never proves runtime,
visual, or file behavior.

**Ledger sync 2026-10-21.** This file was reconciled against the real implementation state. Every
`[x]` below points at an evidence file in [evidence/](evidence/) (or at a source-verified fact
named inline). Every task whose acceptance is a runtime / visual / real-file behavior is left
`[ ]` with a one-line reason, even where the code change itself is landed. Two evidence files
named in the original task text **do not exist**: `evidence/p0-baseline.md` and
`evidence/p0-audit-closure.md`; the baseline build and test logs exist instead as
`evidence/p0-build.log` and `evidence/p0-test.log`.

**Startup crash found and fixed (code-level, 2026-10-21).** The viewer previously threw
`System.ArgumentException: Delegate to an instance method cannot have null 'this'` at
`ViewerUiManager..ctor`: the constructor captured `host.MinimapAndStatus.DrawStatusContent` at
`Workbench/Services/UiManager/ViewerUiManager.cs:24`, but `ViewerApp` constructs
`ViewerUiManager` at `ViewerApp.cs:581` while `_minimapAndStatus` is only assigned at
`ViewerApp.cs:621`. The constructor now resolves the provider at draw time through a lambda
(`ViewerUiManager.cs:24-26`), and the solution builds with 0 errors. **The viewer has not been
launched since the fix**, so every runtime/visual gate below is still operator-owned proof, not
a code blocker.

## P0 — Baseline and guardrails

- [x] T001 Record the baseline build and test result on `280-ui-rework`.
  - Command: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
  - Command: `dotnet test wow-viewer/WowViewer.slnx -c Debug`
  - Receipt: `evidence/p0-build.log` (build, 0 errors), `evidence/p0-test.log`
    (Core 7 failed / 1819 passed / 1 skipped / 1827 total; Editor 91/91; Curation 39/39;
    PM4 165/165). The task's originally named `evidence/p0-baseline.md` was never written; the
    two raw logs are the baseline of record.
- [x] T002 Add a UI-string inventory script that lists every visible label and menu path.
  - Output: `evidence/ui-strings-baseline.txt` (3,489 data rows / 3,300 callsites, 117 with a
    lexical menu path, source SHA-256 `D95BFAF…DC44C`).
  - Receipt: `evidence/p0-inventory.md` (script `eng/ui/Get-UiStringInventory.ps1`; self-test
    exit 0 with 16 checks; baseline SHA-256 `F7C00CF3…F5C1`).
- [ ] T003 Confirm the audit's duplicate table is complete against the running viewer; add any
  missing row to [audit.md](audit.md).
  - Not done: the named receipt `evidence/p0-audit-closure.md` does not exist, and the running-
    viewer confirmation is blocked by the viewer not having been launched since the fix.

**Gate P0**: build and tests pass; the string inventory exists; the audit has no `unknown` rows.
*Status: partially met.* The build and the string inventory are met. The 7 Core test failures
are pre-existing format/data failures, not UI. The audit carries no `unknown` row (the only
occurrence of the word is the status legend in `audit.md:4`), but T003's running-viewer
confirmation was never recorded, so the gate is **not closed**.

## P1 — File operations repair

- [x] T010 Replace the "Open File..." text popup with `ImGuiPathPicker`.
  - File: `Workbench/Services/MainMenu/MainMenuBarService.cs`
  - Receipt: `evidence/p1-file-operations.md`; source-verified at
    `MainMenuBarService.cs:207` — `ImGuiPathPicker.Instance.Open("Open File", …)`. The old
    typed-path modal is gone.
- [ ] T011 Audit every Export action: run it, confirm a file is written, add the status-line path
  report.
  - Code landed (10 terrain exports route through the repaired service, `evidence/p1-file-operations.md`),
    but no export was executed against real data, so "each export writes a real file" is unproven.
    Operator-owned; the viewer has not been launched since the startup-crash fix.
- [ ] T012 Audit every Import action: run it, confirm it applies, or delete it.
  - Code landed (3 explicit session-only apply chains with non-zero-work enforcement,
    `evidence/p1-file-operations.md`), but no import was executed against real terrain.
  - Operator-owned; the viewer has not been launched since the startup-crash fix.
- [ ] T013 Add explicit failure reporting to every export/import action.
  - Partially landed: the reporter has 7 passing behaviour checks including a thrown write
    exception, missing output and empty output (`evidence/p1-file-operations.md`,
    `evidence/p1-reporter-artifact.md`), but the "no silent no-op remains" sweep over every
    action in the running viewer is unproven. Operator-owned.
- [x] T014 Record the operator-reported functional state of each File-menu action in
  [audit.md](audit.md) §2.
  - Values: `works`, `unreliable`, `dead`.
  - Receipt: `audit.md` §2.1 "File-menu functional state (post-implementation)" — records the
    operator's report (File mostly works, Export unreliable, Import non-functional, Open File
    required a typed path) and the post-implementation disposition, with runtime confirmation
    still operator-owned.

**Gate P1**: the operator opens a file through the picker, exports one artifact of each kind, and
imports one artifact of each kind; every action produces or consumes a real file. Receipt:
`evidence/p1-file-operations.md`. *Status: open — the gate is a runtime gate and the viewer has
not been launched since the startup-crash fix.*

## P2 — Bottom bar and Options sub-menu

- [x] T020 Add the destination buttons to the bottom bar: Viewer, Editor, Archaeology.
  - File: `Workbench/Services/UiManager/ViewerUiManager.cs`
  - Receipt: `evidence/p4-destinations.md` (bottom bar enumerates `WorkbenchTab` = Viewer /
    Editor / Archaeology); source-verified at `ViewerUiManager.cs:256-266`.
- [x] T021 Add the Options button and the sub-menu anchored next to the chat.
  - Files: `ViewerChromeService.cs`, `FakeChatWindowService.cs`
  - Receipt: `evidence/u-shell-integration.md` (`UiPopupRegistry` + Options region);
    `evidence/u-ui-manager.md` §2 (manager draws `UiRegion.Options`); source-verified at
    `UiLayout.cs:56-58` (the Options rect is derived from the chat rect) and
    `ViewerUiManager.cs:263`.
- [x] T022 Move the render toggles into the Options sub-menu.
  - Receipt: `evidence/p2-options.md` — 23 asserted render/object/simulation state paths retained
    in the Options groups; the Options source contains 0 chrome window/popup/style ownership
    calls.
- [x] T023 Move the overlay, layer, and transparency popups into the Options sub-menu; delete the
  three bottom-bar popups.
  - Receipt: `evidence/p2-options.md` — opacity and layer controls now draw directly in Options;
    the Chrome service has zero `ImGui.Begin`/`BeginPopup`/`OpenPopup`/`SetNextWindowPos`/
    `SetNextWindowSize`/`PushStyleColor` ownership calls.
- [x] T024 Keep the FPS/stats readout in the bottom bar in every destination.
  - Receipt: `evidence/p2-options.md` (`DrawStatsContent()` content provider);
    source-verified at `ViewerUiManager.cs:265` — the bottom bar region is drawn for every
    destination and the stats provider is called from it.
- [x] T025 Move Render Quality and Lighting from Utilities into Options > Display.
  - Receipt: `evidence/p2-options.md` — the Display group delegates to the existing
    `DrawRenderQualityContent` / `DrawLightingContent` / `DrawUiThemeSettingsContent` APIs.
- [x] T026 Move Audio from Utilities into Options > Audio.
  - Receipt: `evidence/p2-options.md` — the Audio group delegates to the existing
    `DrawAudioContent` API.
- [x] T027 Move the bottom-bar grid, layer, overlay, object, and transparency toggles into the
  Options groups defined in [information-architecture.md](information-architecture.md) §7.
  - Receipt: `evidence/p2-options.md` — all 23 preserved control state paths asserted present in
    the seven Options groups (Display, Overlays, Layers, Grids, Objects, Audio, Simulation).
- [x] T028 Delete the three bottom-bar popups (Overlays, Layers, Transparency) after the move.
  - Receipt: `evidence/p2-options.md` — zero popup ownership calls remain in the Chrome service.
- [x] T029 Delete the `Utilities` workbench tab and `UtilitiesBottomTab` after every page has a
  new home.
  - Receipt: `evidence/p4-destinations.md` (all seven compatibility page enums deleted, 0
    matches); source-verified — `WorkbenchTab.cs` now declares only
    `Viewer, Editor, Archaeology`, and the Utilities pages have new homes in
    `evidence/p8-styling-naming.md` / `information-architecture.md` §5.

**Gate P2**: the operator reaches Viewer, Editor, Archaeology, and Options from the bottom bar in
at most two clicks; the FPS/stats readout is visible; the Options sub-menu does not cover the chat
or the status bars; no Utilities page remains. Receipt: `evidence/p2-options.md`.
*Status: open — the click-distance, overlap and visibility clauses are visual facts only the
operator can confirm, and the viewer has not been launched since the startup-crash fix. The "no
Utilities page remains" clause is met (source-verified).*

## P3 — Top menu reduction

- [x] T030 Reduce the top menu to File / View / Help.
  - File: `Workbench/Services/MainMenu/MainMenuBarService.cs`
  - Receipt: `evidence/p3-top-menu.md` (Roslyn check `roots=File,View,Help`) and
    `evidence/p3-menu-inventory.md` (4 files, 64 rows, 59 callsites, source SHA-256
    `8762EBE8…BF3ADF`).
- [x] T031 Delete Tools > Panels; move each surviving action to its owning destination.
  - Receipt: `evidence/p3-top-menu.md` — the whole Tools menu, including Panels, Converters,
    Export and Import trees, was removed; the destination wiring table in the same receipt
    names the surviving action homes.
- [x] T032 Delete the Tools > Export and Tools > Import trees; move each action to the owning
  destination.
  - Receipt: `evidence/p3-top-menu.md` — the File-menu DAT export / height-scale subtree was
    also removed; exports and imports now live with their owning destinations
    (`evidence/p1-file-operations.md`).
- [x] T033 Delete the legacy shell toggles ("Tab System (069)", "Dockable Shell Panels").
  - Receipt: `evidence/p3-top-menu.md`; source-verified — 0 matches for either label in
    `wow-viewer/src/viewer/WoWViewer`.
- [x] T034 Delete the View menu entries that duplicate a destination surface.
  - Receipt: `evidence/p3-top-menu.md` — View duplicates removed (wireframe, Log, Perf, Lighting,
    PM4, File Browser, Model Info, Asset Catalog), each with a named surviving owner.

**Gate P3**: the top menu has three menus; no export/import action lives in the top menu; the
string inventory shows the deleted paths are gone. Receipt: `evidence/p3-top-menu.md`.
*Status: met for the structural clauses (three menus; `menu_exports_imports=0`; deleted paths
absent from the captured inventory). The inventory in `evidence/` is the pre-refactor
`ui-strings-baseline.txt` plus the MainMenu-scoped capture — no post-refactor full-tree string
inventory was regenerated, so the "string inventory shows the deleted paths are gone" clause is
proven only for the MainMenu subtree.*

## P4 — Destination separation

- [x] T040 Make the left navigator destination-aware.
  - File: `Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs`
  - Receipt: `evidence/p4-destinations.md`; source-verified — the navigator switches on the
    active destination and draws only that destination's sections.
- [x] T041 Make the right inspector destination-aware.
  - File: `Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs`
  - Receipt: `evidence/p4-destinations.md`; source-verified — the manager draws the Viewer tab
    strip for Viewer and `DrawDestinationContent(Destination)` otherwise.
- [x] T042 Delete cross-destination controls from both sidebars.
  - Receipt: `evidence/p4-destinations.md` — both sidebars are content-only; 0 matches for the
    legacy cross-destination routes and the deleted floating-window wrappers.
- [x] T043 Delete the compatibility enums (`ModelBottomTab`, `WorldBottomTab`, `ToolsBottomTab`,
  `UtilitiesBottomTab`, `TerrainBottomTab`, `Pm4BottomTab`, `ArcheologyBottomTab`).
  - File: `Workbench/WorkbenchNavigator.cs`
  - Receipt: `evidence/p4-destinations.md` — 0 matches for all seven enums; source-verified in
    this session.
- [x] T044 Delete the legacy tab redirects and `Canonicalize`.
  - Receipt: `evidence/p4-destinations.md` — 0 matches for `Canonicalize` and the `OpenWorkbenchTab`
    overloads; source-verified in this session.
- [x] T045 Reset or migrate persisted layouts that point at deleted pages.
  - Receipt: `evidence/u-shell-integration.md` and `evidence/p4-destinations.md` — versioned
    `output/settings/ui-layout-v1.json` with a version field; the manager deliberately does not
    read `viewer_settings.json` navigation, `imgui.ini` docking or old shell rectangles; old page
    indices are reset.
- [x] T046 Add the left-sidebar Diagnostics section (Log, Perf, Runtime Stats) and delete the
  three Utilities pages.
  - Receipt: `evidence/p4-destinations.md`; source-verified — the navigator draws a `Diagnostics`
    section with Log, Performance and the runtime-stats content.
- [x] T047 Add the Viewer left-sidebar Scene Tools section (Taxi, Capture, Camera Path) and
  delete the two Utilities pages.
  - Receipt: `evidence/p4-destinations.md`; source-verified — a `Scene Tools` section exists with
    a Taxi / Capture page combo, and the Capture page opens the capture panel whose tab strip
    includes "Camera Path". This matches
    [information-architecture.md](information-architecture.md) §5, which groups camera path with
    capture rather than giving it its own Scene Tools page.
- [x] T048 Add the Viewer left-sidebar Assets section (Asset Catalog) and delete the Utilities
  page.
  - Receipt: `evidence/p4-destinations.md`; source-verified — an `Assets` section exists and
    calls the assets content.
- [x] T049 Make the left sidebar show only the common sections plus the current destination's
  sections, per [information-architecture.md](information-architecture.md) §4.
  - Receipt: `evidence/p4-destinations.md`; source-verified — the navigator's destination switch
    matches the IA §4 map.

**Gate P4**: switching destination changes the visible sidebar content; no editor-only control
appears in Viewer or Archaeology; the build has no references to the deleted enums; the left
sidebar matches the IA map. Receipt: `evidence/p4-destinations.md`. *Status: met structurally
(the build has no references to the deleted enums; the sidebar content map matches). The
"switching destination changes the visible sidebar content" clause is a visual fact the operator
must confirm, and the viewer has not been launched since the startup-crash fix.*

## P5 — Archaeology dedupe

- [x] T050 Delete the legacy 5-tab strip (`DrawArcheologySubTabContent`).
  - Receipt: `evidence/p5-archaeology.md` — 0 matches in the scoped source; the canonical
    5-case dispatcher is kept.
- [x] T051 Delete the floating UniqueId Archaeology window (`DrawUniqueIdArchaeologyWindow`).
  - Receipt: `evidence/p5-archaeology.md` — deleted with its private content bridge.
- [x] T052 Delete the Quick-section archaeology launchers (`DrawArchaeologyQuickSection`).
  - Receipt: `evidence/p5-archaeology.md` — deleted.
- [x] T053 Split the Cartography "Layers" page into named sections: layer stack, DBC phase
  discovery, add phase map, donor tile grid picker.
  - Receipt: `evidence/p5-archaeology.md` — six sibling shared-widget sections exist (Layer
    Stack, DBC Phase Discovery, Add Phase Map, Donor Tile Grid Picker, Synthesized Minimap, Save
    Merged Output) and the donor picker no longer nests a collapsing header.
- [x] T054 Remove the host bridges for the deleted members.
  - File: `ArchaeologyPanelService.Host.cs`
  - Receipt: `evidence/p5-archaeology.md` — legacy-only window/export flags, temporal-stratigraphy
    and launcher bridges removed; shared analysis/capture bridges kept.

**Gate P5**: the 6-page destination strip is the only archaeology navigation; the string inventory
shows the deleted entry points are gone. Receipt: `evidence/p5-archaeology.md`. *Status: met
structurally (the 0-match structural check plus the canonical dispatcher). The archaeology
destination is **5 pages**, not 6 — PM4 is an explicit cross-destination button — so the gate
wording is stale relative to the implementation and to
[information-architecture.md](information-architecture.md) §5. The "string inventory shows the
deleted entry points are gone" clause is proven by the scoped source check, not by a regenerated
full-tree inventory.*

## P6 — Editor section rename

- [x] T060 Rename "Tasks & Workspace" to its real owner.
  - Receipt: `evidence/p6-editor.md` — renamed to "Archaeology Tasks"; source-verified.
- [x] T061 Rename "3D Object Library" to its real owner.
  - Receipt: `evidence/p6-editor.md` — renamed to "Rosetta Object Library"; source-verified.
- [x] T062 Rename "Imports & Exports" to its real owner.
  - Receipt: `evidence/p6-editor.md` — renamed to "Archaeology Imports"; source-verified.
- [x] T063 Delete the one-button "PM4 Placement Tools" section.
  - Receipt: `evidence/p6-editor.md` — the section is deleted; source-verified (0 matches).
- [x] T064 Remove the second PM4 export home from Data I/O.
  - Receipt: `evidence/p6-editor.md` — `EditorDataIoPage` no longer draws PM4 exports; the live
    export command set is called only from PM4 Selection; source-verified (0 matches for
    `DrawPm4Exports` / "PM4 Exports" / "Imports & Exports" in the Data I/O page).
- [x] T065 Delete `MigrateLegacyEditorPageIndex` after the layout reset.
  - Receipt: `evidence/p4-destinations.md` — the method is gone (0 matches), and the versioned
    layout reset landed with it. `evidence/p6-editor.md` §T065 recorded it as deliberately
    retained *pending* that reset; the reset has since landed, so the deletion is now complete.
- [x] T066 Move PM4 matching offline: remove the Correlation, Reconcile, and Alignment tabs, the
  match-curation section, and `Pm4ObjectMatchStore.cs`; the offline tool is
  [Spec 281](../281-museum-adt-patcher/spec.md).
  - Receipt: `evidence/p6-editor.md` — five inspection tabs remain (Overlay, Selection, Info,
    Outliner, Object Groups); the alignment/correlation UI, the manual curation UI and
    `Pm4ObjectMatchStore.cs` are deleted; source-verified in this session.

**Gate P6**: the Editor pages name their real content; PM4 has one export home; the viewer has no
matching/alignment/reconciliation tabs; the build has no reference to the deleted migration.
Receipt: `evidence/p6-editor.md`. *Status: met structurally.*

## P7 — Floating-window dedupe

- [x] T070 Delete the 12 Tools > Panels floating windows that duplicate destination pages.
  - Receipt: `evidence/p7-floating-windows.md` §1.1 — the floating-window wrappers are gone
    (0 matches for `DrawSettingsWindow`, `DrawCaptureAutomationWindow`, `DrawCameraPathWindow`,
    `DrawWeakSignalWindow`, `DrawUniqueIdArchaeologyWindow`, `DrawLogViewer`, `DrawPerfWindow`,
    `DrawDockspaceHost`, `DrawLegacyLeftSidebar`, `DrawLegacyRightSidebar`); source-verified in
    this session. `evidence/p4-destinations.md` and `evidence/u-shell-integration.md` carry the
    earlier structural evidence.
- [x] T071 Keep only Log and Perf as global floating windows.
  - Receipt: `evidence/p7-floating-windows.md` §2 — Log and Performance are the only optional
    global windows, both manager-owned with content-only providers; source-verified (the
    manager's diagnostics window set contains exactly Log and Performance).
- [x] T072 Delete the PM4 Actions popup; keep one PM4 entry point.
  - Receipt: `evidence/p7-floating-windows.md` §1.2 — the popup and its "PM4 Workbench" /
    "Reload PM4" actions are deleted; `WorldObjectsPanelService.cs:209-212` now draws one
    `Open PM4 Workbench` button. The only residual `"PM4 Actions"` string is an inspector payload
    section name (`InspectorPayloadsService.cs:74`), not a popup.

**Gate P7**: no floating window duplicates a destination page; the string inventory shows the
deleted windows are gone. Receipt: `evidence/p7-floating-windows.md`. *Status: met structurally
(T070–T072: 0 wrapper matches, Log/Performance are the only optional global windows, the PM4
Actions popup is gone). The "string inventory" clause is proven by the scoped source check, not
by a regenerated full-tree inventory. No runtime/visual proof — operator-owned.*

## P8 — Styling and naming

- [x] T080 Add any missing primitives to `SharedUiWidgets` (section header, body text, button row,
  empty state).
  - Receipt: `evidence/p8-styling-naming.md` §2 — body text, button row, section title,
  muted/success text, empty state and surface style added; **no existing signature changed**, so
  every other service still compiles unchanged.
- [ ] T081 Replace local colors, fonts, and headers in every sidebar and page with the shared
  primitives.
  - **Not done.** Only the five manager-owned surfaces were converted.
  `evidence/p8-styling-naming.md` §5 lists **34 files / 179 hits** of bespoke styling remaining
  outside those surfaces; a re-measurement in this session over the same pattern gives 41 files /
  243 hits. The 1 remaining hit inside the manager surfaces is the accepted "Area POI Details"
  tree node. "Every sidebar" is **not** met.
- [x] T082 Rename WC3 data support to "WC3 Beta" in the UI.
  - Receipt: `evidence/p8-styling-naming.md` §8 — 10 occurrences carry the user-visible
  "WC3 Beta" text; the 5 remaining non-`Beta` occurrences in the viewer sources are code
  comments, not labels. Source-verified in this session: `Wc3DataSource.Name` = `"WC3 Beta: …"`.
- [ ] T083 Rename WC3 data support to "WC3 Beta" in the documentation.
  - **Not done.** `evidence/p8-styling-naming.md` §8: the research/architecture documents still
  carry 24 + 3 non-`Beta` `WC3` mentions, plus prose "Warcraft 3" references. That is
  parent-owned and was deliberately not edited.
- [ ] T084 Final dead-control sweep: remove or wire every remaining `dead` row.
  - **Not done.** `evidence/p8-styling-naming.md` §9 states this explicitly. The P8 pass added
  no control and removed none.
- [ ] T085 Final string search: zero `WC3` labels without `Beta`.
  - **Not done for the documentation corpus.** The viewer UI labels are clean (see T082), but the
  research/architecture documents still contain non-`Beta` `WC3` mentions
  (`evidence/p8-styling-naming.md` §8), so the final search is not zero-wide.

**Gate P8**: every sidebar uses the shared primitives; zero bespoke section styles; zero `WC3`
labels without `Beta`; zero dead controls. Receipt: `evidence/p8-styling-naming.md`.
*Status: **not met** — 34 files / 179 bespoke-style hits remain outside the five manager-owned
surfaces, the documentation rename and the dead-control sweep are not done, and the final string
search is not zero-wide.*

## P9 — Visual layout

- [ ] T100 Apply the design tokens from [layout.md](layout.md) §8: colour, type, spacing.
  - **Partial — not ticked.** Colour, spacing and surface tokens are consumed
    (`SharedUiWidgets.cs:124-238`; `UiTokens.cs:15-49`), but the type scale
    (`Caption`/`Body`/`Label`/`Title`, `UiTokens.cs:27-31`) has **0 consumers** in the viewer
    tree; no font size is applied from it. Receipt: `evidence/p9-visual-layout.md` §7.
- [x] T101 Move the minimap to the floating top-right instrument; add collapse, zone label, and
  coordinates.
  - Receipt: `evidence/p9-visual-layout.md` §4 — floating top-right, collapse/expand, dock/undock,
    hide, tile coordinates, and the zone label + coordinates under the map
    (`ViewerUiManager.cs:226-233`). Visual confirmation is operator-owned.
- [x] T102 Delete the Utilities Minimap page; the top-right minimap is canonical.
  - Receipt: `evidence/p9-visual-layout.md` §7 — the `Utilities` tab and its pages are gone
    (T029); 0 matches for a minimap page enum/label in the viewer tree. The only minimap surface
    is the manager-owned `UiRegion.Minimap`.
- [x] T103 Convert the chat to a docked panel; keep Auto-Stream and Spawn BBs Only.
  - **Operator correction 2026-10-21:** the original "resizable bottom-left dock" was rejected —
    the operator reported it as a floating window that covered the UI and could not be moved. The
    chat is now a fixed docked panel spanning the scene between the sidebars, drawn with
    `NoMove | NoResize`; the persisted floating rect is deleted and the layout document is v3.
  - Receipt: `evidence/p9-visual-layout.md` §5 — docked rect derived from the sidebar widths and
    scene height (`UiLayout.cs:71-80`), both population controls, collapse state persisted as
    `ChatCollapsed` (`ViewerUiManager.Persistence.cs:30`, `:51`, `:80`, `:109`).
    Visual confirmation is operator-owned.
- [x] T104 Add sidebar collapse rails and persistent widths.
  - Receipt: `evidence/p9-visual-layout.md` §3 — 24 px rails (`UiLayout.cs:32-33`),
    collapse/expand buttons (`ViewerUiManager.cs:193-231`), versioned persistence
    (`ViewerUiManager.Persistence.cs:10-11`, `:30-64`, `:91-114`). Visual confirmation is still
    operator-owned.
- [x] T105 Add the responsive breakpoints and default collapse from [layout.md](layout.md) §9.
  - Receipt: `evidence/p9-visual-layout.md` §2 — four bands and their defaults are implemented
    in `UiLayout.cs:34`, `:74-75`. The gate's "layout holds at 1280, 1600, and 1920 px" is a
    visual fact the operator must confirm; the viewer has not been launched since the fix.
- [ ] T106 Optional: `Ctrl+K` command palette over destinations and actions.
  - **Not implemented.** 0 matches for `CommandPalette`, `Ctrl+K` or `ImGuiKey_K` in the viewer
    tree (`evidence/p9-visual-layout.md` §7).

**Gate P9**: the minimap is top-right and collapsible; the chat is a bottom-left dock; both
sidebars collapse to rails; the layout holds at 1280, 1600, and 1920 px. Receipt:
`evidence/p9-visual-layout.md`. *Status: **met structurally except T100 and T106** — T101–T105 are
complete; T100 (type scale) is defined but not applied because the viewer uses one ImGui font;
T106 is not implemented. The visual gate at 1280/1600/1920 is operator-owned.*

## U0–U6 — UI manager consolidation

- [x] T110 Create `ViewerUiManager`, `UiRegion`, `UiRegionState`, and `UiTokens`; draw the empty
  regions with the correct rects.
  - Receipt: `evidence/u-shell-integration.md` — `ViewerUiManager`, `UiLayout` (which defines
  `UiRegion` / `UiRegionState`) and `UiTokens` exist with the documented region rects.
- [x] T111 Move the top bar, bottom bar, and status bar into the manager; delete their window
  code.
  - Receipt: `evidence/u-shell-integration.md`; source-verified — the manager draws the top, bottom
  and status regions and the old `DrawToolbar` / `DrawBottomBar` wrappers are gone.
- [x] T112 Move the left and right sidebars into the manager; delete `ShellLayoutService`.
  - Receipt: `evidence/u-shell-integration.md` and `evidence/p4-destinations.md` —
  `ShellLayoutService` / `ShellPanelId` / `UseDockspaceUi` have 0 matches in the viewer tree;
  source-verified in this session.
- [x] T113 Move the minimap into the manager; support floating top-right and right-sidebar dock.
  - Receipt: `evidence/u-ui-manager.md` §2 and `evidence/u-shell-integration.md` — the manager
  draws `UiRegion.Minimap` itself and the layout provides both the floating top-right rect and
  the right-sidebar dock rect. *Not visually confirmed — operator-owned.*
- [x] T114 Move the chat into the manager; anchor the Options sub-menu.
  - Receipt: `evidence/u-ui-manager.md` §2 — the manager draws `UiRegion.Chat` itself; the Options
  rect is derived from the chat rect (`UiLayout.cs:55-58`). *Not visually confirmed.*
- [x] T115 Create one popup registry; move Settings, About, open-file, overlays, layers, and
  transparency into it.
  - Receipt: `evidence/u-shell-integration.md` (`UiPopupRegistry` owns modal lifetime, nesting,
  Escape and close callbacks; Settings, About, Keyboard Shortcuts, Forget Base and provider popup
  requests are routed through it) and `evidence/p2-options.md` (the overlays, layers and
  transparency content moved into the Options region rather than into popups). The open-file
  action uses the shared path picker (`evidence/p1-file-operations.md`), not a popup.
- [x] T116 Add `UiControlDoc` to every provider; generate `docs/WoWViewer/UI-REFERENCE.md`.
  - Receipt: `evidence/u-ui-manager.md` §3 and §6 — the registry, the generated reference and the
  in-app reference are in place. Re-verified in this session: the inventory has **123 rows / 115
  distinct labels**, the registry holds **115** entries (all `visible: true`), and the generated
  `docs/WoWViewer/UI-REFERENCE.md` reports "Controls documented: **115**". The scanner now reads
  the first argument of each widget call, including conditional labels, so the previously
  invisible minimap/sidebar labels are registered (`evidence/u-ui-manager.md` §8).
- [x] T117 Add a build check that fails when a visible control has no registry entry.
  - Receipt: `evidence/u-ui-manager.md` §4/§6 — `eng/ui/Verify-UiReference.ps1` is wired into
  both viewer `.csproj` files as a `VerifyUiReference` target; negative probe A (a registry entry
  removed) exits 1 with `visible control 'Log' has no UI control registry entry`.
- [x] T118 Add a build check that regenerates `UI-REFERENCE.md` and fails on drift.
  - Receipt: `evidence/u-ui-manager.md` §4/§6 — negative probe B (a stale generated document)
  exits 1, and a drift-injected build fails with
  `error : UI reference build check FAILED (Spec 280 T117/T118)`.
- [x] T119 Delete the legacy floating windows, the compatibility enums, and the dockspace path.
  - Receipt: `evidence/p4-destinations.md` and `evidence/u-shell-integration.md` — 0 matches for
  `ShellLayoutService`, `ShellPanelId`, `UseDockspaceUi`, `DrawDockspaceHost` and the seven
  compatibility enums; source-verified in this session.

**Gate U**: one service owns every chrome window; no provider calls `ImGui.Begin` for chrome; one
token set; one input-ownership rule; every visible control has a registry entry; the generated
reference matches. Receipt: `evidence/u-ui-manager.md`. *Status: met structurally for every
clause. The build check is green (exit 0, 0 errors, 0 warnings) and the scanner now covers the
previously invisible conditional labels. The viewer has not been launched since the startup-crash
fix, so runtime/visual confirmation remains operator-owned.*

## Final verification

- [x] T090 `dotnet build wow-viewer/WowViewer.slnx -c Debug` — 0 errors.
  - Re-verified in this session: `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore`
    → exit 0, `Build succeeded`, 494 warnings, **0 errors**. The viewer projects also build
    0 errors on their own (`evidence/u-ui-manager.md` §6 commands 9 and 13).
- [ ] T091 `dotnet test wow-viewer/WowViewer.slnx -c Debug` — all pass.
  - **Not "all pass."** Re-verified in this session with an absolute path:
    `dotnet test I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug --no-build` →
    Core **7 failed / 1822 passed / 1 skipped / 1830 total**; Editor 91/91; Curation 39/39;
    PM4 165/165. The 7 failures are the **same pre-existing set** recorded in the P0 baseline
    (`evidence/p0-test.log`: 7 failed / 1819 passed) — `ModelFootprintReader`,
    `AdtV23Summary`, `EnrichmentStreamFormat`, 2 × `WtfLineClassifier`, `WmoRealData`,
    `LkToAlphaRoundTrip` — all format/data tests, none UI-related. The +3 passes are the new
    `UiControlDocumentationTests` (3/3 pass).
  - **Environment note (verified, not a regression):** invoking the same suite through a
    *relative* path (`i:\…`) produces **9** failures, because `Wc3FileSourceTests` and
    `Wc3AssetExtractionTests` compare a drive-letter-cased path (`i:\` expected vs `I:\` actual).
    Re-running those two tests with an absolute `I:\` path gives 35/35 pass. The count is
    therefore path-casing dependent, not code dependent.
- [ ] T092 Operator visual confirmation of the bottom bar, Options sub-menu, and destination
  sidebars on a real client.
  - Operator-owned. The startup crash is fixed in code (`ViewerUiManager.cs:24-26`) and the build
    is green, but the viewer has not been launched, so no visual confirmation exists yet.
- [ ] T093 Operator functional confirmation of open, export, and import on a real client.
  - Operator-owned. The startup crash is fixed in code, but no real-client run has been made.
- [ ] T094 Update [audit.md](audit.md) §0.11 status to `implemented` and record the final receipt.
  - **Not done.** `audit.md` §0.11 was updated on 2026-10-21 to the honest status
    `partial — implementation landed, verification outstanding` (not `implemented`), because the
    audit's own acceptance criteria are not met: the running-viewer confirmation (T003) was never
    recorded, the P8/P9 gates are open, T100/T101/T103/T106 are incomplete, and the viewer has not
    been launched since the startup-crash fix. The P7 and P9 receipts now exist. The whole-spec
    receipt is `evidence/final-receipt.md`.

**Gate Final**: T090–T094 pass; the receipt is `evidence/final-receipt.md`. *Status: **not met** —
T091 does not literally pass (7 pre-existing Core failures), T092/T093 are operator-owned and the
viewer has not been launched since the fix, and T094 is not done.*
