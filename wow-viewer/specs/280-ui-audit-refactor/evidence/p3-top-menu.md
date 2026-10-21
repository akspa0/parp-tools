# P3 menu/content extraction handoff — 2026-10-21

**Content extraction delivered; parent integration and real UI proof remain open.** Top menu
source now contains File / View / Help only. No menu-owned chrome/window/popup lifecycle remains.
No dependency on a future shell type was introduced. The old standalone shell call is **not**
valid runtime integration; it remains only as an obsolete compile-compatible alias.

## Parent APIs and call order

All methods below are public on the existing internal service; no host interface edit is required.

1. Call [DrawContent](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs:72)
   **inside** the manager's active menu bar. It never begins/ends that bar or any window.
2. Call [DrawDeferredActions](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs:197)
   once per frame at the stable manager root, including when UI chrome is hidden, before drawing
   the existing shared picker. Do not call it once per destination or once per popup.
3. Drain [TryTakePopupRequest](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:62)
   into the manager's popup queue. IDs: AboutPopupId, ForgetBasePopupId,
   KeyboardShortcutsPopupId. The manager owns opening, dimensions, beginning/ending, focus, and close.
   Queue requests rather than opening multiple modal windows simultaneously.
4. Inside the active manager popup, call
   [DrawPopupContent](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:87).
   A true return requests closure. Always call
   [OnPopupClosed](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:115)
   on close/escape/title-bar dismissal too; this clears pending saved-base confirmation state.
   Keyboard content is scrollable; use an appropriately sized manager window (previously 560×420).
   RequestAbout and RequestKeyboardShortcuts are also exposed for manager shortcuts/help.

## Destination wiring

| Destination | API / integration |
|---|---|
| Editor > Terrain Tools | Inject [DrawTerrainFileOperationsContent](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:16) as the optional drawFileOperations constructor argument of [EditorTerrainToolsPage](../../../src/viewer/WoWViewer/Workbench/Pages/EditorTerrainToolsPage.cs:14). It draws 10 exports + 3 session-only imports before the existing Terrain Lab. |
| Viewer model surface | Call [DrawModelExportContent](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:17) exactly once at the owning model destination. |
| Viewer world surface / World Overview | Call [DrawWorldExportContent](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:42) exactly once. Includes GLB scopes and the two existing DAT export actions. |
| World settings menu | [DrawDatHeightScaleMenuContent](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:57) delegates the existing submenu. Call only inside a manager-owned settings menu, not as an independent window. |
| Editor > Converters | Inject [DrawConvertersContent](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs:20) into [EditorConvertersPage](../../../src/viewer/WoWViewer/Workbench/Pages/EditorConvertersPage.cs:14), or redirect the existing host draw to it. Uses SharedUiWidgets; preserves Map/WMO preparation and dialog requests. No new converter/format support. |
| Archaeology > Cartography | Existing synthesized-minimap page is the survivor; its Tools/dashboard launchers were deleted. No second minimap-export home added. |
| PM4 workbench | Existing PM4 export commands remain the survivor; no new PM4 export home added here. |

The optional page arguments intentionally keep the parent build source-compatible. **If the
parent does not inject the terrain delegate, terrain exports/imports will not appear.** The
old converter fallback also remains until parent wiring switches it to the new content.

## Exact references outside this write scope

- [ViewerApp.cs:1380](../../../src/viewer/WoWViewer/ViewerApp.cs:1380): replace the standalone
  DrawMenuBar call with manager invocation/call order above. The obsolete alias draws only content;
  it no longer owns chrome, pumps deferred work, or opens popups.
- [EditorWorkbenchPages.cs:21–23](../../../src/viewer/WoWViewer/Workbench/Pages/EditorWorkbenchPages.cs:21):
  pass the terrain and converter delegates, or wire the existing parent host route.
- [ViewerApp_Editor.cs:57–58](../../../src/viewer/WoWViewer/ViewerApp_Editor.cs:57),
  [EditorDataIoPage.cs:41](../../../src/viewer/WoWViewer/Workbench/Pages/EditorDataIoPage.cs:41), and
  [ViewerAppContext.cs](../../../src/viewer/WoWViewer/Workbench/Pages/ViewerAppContext.cs): remove the
  obsolete ImportsAndExports dashboard route after updating the page. The temporary
  [EditorPanels bridge](../../../src/viewer/WoWViewer/Workbench/Services/EditorPanels/EditorPanelsService.cs:1071)
  contains guidance only, not duplicate controls. Redirect the converter route or inject its delegate.
- [WorkbenchPanelsService.cs:1282](../../../src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs:1282):
  parent can delete the old converter wrapper after redirecting its callers; it still contains
  obsolete Tools-path hints and three nonfunctional informational cards. New converter content
  exposes only the two wired actions.
- [MainMenuBarService.Host.cs](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Host.cs)
  and [EditorPanelsService.Host.cs](../../../src/viewer/WoWViewer/Workbench/Services/EditorPanels/EditorPanelsService.Host.cs)
  were read, **not edited**. Parent must prune unused legacy enum/navigation/flag bridges when
  deleting the old shell types. Do not infer use merely from those preserved bridge declarations.
- Old TerrainTileIoService import-scope windows/request functions were not edited. New terrain
  content never opens those windows; parent may remove obsolete hosting after callers migrate.

## Removed surfaces / preserved actions

- Entire Tools menu, including Panels, Converters, Export and Import trees.
- File-menu DAT export/height-scale subtree (DAT open remains a flat File action).
- View duplicates: wireframe, Log, Perf, Lighting, PM4, File Browser, Model Info, Asset Catalog.
  Options/Diagnostics/owning Viewer destinations are parent-owned survivors.
- View legacy toggles: Tab System (069), Dockable Shell Panels.
- Top-mode Viewer/Editor/Archaeology buttons and their local styling (bottom navigation is parent-owned).
- Duplicate Data I/O terrain/GLB/synthesized-minimap controls.
- MainMenu legacy-sidebar utility adapter file deleted; popup bodies are content APIs instead
  of independent windows. Credits and shortcut catalogue content are preserved.

## Verification / criterion → evidence

Focused viewer build: final exit **0**, **317 warnings / 0 errors**; see
[build receipt](p1-focused-build.md). Seven reporter checks and structural checks: exit **0**;
see [P1 receipt](p1-file-operations.md). Scoped whitespace diff check: exit **0**.

Menu inventory command (exit **0**, temporary TXT then wrapped in the linked Markdown artifact):

```powershell
pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiStringInventory.ps1 `
  -ViewerPath wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu `
  -OutputPath wow-viewer/src/viewer/WoWViewer/Workbench/Services/FileOperations/p3-menu-inventory.txt
```

[Captured inventory](p3-menu-inventory.md): **4 source files, 64 rows, 59 callsites**;
source SHA256 **8762EBE87D3CDCCF5662F91ADDD09F3CDBEA531B7850165E8DF8280144BF3ADF**.
The temporary TXT was removed. An initial attempt to use a Markdown output extension was
rejected by the inventory tool before writing; the tool was not modified.

| Criterion | Evidence | Remaining |
|---|---|---|
| Three top menus | Roslyn check: roots=File,View,Help; captured source inventory | Parent/runtime reachability |
| No export/import top-menu path | Captured paths contain only File/View/Help global actions; source check menu_exports_imports=0 | Parent connects destination APIs |
| No independent chrome | Structural check chrome_owners=0, including popup lifecycle calls | Manager popup/input/size behavior |
| Legacy toggles/duplicates removed | Inventory paths + deleted adapter + source checks | Parent removes leftover hosting/bridges outside scope |
| Surviving exports preserved | Destination APIs above; focused build | All export confirmation/loadability is operator-owned |

No full P1/P3 runtime gate, visual proof, client file correctness, or completed phase is claimed.
No task/status/memory edits, no host edits, no commits, no client run. Parent's next bounded step
is the integration above, followed by operator export/import and popup/navigation smoke proof.
