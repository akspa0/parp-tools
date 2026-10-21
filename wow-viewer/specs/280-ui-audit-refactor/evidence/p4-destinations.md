# P4 destination separation + P7 floating-window dedupe

Date: 2026-10-21
Scope: Spec 280 P4, P7, and the U0–U2/U4/U6 shell integration delegated to this task.
No task/status/memory edits, no commits, no real-client run, no Terra.

## Result

The shell now has one owned manager and three destinations. The old page enums,
translating navigation overloads, dockspace, legacy sidebars, and floating duplicate
windows are gone. Feature content is retained behind the new destination pages.

## Destination map

| Destination | Right-sidebar pages | Left-sidebar destination sections |
|---|---|---|
| Viewer | Inspector, World, Model, PM4 | Scene Tools (Taxi, Capture), Assets |
| Editor | Placement & Objects, Terrain Tools, Data I/O, Converters | Editor Workspace, Project Output |
| Archaeology | Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback & Capture, Cartography | Archaeology Workspace, Layer Projects |

- [`WorkbenchTab.cs`](../../../src/viewer/WoWViewer/Workbench/WorkbenchTab.cs) now declares only `Viewer`, `Editor`, `Archaeology`; `ViewerPanel` declares `Inspector`, `World`, `Model`, `PM4`.
- [`WorkbenchNavigator.cs`](../../../src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs) now exposes only `GetPageLabels`; all seven compatibility page enums, `Canonicalize`, and the old label helpers are deleted.
- [`ViewerUiManager.cs`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs) owns the top bar, bottom bar, status bar, left/right sidebars, Options, and the popup registry. It draws `MainMenu.DrawContent()` inside its menu bar and calls `DrawDeferredActions()` once at the root.
- [`WorkbenchPanelsService.cs`](../../../src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs) is content-only: `DrawViewerContent`, `DrawDestinationContent`, `DrawSceneToolsContent`, `DrawAssetsContent`, `DrawCameraContent`, and `DrawConvertersSubTabContent`.
- [`NavigatorPanelService.Sidebars.cs`](../../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs) is content-only: Workspace, World Overview, File Browser, destination sections, Diagnostics. The World Overview no longer embeds a second minimap.
- [`EditorWorkbenchPages.cs`](../../../src/viewer/WoWViewer/Workbench/Pages/EditorWorkbenchPages.cs) receives the terrain file-operation and converter delegates at construction; [`ViewerApp_Editor.cs`](../../../src/viewer/WoWViewer/ViewerApp_Editor.cs) injects `MainMenuBarService.DrawTerrainFileOperationsContent` and `DrawConvertersContent`.
- Viewer World page 3 calls `DrawWorldExportContent`; Viewer Model page 2 calls `DrawModelExportContent`; Editor Terrain Tools calls the repaired terrain file operations.

## Deleted shell and duplicate surfaces

- `Workbench/Services/ShellLayout/ShellLayoutService.cs` deleted.
- `ViewerApp_Sidebars.cs` deleted (only legacy utility state remained).
- `MainMenuBarService.Sidebars.cs` deleted.
- `Pm4WorkbenchService.Alignment.cs` deleted (old correlation/alignment surface).
- `Pm4ObjectMatchStore.cs` deleted; `Pm4ObjectMatchStore`/`Pm4ObjectMatchEntry` fields, host bridges, settings persistence, and the Rosetta “Match to selected PM4 object” action are removed.
- `DrawDockspaceHost`, `DrawLegacyLeftSidebar`, `DrawLegacyRightSidebar`, `DrawToolbar`, `DrawBottomBar`, `DrawFixedSidebarSplitters`, `DrawSettingsWindow`, `DrawCaptureAutomationWindow`, `DrawCameraPathWindow`, `DrawWeakSignalWindow`, `DrawUniqueIdArchaeologyWindow`, `DrawLogViewer`, and `DrawPerfWindow` are gone.
- `OpenWorkbenchTab` overloads, `NormalizeWorkbenchStateAfterLoad`, `MigrateLegacyEditorPageIndex`, `FixedBottomDrawerTab`, `UseDockspaceUi`, `UseTabUi`, and the old shell settings fields are gone.
- Log and Performance are the only optional global windows; both are manager-owned and use content-only providers.

## Persistence reset

[`ViewerUiManager.Persistence.cs`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs) writes `output/settings/ui-layout-v1.json` with a version field. It deliberately does not read `viewer_settings.json` navigation, `imgui.ini` docking, or old shell rectangles. Old page indices are reset to Viewer/Inspector and Archaeology/Cartography. `Reset Shell Layout` calls `ViewerUiManager.ResetLayout()`.

## Verification

Commands and results:

```powershell
pwsh -NoProfile -File wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/Verify-Integration.ps1
# exit 0
# Focused semantic check (frozen legacy minimap host excluded in memory): 0 errors; 360 trees
# Pure layout checks passed: 4361 assertions

dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -v:minimal
# exit 1
# 102 warnings, 1 error
# Workbench/Services/MinimapStatus/MinimapAndStatusService.cs(76,13):
# CS0246: The type or namespace name 'ShellLayoutService' could not be found
```

The semantic check is diagnostic: it excludes only the frozen minimap provider’s obsolete
`ShellLayoutService` field and `DrawMinimapWindow` method in memory. It does not modify files,
install stubs, or claim a passing viewer build. The full build’s only error is that frozen
provider, which is parent-owned.

Structural checks (single pass over `wow-viewer/src/viewer/WoWViewer/**/*.cs`):

- `OpenWorkbenchTab`, `Canonicalize`, all seven compatibility page enums, `Pm4ObjectMatchStore`,
  `Pm4ObjectMatchEntry`, `TrySavePm4ObjectMatch`, `Match to selected PM4 object`,
  `MigrateLegacyEditorPageIndex`, `DrawDockspaceHost`, legacy sidebar/toolbar/bottom-bar methods,
  and the floating-window wrappers: **0 matches**.
- `ShellLayoutService`/`ShellPanelId`: only the frozen `MinimapStatus` provider remains.
- `DrawLogViewerContent` remains as content; the manager owns its window.

## Criterion → evidence

| Contract | Evidence | Remaining |
|---|---|---|
| Three destinations only | `WorkbenchTab` source; manager bottom bar enumerates it | Operator visual check |
| Viewer four tabs with focus select | `ViewerPanel` source; `BeginPanelTab` uses `SetSelected` | Operator click check |
| Editor four pages | `WorkbenchNavigator.GetPageLabels(Editor)`; `EditorWorkbenchPages.Draw` | Operator page check |
| Archaeology five pages, no PM4 page | `GetPageLabels(Archaeology)` has five labels; dispatcher cases 0–4 | Operator page check |
| Left destination sections | `NavigatorPanelService.DrawContent` switch | Operator visual check |
| No duplicate minimap in World Overview | `DrawWorldOverviewContent` no longer calls minimap surface | Parent minimap conversion |
| One input authority | `ViewerUiManager.CanSceneConsumeMouse`/`IsSceneKeyboardCaptureBlocked`; `ViewerApp` call sites delegate | Operator input check |
| Scene edge-to-edge | `UiLayout` scene rect is full width; sidebars overlay | Operator visual check |
| Old layouts reset | Versioned `ui-layout-v1.json`; no legacy reads | Operator restart check |
| Floating duplicates removed | Structural check above | Operator visual check |
| Feature content retained | Viewer/Editor/Archaeology content delegates; terrain/converter injection | Operator action check |

## Parent-owned follow-ups

1. Convert `MinimapStatus` to content-only: remove `ShellLayoutService`, `ShellPanelId`,
   `UseDockspaceUi`, `DrawMinimapWindow`, and `DrawStatusBar` window ownership. Use
   `ViewerUiManager.MinimapHost`, `ReportHostedRegion(UiRegion.Minimap, ...)`, and
   `GetRegionRect(UiRegion.Minimap)`.
2. Convert chat to `ViewerUiManager.ChatHost`/`ReportHostedRegion(UiRegion.Chat, ...)`; the
   existing `SqlSpawnStreaming.DrawChatOverlay` call remains until then.
3. Extract the status-bar content from `MinimapAndStatusService.DrawStatusBar` and assign it to
   `ViewerUiManager.StatusContent`; the manager currently shows `StatusMessage`.
4. P8 style/token pass: the manager uses existing `SharedUiWidgets`; no new token set was added.
5. Documentation/UI-reference generation and real-client visual/input proof remain parent/operator-owned.
