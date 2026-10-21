# U0–U2/U4/U6 shell integration

Date: 2026-10-21
Scope: owned `ViewerUiManager`, pure layout, popup registry, input authority, versioned
persistence, and deletion of the old shell. Parent retains minimap/chat/style/docs integration.

## Manager ownership

[`ViewerUiManager.cs`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs)
owns every chrome window:

- top bar: `MainMenuBarService.DrawContent()` inside the manager menu bar
- bottom bar: Viewer / Editor / Archaeology, Options, FPS
- status bar: always visible; `StatusContent` hook for the parent’s extracted status content
- left sidebar: `NavigatorPanelService.DrawContent()`
- right sidebar: Viewer tab strip or destination page combo + content
- Options: `ViewerChromeService.DrawOptionsContent()`
- popup registry: Settings, About, Keyboard Shortcuts, Forget Base, and provider popup requests
- optional global windows: Log and Performance only

`MainMenuBarService.DrawDeferredActions()` is called exactly once at the manager root.
`MainMenuBarService.TryTakePopupRequest()` drains provider requests into the registry.

## Pure layout

[`UiLayout.cs`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs) computes
logical-pixel rectangles for scene, top bar, bottom bar, status bar, left/right sidebars, Options,
minimap, and chat. The scene rectangle is always full width between the top and bottom/status bars;
sidebars overlay it and never shrink the projection. Responsive defaults:

| Width | Left | Right | Sidebar widths |
|---|---|---|---|
| ≥ 1920 | expanded | expanded | 280 / 360 |
| 1600–1919 | expanded | expanded | 240 / 320 |
| 1280–1599 | collapsed rail | expanded | 240 / 320 |
| < 1280 | collapsed rail | collapsed rail | 240 / 320 |

Collapsed sidebars remain 24 px rails. The status bar is always visible. Invalid/NaN widths are
clamped to finite values.

## Input authority

[`ViewerUiManager.Input.cs`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Input.cs)
is the single scene-input authority:

- `CanSceneConsumeMouse` requires the point to be inside the scene rect, no popup open, no ImGui
  mouse capture, and no visible chrome/hosted region containing the point.
- `IsSceneMouseCaptureBlocked` is its inverse.
- `IsSceneKeyboardCaptureBlocked` covers popups, ImGui keyboard capture, and text input.
- `TryGetSceneViewportRect` and `TryGetSceneFramebufferViewport` use the pure layout.
- `QueueImGuiMouseButtonEvent`, `FlushPendingImGuiMouseButtonEvents`, and
  `SyncImGuiWindowMetrics` are retained from the proven controller plumbing.

`ViewerApp` mouse, wheel, keyboard, render, resize, and capture call sites now delegate to the
manager. The old dockspace bypass is gone.

## Persistence

[`ViewerUiManager.Persistence.cs`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs)
writes `output/settings/ui-layout-v1.json` atomically through a `.pending` file. It stores only:

- version
- left/right visibility and widths
- destination
- Viewer panel
- three destination page indices

It does not read old `viewer_settings.json` navigation, `imgui.ini` docking, or shell rectangles.
Old page indices are reset. `Reset Shell Layout` calls `ResetLayout()`.

## Popup registry

[`UiPopupRegistry.cs`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/UiPopupRegistry.cs)
owns modal lifetime, nesting, Escape close, and close callbacks. Settings is requested by setting
`SettingsWindow._showSettingsWindow`; the manager converts that request into the registry. About,
Keyboard Shortcuts, and Forget Base use the existing `MainMenuBarService` public popup APIs.
Provider popup buttons use `ViewerUiManager.DrawPopupButton`.

## Parent hooks

The manager exposes these APIs for the parent’s minimap/chat/status conversion:

```csharp
Action<UiRect>? MinimapHost { get; set; }
Action<UiRect>? ChatHost { get; set; }
Action? StatusContent { get; set; }
void ReportHostedRegion(UiRegion region, UiRect rect, bool visible);
UiRect GetRegionRect(UiRegion region);
```

`ReportHostedRegion` makes a hosted minimap/chat rectangle block scene input. `GetRegionRect`
returns the manager’s current logical rectangle. The manager does not draw minimap or chat yet.

## Remaining parent-owned callsites

| Location | Remaining work |
|---|---|
| `Workbench/Services/MinimapStatus/MinimapAndStatusService.cs:76` | Remove `ShellLayoutService` field; use `ViewerUiManager` hosting APIs. |
| `Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs:628–654` | Remove `ShellPanelId`, `UseDockspaceUi`, `PrepareDockableShellPanelWindow`, and `CaptureDockPanelState`; draw inside `MinimapHost`. |
| `MinimapAndStatusService.DrawStatusBar` | Extract content and assign to `ViewerUiManager.StatusContent`; the manager currently shows `StatusMessage`. |
| `SqlSpawnStreamingService.DrawChatOverlay` | Convert to `ChatHost`/`ReportHostedRegion(UiRegion.Chat, ...)`. |
| P8 style pass | Manager uses existing `SharedUiWidgets`; no new token set was introduced. |
| Docs/UI reference | Parent generates the control registry/reference and updates docs. |

## Verification

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

The semantic check is diagnostic only. It excludes the frozen minimap provider’s obsolete
`ShellLayoutService` field and `DrawMinimapWindow` method in memory; it does not modify files,
install stubs, or claim a passing viewer build. The full build’s only error is that frozen
provider.

## Criterion → evidence

| Contract | Evidence | Remaining |
|---|---|---|
| One service owns chrome windows | `ViewerUiManager.Draw` region calls; no provider `ImGui.Begin` for chrome outside Log/Perf | Operator visual check |
| Scene edge-to-edge | `UiLayout` scene rect full width; layout assertions | Operator visual check |
| Sidebars float, not shrink scene | Layout assertions compare expanded/collapsed scene rects | Operator visual check |
| One input authority | `ViewerUiManager.Input`; `ViewerApp` delegates | Operator input check |
| Popup registry owns Settings/About/help/Options | `RegisterPopups`; `TryTakePopupRequest`; `DrawPopupButton` | Operator popup check |
| Versioned layout reset | `ui-layout-v1.json`; no legacy reads | Operator restart check |
| Old shell deleted | `ShellLayoutService` file deleted; structural check | Parent minimap conversion |
| Log/Perf manager-owned | `DrawDiagnosticsWindows`; content-only providers | Operator window check |
| No new style tokens | Manager uses `SharedUiWidgets`; no `UiTokens` file added | P8 parent pass |
| No task/status/memory edits | This receipt only | — |
