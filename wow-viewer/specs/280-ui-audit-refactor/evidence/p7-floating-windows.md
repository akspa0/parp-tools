# P7 floating-window dedupe receipt — 2026-10-21

**Spec**: [280-ui-audit-refactor](../spec.md) · **Plan**: [plan.md](../plan.md) ·
**Tasks**: T070–T072 · **Branch**: `280-ui-rework`

**Scope of this receipt.** P7 deletes the floating windows that duplicated destination pages,
keeps exactly two optional global windows, and removes the PM4 Actions popup so PM4 has one entry
point. This receipt is **structural only**: it proves the source state and the build. It does
**not** prove runtime or visual behaviour — the viewer has not been launched since the startup
crash fix (see [final-receipt.md](final-receipt.md) §3).

---

## 1. What was deleted

### 1.1 The duplicate floating windows (T070)

The following window wrappers are gone from `wow-viewer/src/viewer/WoWViewer` (0 matches for each
name as a method declaration or call):

| Deleted wrapper | Duplicated destination page |
|---|---|
| `DrawSettingsWindow` | Options > Settings |
| `DrawCaptureAutomationWindow` | Viewer > Scene Tools > Capture |
| `DrawCameraPathWindow` | Viewer > Scene Tools > Capture > Camera Path |
| `DrawWeakSignalWindow` | Archaeology > Cartography |
| `DrawUniqueIdArchaeologyWindow` | Archaeology > UniqueId |
| `DrawLogViewer` (window wrapper) | Viewer > Diagnostics > Log |
| `DrawPerfWindow` | Viewer > Diagnostics > Performance |
| `DrawDockspaceHost` | the manager-owned shell |
| `DrawLegacyLeftSidebar` | the manager-owned left sidebar |
| `DrawLegacyRightSidebar` | the manager-owned right sidebar |
| legacy toolbar wrapper | the manager-owned top bar |
| legacy bottom-bar wrapper | the manager-owned bottom bar |

The 12 entries above are the 12 duplicate floating windows named by T070. The two diagnostics
windows were not deleted as *content*; their **window wrappers** were deleted and the manager now
owns the window (see §2).

### 1.2 The PM4 Actions popup (T072)

`Workbench/Services/WorldObjects/WorldObjectsPanelService.cs` no longer opens a "PM4 Actions"
popup. The popup and its "PM4 Workbench" / "Reload PM4" actions were replaced by one button:

```csharp
// One PM4 entry point: the Inspector > PM4 page. The old "PM4 Actions" popup
// duplicated it and is deleted (Spec 280 T072).
if (ImGui.Button("Open PM4 Workbench"))
    OpenPm4Workbench(_worldScene.Pm4Overlay.HasSelectedPm4Object ? Pm4WorkbenchTab.Selection : Pm4WorkbenchTab.Overlay);
```

Source: `Workbench/Services/WorldObjects/WorldObjectsPanelService.cs:209-212`.

**Residual string, not a popup.** The literal `"PM4 Actions"` still appears once as an
**inspector payload section name** in
`Workbench/Services/InspectorPayloads/InspectorPayloadsService.cs:74`
(`builder.AddSection("PM4 Actions")`). That is a data-payload section label, not a toolbar popup
and not a second PM4 entry point. It is recorded here so the "0 matches" claim is not overstated.

## 2. What remains

Exactly two optional global windows remain, and both are **manager-owned content-only providers**:

| Window | Owner (window) | Content provider | Source |
|---|---|---|---|
| Log | `ViewerUiManager.DrawDiagnosticsWindows` | `IViewerAppHost.DrawLogViewerContent()` → `LogViewerService.DrawLogViewerContent()` | `Workbench/Services/UiManager/ViewerUiManager.cs:315-323` |
| Performance | `ViewerUiManager.DrawDiagnosticsWindows` | `IViewerAppHost.PerfPanel.DrawPerfContent()` | `Workbench/Services/UiManager/ViewerUiManager.cs:324-330` |

`DrawDiagnosticsWindows` contains exactly those two `ImGui.Begin` calls. Neither provider opens a
window, sets a window position/size, or owns a popup; the manager does. Both windows are drawn
inside the shared token surface (`SharedUiWidgets.PushSurfaceStyle()` / `PopSurfaceStyle()`).

## 3. Structural verification

| # | Check | Result |
|---|---|---|
| 1 | Search `wow-viewer/src/viewer/WoWViewer/**/*.cs` for `DrawSettingsWindow\|DrawCaptureAutomationWindow\|DrawCameraPathWindow\|DrawWeakSignalWindow\|DrawUniqueIdArchaeologyWindow\|DrawLogViewer\|DrawPerfWindow\|DrawDockspaceHost\|DrawLegacyLeftSidebar\|DrawLegacyRightSidebar` | **0 wrapper matches.** The only `DrawLogViewer` hit is `DrawLogViewerContent` (the content provider), which is expected. |
| 2 | Search the same tree for `PM4 Actions` | **2 hits, neither a popup:** the deletion comment at `WorldObjectsPanelService.cs:209` and the payload section name at `InspectorPayloadsService.cs:74`. |
| 3 | `ViewerUiManager.DrawDiagnosticsWindows` window set | Exactly `Log` and `Performance` (`ViewerUiManager.cs:315-331`). |
| 4 | `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:m -clp:ErrorsOnly` | **exit 0**, `Build succeeded. 494 Warning(s) 0 Error(s)` (2026-10-21). |
| 5 | `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | **exit 0**, `UI_REFERENCE_CHECK inventory_rows=123 distinct_labels=115 registry_entries=115` / `PASSED errors=0 warnings=0`. |

## 4. Criterion → evidence

| Task | Criterion | Verdict | Evidence / gap |
|---|---|---|---|
| **T070** | Delete the 12 Tools > Panels floating windows that duplicate destination pages | **Done (structural)** | §1.1: 0 matches for all 12 wrapper names; check 1. |
| **T071** | Keep only Log and Perf as global floating windows | **Done (structural)** | §2: `DrawDiagnosticsWindows` contains exactly Log and Performance, both manager-owned with content-only providers; check 3. |
| **T072** | Delete the PM4 Actions popup; keep one PM4 entry point | **Done (structural)** | §1.2: the popup is replaced by the single `Open PM4 Workbench` button at `WorldObjectsPanelService.cs:211`; the only residual `"PM4 Actions"` string is a payload section name, not a popup. |

## 5. What this receipt does NOT prove

- **No runtime proof.** The viewer was not launched. The startup crash was fixed in code
  (`ViewerUiManager` now resolves the status provider at draw time), but startup is still
  operator-owned proof.
- **No visual proof.** That the Log and Performance windows look correct, that no duplicate
  window is still reachable from a menu, and that the single PM4 button opens the intended page
  are visual/interaction facts.
- **No real-client proof.** No PM4 workflow was executed.
- **No claim about the payload section label.** `InspectorPayloadsService.cs:74` still names a
  section "PM4 Actions"; it is not a popup and was not renamed by this task.
