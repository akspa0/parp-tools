# Receipt — Perf panel moved out of the PM4 workbench service (2026-09-27)

Operator (on being told the render switches were in the "PM4 workbench sidebar"): "How the fuck does that belong in
a PM4 feature, when it has to do with rendering of the whole fucking scene?"

Cause: the U-01 extraction (`aeecd8e`) put the Utilities > Perf content (frame history, submission counters, scene
render switches) into `Pm4WorkbenchService` because it came from `ViewerApp_Sidebars.cs`. What users see was already
Utilities > Perf; only the owning class was wrong (and the previous handoff named the class, not the UI).

## Files changed
- `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Sidebars.cs` → `Workbench/Services/Perf/PerfPanelService.FrameHistory.cs` (class renamed; content unchanged).
- `Workbench/Services/Perf/PerfPanelService.cs` (new): host bridge + `DrawPerfWindow` / `DrawPerfContent`, moved verbatim from `Pm4WorkbenchService.cs`.
- `Pm4WorkbenchService.cs` / `.Host.cs`: moved members and the now-unused `_showPerfWindow` bridge removed.
- `ViewerApp.cs`, `ViewerApp_Host.cs`, `IViewerAppHost.cs`, `WorkbenchPanelsService.cs` / `.Host.cs`: composition-root wiring (same pattern as every other service).

## Verification
| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 0 | 0 errors; compiler warnings identical to base |
| `dotnet test WowViewer.slnx -c Debug` | 1 | 26 failures, identical to the pre-existing E4 environmental set |

Behavior-preserving move; UI location unchanged (Utilities > Perf > Frame history > Submission efficiency).
