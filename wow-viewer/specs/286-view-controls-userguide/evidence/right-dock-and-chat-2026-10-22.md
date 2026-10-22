# Right dock and collapsed chat — source/test receipt

Date: 2026-10-22
Spec: `286-view-controls-userguide`
Amendment: FR-019–021 / SC-011 (operator-directed)
Owner: agent (source/tests only)

## Operator wording

> "the inspector and any right-sided stuff should fit under the minimap and above the bottom
> bars, neatly tucked in that spot. resizeable by the user, if they want it to be resized.
> Simple and neat. The chat should start collapsed."

## Files changed

| File | Change |
| --- | --- |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs` | Added `UiRegion.RightDock`; geometry sits below the minimap, above the bottom bar, left of the icon rail; width clamps to 240–864 logical px at 1080p. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Added `_rightDockOpen` / `_rightDockWidth`; draws the dock region; left-edge splitter resizes and persists; `OpenViewerDock` / `ToggleRightDock` / `CloseRightDock`; chat defaults collapsed. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs` | Right rail toggles the dock instead of floating windows; removed the four floating `DrawViewerWindow` calls; selected state follows the active dock panel. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Overlap guard: `Navigate` closes the dock when the destination is not Viewer; `ToggleRightDock` navigates to Viewer first when invoked from another tab. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs` | Layout v6 (`ui-layout-v6.json`); persists `RightDockOpen` / `RightDockWidth`; defaults `ChatCollapsed = true`. |
| `wow-viewer/tests/WowViewer.Core.Tests/ViewerUiLayoutTests.cs` | Added dock geometry, hidden-state and width-clamp regressions. |
| `wow-viewer/specs/286-view-controls-userguide/spec.md` | Dated operator amendment FR-019–021 / SC-011. |

## Verification

| Command | Result |
| --- | --- |
| `dotnet build ./WowViewer.slnx -c Debug` | **Exit 0.** `Build succeeded.` |
| `dotnet build ./WowViewer.slnx -c Debug -v q -clp:ErrorsOnly` (after overlap guard) | **Exit 0.** `Build succeeded.` |
| `dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~ViewerFooterLayoutTests\|FullyQualifiedName~ViewerUiLayoutTests"` | **Exit 0.** `Passed! - Failed: 0, Passed: 35, Skipped: 0, Total: 35`. Re-run after the overlap guard: same 35/35. |
| `pwsh -NoProfile -File ./eng/ui/Verify-UiReference.ps1` | **Exit 0.** `inventory_rows=125 distinct_labels=116 registry_entries=116`; `PASSED errors=0 warnings=0`. |

## Criterion → evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| FR-019 — one docked right panel below minimap, above bottom bars, left of the rail | `UiLayout` `RightDock` rect; `RightDock_SitsBelowMinimapAndAboveBottomBar` at 1080p/1440p/4K; floating viewer windows removed. | Source/test PASS. Runtime visual proof open. |
| FR-020 — user-resizable, persisted width | `DrawRightDock` splitter adjusts `_rightDockWidth`; `RightDock_ClampsRequestedWidth`; layout v6 persists the value. | Source/test PASS. Runtime drag/restart proof open. |
| FR-021 — chat starts collapsed | `_chatCollapsed = true`; `ApplyDefaults` sets `true`; `SavedLayout.ChatCollapsed` defaults `true`; layout v6 resets old state. | Source/test PASS. Runtime visual proof open. |
| SC-011 | Not claimed. Requires the running viewer. | **OPEN — operator-owned.** |

## Notes

- The right icon rail remains the accepted 44 px action bar; the dock is the content surface to
  its left, so the accepted shell is preserved.
- The dock splitter is a 6 px invisible button on the dock's left edge; the cursor changes to
  `ResizeEW` while hovering or dragging.
- No runtime, visual, FPS, audio or input proof is claimed from build or unit tests.
