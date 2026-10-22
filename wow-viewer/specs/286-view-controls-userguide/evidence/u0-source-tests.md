# U0 source/test receipt — non-scrolling footer and FPS disclosure

Date: 2026-10-22
Spec: `286-view-controls-userguide`
Phase: U0 (T001–T007)
Owner: agent (source/tests only)

## Scope

U0 only: the bottom action bar and status bar fit their actual client region, neither footer
region scrolls, and runtime detail is disclosed on FPS hover from the counters the status bar
already collects. No U1–U5 work, no new debug sidebar, no second telemetry pipeline.

## Files changed

| File | Change |
| --- | --- |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerFooterLayout.cs` | New pure fit policy: measured item placement, compact-before-hide, essential items never hidden, status right-alignment, bounded disclosure rect. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/MinimapStatus/FooterDiagnosticsSnapshot.cs` | New immutable projection of existing status counters; compact line plus full detail lines. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/MinimapStatus/FooterDiagnosticsPresenter.cs` | New bounded FPS-hover tooltip; reads the snapshot only. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Bottom bar and status bar now use `NoScrollbar \| NoScrollWithMouse`; `DrawBottom` measures and places every item through the fit policy. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs` | `DrawBottomActions` draws the four actions at fitted placements; literal labels retained for the control scanner. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs` | `DrawStatusContent` right-aligns inside the actual content width instead of `io.DisplaySize.X`; builds the snapshot; removed the now-dead `BuildCompactRuntimeStatus`. |
| `wow-viewer/tests/WowViewer.Core.Tests/ViewerFooterLayoutTests.cs` | New regressions for fit, compaction, hiding, status alignment, disclosure bounds and snapshot content. |
| `wow-viewer/tests/WowViewer.Core.Tests/ViewerUiLayoutTests.cs` | Added footer-region containment theory at 1080p/1440p/4K. |
| `wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj` | Source-links the two new pure files into the headless test project. |

## Verification commands and results

| Command | Result |
| --- | --- |
| `dotnet build ./WowViewer.slnx -c Debug` | **Blocked by environment, not code.** The viewer project compiled with no C# errors; the build then failed at the copy step because the operator's running viewer (`ParpToolsWoWViewer`, PID 31760) holds `bin\Debug\net10.0\*.dll`. Errors were `MSB3027`/`MSB3021` file locks only. |
| `dotnet build ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-dependencies -v q -clp:ErrorsOnly` | **Exit 0.** `Build succeeded. 0 Error(s)`. This compiles the two new pure files and the new tests against the existing outputs. |
| `dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~ViewerFooterLayoutTests\|FullyQualifiedName~ViewerUiLayoutTests"` | **Exit 0.** `Passed! - Failed: 0, Passed: 28, Skipped: 0, Total: 28`. |
| `pwsh -NoProfile -File ./eng/ui/Verify-UiReference.ps1` | **Exit 0.** `inventory_rows=125 distinct_labels=116 registry_entries=116`; `PASSED errors=0 warnings=0`. |

The full-solution build must be re-run by the operator after closing the running viewer. The
source compiles; only the output copy is locked.

## Criterion → evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| FR-014 — footer bars expose no horizontal/vertical scrollbars; essential controls fit the actual region | `ViewerUiManager.cs` passes `NoScrollbar \| NoScrollWithMouse` to both footer regions; `ViewerFooterLayout.Fit` keeps every essential item visible and bounds content to `AvailableWidth`; `BottomBar_FitsInsideItsRegionAtCommonResolutions`, `BottomBar_FitsAtDefaultAndCustomEffectiveScales`, `NarrowRegion_CompactsBeforeHidingAndKeepsEssentials`, `SecondaryItems_AreHiddenBeforeEssentials` pass. | Source/test PASS. Runtime visual proof open (T008). |
| FR-015 — runtime/debug detail readable in an FPS-hover tooltip; reuse existing diagnostics | `FooterDiagnosticsSnapshot` projects existing counters; `FooterDiagnosticsPresenter.DrawHoverDisclosure` draws them in a size-constrained tooltip; `Snapshot_CompactLineKeepsFpsAreaAndPending` and `Snapshot_WithoutSceneKeepsFpsReachableAndShowsStatusMessage` pass. | Source/test PASS. Runtime hover proof open (T008). |
| FR-018 — contextual disclosure reuses existing data/primitives, stays inside client bounds, respects input ownership | Disclosure reads the same snapshot; `DisclosureRect_StaysInsideTheClientArea` bounds the rect at 1080p/1440p/4K; the tooltip is drawn from the status region, which already owns UI input. | Source/test PASS. Runtime proof open (T008). |
| SC-008 — operator confirms reachable FPS, pending counts and no scrollbars | Not claimed. Requires the running viewer. | **OPEN — operator-owned (T008).** |
| SC-010 — no footer input leakage into the scene | Not claimed. Requires the running viewer. | **OPEN — operator-owned (T008).** |

## Notes

- The FPS disclosure is a hover affordance on the existing FPS readout, not a new button. The
  control registry therefore stays unchanged; `Verify-UiReference.ps1` reports zero drift and
  zero warnings. Adding a readout row would create a stale-entry warning because the scanner
  only inventories interactive controls.
- The four bottom actions keep literal labels so `Get-UiControlInventory.ps1` continues to find
  `Spawns`, `Pages`, `Left Bar` and `Right Bar`.
- No runtime, visual, FPS, audio or input proof is claimed from build or unit tests.
