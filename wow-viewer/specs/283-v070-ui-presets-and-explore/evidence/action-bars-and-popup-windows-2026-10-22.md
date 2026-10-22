# Spec 283 action-bars and popup windows receipt — 2026-10-22

## Operator corrections implemented

- Replaced full-height data sidebars with narrow anchored edge action bars; their line-art icons have
  full accessible names and tooltips. Clicking one opens an independently closeable window backed by
  the existing Workspace, Assets, Taxi/Capture, Inspector, World, Model or PM4 content owner.
- Made the world viewport full-width behind overlays, fixed the minimap at upper-right, and fixed chat
  at lower-left independently of saved sidebar widths. Layout schema v5 ignores the unusable v4 state.
- Replaced destination-only bottom controls with a compact row retaining preset selection, spawn,
  page-window, edge-rail visibility and Options actions. The rejected Spellbook was removed entirely.
- Added an early visibility guard to the WDL map preview so false `ShowWdlPreview` state stops drawing.
- Follow-up integration review restored separate Maps, File Browser, and Diagnostics launch paths;
  reused the existing Navigator content instead of orphaning its map list and file browser.
- Taxi uses a taxi-only provider; Capture remains a separate window whose keyboard context follows
  its visibility. Viewer Pages focuses the existing inspector rather than rendering another copy.
- Options uses a titled, independently closeable window. Chat uses its full log width with no tools
  column or duplicate population-settings row. Map preview reserves footer space for Load/Cancel.
- Shared icon buttons now provide a border, hover/open feedback, and full-name tooltips. No
  two-character rail labels or Spellbook launcher remain.

## Files changed

- `src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs`
- `src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs`
- `src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs`
- `src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs`
- `src/viewer/WoWViewer/Workbench/Services/WdlPreview/WdlPreviewService.WdlPreview.cs`
- `src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Options.cs`
- `src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs`, `ViewerApp_Host.cs`, and `ViewerApp.cs`
- `src/viewer/WoWViewer/Workbench/Services/Spellbook/SpellbookService.cs` (deleted)
- `src/viewer/WoWViewer/UI/UiControlScanner.cs`
- `src/viewer/WoWViewer/UI/SharedUiWidgets.cs`
- `src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs`
- `src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs`
- `src/viewer/WoWViewer/Workbench/Services/Dialogue/FakeChatWindowService.cs`
- `src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs`
- `eng/ui/Get-UiControlInventory.ps1`, `eng/ui/ui-controls.json`, `docs/WoWViewer/UI-REFERENCE.md`
- `tests/WoWViewer.Core.Tests/WowViewer.Core.Tests.csproj`,
  `tests/WoWViewer.Core.Tests/ViewerUiLayoutTests.cs`
- `tests/WowViewer.Core.Tests/UiControlDocumentationTests.cs`
- Spec 283 `spec.md`, `plan.md`, `visual-analysis.md`, `tasks.md`, this receipt; `specs/STATUS.md`;
  `memory-bank/activeContext.md` and `memory-bank/progress.md`.

## Verification

| Command | Exit | Result |
|---|---:|---|
| `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:m -clp:ErrorsOnly` | 0 | Build succeeded; 411 warnings, 0 errors. |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~ViewerUiLayoutTests\|FullyQualifiedName~UiControlDocumentation" -v minimal` | 0 | 13 passed, 0 failed, 0 skipped. The backslash before the pipe is Markdown table escaping only. |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | 125 inventory rows, 116 distinct labels, 116 registry entries; 0 errors, 0 warnings. |

Raw logs: [build](../../../output/validation/spec283-action-bars/solution-build.log),
[tests](../../../output/validation/spec283-action-bars/focused-tests.log),
[registry](../../../output/validation/spec283-action-bars/ui-reference.log).

## Criterion → evidence

| Criterion | Evidence | Result |
|---|---|---|
| Scene remains full width; rail widths, chat and minimap geometry are independent of stale saved sidebar widths. | Passing geometry tests assert full scene width, narrow rails, fixed chat/minimap rects across different stored widths, and tool windows inside the client bounds at six sizes (800x600 through 3840x2160). | Pass (deterministic geometry, not visual proof). |
| Every icon opens the corresponding closeable tool window; its title-bar close works in the running app. | Callbacks and `ImGui.Begin(..., ref state.Open, ...)` are present in `ViewerActionBarsService`. | Static implementation only; runtime clicks remain operator validation. |
| Spawn preview closes after load/cancel/X and does not reopen while its flag is false. | `DrawWdlPreviewDialog()` returns before `Begin` when `ShowWdlPreview` is false. | Static implementation only; runtime validation remains open. |
| No Spellbook UI remains. | Spellbook code/wiring and registry launchers removed; source search returns no viewer-code hits. | Static check. |
| UI reference and application compile. | UI verifier, solution build, and full-name icon/source-line tests above. | Pass (build/static). |

## Not claimed

No desktop launch, supplied-client visual review, or interactive icon/window walk was performed in
this run. The operator's visual/runtime acceptance remains open. Converter readiness and automated
0.5.3 DBC edits are separate work and remain unimplemented/unverified.

## Subsequent operator feedback — 2026-10-22

The operator described the new UI as "very nice looking and pretty functional" and said it "feels
just right for what we're doing here." This accepts the overall scene-first/icon-bar direction and
reopens the scale/discoverability work with concrete feedback. It does not constitute a complete
per-control, per-display or client-source verification matrix, so the remaining runtime gates are
not bulk-closed. The requested View/scale controls, auto-open Maps and screenshot guide are specified
as [Spec 286](../../286-view-controls-userguide/spec.md), preserving this UI rather than redesigning it.
