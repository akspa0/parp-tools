# U0c receipt — dock content and footer fit

Date: 2026-10-22
Spec: `286-view-controls-userguide` · Phase: U0c (T053–T056)
Owner: agent (source/tests/headless probe only)

## Origin

Operator's first real run of the U0/U0b build, 3440×1392 window: "none of the inspector or model
buttons do anything … the bottom bar is STILL CUT OFF … no reason for the Left Bar / Right Bar
buttons, Tab hides both … the buttons can be smaller." The U0/U0b receipts were source/test only
and the headless layout tests did not exercise ImGui, so neither defect was visible to them.

## Root causes

1. **Dock looked dead.** `DrawRightDock` drew a full-height `InvisibleButton` splitter and then
   `SameLine()` for the panel. ImGui's line height became the whole window, so only the page
   selector (the first widget) appeared on that line and every later widget was laid out below the
   window. The dock window also had no scrolling, so tall panels could not be reached.
2. **Status bar clipped.** The region was `22 × scale` px tall but held a frame-height button plus
   the shared `8 × scale` window padding top and bottom. The content was taller than the region
   and was clipped at the window edge (ultrawide scale 1.79: 39.4 px region, 14.3 px padding,
   34.0 px button row).
3. **Bottom bar buttons** were a fixed `36 × scale` with `82 × scale` minimum widths, plus two
   redundant bar-toggle buttons.

## Files changed

| File | Change |
| --- | --- |
| `src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Dock: scrolling child body, splitter submitted last as a left-edge overlay with a hover highlight. Footer windows use tight vertical padding. Bottom bar uses `ImGui.GetFrameHeight()` and label-sized widths (`MeasureFooterButton`); Left Bar / Right Bar removed. `UpdateLayout` passes the measured footer row (`MeasureFooterRow`: base font × effective font scale + frame padding). |
| `src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs` | `footerRowHeight` parameter; bottom and status bar heights = row + 2 × per-bar vertical padding (`BottomBarPadY` 4, `StatusBarPadY` 2, floor `MinFooterRow` 22). |
| `src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs` | `DrawBottomActions` uses the frame height; Left Bar / Right Bar buttons removed (`ToggleLeftSidebar`/`ToggleRightSidebar` stay: the `[` / `]` keys call them). |
| `src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs` | `AlignTextToFramePadding()` so the status text sits on the audio button's baseline. |
| `eng/ui/ui-controls.json`, `docs/WoWViewer/UI-REFERENCE.md` | Left Bar / Right Bar entries removed; reference regenerated (119 labels / 119 entries). |
| `tests/WowViewer.Core.Tests/ViewerUiLayoutTests.cs`, `ViewerFooterLayoutTests.cs` | New footer-row regressions; footer test item list no longer has the two toggle ids. |
| `specs/286-…/spec.md`, `tasks.md`, `evidence/imgui-layout-probe/*.txt` | Operator amendment (FR-022–024, SC-012), U0c tasks, probe source kept for reproduction. |

## Verification commands and results

PowerShell 7, from `wow-viewer`. The operator's viewer (PID 55936) was running and locks
`bin\Debug\net10.0`, so the normal build's copy step fails with `MSB3027`/`MSB3021` (not a C#
error). The viewer was therefore compiled into a scratch output directory.

| Command | Result |
| --- | --- |
| `dotnet test … --filter 'FullyQualifiedName~ViewerFooterLayoutTests\|…ViewerUiLayoutTests\|…ViewPreferencePolicyTests\|…UiControlDocumentationTests'` (normal build) | **Exit 1** — output copy blocked by the running viewer (`MSB3027`). Not a code failure. |
| `dotnet build ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-dependencies -v q -clp:ErrorsOnly` then `dotnet test … --no-build` with the same filter | **Exit 0.** `Passed! Failed: 0, Passed: 90, Total: 90`. The pure policy files are source-linked, so they are the code under test. |
| `dotnet build ./src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug -p:OutDir=<scratchpad>\viewer-build\` | **Exit 0.** `Build succeeded. 0 Error(s)` (viewer compiles, including the registry build check). |
| `pwsh -NoProfile -File ./eng/ui/Verify-UiReference.ps1` | **Exit 0.** `inventory_rows=128 distinct_labels=119 registry_entries=119`; `PASSED errors=0 warnings=0`. |
| `dotnet run -c Release` in the headless ImGui probe (`evidence/imgui-layout-probe/`, ImGui.NET 1.89.9.3, the version the viewer ships) | **Exit 0.** Output below. |
| `dotnet build ./WowViewer.slnx -c Debug` | **Not re-run** this pass: the same viewer lock would fail the copy step. Re-run after closing the viewer. |

### Headless ImGui probe output (3440×1392, display scale 1.79, user text 1.0×)

```
OLD dock: selector y / window bottom           314 / 1000  visible=True
OLD dock: first content line y                 992         (at the window's bottom edge, inside the bottom padding: effectively cut off)
NEW dock: selector y / window bottom           314 / 1000  visible=True
NEW dock: first content line y                 344
NEW dock: body scrolls (ScrollMaxY>0)          1721  scrolls=True
NEW dock: grip x-range vs content x            grip 1914-1925, content starts 1925  overlap=False
statusOld: button bottom / text bottom         1400.4 / 1395.0  fitsInside=False  (window bottom 1392.0)
statusNew: button bottom / text bottom         1387.6 / 1382.2  fitsInside=True   (window bottom 1392.0)
bottomNew: button bottom / window bottom       1345.2 / 1352.6  fitsInside=True
```

The probe replays the same ImGui call patterns the viewer uses; it is not the viewer. It supports
the layout claims but does not replace the operator's run.

## Criterion → evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| FR-022 dock panels show content and scroll | Probe: first content line y 992 → 344; `ScrollMaxY` 1721 in the child body; grip overlay does not overlap content | Probe PASS. **Operator run open (T057).** |
| FR-023 footer bars sized from the measured row; status fully visible | `FooterBars_AreNeverShorterThanTheirButtonRow` (1080p, 3440×1392, 4K × text 1.0×/2.2×), `FooterBars_GrowWithTextSizeAndShrinkTheSceneInstead`, `FooterBars_FallBackToTheFloorBeforeTheFontIsMeasured`; probe `statusNew fitsInside=True` vs `statusOld False` | Policy + probe PASS. **Operator run open (T057).** |
| FR-024 no Left Bar / Right Bar; standard-height buttons | Items, buttons and registry entries removed; `ImGui.GetFrameHeight()` used; probe `bottomNew` fits | Source PASS. **Operator run open (T057).** |
| SC-012 | Not claimed | **OPEN — operator (T057).** |

## Known limits

- `MeasureFooterRow` assumes the first atlas font is the one in use. The viewer source loads no
  custom font (no `AddFontFromFileTTF`/`AddFontDefault`/`PushFont` calls), so it is ImGui's default
  font; if a custom primary font is ever added, the row measurement must follow it.
- Style metrics other than the footer padding still scale with display only (see U1 receipt).
- No viewer launch, screenshot, or full-solution build was performed in this pass.
