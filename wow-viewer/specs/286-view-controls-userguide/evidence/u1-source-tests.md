# U1 source/test receipt — one text-scale owner and View routes

Date: 2026-10-22
Spec: `286-view-controls-userguide`
Phase: U1 (T009–T015)
Owner: agent (source/tests only)

## Scope and gate note

U1 only: one owner for the user's interface text size, View menu routes to existing visual
settings, and three explicitly named resets. No renderer, camera, converter or U2–U5 work.

**Gate note (operator direction, 2026-10-22):** the tasks list T008 (U0 operator runtime proof)
as a stop gate before U1. The operator instructed "read memory bank files and continue
implementation work" with T008 still open, so U1 source work proceeded. **T008 remains unchecked
and open** (SC-008/010/011 unproven); T016 below is the next gate and U2 does not start before it.

## Files changed

| File | Change |
| --- | --- |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Settings/ViewPreferencePolicy.cs` | New pure policy: supported range 0.85–2.20, default 1.0, damaged-value fallback (≤0.5 / non-finite → 1.0, as the legacy loader did), `NormalizePersisted` with a `corrected` flag, `EffectiveFontScale(user, display)`, the reset truth table (`ViewResetKind` → `ViewResetEffect`), presets, percent formatting. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Settings/ViewPreferencesService.cs` | New owner over the existing `IViewerAppHost.UiFontScale` value: `SetUserScale`, `Commit`, `LoadPersisted` (logs a correction once), `Reset(ViewResetKind)`. Never touches ImGui. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Owns the service (`ViewPreferences`); font composition now calls `ViewPreferencePolicy.EffectiveFontScale` (same user × display clamp 0.75–4.0 as before). No new `ViewerApp`/`IViewerAppHost` members. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Settings/ViewerSettingsService.cs` | Load normalises through the owner and **no longer writes `ImGui.GetIO().FontGlobalScale` directly**. Save is unchanged (`UiFontScale = _uiFontScale`). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Settings/SettingsWindowService.cs` | The text-size slider and presets go through the owner; the five hand-copied preset blocks (each wrote `FontGlobalScale` and saved) are one loop; added **Reset Text Size**. Slider writes the file once when the drag ends. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.View.cs` | New partial: View menu content (see below). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs` | View block now delegates to `DrawViewMenuContent()`; **Reset Shell Layout** replaced by the three named resets. |
| `wow-viewer/eng/ui/ui-controls.json`, `eng/ui/New-UiControlRegistry.ps1`, `docs/WoWViewer/UI-REFERENCE.md` | Registry: removed **Reset Shell Layout**; added **Reset Text Size Only**, **Reset Layout Only**, **Reset Layout and Text Size**, **Text Scale**, **Visual Options...**, **Camera and Fog...**. Reference regenerated with `Update-UiReference.ps1 -Write`. |
| `wow-viewer/docs/WoWViewer/USERGUIDE.md` | One sentence: the View menu paragraph no longer names the removed item. Full guide work stays in U5 (T041). |
| `wow-viewer/tests/WowViewer.Core.Tests/ViewPreferencePolicyTests.cs`, `WowViewer.Core.Tests.csproj` | New policy tests; the pure policy file is source-linked like the U0 files. |

### View menu as built

`Reset Camera`, `Hide UI Chrome`, `Left Sidebar`, `Right Sidebar` (unchanged) · **Visual Options...**
(opens the existing Options window: Display → render quality/lighting/theme/minimap, Overlays,
Layers, Grids, Objects, Audio, Simulation) · **Camera and Fog...** (opens the existing Workspace
window) · **Text Size ▸** slider `Text Scale` + 100/120/135/150/175% presets with the current one
checked · **Reset Text Size Only** · **Reset Layout Only** · **Reset Layout and Text Size**.
Every item is at most one submenu deep and delegates to the existing owner; the menu stores no value.

## Verification commands and results

Run from `wow-viewer`, PowerShell 7. Failures are listed because they happened.

| # | Command | Result |
| --- | --- | --- |
| 1 | `dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~ViewPreferencePolicyTests\|FullyQualifiedName~ViewerUiLayoutTests'` (first run) | **Exit 1.** The viewer project's UI-reference build check failed: the generated `UI-REFERENCE.md` was stale after the registry edit. Expected; fixed by command 2. |
| 2 | `pwsh -NoProfile -File ./eng/ui/Update-UiReference.ps1 -Write` | Exit 0, wrote `docs/WoWViewer/UI-REFERENCE.md`. |
| 3 | same `dotnet test` as 1 | **Exit 1.** `CS0051 Inconsistent accessibility` — the theory took the internal `ViewResetKind`. Test rewritten as a fact. |
| 4 | same `dotnet test` as 1 | **Exit 0.** `Passed! Failed: 0, Passed: 60, Total: 60`. |
| 5 | `pwsh -NoProfile -File ./eng/ui/Verify-UiReference.ps1` (R) | **Exit 0.** `inventory_rows=130 distinct_labels=121 registry_entries=121`; `PASSED errors=0 warnings=0`. |
| 6 | `dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~ViewerFooterLayoutTests\|FullyQualifiedName~ViewerUiLayoutTests\|FullyQualifiedName~ViewPreferencePolicyTests\|FullyQualifiedName~UiControlDocumentationTests'` | **Exit 0.** `Passed! Failed: 0, Passed: 82, Total: 82` (U0 footer regressions and UI documentation tests still pass). |
| 7 | `dotnet build ./WowViewer.slnx -c Debug` (B) | **Exit 0.** `Build succeeded. 0 Error(s)`, 433 warning(s) (pre-existing categories: CS0649/CS0436/CS8602 etc.; not triaged here). |

## Criterion → evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| FR-001 — View is the home for camera, presentation, interface visibility and scale; no Settings detour | `MainMenuBarService.View.cs`; registry rows under `Top menu > View` | Source PASS. Visible reachability open (T016). |
| FR-002 — View shows and adjusts the current text scale | `Text Scale` slider reads `ViewPreferencesService.UserScale`, applies each drag step | Source PASS. Appearance open (T016). |
| FR-003 — scale reset restores the user adjustment, keeps display-aware sizing, keeps the client | `ScaleReset_RestoresTheUserDefaultButKeepsDisplayAwareSizing` (1080p/1440p/4K: effective scale after reset == display scale); `Reset` never touches the data source or world | Policy PASS. Loaded-scene retention open (T016). |
| FR-004 — layout-only, scale-only and combined resets are distinct and named | `Resolve_NamesExactlyWhatEachResetChanges`; three menu labels with tooltips; `ResetLayout()` does not touch `UiFontScale` (read in `ViewerUiManager.Persistence.cs`, unchanged) | Source/policy PASS. Exact on-screen effects open (T016). |
| FR-005 — one setting, no conflicting copies | View slider, Settings slider/presets and the loader all use `ViewPreferencesService` over the one `UiFontScale` field; direct `FontGlobalScale` writes removed from Settings and the loader | Source PASS (grep: only `ViewerUiManager` writes `FontGlobalScale`). |
| FR-006 — consistent across menus, tool windows and restart; legible at supported scales | `NormalizeUserScale`/`NormalizePersisted` tests; `EffectiveFontScale` tests at 1080p/1440p/4K incl. the 4.0 ceiling | Policy PASS. **Restart persistence and legibility/overlap at 0.85–2.20 open (T016).** |
| FR-013 — accepted scene-first model preserved | No change to icon bars, tooltips, rails, chat or the dock; no Spellbook/abbreviated labels | Source PASS. |
| SC-001 — scale, three resets and visual groups reachable within View + one submenu | Menu structure above | Source PASS. **OPEN — operator (T016).** |
| SC-002 — resets produce exactly their named effects, keep client/map, persist across restart | Truth table test | Policy PASS. **OPEN — operator (T016).** |
| SC-003 — no scale-induced overlap/clipping at 1080p/1440p/4K | Not claimed | **OPEN — operator (T016).** |

## Known limits and risks (for T016)

- Style metrics (item spacing, frame padding, scrollbar size) scale with the **display** factor only;
  text scales with user × display. This was already true of the old Settings slider. At user 2.20 the
  glyphs outgrow the padding; whether that clips or overlaps any tool window is exactly what T016
  must observe. No metric change was made without evidence.
- A legacy saved scale outside 0.85–2.20 (for example 0.75 or 2.5) is corrected in memory on load
  and logged once; the file is rewritten by the next normal save. No forced save at load.
- `Visual Options...` and `Camera and Fog...` open the owning windows; they do not scroll to or
  collapse individual sections (all Options sections are open by default).
- Build warnings were not triaged; the 0-error build is the assertion.
- No viewer launch, no client, no screenshots, no display/DPI run was performed.
