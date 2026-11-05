# Spec 292 Amendment — Bottom Toolbar Reversion Receipt

**Date:** 2026-11-05  
**Scope:** Operator amendment to restore nine scene toggles plus M2/MDX and WMO opacity controls on the bottom toolbar; remove the Texture Masks toolbar button. Existing L0–L7/Alpha quick checks are retained. The detailed Layers panel remains available in the sidebar.

## Files changed

| File | Change |
|---|---|
| [`BottomBarService.cs`](../../src/viewer/WoWViewer/Workbench/Services/Chrome/BottomBarService.cs) | Restores Sky/Fog/Water/Doodads/WMOs/Terrain/PM4/Grid/Mute toggles, adds M2/MDX and WMO opacity sliders, removes the Texture Masks button, retains resident texture/alpha checks, and applies saved state on each scene/map activation. |
| [`ViewerSettingsService.cs`](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewerSettingsService.cs) | Persists all nine quick toggle states in `viewer_settings.json`; existing opacity persistence remains authoritative. |
| [`ViewerFooterLayoutTests.cs`](../../tests/WoWViewer.Core.Tests/ViewerFooterLayoutTests.cs) | Updates footer fit coverage for the nine quick toggles, optional L0–L7/Alpha checks, and two opacity sliders at common resolutions/scales. |
| [`ui-controls.json`](../../eng/ui/ui-controls.json), [`UI-REFERENCE.md`](../../docs/WoWViewer/UI-REFERENCE.md) | Documents bottom toggle and opacity controls; regenerated the UI reference. |
| [`spec.md`](../spec.md), [`tasks.md`](../tasks.md), [`STATUS.md`](../../STATUS.md), [`activeContext.md`](../../../memory-bank/activeContext.md), [`progress.md`](../../../memory-bank/progress.md) | Records the operator amendment, completed source/build checks, and outstanding runtime-proof gate. |

## Verification

| Command | Exit | Result |
|---|---:|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ViewerFooterLayoutTests" --nologo` | 0 | 18 passed, 0 failed, 0 skipped. |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | Inventory 171 rows / 156 distinct labels; 156 registry entries; 0 errors, 0 warnings. |
| `dotnet build wow-viewer/WoWViewer.slnx -c Debug --no-restore --nologo -v minimal` | 0 | Full solution build succeeded, 0 errors, 2 NU1902 package vulnerability warnings. |

An additional combined UI-test attempt ran `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ViewerFooterLayoutTests|FullyQualifiedName~UiControlDocumentationTests" --nologo` and exited 1: 22 passed, 1 failed. The unrelated failure is `UiControlDocumentationTests.IconActions_AreDocumentedByFullName_AndExactSourceLine` at line 73, whose expected action-bar list omits the already-present `Archaeology` icon. The registry verifier itself passes; no icon-action test/source was changed for this amendment.

## Requirement → evidence

| Requirement | Evidence |
|---|---|
| Restore nine quick scene toggles | [`BottomBarService.DrawBottom`](../../src/viewer/WoWViewer/Workbench/Services/Chrome/BottomBarService.cs) now renders Sky, Fog, Water, Doodads, WMOs, Terrain, PM4, Grid, and Mute with immediate scene effects and settings save. |
| Restore M2/MDX and WMO opacity controls | Bottom-bar sliders update `M2Renderer.GlobalOpacity` + `MdxRenderer.GlobalOpacity` together and `WmoRenderer.GlobalOpacity` independently; existing settings load/save persists values. |
| Remove Texture Masks button | The bottom bar no longer creates or opens the Texture Masks button. The Layers sidebar remains the authoritative detailed texture/alpha workbench. |
| Keep existing quick layer checks | L0–L7 (resident layers) and Alpha checks remain as secondary bottom-bar actions and are also available in Layers. |
| Persist quick toggle state | Added nine toggle values to the viewer settings DTO; `ApplyToScene` applies them when the scene/map changes and `CaptureSceneState` records live values before settings save. |
| UI inventory | `Verify-UiReference.ps1` passes with zero errors/warnings. |

## Operator gate owed

Run the viewer on a real client, verify the nine scene toggles and both opacity sliders affect rendering immediately, confirm values survive restart, and confirm the Texture Masks button is absent while L0–L7/Alpha and the Layers sidebar still work. Record the client fingerprint and screenshot/artifacts. This receipt does not claim runtime or visual proof.
