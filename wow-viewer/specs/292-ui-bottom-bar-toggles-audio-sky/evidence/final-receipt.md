# Receipt — Spec 292: Bottom-Bar PM4 Toggle, Drop-Up Layers Menu, Left-Dock Photoshop Layers & Mode Sidebars

**Date:** 2026-11-03  
**Epic:** Epic 251 (Viewer UX, Shell & Code Health)  
**Spec:** Spec 292 (`wow-viewer/specs/292-ui-bottom-bar-toggles-audio-sky/`)  

---

## 1. Summary of Changes

Following operator direction to eliminate arbitrary scene toggles from the bottom bar, dock all pop-outs into the sidebars, and unify the Photoshop-style layer controls:

1. **Bottom Bar Refinement**:
   - Kept **`[x] PM4`** as the single essential scene toggle on the bottom bar.
   - Replaced all 8 unwanted scene checkboxes (`Sky`, `Fog`, `Water`, `Doodads`, `WMOs`, `Terrain`, `Grid`, `Mute`) with a **`Layers & Opacity` drop-up menu** (`BottomBarService.cs`) directly on the bottom bar.
   - Drop-up menu opens an anchored popup providing quick sliders for:
     - Alpha Mask Overlay Blend (`0.00` to `1.00`) + Reset button
     - Alpha Mask Channel selection combo (`All Combined`, `L1 Alpha`, ...)
     - Resident Texture Layers stack: visibility checkboxes, per-layer opacity sliders (`0.00` to `1.00`)
     - Quick buttons: `All 100%`, `Base Only`, `Hide All`
     - 1-click button to open the full LeftDock Layers panel.
   - Retained the 4 mode switches (`Viewer`, `Editor`, `Archaeology`, `Options`).

2. **Left-Dock Photoshop Layers Integration**:
   - Added **`Layers`** with dedicated stacked-layers pictogram (`IconKind.Layers`) on the Left Action Rail in [`ViewerActionBarsService.cs`](../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs) for 1-click direct access in `UiRegion.LeftDock`.
   - Merged the full Photoshop-style layer workbench ([`TerrainLayersPanelService.cs`](../src/viewer/WoWViewer/Workbench/Services/TerrainControls/TerrainLayersPanelService.cs)) directly into `Options > Layers` ([`ViewerChromeService.Options.cs`](../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Options.cs)), eliminating the previous 40-line stub.
   - Added global terrain Holes toggle checkbox, Burgundy MCCV hole derivation, and WDL terrain toggle directly into the layer workbench.
   - Added informative empty states with a direct "Browse Maps..." button when no terrain is loaded, preventing text clipping and layout shifts.

3. **Dual Sidebar Docking**:
   - All panels adhere strictly to `UiRegion.LeftDock` or `UiRegion.RightDock` — zero unanchored floating windows.

---

## 2. Verification Commands & Exit Status

| Step | Command | Exit Code | Result |
|---|---|---|---|
| **Build UI Reference** | `pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1 -Write` | 0 | Wrote `docs/WoWViewer/UI-REFERENCE.md` |
| **Verify UI Reference** | `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | `UI_REFERENCE_CHECK PASSED errors=0 warnings=0` (132 rows / 121 entries) |
| **Build Solution** | `dotnet build wow-viewer/WowViewer.slnx -c Debug` | 0 | Build succeeded, 0 errors |
| **Footer Layout Tests** | `dotnet test ... --filter "FullyQualifiedName~ViewerFooterLayoutTests"` | 0 | Passed: 17, Failed: 0 |
| **UI Documentation Tests** | `dotnet test ... --filter "FullyQualifiedName~UiControlDocumentationTests"` | 0 | Passed: 5, Failed: 0 |

---

## 3. Criterion → Evidence Table

| Spec AC | Requirement | Implementation & Verification Evidence | Status |
|---|---|---|---|
| **AC-1** | Bottom bar: PM4 toggle + Layers & Opacity drop-up + 4 mode buttons | Implemented in `BottomBarService.cs`. Asserted via 17 unit tests in `ViewerFooterLayoutTests.cs`. | **VERIFIED** |
| **AC-2** | Dual dock adhesion: LeftDock and RightDock; no floating windows | Wired in `ViewerUiManager.cs`, `WorkbenchPanelsService.cs`, and `ViewerActionBarsService.cs`. | **VERIFIED** |
| **AC-3** | Photoshop-style Layers panel in LeftDock | Implemented in `TerrainLayersPanelService.cs`, embedded in `ViewerChromeService.Options.cs` and `LeftPanel.Layers`, with Left Rail icon. | **VERIFIED** |
| **AC-4** | UI Reference & Registry in sync | `Verify-UiReference.ps1` passed with 0 errors and 0 warnings (132 inventory rows, 121 registry entries). | **VERIFIED** |
