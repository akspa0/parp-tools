# Fix — Toolbar hovered-WMO doodad-set combo collapses when the mouse leaves the WMO

**Date**: 2026-09-07 · **Report**: "the drop-down to set the doodadset, disappears once you mouse
away from the boat to change the setting. very elusive and annoying"

## Root cause

The toolbar quick combo (Spec 231 FR-6, `##HoveredWmoDoodadSet`) sourced its WMO from
`_worldScene.HoveredAssetInfo` every frame. Hover follows the cursor, so moving the cursor off the
WMO to reach the dropdown retargeted (or removed) the source — the combo list collapsed
mid-interaction.

## Fix ([ViewerApp_Sidebars.cs](../../../src/viewer/WoWViewer/ViewerApp_Sidebars.cs))

- While `ImGui.IsPopupOpen("##HoveredWmoDoodadSet")` is true, the combo draws against a **frozen**
  `WmoRenderer` reference (`_hoveredWmoDoodadSetComboWmo` + source path for the tooltip) instead of
  the live hover.
- When the popup is closed, the frozen reference refreshes from the current hover and is cleared
  when nothing WMO-ish is hovered.

New god-class members: the two freeze fields + the `DrawHoveredWmoDoodadSetCombo` helper —
bug-fix state required by the interaction, noted here for the Spec 228 ledger.

## Verification

- `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` — **0 Errors** (exit 0).
- Interactive verification (hover WMO → open dropdown → move mouse to the list → change set) is
  **operator-owned input proof** — build/test output does not establish hover behavior.
