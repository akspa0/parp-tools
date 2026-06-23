# Plan 073 — UI Surface Revamp

## Sub-plan 073a: Toolbar / left sidebar dedup and alignment

**Goal:** Decide where workspace controls live and make the toolbar + left sidebar not duplicate each other.

**Approach:**
- Toolbar becomes scene/viewport chrome: cells toggle, quick actions, status-ish buttons.
- Left sidebar owns file/source/workspace chrome: Open Game Folder, Open File, map discovery, workspace bars.
- Remove duplicated "Open Game Folder" from toolbar or left sidebar (pick one).
- Fix button alignment/spacing in toolbar.

## Sub-plan 073b: Tools tab converter integration

**Goal:** Surface converter commands in the Tools tab so users don't have to drop to CLI.

**Approach:**
- Add a "Converters" sub-tab or section under Tools.
- List major converter commands (Alpha↔LK, M2↔MDX, WMO V14↔V17, Split ADT, Terrain Patch, etc.).
- Each entry opens a small in-panel form with the required paths/options and runs the CLI tool via `Process.Start` or an internal call if already referenced.

## Sub-plan 073c: Tab/sub-tab alignment and polish pass

**Goal:** All top tabs and sub-tabs align consistently; no clipped labels or weird spacing.

**Approach:**
- Audit `DrawTopTabBar`, `DrawBottomTabBar`, and sub-tab headers for alignment.
- Standardize padding/spacing constants.
- Fix any label truncation or overlapping buttons.

## Sub-plan 073d: Model/World/Terrain panel alignment polish

**Goal:** Content inside each workbench sub-tab lines up cleanly and doesn't overflow its panel.

**Approach:**
- Add `ImGui.BeginChild` wrappers where content scrolls.
- Align tables/grids to available width.
- Fix any hard-coded widths that break when sidebar width changes.

## Validation

- Build `WoWViewer.csproj` clean after each sub-plan.
- Toggle `View > Legacy UI` and verify legacy mode unchanged.
- Run app and visually confirm each changed area.
