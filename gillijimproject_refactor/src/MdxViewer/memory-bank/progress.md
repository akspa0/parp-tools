# Progress — MdxViewer

Last updated: 2026-07-04
Keep last-week compatibility truth only. Older milestones moved to `memory-bank/archive/2026-07-04-older-history.md`.

## 2026-07-04

- Added bottom display bar in `ViewerApp_Sidebars.cs`; terrain/world toggles no longer need duplicate homes.
- Reworked top toolbar into launcher strip for minimap, terrain workbench, PM4 workbench, and capture automation.
- Replaced duplicate workspace toggles with window-launch actions.
- Restored omitted PM4 windows by wiring `DrawPm4ObjectMatchWindow()` and `DrawPm4WmoCorrelationWindow()` back into `DrawUI()` and `Tools`.
- Attempted validation used `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`, but broad missing refs outside this slice still block compile proof.

## Current handoff

- `MdxViewer` stays compatibility only.
- Use `wow-viewer` for new ownership.
- Use archive file for Apr/May history.
