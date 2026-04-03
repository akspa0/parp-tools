---
description: "Plan wow-viewer editor-mode UI and panel organization. Use when the ask is viewer-vs-editor workspace switching, preset panel layouts, editor tool clustering, PM4 and terrain tool surfacing, or making wow-viewer feel like the modern map viewer-editor instead of a pile of debug windows."
name: "wow-viewer Editor UI Surface Plan"
argument-hint: "Optional panel family, workspace mode, dock preset, or editing workflow to prioritize"
agent: "codex"
---

Design the `wow-viewer` editor UI surface so the current viewer and editing workflows become organized workspaces instead of scattered debug panels.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
5. `wow-viewer/README.md`
6. `AGENTS.md`

## Goal

Define how `wow-viewer` should expose distinct viewer and editor workspaces, preset panel layouts, and task-centered tool groupings so map editing becomes intentional and discoverable.

## Existing UI Inputs To Account For

- left and right sidebars plus docked inspectors in `MdxViewer`
- chunk clipboard panel and editor overlays
- alpha-mask import or export menus and terrain export surfaces
- PM4 workbench tabs and overlay controls
- asset catalog, minimap, capture automation, and map export workflows

## Non-Negotiable Constraints

- The new UI organization must sit on shared services; it cannot become a new source of format or save logic.
- Viewer mode and editor mode should be workspace profiles or panel presets, not separate apps.
- Editing affordances must make dirty state, current target, and save scope obvious.
- Existing useful tools such as chunk clipboard and alpha-mask import should be preserved, but reorganized around editing tasks instead of buried in generic menus.
- Avoid a generic "toolbox" dump. Group tools by task: terrain, objects, PM4 evidence, save/publish, inspection.

## What The Plan Must Produce

1. workspace or mode model for viewer vs editor
2. panel and dock preset layout by mode
3. task-grouped tool families for editing
4. selection, target, dirty-state, and save-status affordances
5. which legacy panels survive, merge, move, or die
6. the first UI-only slice to implement safely
7. risks if the current scattered UI is left in place

## Deliverables

Return all items:

1. mode or workspace model
2. panel tree per mode
3. task-grouped editor tool map
4. status or affordance design
5. legacy panel migration decisions
6. first UI slice
7. remaining UI risks

## First Output

Start with:

1. the editor tasks that most need a dedicated workspace
2. the current panels or menus that should be regrouped first
3. the minimum mode-switch model that would already improve the tool
4. what the UI slice should not try to solve yet