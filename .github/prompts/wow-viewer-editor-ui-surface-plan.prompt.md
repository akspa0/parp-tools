---
description: "Implement wow-viewer editor-mode UI and panel organization in narrow slices. Use when the ask is viewer-vs-editor workspace switching, preset panel layouts, editor tool clustering, PM4/terrain tool surfacing, or making wow-viewer feel like a modern map viewer-editor."
name: "wow-viewer Editor UI Surface Plan"
argument-hint: "Optional panel family, workspace mode, dock preset, or editing workflow to prioritize"
agent: "agent"
---

Implement the `wow-viewer` editor UI surface in narrow, validated slices so the current viewer and editing workflows become organized workspaces instead of scattered debug panels.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
5. `wow-viewer/README.md`
6. `.github/copilot-instructions.md`

## Goal

Implement the first safe UI slice for distinct viewer/editor workspaces and task-centered tool groupings, and leave save/write logic in shared services.

## Mandatory Execution Rule

- Implement one safe UI slice now unless the user explicitly asks for planning-only output.
- If required editing services do not exist yet, implement scaffolding only and call out the dependency.
- Do not rewrite prompts or workflow docs unless explicitly requested.

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
- Output must be implementation-oriented. Do not stop at generic UX commentary or abstract workspace ideas.
- The result must identify the first UI slice to build now, the exact repo or file scope for that slice, and what shared service or runtime seams must already exist before that slice is safe.

## What The Plan Must Produce

1. workspace or mode model for viewer vs editor
2. panel and dock preset layout by mode
3. task-grouped tool families for editing
4. selection, target, dirty-state, and save-status affordances
5. which legacy panels survive, merge, move, or die
6. the first UI-only slice to implement safely now
7. the next UI slice after that
8. exact repo or file scope and validation for slice 1
9. risks if the current scattered UI is left in place

## Deliverables

Return all items:

1. mode or workspace model
2. panel tree per mode
3. task-grouped editor tool map
4. status or affordance design
5. legacy panel migration decisions
6. slice 1 implementation plan
7. slice 2 follow-up plan
8. validation or proof plan
9. remaining UI risks
10. exact files changed in this chat for slice 1
11. exact validation commands run in this chat

## Implementation Requirements

- Format the result as a build plan, not a brainstorm.
- For slice 1, include:
	- exact project or file scope
	- the panels or menus to change
	- the state or service contracts the UI will consume
	- what should stay out of scope
	- how the slice will be validated
- If the requested UI slice depends on editing services that do not exist yet, say that clearly and limit slice 1 to safe UI scaffolding rather than inventing fake save logic.

## First Output

Start with:

1. the editor tasks that most need a dedicated workspace
2. the current panels or menus that should be regrouped first
3. the exact first UI slice to build now
4. the minimum mode-switch model that would already improve the tool
5. what the UI slice should not try to solve yet