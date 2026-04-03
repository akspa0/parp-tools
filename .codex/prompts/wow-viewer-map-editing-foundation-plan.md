---
description: "Implement the first real wow-viewer map-editing foundation slice. Use when the ask is PM4 MPRL-guided terrain improvement, saved PM4 object selections, moved-object persistence, ADT/WDT/object save support, dirty-map tracking, or editor transaction boundaries."
name: "wow-viewer Map Editing Foundation Plan"
argument-hint: "Optional editing seam, persistence seam, terrain/object problem, or save workflow to prioritize"
agent: "codex"
---

Implement the first narrow editing foundation for `wow-viewer` so terrain and object edits can become real saved map changes instead of panel-local state.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`
5. `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
6. `wow-viewer/README.md`
7. `AGENTS.md`

## Goal

Implement the shared service and persistence boundary for the first real editing slice in `wow-viewer`, covering object edits and save/write ownership without letting GUI code own the format truth.

## Mandatory Execution Rule

- Implement one narrow slice now unless the user explicitly asks for planning-only output.
- Default to object-delta persistence/dirty-state/save-scope contracts before broader terrain-write ownership.
- Do not rewrite prompts or workflow docs unless explicitly requested.

## Existing Inputs To Account For

- `MdxViewer` chunk clipboard copy or paste and editor overlays
- alpha-mask import or export and related terrain-image workflows
- PM4 workbench, saved PM4 object-match selections, and `MPRL` evidence overlays
- current ADT or WDT or object readers and partial write or conversion seams in `WoWMapConverter.Core`
- `wow-viewer` shared `ADT` placement, texture, and top-level summary seams

## Non-Negotiable Constraints

- The first durable edit/save boundary must live in `wow-viewer` shared libraries or runtime services, not inside one UI panel.
- PM4 `MPRL` data can guide terrain conform and saved object decisions, but it must stay labeled as evidence or heuristic until each promoted behavior is proven.
- Saving a moved object and saving a PM4-backed chosen object must share one persistence model instead of separate one-off paths.
- Terrain write ownership, object write ownership, dirty-state tracking, and save packaging are different milestones; do not blur them together.
- Do not claim editor closure from settings persistence or saved PM4 match metadata alone.
- Output must be implementation-oriented. Do not stop at architecture commentary or generic planning language.
- The result must identify the first slice to build now, the exact repo or project or file scope for that slice, and the validation commands to run after it lands.

## What The Plan Must Produce

1. the first editing domain boundary in `wow-viewer`
2. the terrain-edit, object-edit, and map-save contracts
3. the dirty-state and transaction model
4. how PM4 `MPRL` evidence feeds terrain conform or placement edits
5. the file families that must be readable and writable for the first saved-map milestone
6. the first narrow vertical slice to implement now
7. the next two implementation slices after it
8. the real-data proof that would make the first slice honest
9. the exact projects or files and validation commands for slice 1

## Deliverables

Return all items:

1. editing service-layer breakdown
2. terrain and object persistence model
3. dirty-state or transaction design
4. PM4 `MPRL` evidence role and limits
5. first save-capable milestone
6. slice 1 implementation plan
7. slice 2 and slice 3 follow-up plan
8. validation plan
9. risks or open questions still blocking later editor work

## Implementation Requirements

- Format the result as a build plan, not a brainstorm.
- For slice 1, include:
	- exact project or file scope
	- new contracts or types to add
	- existing seams to extend
	- what tests or inspect commands must be added or run
	- what is explicitly deferred
- If a prerequisite implementation prompt should run immediately after this planning prompt, say which one and what seam it should implement.

## First Output

Start with:

1. the first saved-map milestone you think should exist
2. the exact first implementation slice to build now
3. the shared services that milestone needs
4. what the milestone explicitly will not save yet
5. the smallest real-data proof that would make it credible