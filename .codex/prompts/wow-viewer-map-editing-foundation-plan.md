---
description: "Plan the first real wow-viewer map-editing foundation slice. Use when the ask is PM4 MPRL-guided terrain improvement, saved PM4 object selections, moved-object persistence, ADT or WDT or object save support, dirty-map tracking, or editor transaction boundaries."
name: "wow-viewer Map Editing Foundation Plan"
argument-hint: "Optional editing seam, persistence seam, terrain/object problem, or save workflow to prioritize"
agent: "codex"
---

Design the first narrow editing foundation for `wow-viewer` so terrain and object edits can become real saved map changes instead of panel-local state.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`
5. `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
6. `wow-viewer/README.md`
7. `AGENTS.md`

## Goal

Define the shared service, persistence, and proof boundary for the first real editing slice in `wow-viewer`, covering terrain edits, object edits, and save/write ownership without letting GUI code own the format truth.

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

## What The Plan Must Produce

1. the first editing domain boundary in `wow-viewer`
2. the terrain-edit, object-edit, and map-save contracts
3. the dirty-state and transaction model
4. how PM4 `MPRL` evidence feeds terrain conform or placement edits
5. the file families that must be readable and writable for the first saved-map milestone
6. the first narrow vertical slice to implement
7. the real-data proof that would make that slice honest

## Deliverables

Return all items:

1. editing service-layer breakdown
2. terrain and object persistence model
3. dirty-state or transaction design
4. PM4 `MPRL` evidence role and limits
5. first save-capable milestone
6. validation plan
7. risks or open questions still blocking later editor work

## First Output

Start with:

1. the first saved-map milestone you think should exist
2. the shared services that milestone needs
3. what the milestone explicitly will not save yet
4. the smallest real-data proof that would make it credible