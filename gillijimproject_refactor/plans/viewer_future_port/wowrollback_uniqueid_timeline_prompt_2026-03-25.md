# WoWRollback UniqueID Timeline Prompt

Use this prompt in a fresh planning chat when the goal is to port practical WoWRollback `UniqueID` timeline functionality into the active `MdxViewer` world scene.

## Prompt

Design a concrete implementation plan for bringing the WoWRollback `UniqueID` timeline concept into `gillijimproject_refactor/src/MdxViewer` using the current viewer UI, world-loading flow, and data-source methodology.

The plan must assume:

- the viewer already loads terrain, WMOs, MDX/M2 objects, taxi overlays, PM4 overlays, and optional Alpha-Core SQL world spawns
- WoWRollback already contains planning and pipeline material around `UniqueID` ranges, preset JSON, timelines, and viewer filtering
- the current user goal is not a separate rollback web viewer; it is an in-viewer rollback slider or labeled-range filter that can hide objects outside selected `UniqueID` bands
- the user wants the data interpreted like development sediment layers, where groups of `UniqueID` ranges expose a hidden timeline that is not uniform across every tile
- Alpha-Core SQL appears to contain more object identity data than earlier assumptions allowed, including `UniqueID` values for NPCs as well as gameobjects

## What The Plan Must Produce

1. A target architecture for viewer-side `UniqueID` timeline filtering.
2. A first practical slice for `v0.4.6`.
3. A data-ingestion and caching plan for Alpha-Core SQL.
4. A UI plan for a rollback slider, labeled ranges, or preset-based selection.
5. A tile-awareness strategy so non-uniform `UniqueID` layers can still be explained cleanly.
6. A validation plan that uses real data and does not over-claim.
7. A risk register covering data correctness, performance cost, and missing semantics.

## Required Constraints

- keep the work inside the active viewer and current data-loading workflow
- do not revive the old standalone WoWRollback web-viewer architecture as the main delivery target
- do not reparse giant SQL dumps every frame or every map reload if a built-in indexed cache is a credible option
- evaluate a built-in SQLite database or equivalent indexed local store created from Alpha-Core SQL on first import/load
- keep `UniqueID` filter logic separate from terrain decode, PM4 decoding, and unrelated renderer rewrites
- do not assume that one global slider alone is enough if tile-local ranges and labels are materially more useful
- do not claim that SQL `UniqueID` rows are trustworthy until their map/tile/model joins are actually checked in the active viewer path

## Specific Questions The Plan Must Answer

1. Where should the imported SQL data live after first load: transient memory only, viewer settings cache, SQLite database, or another indexed format?
2. How should the viewer distinguish world ADT/WDT placements from SQL-driven NPC/gameobject placements when applying `UniqueID` filters?
3. Should the first UI slice be a max-UniqueID slider, a min/max range, labeled preset ranges, or a combined approach?
4. How should the viewer communicate that some tiles simply do not contain every `UniqueID` layer?
5. What existing WoWRollback docs, presets, or pipeline outputs can be reused directly without dragging in stale viewer architecture?
6. What is the first implementation slice that proves the feature is real without requiring pathing, server emulation, or massive renderer work?

## Existing Repo Anchors To Use

- `WoWRollback/docs/presets/README.md`
- `WoWRollback/docs/pipeline/alpha-to-lk-end-to-end.md`
- `plans/development_repair_plan.md`
- `src/MdxViewer/README.md`

## Suggested Deliverable Structure

1. Current-state inventory
2. Data model and cache strategy
3. Viewer UI / UX plan
4. First implementation slice
5. Later expansion options
6. Risks and unknowns
7. Real-data validation plan

## Validation Rules

- if the first slice only compiles, say that is not enough
- require real-data validation against Alpha-Core SQL plus the fixed development terrain paths before calling the feature usable
- do not describe tile-local chronology, preset labels, or SQL joins as solved unless the viewer actually proves them on loaded data