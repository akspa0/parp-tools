# v0.4.6 / v0.5.0 Roadmap Prompt

Use this prompt in a fresh planning chat when the goal is to split the next clean branch of work into a realistic `v0.4.6` milestone and a broader `v0.5.0` milestone.

## Prompt

Design a concrete roadmap for the next branch after `v0.4.5` in `gillijimproject_refactor/src/MdxViewer`.

This roadmap must assume:

- `v0.4.5` was the stabilization and release milestone
- `v0.4.6` should deliver the first visible post-release feature slice without becoming a second giant rewrite
- `v0.5.0` should carry the larger renderer / performance / fidelity expansion once the foundations for that work are in place
- the work is happening on a clean follow-on branch rather than directly on `main`

## The Roadmap Must Produce

1. A short statement of milestone intent for `v0.4.6`.
2. A short statement of milestone intent for `v0.5.0`.
3. A list of core goals for each milestone.
4. Dependencies between those goals.
5. A clear boundary between what belongs in `v0.4.6` and what must wait for `v0.5.0`.
6. Validation expectations for each goal.
7. Explicit non-goals for both milestones.

## Current Direction To Include

The roadmap should treat the following as high-priority candidates:

### Strong `v0.4.6` candidates

- port the practical WoWRollback `UniqueID` timeline feature into the active viewer UI
- add a rollback slider or labeled range filter so users can hide objects outside chosen `UniqueID` bands
- use Alpha-Core SQL data more fully, including built-in cached ingestion instead of reparsing giant SQL dumps every session
- investigate a first-run SQL to SQLite cache or equivalent built-in indexed data store
- tighten SQL-driven NPC / gameobject fidelity enough to stop obviously wrong all-gear / all-options presentation
- land a first performance recovery slice that reduces the current sub-30 FPS problem on real world scenes

### Strong `v0.5.0` candidates

- enhanced terrain shader and lighting path
- renderer architecture split between historical and enhanced modes
- deeper performance work beyond the first emergency recovery slice
- scene liveness follow-up for NPCs if the required data actually exists
- more ambitious shader-family and lighting rollout for terrain, WMO, models, and liquids

## Required Constraints

- do not let `v0.4.6` become a shadow `v0.5.0`
- do not let `v0.5.0` become an unbounded wishlist
- keep WoWRollback viewer integration on the active viewer UI/data-loading path instead of reviving a separate legacy viewer stack
- do not assume Alpha-Core SQL alone contains enough information to solve equipment, animation, or pathing until that is verified
- do not promise server-emulation or PM4-driven NPC navigation in `v0.4.6` unless a much smaller proven slice is identified first
- performance recovery must be treated as a first-class dependency, not a nice-to-have after fidelity work

## Existing Companion Planning Files

Reuse these where relevant:

- `plans/post_v0_4_5_plan_set_2026-03-25.md`
- `plans/wowrollback_uniqueid_timeline_prompt_2026-03-25.md`
- `plans/alpha_core_sql_scene_liveness_prompt_2026-03-25.md`
- `plans/viewer_performance_recovery_prompt_2026-03-25.md`
- `plans/enhanced_terrain_shader_lighting_prompt_2026-03-25.md`

## Suggested Deliverable Structure

1. Milestone split
2. `v0.4.6` core goals
3. `v0.5.0` core goals
4. Shared dependencies
5. Validation expectations
6. Explicit non-goals

## Validation Rules

- if a milestone item only has build validation, say that is not enough
- for terrain, lighting, spawn fidelity, and pathing work, require real-data runtime validation before describing the item as complete
- do not treat old archived plans as proof that a feature is already implemented in the active viewer