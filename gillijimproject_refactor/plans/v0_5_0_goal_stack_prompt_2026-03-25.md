# v0.5.0 Goal Stack Prompt

Use this prompt in a fresh planning chat when the goal is to define the `v0.5.0` goal stack specifically, after the near-term `v0.4.6` branch plan has already been separated.

## Prompt

Design a concrete `v0.5.0` goal stack for `gillijimproject_refactor/src/MdxViewer`.

This plan must assume that `v0.4.5` is the stabilization-and-release milestone, `v0.4.6` is the first post-release feature slice, and `v0.5.0` is the broader milestone for meaningful capability growth once the most visible post-release foundation work is underway.

## The Goal Stack Must Produce

1. A short list of top-level `v0.5.0` goals.
2. A separation between core goals and stretch goals.
3. Dependencies between those goals.
4. Clear boundaries between `v0.4.5` deferred work and genuine `v0.5.0` work.
5. A validation expectation for each goal.

## Current Direction To Include

The plan should treat the following as strong `v0.5.0` candidates:

- enhanced renderer architecture and mode split
- enhanced terrain first slice
- shader-family and lighting roadmap execution
- deeper viewer performance recovery beyond any `v0.4.6` emergency slice
- SQL actor fidelity / liveness follow-up only if the first correctness work and performance budgets are already in place
- minimap hardening beyond the immediate release blocker fix
- selected-object / world-inspection quality-of-life follow-up where it materially improves debugging and iteration

The plan should explicitly treat these as likely `v0.4.6` work instead of `v0.5.0` work unless scope grows unexpectedly:

- WoWRollback `UniqueID` range filtering inside the active viewer
- Alpha-Core SQL import caching / SQLite indexing
- first-pass SQL actor fidelity corrections
- first performance triage slice aimed at the current obvious frame-time problem

The plan should also explicitly state what should stay out of `v0.5.0` unless stronger evidence appears.

## Existing Companion Planning Files

Reuse the current enhanced-renderer planning bundle where relevant:

- `plans/post_v0_4_5_plan_set_2026-03-25.md`
- `plans/v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
- `plans/enhanced_renderer_plan_set_2026-03-25.md`
- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`
- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`
- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`

## Required Constraints

- Do not let `v0.5.0` become an unbounded wishlist.
- Keep release stabilization work and `v0.4.6` foundation work separate from post-release fidelity growth.
- Prefer goals with a visible vertical slice and a credible validation path.
- Be explicit when a goal is still blocked on more reverse engineering.
- Do not let speculative server-like NPC pathing dominate the milestone unless the prerequisite data and performance work are already solved.

## Suggested Deliverable Structure

1. Milestone intent
2. Core goals
3. Stretch goals
4. Dependency map
5. Validation expectations
6. Explicit non-goals

## Validation Rules

- If a goal only has build validation, say that is not enough.
- For terrain, lighting, and shader work, require real-data runtime validation before describing the goal as complete.