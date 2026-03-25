# v0.5.0 Goal Stack Prompt

Use this prompt in a fresh planning chat when the goal is to define the next milestone after `v0.4.5`.

## Prompt

Design a concrete `v0.5.0` goal stack for `gillijimproject_refactor/src/MdxViewer`.

This plan must assume that `v0.4.5` is a stabilization-and-release milestone, while `v0.5.0` is the next milestone for meaningful capability growth once the most visible release blockers are under control.

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
- minimap hardening beyond the immediate release blocker fix
- selected-object / world-inspection quality-of-life follow-up where it materially improves debugging and iteration

The plan should also explicitly state what should stay out of `v0.5.0` unless stronger evidence appears.

## Existing Companion Planning Files

Reuse the current enhanced-renderer planning bundle where relevant:

- `plans/enhanced_renderer_plan_set_2026-03-25.md`
- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`
- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`
- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`

## Required Constraints

- Do not let `v0.5.0` become an unbounded wishlist.
- Keep release stabilization work separate from post-release fidelity growth.
- Prefer goals with a visible vertical slice and a credible validation path.
- Be explicit when a goal is still blocked on more reverse engineering.

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