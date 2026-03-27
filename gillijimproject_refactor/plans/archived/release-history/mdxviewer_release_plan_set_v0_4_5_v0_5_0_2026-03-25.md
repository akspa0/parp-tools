# MdxViewer Release Plan Set: v0.4.5 And v0.5.0

Use this file as the entry point for Copilot planning sessions that need to separate immediate release stabilization from the larger post-release roadmap.

## Plan Set

### 1. v0.4.5 Release Stabilization

Use this when the session should focus on what must be true before a GitHub release feels stable enough to ship:

- `plans/v0_4_5_release_stabilization_prompt_2026-03-25.md`

### 2. Fullscreen Minimap Repair

Use this when the session should focus specifically on the fullscreen minimap bug as an active release blocker:

- `plans/fullscreen_minimap_repair_prompt_2026-03-25.md`

### 3. v0.5.0 Goal Stack

Use this when the session should plan the next milestone after `v0.4.5` without turning post-release goals into release blockers:

- `plans/v0_5_0_goal_stack_prompt_2026-03-25.md`

## Usage Order

1. Start with the `v0.4.5` stabilization prompt when the goal is shipping confidence.
2. Use the fullscreen minimap repair prompt if the session is specifically about the unresolved minimap bug.
3. Move to the `v0.5.0` goal stack only after the release blocker list is clearly separated from later work.

## Current Release Rule

- Treat the fullscreen minimap as an unresolved `v0.4.5` release blocker until it has real runtime validation.
- Do not let `v0.5.0` renderer/fidelity goals dilute the release-stability work.