# Post-v0.4.5 Plan Set

Use this file to choose the right prompt for the next milestone after `v0.4.5`.

## Use This When

The goal is no longer release triage for `v0.4.5`, but planning the next clean branch of work for `v0.4.6` and `v0.5.0`.

## Prompt Selection

### 1. Branch-level milestone split

Use:

- `plans/v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`

Choose this when the question is how to divide near-term `v0.4.6` work from broader `v0.5.0` work.

### 2. WoWRollback and UniqueID timeline integration

Use:

- `plans/wowrollback_uniqueid_timeline_prompt_2026-03-25.md`

Choose this when the task is porting rollback-style UniqueID filtering, timeline layering, or Alpha-Core SQL-backed placement metadata into the active viewer.

### 3. Alpha-Core SQL spawn fidelity and scene liveness

Use:

- `plans/alpha_core_sql_scene_liveness_prompt_2026-03-25.md`

Choose this when the task is making SQL NPCs / gameobjects more faithful, using more of the SQL payload, or evaluating animation/pathing possibilities.

### 4. Viewer performance recovery

Use:

- `plans/viewer_performance_recovery_prompt_2026-03-25.md`

Choose this when the task is reducing world-scene frame cost, scene-state churn, or renderer overhead before piling on more fidelity.

### 5. Enhanced terrain shader and lighting work

Use:

- `plans/enhanced_terrain_shader_lighting_prompt_2026-03-25.md`
- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`
- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`
- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`

Choose these when the task is the enhanced renderer path itself rather than broader milestone selection.

## Practical Direction

Current post-`v0.4.5` planning should generally assume:

- `v0.4.6` is the first delivery slice for visible WoWRollback/UniqueID timeline value inside the active viewer
- `v0.5.0` is the broader renderer/performance/fidelity milestone
- SQL spawn fidelity, equipment correctness, and animation/pathing ideas must be kept separate from what is already proven in the current viewer
- performance work is not optional if scene liveness and richer shader paths are going to land without turning the viewer into a worse slideshow

## Validation Rules

- planning files are not proof of implementation
- build-only validation is not enough for terrain, lighting, spawn fidelity, or pathing claims
- do not describe speculative server-like NPC behavior as already feasible unless the required data and runtime hooks are actually identified