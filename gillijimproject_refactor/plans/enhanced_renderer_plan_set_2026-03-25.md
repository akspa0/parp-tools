# Enhanced Renderer Plan Set

Use this file as the entry point for Copilot planning work on enhanced terrain shading, shader-family reconstruction, and lighting-model expansion in `MdxViewer`.

## Plan Set

### 1. Master Brief

Start here when the session needs the full problem frame:

- `plans/enhanced_terrain_shader_lighting_prompt_2026-03-25.md`

Use it when the task is broad and needs architecture, first-slice, risk, and validation guidance in one pass.

### 2. Architecture Prompt

Use this when the session should focus on runtime boundaries, ownership, and mode separation:

- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`

### 3. First Slice Prompt

Use this when the session should produce a concrete landable implementation plan for enhanced terrain only:

- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`

### 4. Shader / Lighting Roadmap Prompt

Use this when the session should plan the follow-on work after the first terrain slice:

- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`

## Suggested Copilot Workflow

1. Start with the master brief if the session has no context.
2. Use the architecture prompt to lock down `Historical` versus `Enhanced` boundaries.
3. Use the first-slice prompt to produce a concrete terrain-only implementation plan.
4. Use the shader / lighting roadmap prompt after the first slice is stable enough to plan the next stages.

## Guardrails

- Keep the historical renderer intact.
- Keep terrain decode work separate from terrain shading work.
- Do not claim parity with Blizzard clients based only on translated shaders.
- Require real-data validation for terrain-facing claims.
- Treat build-only validation as build-only.