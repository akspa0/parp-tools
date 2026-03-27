# v0.5.0 Goal Stack Prompt

Use this prompt in a fresh planning chat when the goal is to define the `v0.5.0` goal stack specifically, after the near-term `v0.4.6` branch plan has already been separated.

## Prompt

Design a concrete `v0.5.0` goal stack where the production viewer/tooling stack moves out of `parp-tools` into `https://github.com/akspa0/wow-viewer` built around one canonical shared library.

This plan must assume that `v0.4.5` is the stabilization-and-release milestone, `v0.4.6` is the first post-release feature slice in the current repo, and `v0.5.0` is the milestone where the shipping codebase breaks into `wow-viewer` because `parp-tools` has outgrown its role as the release home.

## The Goal Stack Must Produce

1. A short list of top-level `v0.5.0` goals.
2. A separation between core goals and stretch goals.
3. Dependencies between those goals.
4. Clear boundaries between `v0.4.5` deferred work and genuine `v0.5.0` work.
5. A validation expectation for each goal.

## Current Direction To Include

The plan should treat the following as strong `v0.5.0` candidates:

- new production repo layout and migration strategy for `wow-viewer`
- top-level repo structure where the main renderer app has one obvious home and the supporting libraries/tools are clearly separated
- canonical shared library that absorbs domain logic now scattered across `MdxViewer`, `WoWMapConverter.Core`, `gillijimproject-csharp`, rollback/reference slices, and other imported/internal project code
- explicit viewer split so the viewer becomes a consumer of that shared library instead of the home of format/runtime truth
- tool/CLI split so converters, exporters, and inspection tools ride the same contracts as the viewer
- explicit policy that upstream externals such as `Warcraft.NET`, `DBCD`, `WoWDBDefs`, and `Alpha-Core` stay under `libs/` and track original repos where practical, while our own parsing/writing/conversion logic is rebuilt into first-party ownership
- explicit bootstrap/dependency policy for upstream externals and support repos, including `WoWTools.Minimaps`, `SereniaBLPLib`, and automatic cloning of `wow-listfile`
- deeper performance overhaul beyond any `v0.4.6` emergency slice, planned against the new architecture instead of endless local surgery in `parp-tools`
- enhanced renderer architecture and mode split only if it is anchored to the new shared-library/runtime boundaries
- enhanced terrain first slice and shader-family / lighting execution only if the migration plan identifies where those systems now belong canonically
- SQL actor fidelity / liveness follow-up only if the first correctness work, architecture split, and performance budgets are already in place

The plan should explicitly treat these as likely `v0.4.6` work instead of `v0.5.0` work unless scope grows unexpectedly:

- WoWRollback `UniqueID` range filtering inside the active viewer
- Alpha-Core SQL import caching / SQLite indexing
- first-pass SQL actor fidelity corrections
- first performance triage slice aimed at the current obvious frame-time problem

The plan should also explicitly state what should stay out of `v0.5.0` unless stronger evidence appears.

The plan may treat upstream collaboration such as extending `Noggit` / `noggit-red` with alpha-era asset support as a stretch goal, but not as the main delivery target for the milestone.

The plan may also treat targeted reuse or extension of existing upstream tooling as a stretch/secondary track, including:

- evaluating `MapUpconverter` for `3.3.5 -> later-client` outputs once Alpha data can be safely brought up to `3.3.5`
- evaluating an enhanced `ADTMeta` path if it materially helps the new metadata/runtime pipeline
- continuing to learn from `wow.export` and `wow.tools.local` where they remain useful references or narrow integration seams

## Existing Companion Planning Files

Reuse the current enhanced-renderer planning bundle where relevant:

- `plans/post_v0_4_5_plan_set_2026-03-25.md`
- `plans/v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
- `plans/v0_5_0_new_repo_library_migration_prompt_2026-03-25.md`
- `plans/unified_format_io_overhaul_prompt_2026-03-23.md`
- `plans/enhanced_renderer_plan_set_2026-03-25.md`
- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`
- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`
- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`

## Required Constraints

- Do not let `v0.5.0` become an unbounded wishlist.
- Keep release stabilization work and `v0.4.6` foundation work separate from post-release fidelity growth.
- Prefer goals with a visible vertical slice and a credible validation path.
- Be explicit when a goal is still blocked on more reverse engineering.
- Treat `parp-tools` as the R&D repo, not the long-term release repo for this milestone.
- Prefer a repo shape where the shipping renderer app lives in one obvious location and the library/tool/dependency boundaries are readable at a glance.
- Distinguish domain logic to absorb into the new first-party library from commodity dependencies that can remain external.
- Assume the first-party read/parse/write/convert stack is being rebuilt into our own library, not preserved as a patchwork of current internal base libraries.
- Do not let speculative server-like NPC pathing dominate the milestone unless the prerequisite data and performance work are already solved.
- Do not define `v0.5.0` as “move repos and also finish every renderer dream” in one pass.

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