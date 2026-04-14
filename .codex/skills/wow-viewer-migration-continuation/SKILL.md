# wow-viewer Migration Continuation

## When To Use

- The user wants to continue `wow-viewer` work from a prior session.
- The task is choosing the next migration slice.
- The task is updating instructions, prompts, skills, or memory-bank files for `wow-viewer`.
- The task is deciding whether the next narrow slice is PM4 work or a shared-I/O format slice.
- The task is deciding whether dataset export, terrain-supervision artifacts, manifest or harvest work should move into a new `wow-viewer` tool instead of staying in `WoWMapConverter`.
- The task is deciding how archive-backed client roots should be staged or referenced for real-data validation across multiple builds.
- The task is clarifying whether work belongs in `wow-viewer`, `gillijimproject_refactor`, or both.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`
4. `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`
5. `gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md`
6. `wow-viewer/README.md`
7. `AGENTS.md`
8. `.codex/prompts/wow-viewer-tool-suite-plan-set.md`
9. `.codex/prompts/wow-viewer-dataset-builder-plan.md`
10. `.codex/prompts/wow-viewer-pm4-library-implementation.md`
11. `.codex/prompts/wow-viewer-shared-io-implementation.md`
12. `.codex/skills/wowarchive-client-staging/SKILL.md` when the continuation depends on archive-backed client roots or staging policy

## Routing Rules

- Use `.codex/prompts/wow-viewer-pm4-library-implementation.md` when the next task is a real `Core.PM4` implementation slice.
- Use `.codex/prompts/wow-viewer-shared-io-implementation.md` when the next task is a real `Core` or `Core.IO` implementation slice.
- Use `.codex/prompts/wow-viewer-dataset-builder-plan.md` when the next task is dataset corpus export cutover, terrain-supervision artifact ownership, manifest or harvest cutover, minimap cleanup or mask-stitch ownership, or planning the first dedicated `wow-viewer` dataset tool over shared services.
- Use `.codex/prompts/wow-viewer-tool-suite-plan-set.md` when the task is broader repo planning, tool inventory, CLI or GUI surface design, bootstrap layout, or migration sequencing.
- Use `.codex/skills/wowarchive-client-staging/SKILL.md` when the next task depends on WoWArchive, `MountAll.bat`, mounted client performance, temp client staging, or pruning staged client copies before large validation runs.
- Stay in `gillijimproject_refactor` only when the user explicitly wants active-viewer runtime work, terrain work, or format-compatibility changes in the current production path.

## Continuation Rules

1. Start from the current validated state, not from older bootstrap assumptions.
2. Preserve `wow-viewer` as the canonical implementation target.
3. Keep validation claims precise.
4. Keep continuity assets current.
5. Keep always-on instructions ahead of the narrower workflow docs.
6. Prefer staged local client roots over mounted archive roots for wide real-data work.

## Guardrails

- Do not route every `wow-viewer` request back into generic migration planning when the user is clearly asking for an implementation slice.
- Do not re-open old repo-shape debates when the active task is already inside `Core.PM4` or `Core.IO`.
- Do not fall back to `MdxViewer` as the default PM4 source of truth when the user is explicitly prioritizing direct library completion in `wow-viewer`.
- Do not claim runtime validation where none exists.
