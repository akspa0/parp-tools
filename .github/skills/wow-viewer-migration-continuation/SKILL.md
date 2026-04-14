---
name: wow-viewer-migration-continuation
description: 'Use when resuming wow-viewer migration work across chat sessions, choosing the next slice, or updating prompts, instructions, skills, or memory-bank files for the new repo. Includes routing rules for PM4 implementation, shared-I/O implementation for ADT or WDT or WMO or BLP or DBC families, dataset-builder cutover, archive-backed client staging, and broader tool-suite planning.'
argument-hint: 'Describe the wow-viewer area to resume, plan, or document'
user-invocable: true
---

# wow-viewer Migration Continuation

## When To Use

- The user wants to continue `wow-viewer` work from a prior session.
- The task is choosing the next migration slice.
- The task is updating `.github` instructions, prompts, or skills for `wow-viewer`.
- The task is deciding whether the next narrow slice is PM4 work or a shared-I/O format slice such as ADT root, `_tex0.adt`, `_obj0.adt`, WDT, WMO, BLP, DBC, or DB2.
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
7. `.github/copilot-instructions.md`
8. `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
9. `.github/prompts/wow-viewer-dataset-builder-plan.prompt.md`
10. `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
11. `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`
12. `.github/skills/wowarchive-client-staging/SKILL.md` when the continuation depends on archive-backed client roots or staging policy

## Routing Rules

- Use `wow-viewer-pm4-library-implementation.prompt.md` when the next task is a real `Core.PM4` implementation slice such as `pm4 inspect`, `pm4 audit`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, `pm4 export-json`, a PM4 test slice, or a narrow solver or consumer extraction.
- Use `wow-viewer-shared-io-implementation.prompt.md` when the next task is a real `Core` or `Core.IO` implementation slice such as ADT root or split-ADT summary work, WDT summary work, WMO or BLP or DBC or DB2 detection work, a shared chunk-reader, a non-PM4 inspect or converter slice, or a shared-format regression update.
- Use `wow-viewer-dataset-builder-plan.prompt.md` when the next task is dataset corpus export cutover, terrain-supervision artifact ownership, manifest or harvest cutover, minimap cleanup or mask-stitch ownership, or planning the first dedicated `wow-viewer` dataset tool over shared services.
- Use `wow-viewer-tool-suite-plan-set.prompt.md` when the task is broader repo planning, tool inventory, CLI or GUI surface design, bootstrap layout, or migration sequencing.
- Use `.github/skills/wowarchive-client-staging/SKILL.md` when the next task depends on WoWArchive, `MountAll.bat`, mounted client performance, temp client staging, or pruning staged client copies before large validation runs.
- Stay in `gillijimproject_refactor` only when the user explicitly wants active-viewer runtime work, terrain work, or format-compatibility changes in the current production path.

## Continuation Rules

1. Start from the current validated state, not from older bootstrap assumptions.
   `wow-viewer` already exists, `Core.PM4` already has real code, and there are already passing PM4 and shared-I/O tests.

2. Preserve `wow-viewer` as the canonical implementation target.
   The default continuation path is direct library work in `wow-viewer`, not broader `MdxViewer` hookup or viewer-first parity work.

3. Keep validation claims precise.
   A `wow-viewer` build or test pass is the primary implementation validation for `wow-viewer` work. An `MdxViewer` build is optional consumer-compile validation only when integration changed or the user asked for it. Neither is runtime signoff by itself.

4. Keep continuity assets current.
   If the active workflow, commands, or status changed, update the memory bank and relevant prompt or skill instead of leaving future sessions stale.

5. Keep the always-on instructions ahead of the skills.
   If a new `wow-viewer` skill or implementation prompt is added, update `.github/copilot-instructions.md` and `wow-viewer/README.md` so future chats can discover that new path without recovering from memory-bank context first.

6. Prefer staged local client roots over mounted archive roots for wide real-data work.
   If a continuation depends on WoWArchive-backed clients for repeated reads, treat the mounted archive as the source and a staged local copy as the working root whenever practical.

## Guardrails

- Do not route every `wow-viewer` request back into generic migration planning when the user is clearly asking for an implementation slice.
- Do not re-open old repo-shape debates when the active task is now inside `Core.PM4`.
- Do not fall back to `MdxViewer` as the default PM4 source of truth when the user is explicitly prioritizing direct library completion in `wow-viewer`.
- Do not claim runtime validation where none exists.