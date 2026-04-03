---
description: "Sequence the wow-viewer tool-suite migration into realistic vertical slices. Use when deciding the order to move old tools, file families, and shared services without exploding scope."
name: "wow-viewer Tool Migration Sequence Plan"
argument-hint: "Optional tool family, dependency problem, or milestone boundary to prioritize"
agent: "codex"
---

Design the migration sequence for the `wow-viewer` tool suite so the old-repo mess is absorbed in a realistic order instead of a giant simultaneous rewrite.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `AGENTS.md`
5. `gillijimproject_refactor/plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`

## Goal

Define the order to migrate shared core services, file-format ownership, CLI tools, and GUI tool panels into `wow-viewer` with explicit dependency management and realistic vertical slices.

## Inputs To Sequence Explicitly

- repo bootstrap and project skeleton
- shared file-format library
- runtime/data-source service layer
- viewer shell cutover
- tool inventory cutover
- CLI/GUI dual-surface workflows
- performance-sensitive work that should wait until after core boundaries are real

## Non-Negotiable Constraints

- Do not schedule a giant all-at-once rewrite.
- Do not let enhanced renderer ambitions consume the tool-suite migration plan.
- Do not migrate tools before the shared service/core boundary they depend on exists.
- Keep real-data validation visible in the phase exits.
- Be explicit about what remains in `parp-tools` during each phase.

## What The Plan Must Produce

1. The phase order.
2. Entry and exit criteria per phase.
3. Dependencies between phases.
4. The first three vertical slices.
5. What should deliberately wait until later.
6. Where real-data validation must occur.
7. What would cause the plan to fail or thrash.

## Deliverables

Return all items:

1. numbered phase plan
2. dependency map
3. first three vertical slices
4. per-phase validation expectations
5. what stays in `parp-tools` during each phase
6. deferred work list
7. major sequencing risks

## First Output

Start with:

1. the first phase that should happen before any serious tool migration
2. the first vertical slice that proves the architecture split is real
3. the tool families that should wait until later phases
4. the biggest sequencing mistake to avoid
