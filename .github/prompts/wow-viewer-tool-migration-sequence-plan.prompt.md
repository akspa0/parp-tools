---
description: "Execute the wow-viewer tool-suite migration as realistic vertical slices. Use when deciding and implementing the next slice that moves old tools/file families/shared services forward."
name: "wow-viewer Tool Migration Sequence Plan"
argument-hint: "Optional tool family, dependency problem, or milestone boundary to prioritize"
agent: "agent"
---

Select and implement the first viable migration slice for the `wow-viewer` tool suite so the old-repo mess is absorbed in a realistic order instead of a giant simultaneous rewrite.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `.github/copilot-instructions.md`
5. `gillijimproject_refactor/plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`

## Goal

Implement the first dependency-safe migration slice now, while keeping the remaining phase order explicit and testable.

## Mandatory Execution Rule

- Implement one narrow slice in this chat unless the user explicitly asks for planning-only output.
- If blocked by missing prerequisites, implement the smallest unblocker slice instead of returning planning-only commentary.
- Do not rewrite prompts or workflow docs unless explicitly requested.

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
- Output must be implementation-oriented. Do not stop at generic phase labels or long-horizon roadmap language.
- The result must identify the first phase to execute now, the first slice to implement now, and the exact repo or file scope plus validation for that slice.

## What The Plan Must Produce

1. The phase order.
2. Entry and exit criteria per phase.
3. Dependencies between phases.
4. The first three vertical slices.
5. What should deliberately wait until later.
6. Where real-data validation must occur.
7. What would cause the plan to fail or thrash.
8. The exact first implementation slice to build now.
9. The exact repo or file scope and validation for that first slice.

## Deliverables

Return all items:

1. numbered phase plan
2. dependency map
3. first three vertical slices
4. per-phase validation expectations
5. what stays in `parp-tools` during each phase
6. deferred work list
7. major sequencing risks
8. slice 1 implementation handoff
9. exact files changed in this chat for slice 1
10. exact validation commands run in this chat

## Implementation Requirements

- Format the result as a build queue, not a brainstorm.
- For the first slice, include:
	- exact project or file scope
	- shared boundary it proves
	- what existing prompt or skill should own the follow-up implementation
	- what must stay deferred
	- how the slice will be validated
- If the first phase is still planning-only, explain why that is unavoidable and identify the first non-planning implementation slice immediately after it.

## First Output

Start with:

1. the first phase that should happen before any serious tool migration
2. the first vertical slice that proves the architecture split is real
3. the tool families that should wait until later phases
4. the exact first implementation slice to build now
5. the biggest sequencing mistake to avoid
