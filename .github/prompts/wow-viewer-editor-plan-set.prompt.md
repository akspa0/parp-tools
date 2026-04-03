---
description: "Route wow-viewer editor-transition work to the right implementation prompt and execute one narrow slice now. Use when the ask is PM4 MPRL-assisted terrain conform, saved PM4 object choices, moved-object persistence, map save/write ownership, editor-mode panel presets, or the broader shift from viewer to viewer-editor."
name: "wow-viewer Editor Plan Set"
argument-hint: "Describe the editor feature family, persistence seam, or UI reorganization slice you want to attack"
agent: "agent"
---

Choose the right detailed prompt for the staged `wow-viewer` editor transition and execute the first viable implementation slice now unless the user explicitly asks for planning-only output.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
5. `wow-viewer/README.md`
6. `.github/copilot-instructions.md`

## Goal

Route the current request to the correct focused prompt in `.github/prompts/`, then implement the first viable narrow slice in this chat so the viewer-to-editor transition lands as validated code changes instead of another planning-only response.

## Companion Prompts

- `wow-viewer-map-editing-foundation-plan.prompt.md`
- `wow-viewer-editor-ui-surface-plan.prompt.md`
- `wow-viewer-cli-gui-surface-plan.prompt.md`
- `wow-viewer-tool-migration-sequence-plan.prompt.md`
- `wow-viewer-shared-io-implementation.prompt.md`
- `wow-viewer-pm4-library-implementation.prompt.md`

## Routing Rules

- Use `wow-viewer-map-editing-foundation-plan.prompt.md` when the problem is PM4 `MPRL`-assisted terrain conform, saved PM4 object choices, moved object persistence, ADT or WDT or object write ownership, dirty-map tracking, save pipelines, or the first real editor transaction boundary.
- Use `wow-viewer-editor-ui-surface-plan.prompt.md` when the problem is viewer-vs-editor mode switching, preset panel layouts, dock profiles, editor affordances, tool clustering, inspector organization, or how existing chunk clipboard and alpha-mask tools should surface in an editor workspace.
- Use `wow-viewer-cli-gui-surface-plan.prompt.md` when the problem is making an editor workflow exist both as headless commands and as GUI panels over the same shared services.
- Use `wow-viewer-shared-io-implementation.prompt.md` when the editor ask has already collapsed to one concrete `Core` or `Core.IO` implementation seam such as ADT or WDT writing, object-placement persistence, map-file detection, or top-level editor save support.
- Use `wow-viewer-pm4-library-implementation.prompt.md` when the ask is specifically the next `Core.PM4` seam needed by editing, such as promoting `MPRL` summaries, terrain-contact helpers, placement solvers, or saved-object evidence contracts into the library.
- Use `wow-viewer-tool-migration-sequence-plan.prompt.md` when the ask is broader phasing for the editor transition across runtime, tools, save paths, and UI cutover.

## Mandatory Execution Rule

- Unless the user explicitly asks for planning-only output, implement one narrow slice in this chat after routing.
- Make code changes in workspace files, run applicable validation commands, and report concrete proof.
- Do not rewrite prompts, instructions, or other workflow docs unless the user explicitly asks for that maintenance task.

## Deliverables

Return all items:

1. the best next prompt to run
2. why it is the correct slice now
3. which companion prompt should follow after it
4. what concrete repo and file scope the next slice should include
5. what should stay out of scope for the next slice
6. what proof level is realistic for that slice
7. whether the chosen prompt should produce a planning artifact or an implementation-ready build plan
8. the exact next action the implementing chat should take after reading the chosen prompt
9. exact files changed for the implemented slice
10. exact validation commands run and their outcome

## Output Rule

- Do not stop at naming the next prompt.
- The answer must hand off a buildable next step with exact repo scope, explicit deferrals, and the proof target for the next slice.
- If the chosen prompt is itself a planning prompt, say that it must produce an implementation-ready slice plan rather than generic architecture commentary.

## First Output

Start with:

1. the exact editor problem you think the user is trying to solve
2. the single best next prompt from the set above
3. the narrow proof that would make that slice real
4. what you are explicitly not claiming yet