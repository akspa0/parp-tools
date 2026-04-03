---
description: "Implement dual-surface wow-viewer workflows where CLI and GUI call the same shared services. Use when refactoring old tools into dual-surface workflows instead of duplicated apps."
name: "wow-viewer CLI GUI Surface Plan"
argument-hint: "Optional tool family, panel, command, or workflow to prioritize"
agent: "agent"
---

Implement one dual-surface workflow in `wow-viewer` so CLI commands and GUI panels share the same service boundary without duplicating format logic.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `.github/copilot-instructions.md`
5. `gillijimproject_refactor/plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`

## Goal

Implement the first end-to-end dual-surface slice now, including one CLI entrypoint and one GUI entrypoint over a shared service.

## Mandatory Execution Rule

- Implement one narrow dual-surface slice now unless the user explicitly asks for planning-only output.
- Run validation for both surfaces where possible.
- Do not rewrite prompts or workflow docs unless explicitly requested.

## Existing GUI/Interactive Inputs To Account For

- `MdxViewer` sidebars, inspectors, exporters, previewers, and conversion panels
- asset-catalog and screenshot workflows
- PM4 and WDL utilities
- terrain import/export and related panel-driven workflows

## Existing CLI/Headless Inputs To Account For

- `WoWMapConverter.Cli`
- `AlphaWdtAnalyzer.Cli`
- `Pm4Research.Cli`
- `WlAnalyzer`
- any old console-style tools in `parpToolbox`, `PM4Tool`, `ADTPrefabTool`, and `WoWRollback`

## Non-Negotiable Constraints

- CLI and GUI must call the same first-party shared services.
- Panel code must not own file parsing, conversion, or export logic.
- CLI tools must not become second-class wrappers around hidden GUI behavior.
- Batch/headless workflows must remain possible for converter/export/audit operations.
- Long-running operations should have service boundaries that work in both interactive and non-interactive hosts.
- Output must be implementation-oriented. Do not stop at abstract surface taxonomy.
- The result must identify the first dual-surface workflow to build now, the exact repo or file scope for that slice, and the validation or proof target after it lands.

## What The Plan Must Produce

1. The shared service boundary for dual-surface tooling.
2. The CLI command families.
3. The GUI panel families.
4. Which tool capabilities should be CLI-only, GUI-only, or dual-surface.
5. Progress/reporting/cancellation design across both surfaces.
6. The first dual-surface workflow to build now.
7. The next dual-surface workflow after that.
8. Exact repo or file scope and validation for workflow 1.
9. The migration risks if surfaces stay duplicated.

## Deliverables

Return all items:

1. service-layer breakdown
2. CLI command tree
3. GUI panel tree
4. dual-surface mapping by tool family
5. cancellation/progress/reporting design
6. slice 1 implementation plan
7. slice 2 follow-up plan
8. validation or proof plan
9. duplication risks still remaining
10. exact files changed in this chat for slice 1
11. exact validation commands run in this chat

## Implementation Requirements

- Format the result as a build plan, not a brainstorm.
- For slice 1, include:
	- exact project or file scope
	- shared services to add or extend
	- CLI command entrypoints to wire
	- GUI panel or surface entrypoints to wire
	- what should stay out of scope
	- how the slice will be validated
- If the chosen workflow depends on shared services that do not yet exist, say which prerequisite implementation prompt should run first.

## First Output

Start with:

1. the tool families that most need dual-surface support
2. the shared services they should sit on
3. which capabilities should stay CLI-only or GUI-only for now
4. the exact first end-to-end workflow to implement now
5. what that first slice should explicitly not try to solve yet
