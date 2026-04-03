---
description: "Plan how wow-viewer tools should exist both as CLI commands and GUI panels over the same shared services. Use when refactoring old tools into dual-surface workflows instead of duplicated apps."
name: "wow-viewer CLI GUI Surface Plan"
argument-hint: "Optional tool family, panel, command, or workflow to prioritize"
agent: "codex"
---

Design the dual-surface tool plan for `wow-viewer` so the reformed tool suite can expose both CLI commands and GUI panels without duplicating format logic or workflow code.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `AGENTS.md`
5. `gillijimproject_refactor/plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`

## Goal

Define how the new tool suite exposes the same shared capability through both CLI and GUI surfaces, using one service layer instead of separate implementations for panels, commands, and one-off executables.

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

## What The Plan Must Produce

1. The shared service boundary for dual-surface tooling.
2. The CLI command families.
3. The GUI panel families.
4. Which tool capabilities should be CLI-only, GUI-only, or dual-surface.
5. Progress/reporting/cancellation design across both surfaces.
6. The first dual-surface workflow to build.
7. The migration risks if surfaces stay duplicated.

## Deliverables

Return all items:

1. service-layer breakdown
2. CLI command tree
3. GUI panel tree
4. dual-surface mapping by tool family
5. cancellation/progress/reporting design
6. first workflow to implement in both surfaces
7. duplication risks still remaining

## First Output

Start with:

1. the tool families that most need dual-surface support
2. the shared services they should sit on
3. which capabilities should stay CLI-only or GUI-only for now
4. the first end-to-end workflow you would implement both ways
