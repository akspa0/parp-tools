---
description: "Design the concrete wow-viewer repo, solution, and bootstrap layout for the reformed tool suite. Use when deciding where the app, shared library, tools, libs, scripts, and research leftovers should live."
name: "wow-viewer Bootstrap Layout Plan"
argument-hint: "Optional repo-shape concern, tool family, or bootstrap constraint to prioritize"
agent: "agent"
---

Design the concrete repository and solution bootstrap plan for `https://github.com/akspa0/wow-viewer`.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `.github/copilot-instructions.md`
5. `gillijimproject_refactor/plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`

## Goal

Produce a concrete repo and solution layout for `wow-viewer` so the shipping app, shared library, standalone tools, upstream libs, bootstrap scripts, and archaeology-only leftovers stop living in one tangled tree.

## Old Repo Inputs To Account For

- `gillijimproject_refactor/src/MdxViewer`
- `gillijimproject_refactor/src/WoWMapConverter`
- `gillijimproject_refactor/src/gillijimproject-csharp`
- `gillijimproject_refactor/AlphaWDTAnalysisTool`
- `gillijimproject_refactor/AlphaWdtInspector`
- `gillijimproject_refactor/DBCTool`
- `gillijimproject_refactor/DBCTool.V2`
- `gillijimproject_refactor/BlpResizer`
- `gillijimproject_refactor/AlphaLkToAlphaStandalone`
- `gillijimproject_refactor/src/Pm4Research.Cli`
- `gillijimproject_refactor/src/MDX-L_Tool`
- `gillijimproject_refactor/tools/WlAnalyzer`
- top-level archaeology/source inputs such as `parpToolbox`, `PM4Tool`, `ADTPrefabTool`, and `WoWRollback`

## Non-Negotiable Constraints

- The main viewer app must have one obvious home.
- First-party shared code must not be mixed into `libs/`.
- Upstream repos must remain external and trackable under `libs/`.
- Tool executables must not each reinvent their own mini-core.
- R&D and archaeology leftovers should not quietly become production dependencies.
- Do not mirror the old repo structure into the new repo out of laziness.

## What The Plan Must Produce

1. The exact top-level repo tree.
2. The exact solution/project grouping.
3. Which projects are app-only, core-only, tool-only, or research-only.
4. Which upstream repos are baseline bootstrap dependencies versus optional evaluation dependencies.
5. What bootstrap scripts should do on a clean clone.
6. What stays in `parp-tools` instead of moving.
7. The first bootstrap milestone that proves the repo shape is real.

## Deliverables

Return all items:

1. proposed top-level repo tree
2. proposed project/solution tree
3. bootstrap dependency matrix
4. scripts/bootstrap responsibilities
5. what stays in `parp-tools`
6. first bootstrap milestone
7. risks if the tree is done badly

## First Output

Start with:

1. the minimum repo tree you would create on day one
2. the first-party projects that must exist immediately
3. the upstream repos that should be auto-cloned immediately
4. what parts of the old repo should not be copied forward blindly
