---
description: "Inventory old repo tools and decide what becomes a first-class wow-viewer tool, what merges into the viewer, what stays CLI-only, and what remains archaeology-only."
name: "wow-viewer Tool Inventory And Cutover Plan"
argument-hint: "Optional tool family, executable, panel, or domain to prioritize"
agent: "agent"
---

Inventory the old repo tool sprawl and produce a concrete cutover plan for the `wow-viewer` tool suite.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `.github/copilot-instructions.md`
5. `gillijimproject_refactor/plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`

## Goal

Produce an explicit inventory of the tools we built across the old repo and decide which ones become rebuilt first-class tools in `wow-viewer`, which ones fold into existing viewer panels, which ones stay standalone CLIs, and which ones remain archaeology/reference only.

## Tools To Classify Explicitly

- `gillijimproject_refactor/src/MdxViewer` built-in panels and utilities
- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli`
- `gillijimproject_refactor/AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Cli`
- `gillijimproject_refactor/AlphaWdtInspector`
- `gillijimproject_refactor/DBCTool`
- `gillijimproject_refactor/DBCTool.V2`
- `gillijimproject_refactor/BlpResizer`
- `gillijimproject_refactor/AlphaLkToAlphaStandalone`
- `gillijimproject_refactor/src/Pm4Research.Cli`
- `gillijimproject_refactor/src/MDX-L_Tool`
- `gillijimproject_refactor/tools/WlAnalyzer`
- `parpToolbox`
- `PM4Tool`
- `ADTPrefabTool`
- `WoWRollback` sub-tools and modules where still relevant

## Classification Buckets

- rebuild as first-class `wow-viewer` tool
- fold into the main viewer as a panel or workflow
- keep as standalone CLI over shared services
- keep as research/reference only in `parp-tools`
- discard as superseded duplication

## Non-Negotiable Constraints

- Do not let every historical tool survive as its own permanent app if a shared core + fewer surfaces is cleaner.
- Do not collapse everything into the viewer UI if headless execution still matters.
- Do not keep duplicated CLI and GUI logic if both surfaces can ride the same services.
- Be explicit about tools that are still valuable only as archaeology or reverse-engineering evidence.

## What The Plan Must Produce

1. A full tool inventory.
2. A classification for each tool.
3. The destination in `wow-viewer` for each surviving tool.
4. Shared-core dependencies for each tool.
5. Which current viewer panels should survive, be rebuilt, merge, or die.
6. Which legacy executables should not be ported forward.
7. The first three tools or tool families to migrate.

## Deliverables

Return all items:

1. tool inventory table
2. classification per tool
3. destination app/panel/CLI mapping
4. shared-service requirements per tool family
5. archaeology-only leftovers
6. first migration wave
7. major cutover risks

## First Output

Start with:

1. the tool families that clearly deserve first-class survival in `wow-viewer`
2. the biggest duplicated tools that should be merged or killed
3. which old executables look like archaeology, not production assets
4. the first migration wave you would schedule
