---
description: "Define the broader shared wow-viewer library plan for reading and writing supported file types. Use when planning ownership, source-root consolidation, or migration order across viewer, converter, toolbox, and legacy tools."
name: "wow-viewer Shared I/O Library Plan"
argument-hint: "Optional file family, chunk type, or old tool to prioritize"
agent: "codex"
---

Design the owned shared library plan for `wow-viewer` so there is one first-party codebase for reading and writing the file types we currently handle across the old repo.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `AGENTS.md`
5. `gillijimproject_refactor/plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`
6. `gillijimproject_refactor/src/MdxViewer`
7. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core`
8. `gillijimproject_refactor/src/gillijimproject-csharp`

## Goal

Define one owned library stack that becomes the source of truth for file-format reading, writing, conversion, and shared domain contracts instead of leaving that truth scattered across `MdxViewer`, `WoWMapConverter.Core`, `gillijimproject-csharp`, `DBCTool.V2`, `AlphaLkToAlphaStandalone`, `parpToolbox`, `PM4Tool`, and other historical tool roots.

## Current Validated State To Build On

- `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` now already contain:
	- `FourCC`, `ChunkHeader`, and `ChunkHeaderReader`
	- `ChunkedFileReader`
	- WDT or ADT top-level summary contracts and `MapFileSummaryReader`
	- shared `WowFileKind`, `WowFileDetection`, and `WowFileDetector`
- `WowViewer.Tool.Inspect` now consumes that shared surface through `map inspect`.
- `WowViewer.Tool.Converter` now consumes that shared surface through `detect`.
- If the ask is to implement the next narrow shared-format slice instead of planning ownership, use `wow-viewer-shared-io-implementation.md` instead of this broader planning prompt.

## File-Type Families To Cover Explicitly

- Alpha and standard `ADT` / `WDT`
- terrain chunk families and placement chunks
- `M2` / `MDX` / `MDL`
- `WMO` including version seams such as `v14`, `v16`, and `v17`
- `PM4`
- `WDL`
- `DBC` and related table access seams used by tools/viewer runtime
- `BLP`
- export/import surfaces already split across viewer and converter tools

## Non-Negotiable Constraints

- Do not keep multiple first-party parser/writer stacks alive as peers in the new repo.
- Do not confuse upstream dependency reuse with first-party format ownership.
- Preserve readable FourCC handling internally and only reverse at I/O boundaries.
- Keep Alpha and standard terrain handling explicitly separated where the format boundary is real.
- Distinguish canonical readers/writers from partial, speculative, or research-only handlers.
- Treat runtime ingestion helpers and file-format ownership as related but separate seams.

## What The Plan Must Produce

1. The target shared-library breakdown.
2. A format ownership matrix by file family.
3. The source-root mapping for code to absorb, rewrite, wrap, or discard.
4. The writer/read-roundtrip risk matrix.
5. The migration order for file families.
6. The first file-family slice that proves the shared library is real.
7. Validation expectations using real data.

## Source Roots To Inventory Explicitly

- `gillijimproject_refactor/src/MdxViewer`
- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core`
- `gillijimproject_refactor/src/gillijimproject-csharp`
- `gillijimproject_refactor/DBCTool.V2`
- `gillijimproject_refactor/AlphaLkToAlphaStandalone`
- `parpToolbox`
- `PM4Tool`
- `ADTPrefabTool`

## Deliverables

Return all items:

1. target library/module tree
2. file-family ownership matrix
3. absorb vs rewrite vs wrap vs discard map
4. canonical read/write authority list
5. highest-risk format seams
6. first vertical slice
7. real-data validation plan

## First Output

Start with:

1. the worst source-of-truth duplication seams in the current repo
2. which file families should become canonical first
3. which old code roots should be absorbed first versus treated as archaeology/reference
4. what writer/read path must not stay split in the new repo
