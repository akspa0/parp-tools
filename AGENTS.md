# Codex Workspace Instructions

This file is the Codex-facing conversion of the workspace memory-bank rules, `.github/copilot-instructions.md`, and the active `wow-viewer` workflow assets.

## Scope

- The active code paths in this workspace are `gillijimproject_refactor` and `wow-viewer`.
- Treat `wow-viewer` as the primary target when the task mentions the new repo, `Core.PM4`, PM4 library extraction, inspect CLI work, shared I/O ownership, or tool-suite cutover.
- Treat `gillijimproject_refactor`, especially `src/MdxViewer` and `src/WoWMapConverter`, as the legacy or compatibility path when the task explicitly targets the current viewer, terrain work, or old-format behavior.
- Treat `archived_projects`, `WoWRollback/old_projects`, `WMOv14/old_sources`, and `gillijimproject_refactor/next` as non-primary unless the task explicitly targets them.

## Read First

- Before changing viewer, terrain, or format code, read `gillijimproject_refactor/memory-bank/activeContext.md`, `gillijimproject_refactor/memory-bank/progress.md`, `gillijimproject_refactor/memory-bank/data-paths.md`, and `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md` when it exists.
- Before changing `wow-viewer` PM4, shared I/O, or migration workflow, also read `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`, `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`, and `wow-viewer/README.md`.
- Before changing workflow assets, read `.codex/README.md`, `.codex/prompts/wow-viewer-tool-suite-plan-set.md`, `.codex/prompts/wow-viewer-pm4-library-implementation.md`, and `.codex/prompts/wow-viewer-shared-io-implementation.md`.
- If the task touches 3.3.5 terrain texturing or alpha blending, also read `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`.

## Memory Bank Rule

- Treat `gillijimproject_refactor/memory-bank/` as the continuity source for project state.
- Read all relevant memory-bank files before making non-trivial changes; at minimum, read `activeContext.md` and `progress.md` for the area you are touching.
- Keep the memory bank accurate after significant workflow, status, or boundary changes.
- Prefer updating the smallest relevant continuity file instead of leaving stale guidance behind.

## Memory Bank Structure

- Core files: `projectbrief.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`.
- Additional context files such as `data-paths.md`, `agents.md`, `coding_standards.md`, and plan files are part of the working memory surface here.
- `productContext.md` is part of the original Cursor template but is not currently present in this workspace; do not assume it exists.

## Codex Skill Registry

- Use `.codex/skills/wow-viewer-pm4-library/SKILL.md` for `Core.PM4` slices, `pm4 inspect`, `pm4 audit`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, PM4 regression updates, PM4 analyzer work, or narrow PM4 solver extraction from `MdxViewer`.
- Use `.codex/skills/wow-viewer-shared-io-library/SKILL.md` for `Core` or `Core.IO` non-PM4 slices such as ADT root, `_tex0.adt`, `_obj0.adt`, `_lod.adt`, WDT, WMO, BLP, DBC, or DB2 detection or summary work, chunk readers, `map inspect`, `converter detect`, or shared-format regression updates.
- Use `.codex/skills/wow-viewer-migration-continuation/SKILL.md` for continuation routing, next-slice selection, migration regrouping, or workflow-surface updates across chats.
- Use `.codex/skills/terrain-alpha-regression/SKILL.md` for terrain alpha-mask, MCAL, MCLY, split ADT texture, or blending regressions in `gillijimproject_refactor`.

## Prompt Registry

- Use `.codex/prompts/wow-viewer-tool-suite-plan-set.md` to route broader `wow-viewer` planning asks to the right focused prompt.
- Use `.codex/prompts/wow-viewer-pm4-library-implementation.md` for the next narrow `Core.PM4` implementation slice.
- Use `.codex/prompts/wow-viewer-shared-io-implementation.md` for the next narrow shared `Core` or `Core.IO` format slice.

## Build And Validation

- For new `wow-viewer` library or tool work, prefer `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- For legacy parser and format-library work that still explicitly targets `gillijimproject_refactor`, prefer `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`.
- For viewer work, use `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- For `wow-viewer` PM4 inspect or report slices, validate against `gillijimproject_refactor/test_data/development/World/Maps/development` with `WowViewer.Tool.Inspect` commands when the slice changes analyzer or report output.
- Only build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln` for `wow-viewer` work when the task explicitly changes consumer compatibility or the user asks for that compatibility check.
- Do not describe a `wow-viewer` build, test pass, or optional active-viewer compile as real viewer runtime signoff.
- Prefer real-data validation using the fixed paths in `gillijimproject_refactor/memory-bank/data-paths.md`. Do not ask the user for alternate paths unless those fixed paths are missing.

## wow-viewer PM4 Guardrails

- Treat `wow-viewer/src/core/WowViewer.Core.PM4` as the canonical implementation target for new PM4 work.
- Treat `Pm4Research`, `MdxViewer`, `PM4Tool`, `parpToolbox`, and `WoWRollback.PM4Module` as extraction or reference inputs, not as the default owners of PM4 behavior.
- Favor direct library completion in `wow-viewer/src/core/WowViewer.Core.PM4` over broader active-viewer consumer wiring unless the user explicitly asks for integration work or compatibility checks.
- Keep exploratory PM4 interpretations labeled as research or experimental, especially around `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR.Value1`, and final coordinate ownership.
- Each PM4 slice should land with concrete validation in `wow-viewer/tests/WowViewer.Core.PM4.Tests`, `WowViewer.Tool.Inspect`, or both.

## wow-viewer Shared I/O Guardrails

- Favor narrow shared-library slices in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` over tool-local parsing in inspect or converter entrypoints.
- Treat `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` as the canonical owners for new non-PM4 format work; use `gillijimproject_refactor` as reference input only when needed.
- Keep file detection, top-level chunk reading, and summary contracts in shared libraries once they exist; do not duplicate those heuristics across tools.
- Be explicit about proof level: classification, top-level summary, deep payload parsing, and writing are different milestones.
- Each shared-I/O slice should land with concrete validation in `wow-viewer/tests/WowViewer.Core.Tests`, `WowViewer.Tool.Inspect`, `WowViewer.Tool.Converter`, or an appropriate combination.

## Terrain And Alpha Risk Area

- Treat commit `343dadfa27df08d384614737b6c5921efe6409c8` as the pre-regression baseline for terrain alpha-mask behavior unless the user specifies another baseline.
- High-risk files for alpha regressions include `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`, `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`, `src/MdxViewer/Terrain/TerrainRenderer.cs`, `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`, `src/MdxViewer/Terrain/TerrainChunkData.cs`, `src/MdxViewer/Export/TerrainImageIo.cs`, and `src/MdxViewer/ViewerApp.cs`.
- Any change to MCAL decode, edge-fix behavior, `_tex0.adt` texture sourcing, alpha packing, or shader blending must be checked against both Alpha-era terrain and LK 3.3.5 terrain.

## Conventions

- Keep FourCCs readable in memory and only reverse them at I/O boundaries.
- Preserve the existing split between `AlphaTerrainAdapter` and `StandardTerrainAdapter`.
- Favor minimal fixes over broad refactors in the terrain pipeline.
- For `wow-viewer` planning or continuity work, prefer `.codex/prompts/` and `.codex/skills/` as the Codex-facing workflow surface, and keep `gillijimproject_refactor/plans` or the memory bank in sync when the migration state materially changes.
- If behavior, commands, or known risks materially change, update the relevant memory-bank file instead of leaving the old guidance stale.
