# Project Guidelines

## Scope
- The active code paths in this workspace are `gillijimproject_refactor` and `wow-viewer`.
- Treat `wow-viewer` as the primary target when the task mentions the new repo, `Core.PM4`, PM4 library extraction, inspect CLI work, or tool-suite cutover.
- Treat `gillijimproject_refactor`, especially `src/MdxViewer` and `src/WoWMapConverter`, as the active runtime reference path for viewer behavior, terrain work, and format compatibility work.
- Treat `archived_projects`, `WoWRollback/old_projects`, `WMOv14/old_sources`, and `gillijimproject_refactor/next` as non-primary unless the task explicitly targets them.

## First Reads
- Before changing viewer, terrain, or format code, read `gillijimproject_refactor/memory-bank/activeContext.md`, `gillijimproject_refactor/memory-bank/progress.md`, `gillijimproject_refactor/memory-bank/data-paths.md`, and `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`.
- Before changing `wow-viewer` PM4, shared I/O, or migration workflow, read `gillijimproject_refactor/memory-bank/activeContext.md`, `gillijimproject_refactor/memory-bank/progress.md`, `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`, `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`, and `wow-viewer/README.md`.
- Before changing Copilot workflow assets for `wow-viewer`, also read `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`, `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`, and `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`.
- If the task touches 3.3.5 terrain texturing, also read `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`.

## wow-viewer Skill Registry
- Use `.github/skills/wow-viewer-pm4-library/SKILL.md` for `Core.PM4` slices, `pm4 inspect`, `pm4 audit`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, PM4 regression updates, PM4 analyzer work, or narrow PM4 solver extraction from `MdxViewer`.
- Use `.github/skills/wow-viewer-shared-io-library/SKILL.md` for `Core` or `Core.IO` non-PM4 slices such as ADT root, `_tex0.adt`, `_obj0.adt`, `_lod.adt`, WDT, WMO, BLP, DBC, or DB2 detection or summary work, chunk readers, `map inspect`, `converter detect`, or shared-format regression updates.
- Use `.github/skills/wow-viewer-migration-continuation/SKILL.md` for continuation routing, next-slice selection, migration regrouping, or workflow-surface updates across chats.
- Treat these skills as the first discovery surface for `wow-viewer` implementation work, with prompts as the more specific execution surface underneath them.

## wow-viewer Workflow Maintenance Rule
- Whenever a new `wow-viewer` skill, implementation prompt, or workflow asset is added, update `.github/copilot-instructions.md` so the new asset is named and routed here.
- Keep `wow-viewer/README.md`, the relevant `gillijimproject_refactor/plans/wow_viewer_*` continuity plan, and the memory-bank files in sync with any new `wow-viewer` skill or prompt so future chats inherit the new path automatically.
- Do not add a new `wow-viewer` skill without also deciding which existing skill or prompt should hand work to it.

## Build And Validation
- For parser and format-library work, prefer `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`.
- For viewer work, use `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- For `wow-viewer` library or tool work, prefer `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- For `wow-viewer` PM4 inspect or report slices, validate against `gillijimproject_refactor/test_data/development/World/Maps/development` with `WowViewer.Tool.Inspect` commands when the slice changes analyzer or report output.
- The active viewer and converter path currently has little to no first-party automated regression coverage. Do not claim terrain or alpha-mask changes are safe based only on library tests under `lib/*`, archived tests, or documentation examples.
- Do not describe a `wow-viewer` build, test pass, or active-viewer compile as real viewer runtime PM4 signoff. Active `MdxViewer` still has no non-UI command-line path to open the split development map with its required base data source.
- Prefer real-data validation using the fixed paths in `gillijimproject_refactor/memory-bank/data-paths.md`. Do not ask the user for alternate paths unless the existing fixed paths are missing.

## wow-viewer PM4 Guardrails
- Keep `MdxViewer` as the runtime PM4 reference implementation and `Pm4Research` as the library seed until real data proves a change.
- Favor narrow library-first slices in `wow-viewer/src/core/WowViewer.Core.PM4` over broader active-viewer consumer wiring unless the user explicitly asks for integration work.
- Keep exploratory PM4 interpretations labeled as research or experimental, especially around `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR.Value1`, and final coordinate ownership.
- Each PM4 slice should land with concrete validation in `wow-viewer/tests/WowViewer.Core.PM4.Tests`, `WowViewer.Tool.Inspect`, or both.
- If a PM4 slice only proves library or build behavior, say that explicitly instead of implying viewer runtime closure.

## wow-viewer Shared I/O Guardrails
- Favor narrow shared-library slices in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` over tool-local parsing in inspect or converter entrypoints.
- Keep file detection, top-level chunk reading, and summary contracts in shared libraries once they exist; do not duplicate those heuristics across tools.
- Be explicit about proof level: classification, top-level summary, deep payload parsing, and writing are different milestones.
- Each shared-I/O slice should land with concrete validation in `wow-viewer/tests/WowViewer.Core.Tests`, `WowViewer.Tool.Inspect`, `WowViewer.Tool.Converter`, or an appropriate combination.
- If a slice only proves shared detection or summary behavior, say that explicitly instead of implying full format ownership.

## Terrain And Alpha Risk Area
- Treat commit `343dadfa27df08d384614737b6c5921efe6409c8` as the pre-regression baseline for terrain alpha-mask behavior unless the user specifies another baseline.
- High-risk files for alpha regressions include `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`, `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`, `src/MdxViewer/Terrain/TerrainRenderer.cs`, `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`, `src/MdxViewer/Terrain/TerrainChunkData.cs`, `src/MdxViewer/Export/TerrainImageIo.cs`, and `src/MdxViewer/ViewerApp.cs`.
- Any change to MCAL decode, edge-fix behavior, `_tex0.adt` texture sourcing, alpha packing, or shader blending must be checked against both Alpha-era terrain and LK 3.3.5 terrain.

## Conventions
- Keep FourCCs readable in memory and only reverse them at I/O boundaries.
- Preserve the existing split between `AlphaTerrainAdapter` and `StandardTerrainAdapter`.
- Favor minimal fixes over broad refactors in the terrain pipeline.
- For `wow-viewer` planning or continuity work, prefer `.github/prompts/` and `.github/skills/` as the canonical shared workflow surface, and keep `gillijimproject_refactor/plans` or the memory bank in sync when the migration state materially changes.
- For `wow-viewer` shared-format work, keep inspect and converter as thin consumers of `Core` or `Core.IO` seams rather than adding new tool-local parser ownership.
- If behavior, commands, or known risks materially change, update the relevant memory-bank file instead of leaving the old guidance stale.