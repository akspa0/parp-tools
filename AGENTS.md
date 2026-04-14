# Codex Workspace Instructions

This file is the Codex-facing conversion of the workspace memory-bank rules, `.github/copilot-instructions.md`, and the active `wow-viewer` workflow assets.

## Scope

- The active code paths in this workspace are `gillijimproject_refactor` and `wow-viewer`.
- Treat `wow-viewer` as the primary target when the task mentions the new repo, `Core.PM4`, PM4 library extraction, inspect CLI work, shared I/O ownership, tool-suite cutover, dataset-builder cutover, ML corpus export ownership, or shared terrain-supervision artifact generation.
- Treat `gillijimproject_refactor`, especially `src/MdxViewer` and `src/WoWMapConverter`, as the legacy or compatibility path when the task explicitly targets the current viewer, terrain work, or old-format behavior.
- Treat `archived_projects`, `WoWRollback/old_projects`, `WMOv14/old_sources`, and `gillijimproject_refactor/next` as non-primary unless the task explicitly targets them.

## Read First

- Before changing viewer, terrain, or format code, read `gillijimproject_refactor/memory-bank/activeContext.md`, `gillijimproject_refactor/memory-bank/progress.md`, `gillijimproject_refactor/memory-bank/data-paths.md`, and `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md` when it exists.
- Before changing `wow-viewer` PM4, shared I/O, dataset-builder ownership, or migration workflow, also read `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`, `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`, `gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md`, and `wow-viewer/README.md`.
- Before changing `wow-viewer` M2 runtime ownership, model rendering, skin handling, model lighting, shader or effect routing, or M2 performance work, also read `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`.
- Before working from archive-backed game clients, WoWArchive-mounted builds, or broad multi-client real-data validation, read `gillijimproject_refactor/memory-bank/data-paths.md`, `.codex/skills/wowarchive-client-staging/SKILL.md`, `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`, and `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`.
- Before changing workflow assets, read `.codex/README.md`, `.codex/prompts/wow-viewer-tool-suite-plan-set.md`, `.codex/prompts/wow-viewer-dataset-builder-plan.md`, `.codex/prompts/wow-viewer-editor-plan-set.md`, `.codex/prompts/wow-viewer-map-editing-foundation-plan.md`, `.codex/prompts/wow-viewer-editor-ui-surface-plan.md`, `.codex/prompts/wow-viewer-pm4-library-implementation.md`, `.codex/prompts/wow-viewer-shared-io-implementation.md`, `.codex/prompts/wow-viewer-world-runtime-plan-set.md`, `.codex/prompts/wow-viewer-m2-runtime-plan-set.md`, and `.codex/prompts/m2-cross-build-native-investigation.md`.
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

- Use `.codex/skills/wowarchive-client-staging/SKILL.md` for WoWArchive, `MountAll.bat`, WinFsp or `rman-mount`-backed client access, temp staged client copies, pruning stale staged clients, or choosing real client roots for wide validation and export workflows.
- Use `.codex/skills/wow-viewer-pm4-library/SKILL.md` for `Core.PM4` slices, `pm4 inspect`, `pm4 audit`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, PM4 regression updates, PM4 analyzer work, or narrow PM4 solver extraction from `MdxViewer`.
- Use `.codex/skills/wow-viewer-shared-io-library/SKILL.md` for `Core` or `Core.IO` non-PM4 slices such as ADT root, `_tex0.adt`, `_obj0.adt`, `_lod.adt`, WDT, WMO, BLP, DBC, or DB2 detection or summary work, chunk readers, `map inspect`, `converter detect`, or shared-format regression updates.
- Use `.codex/skills/wow-viewer-migration-continuation/SKILL.md` for continuation routing, next-slice selection, migration regrouping, or workflow-surface updates across chats.
- Use `.codex/skills/terrain-alpha-regression/SKILL.md` for terrain alpha-mask, MCAL, MCLY, split ADT texture, or blending regressions in `gillijimproject_refactor`.

## Agent Registry

- Use the `Explore` subagent for broad read-only repo discovery, especially when tracing client-root usage, archive-access seams, or prompt or skill coverage before editing multiple workflow assets.

## Game Client Access And Staging

- Canonical WoWArchive docs live at `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`, and the current mount entrypoint is `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`.
- The current batch mounts the deduplicated bundle read-only into `G:\WoW\WoWArchive-0.X-3.X\Mount`; treat that mount as a source surface, not a high-throughput working root.
- Existing fixed local client roots under `H:\CLIENTS\...` remain valid when they already exist locally.
- For repeated or wide export or audit or inspect or corpus or training-prep work against archive-backed clients, first copy the required client root into `i:/parp/parp-tools/output/tmp/wowarchive-clients` and process the staged copy instead of streaming directly from the mount.
- Delete staged client copies that are no longer needed so the temp area does not silently grow without bound.
- When reporting validation, say whether the proof used a fixed local root, a direct mounted archive path, or a staged local copy.

## Prompt Registry

- Use `.codex/prompts/wow-viewer-tool-suite-plan-set.md` to route broader `wow-viewer` planning asks to the right focused prompt.
- Use `.codex/prompts/wow-viewer-dataset-builder-plan.md` for planning the new `wow-viewer` dataset-builder tool, dataset explorer surface, supervised training-tooling cutover, ML corpus export ownership, terrain-supervision artifact ownership, manifest or harvest cutover, BYOD distribution constraints, or deciding what shared data seams must move out of `WoWMapConverter` before legacy exporter replacement.
- Use `.codex/prompts/wow-viewer-editor-plan-set.md` for broader editor-transition planning, including PM4 `MPRL`-assisted terrain conform, object-save ownership, map persistence, and viewer-vs-editor workspace routing.
- Use `.codex/prompts/wow-viewer-map-editing-foundation-plan.md` for planning the first true terrain or object editing or dirty-map or save pipeline slice.
- Use `.codex/prompts/wow-viewer-editor-ui-surface-plan.md` for planning viewer and editor workspace presets, editor task clustering, and panel reorganization.
- Use `.codex/prompts/wow-viewer-pm4-library-implementation.md` for the next narrow `Core.PM4` implementation slice.
- Use `.codex/prompts/wow-viewer-shared-io-implementation.md` for the next narrow shared `Core` or `Core.IO` format slice.
- Use `.codex/prompts/wow-viewer-world-runtime-plan-set.md` for staged `WorldScene` split work, negative asset lookup suppression such as repeated `.skin` miss churn, explicit terrain/WMO/MDX/overlay runtime service extraction, or WorldScene-to-wow-viewer world-runtime cutover planning.
- Use `.codex/prompts/wow-viewer-m2-runtime-plan-set.md` for staged M2 runtime ownership, exact `%02d.skin` handling, active section classification, material/effect routing, animation/lighting state, scene batching, or M2 consumer-cutover planning.
- Use `.github/prompts/m2-rendering-investigation.prompt.md` for diagnosing invisible M2 models, Ghidra-based native-client M2 render investigation on a live 3.3.5.12340 sandbox, adapter vertex/index validation against native ground truth, or renderer parity fixes for M2-family assets in `MdxViewer`.
- Use `.codex/prompts/m2-cross-build-native-investigation.md` for cross-build native M2 behavior recovery across multiple client branches (for example 3.3.5 through 6.x), including per-build anchor mapping, runtime breakpoint validation, and compatibility-matrix output before implementation.

## wow-viewer M2 And Runtime Guardrails

- Treat `wow-viewer` as the canonical implementation target for all new M2 runtime ownership, including skin-profile loading, section classification, render-pass routing, model lighting, shader or effect selection, and M2 performance work.
- Treat `wow-viewer/src/core/WowViewer.Core.Runtime` and future `wow-viewer` M2-owned library areas as the default home for new M2 renderer code. Do not keep deepening `gillijimproject_refactor/src/MdxViewer` as the design owner for those seams.
- Use `MdxViewer`, `WarcraftNetM2Adapter`, `noggit-red`, legacy tools, and native-client Ghidra work as extraction or reference inputs only unless the user explicitly asks for compatibility work in the old app.
- When a task is primarily reverse engineering or behavior recovery for M2, record the findings in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` and keep the continuity files in sync instead of leaving the evidence only in chat.
- If a change only proves library or build behavior in `wow-viewer`, say that explicitly. Do not imply active-viewer runtime signoff.

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

## wow-viewer Dataset Builder Guardrails

- Treat dataset corpus export, terrain-supervision artifact generation, manifest or harvest ownership, minimap cleanup, and shared mask or stitch or atlas semantics as canonical `wow-viewer` work when the user is asking for new behavior, refactor, or long-range cleanup.
- Put shared dataset contracts and artifact builders in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO`, then expose them through a new dedicated `wow-viewer` dataset tool instead of deepening `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM` as the design owner.
- Use `WoWMapConverter` and `MdxViewer` dataset or export code as extraction or compatibility references only unless the user explicitly asks for a bounded legacy hotfix.
- Do not create or preserve a second permanent dataset-builder pipeline in the legacy repo when the task is really about convergence.
- Keep ML training and inference scripts as downstream consumers; they should not become the canonical owners of format decode, artifact packing, or dataset-manifest semantics.
- Treat dataset-builder, dataset-explorer, and supervised training-tooling surfaces as Bring Your Own Data workflows. Do not plan around shipping copyrighted datasets, harvested corpora, model weights, or model outputs.
- Prefer reproducible configs, manifests, labels, and local-run commands over distributing trained artifacts or precomputed outputs from proprietary data.
- Do not hard-wire the long-range training/tooling architecture to CUDA-only assumptions. Keep backend seams open for alternate runners such as Vulkan or OpenCL or MLX where practical, even if CUDA remains the first implementation host.

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
