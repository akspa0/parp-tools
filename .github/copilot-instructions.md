# Project Guidelines

## Scope
- The active code path in this workspace is `gillijimproject_refactor`, especially `src/MdxViewer` and `src/WoWMapConverter`.
- Treat `archived_projects`, `WoWRollback/old_projects`, `WMOv14/old_sources`, and `gillijimproject_refactor/next` as non-primary unless the task explicitly targets them.

## First Reads
- Before changing viewer, terrain, or format code, read `gillijimproject_refactor/memory-bank/activeContext.md`, `gillijimproject_refactor/memory-bank/progress.md`, `gillijimproject_refactor/memory-bank/data-paths.md`, and `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`.
- If the task touches 3.3.5 terrain texturing, also read `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`.

## Context Lookup Workflow
- Start with concise context indexes before broad searches:
	- `gillijimproject_refactor/memory-bank/context-index.md`
	- `gillijimproject_refactor/src/MdxViewer/memory-bank/context-index.md`
- Use the local lookup tool for fragmented context:
	- Build index: `python gillijimproject_refactor/tools/doc_lookup.py build`
	- Query: `python gillijimproject_refactor/tools/doc_lookup.py query "your terms"`
- Prefer lookup queries over repeated broad file scans when the same topics recur.

## Build And Validation
- For parser and format-library work, prefer `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`.
- For viewer work, use `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- The active viewer and converter path currently has little to no first-party automated regression coverage. Do not claim terrain or alpha-mask changes are safe based only on library tests under `lib/*`, archived tests, or documentation examples.
- Prefer real-data validation using the fixed paths in `gillijimproject_refactor/memory-bank/data-paths.md`. Do not ask the user for alternate paths unless the existing fixed paths are missing.

## Terrain And Alpha Risk Area
- Treat commit `343dadfa27df08d384614737b6c5921efe6409c8` as the pre-regression baseline for terrain alpha-mask behavior unless the user specifies another baseline.
- High-risk files for alpha regressions include `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`, `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`, `src/MdxViewer/Terrain/TerrainRenderer.cs`, `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`, `src/MdxViewer/Terrain/TerrainChunkData.cs`, `src/MdxViewer/Export/TerrainImageIo.cs`, and `src/MdxViewer/ViewerApp.cs`.
- Any change to MCAL decode, edge-fix behavior, `_tex0.adt` texture sourcing, alpha packing, or shader blending must be checked against both Alpha-era terrain and LK 3.3.5 terrain.

## Conventions
- Keep FourCCs readable in memory and only reverse them at I/O boundaries.
- Preserve the existing split between `AlphaTerrainAdapter` and `StandardTerrainAdapter`.
- Favor minimal fixes over broad refactors in the terrain pipeline.
- If behavior, commands, or known risks materially change, update the relevant memory-bank file instead of leaving the old guidance stale.

## Custom Agents
Use these workspace agents (`.github/agents/`) for multi-step integration work:
- **@cherry-pick** — Surgical feature extraction from post-baseline commits. Classifies files by terrain risk, extracts safe files via `git show`, blocks risky terrain hunks without explicit approval.
- **@feature-impl** — Implements new renderer features (TXAN, ribbons, doodads, instancing, MH2O, etc.) following existing MdxViewer code patterns. Reads `implementation_prompts.md` automatically.
- **@terrain-validator** — Read-only terrain pipeline validation. Diffs current code against baseline `343dadfa`, classifies changes as SAFE/SUSPICIOUS/REGRESSION.
- **@build-check** — Post-merge build verification. Runs `dotnet build`, categorizes errors, suggests fixes without modifying code.
- **@commit-analyzer** — Pre-integration commit analysis. Breaks down commits into file-level risk classifications and extraction recommendations.

Typical integration workflow: `@commit-analyzer` → `@cherry-pick` → `@build-check` → `@terrain-validator`. Use `@feature-impl` for features with no existing code to cherry-pick (see `gillijimproject_refactor/src/MdxViewer/memory-bank/implementation_prompts.md`).


## Complex Task Execution (CRITICAL — avoid context loss)
- **Decompose aggressively**: Break multi-file or multi-step work into small, independent subagent tasks. Each subagent should own one bounded piece of work (e.g. one module, one endpoint, one test file).
- **Save early, save often**: After every meaningful unit of progress (a working function, a passing test, a completed module), immediately update `memory-bank/activeContext.md` and/or `memory-bank/progress.md` with what was done, what files changed, and what's next. Do NOT wait until the end of a session.
- **Checkpoint before large operations**: Before starting a batch of changes that could fill the context window, write a session checkpoint to memory-bank summarizing current state, plan, and any partial results.
- **Subagent result capture**: When a subagent completes, record its key outputs (files created/modified, test results, decisions made) to the memory bank before moving to the next subagent task.
- **Prefer many small subagents over one large task**: A 10-step plan should be 10 subagent calls with memory-bank saves between each, not one monolithic implementation pass.
- **Recovery-first mindset**: Write memory-bank state as if the next message could be a fresh context window that needs to pick up where you left off.
- **Always use the questions tool first for prompt/spec work**: For any task driven by `.github/prompts/*.prompt.md`, `memory-bank` plans, artifact plans, or any work with more than 3 meaningful steps, use the questions tool before implementation to confirm scope, acceptance gates, execution order, and blockers.
- **Always use Plan mode for prompt plans**: For complex prompt-planning documents and spec-driven tool or workflow improvements, the questions tool must be used in plan mode so implementation starts from a verified spec rather than assumptions.
- **Use spec-based flows for prompt planning documents**: Treat prompt plans as specs to verify against the live codebase, not as truth. Audit implementation status first, then trim or rewrite stale prompt docs into remaining-work docs before coding when needed.
- **Always use prompt plans**: For any task with more than 3 steps, create or update a detailed prompt plan that breaks the work into bounded subagent tasks and maps phases to files, tests, and acceptance gates.
- **Keep phase completion honest**: For prompt/spec-driven work, do not mark a phase complete until the code path, focused test, or real-data validation is actually proven.