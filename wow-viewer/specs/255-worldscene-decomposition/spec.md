# Spec 255 — WorldScene Decomposition

**Created**: 2026-09-26 · **Branch**: `v0.6.0-dev` · **Status**: Draft — plan awaiting operator approval (T001)
**Parent**: Epic 251 item U-01 ([spec](../251-epic-viewer-ux-and-code-health/spec.md)); continues its E1/E2/E4
steps. Governance: AGENTS.md §9 (scope, receipts) and §10 (god-class freeze).

## Operator direction (2026-09-26)

*"yes, write up a spec for that, so we can get the WorldScene cut down too."* — in reply to the proposal
that `WorldScene.cs` (still 8,275 lines, over 4× the 2,000-line budget) be split beyond the approved U-01
steps, following the same method that took the `ViewerApp` class from 42,973 to 2,512 lines.

## Problem

`src/viewer/WoWViewer/Terrain/WorldScene.cs` holds every world-scene feature in one class: tile streaming,
instance building, visibility, the 1,844-line `Render()`, taxi actors, lighting/fog, skybox, filters,
selection and hover. An agent changing any one of them loads a file four times the budget, and shared
private state makes every change risky. E1 (PM4 overlay, 17,175 → 8,326) and E4 (selection policy,
→ 8,275) removed two features; the rest is still in place.

## Measured baseline (2026-09-26, Roslyn member map of `WorldScene.cs`)

8,275 file lines; 8,061 lines of class members:

| Cluster | Lines | Members |
|---|---|---|
| `Render()` (E2 — already planned) | 1,844 | 1 |
| Hover / pick / wireframe reveal (viewer side of E4) | 1,099 | 49 |
| Tile streaming & instance build (`InitFromAdapter`, `BuildInstances`, `OnTileLoaded`, instance lists, scene-graph index) | 632 | 17 |
| Taxi actors & routes | 404 | 18 |
| Frame visibility collection (scene-graph frame visibility, visible WMO/MDX collection) | 390 | 15 |
| UniqueId / object-path filters + archaeology layer queries | 352 | 35 |
| Frame record types (`WorldRenderFrame`, visibility buckets, nested records) | 351 | 49 |
| Render-path planning & deferred asset loads | 334 | 13 |
| Selected-object resolution | 316 | 12 |
| Selected placement edit / move | 262 | 13 |
| Camera-path collision & terrain height sampling | 243 | 9 |
| Bounds & visibility buckets | 238 | 8 |
| External (SQL) spawns | 186 | 7 |
| Skybox | 179 | 10 |
| Lighting / LIT / fog | 151 | 9 |
| PM4 MPRL yaw helpers left behind by E1 | 134 | 8 |
| Frame stats & time of day | 81 | 4 |
| Constructors, `Dispose`, scattered fields/properties | 865 | — |

## User stories

- **US1 (P1)** — As an agent or developer changing one world-scene feature (taxi actors, lighting, filters,
  selection, streaming), I open that feature's own file, under 2,000 lines, not `WorldScene.cs`.
- **US2 (P1)** — As the operator, I get the same scene: every step is behaviour-preserving and
  independently revertible, with an operator smoke for what it moved.
- **US3 (P2)** — As a maintainer, `WorldScene` ends as a shell (state, composition, frame orchestration)
  that fits the 2,000-line budget once E2 (`Render()` split) lands.

## Acceptance criteria

1. Each step moves one cluster into an owned class; moved bodies are verbatim (line-multiset audit shows
   only intended lines changed) unless the step is explicitly a policy extraction like E4.
2. No new members in `WorldScene` beyond one field per service and host-contract plumbing (AGENTS.md §10);
   no new partial `WorldScene` file; every file < 2,000 lines.
3. Per step: `dotnet build WowViewer.slnx` 0 errors; `dotnet test WowViewer.slnx` failure set unchanged;
   viewer warning set compared; receipt per §9.2.
4. `WorldScene.cs` ≤ ~2,000 lines after all steps including E2.
5. Runtime behaviour (streaming, rendering, FPS, selection, taxi, lighting, sky) is claimed only from
   operator smoke/captures, never from build or tests.

## Constraints and out of scope

- Behaviour-preserving only: no renderer, streaming, lighting or selection change rides on a step.
- Steps touching the render path (instance store, visibility, render-path planning, `Render()` itself) wait
  for R-10's after-capture (Epic 249), exactly as E2 already does, so the lighting fix is measured first.
- Format readers, `WorldAssetManager`, `TerrainManager`, renderers (`WmoRenderer`, `MdxRenderer`, …) are
  not modified (AGENTS.md §4).
- No UI changes; `ViewerApp` callers change only receivers (`_worldScene.X` → `_worldScene.<Service>.X`).
