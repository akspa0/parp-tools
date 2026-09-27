# Receipt — R-10 per-placement WMO batch gate reverted (2026-09-27)

Operator report: modern-data rendering under 1 FPS since R-10 (`2c94ccd`); operator chose "Revert the gate
now". Covers **R10-T009**. No FPS, draw-call or visual result is claimed.

## Files changed

| File | Change |
|---|---|
| `src/viewer/WoWViewer/Terrain/WorldScene.cs` | WMO opaque candidate loop: `canBatch` again requires `_sceneLightManager.Count == 0` (pre-R-10 whole-scene gate, `b53425b`); the per-placement `GetWorldBounds` + `AnyAffecting` query is removed and `reachedByLight` stays `false`; comments updated. No other line changed. |

Kept unchanged: `WorldObjectPassCoordinator` lit partition + tests, R-10a counters, R-10c light culling,
R-10d `SceneLightManager` grid. The batch path's `sceneLights: null` stays exact (a batch now only forms
when no light is active).

## Why (code reading, not a measurement)

- Pre-R-10: any active light kept every WMO per-placement; modern data always has lights.
- R-10b: every unlit placement of a portal-less WMO went to `EndGpuInstanceBatch`, including single-placement
  models (the planner batches any count ≥ 1). That route draws every manually visible group per instance
  with no group frustum admission (`WmoRenderer.SupportsGpuInstancedOpaque` doc comment states the trade),
  and `CollectOpaqueDoodadsForPlacement` submits every doodad within the 6,000-unit range with
  `respectRuntimeDoodadVisibility: false` — no visible-group filter and no view test.
- The operator's symptom (FPS far below the pre-R-10 ~5.5) fits more GPU/draw work, not more CPU light work.

## Commands and exit status (agent host, Linux)

| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true --no-incremental` | 0 | 0 errors; compiler warnings identical to the Spec 255 baseline |
| `dotnet test WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 1 | same 26 pre-existing environmental failures (Core.Tests 1,598/18/1, Core.PM4.Tests 95/8, Editor 91/0, Curation 39/0) |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| WMO path identical to pre-R-10 when lights exist | `canBatch` expression matches `b53425b` | code |
| Modern-data FPS back to at least the pre-R-10 level | — | **not claimed; operator check owed** |

## Operator check (PowerShell 7)

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

Load `wow_classic_beta` 1.60.1 `Azeroth` at the usual camera spot. Frame history → *Submission efficiency*
should show **0 WMO placements batched** whenever scene lights are kept; note FPS and WMO draw calls. If
FPS is still far below the pre-R-10 build, run `b53425b` in a worktree at the same spot to separate R-10
from later changes.
