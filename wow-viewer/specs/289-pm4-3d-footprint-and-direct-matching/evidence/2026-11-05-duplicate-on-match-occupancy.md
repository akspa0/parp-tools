# Receipt — Duplicate-on-match occupancy fix (2026-11-05)

Operator report: "matching things got me multiple copies even when the pm4 already has an
object assigned/paired."

## Root cause

Every duplication/matching path decided whether a PM4 hull was already occupied by comparing the
hull's **world centroid** against each existing placement's **`PlacementPosition`** (the model
origin) with a small radius. WMO and most M2 placement origins sit at the model base, not the
hull centroid, so an already-paired hull routinely measured as "free" and received a duplicate
copy. Three secondary defects compounded it:

1. Only the same tile key `(tileX, tileY)` was scanned; PM4 tiles are sometimes keyed with
   swapped coordinates, so placements on the transposed key were invisible.
2. `SpawnCandidateIntoPm4Hull` and `InstantiateTaughtPairingIntoHull` had **no** occupancy guard,
   and the "Verify & Propagate" UI path spawns into the selected hull and then propagates across
   all tiles including that hull — a guaranteed double spawn.
3. The occupancy radius was derived from the taught model's footprint (or a fixed 8/16 yds), not
   from the hull, so a small model placed off-centre inside a large hull was never detected.

## Fix

- `Pm4DoodadDuplicationEngine` (`src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`):
  - New `PlacementBounds` record (world AABB + origin + `BoundsResolved`).
  - New `IsTargetOccupied(target, placements, originFallbackRadius)` — world-space AABB
    intersection on all three axes; origin-distance fallback only when bounds are unresolved.
  - New `FindUnmatchedTargets` / `DuplicateAcrossTargets` overloads taking `PlacementBounds`.
- `Pm4DoodadDuplicationService` (`src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`):
  - `BuildPlacementBounds` collects placements from both `(tileX, tileY)` and `(tileY, tileX)`.
  - `BuildTarget` builds a world-space `Pm4DoodadTarget` from a `Pm4OverlayObject`.
  - `PopulateTileFromTaughtPairing` and `DuplicateModelToUnmatchedPm4Doodads` now use bounds
    occupancy; newly added instances are appended to the bounds list so a single run cannot
    duplicate within itself.
  - `SpawnCandidateIntoPm4Hull` / `InstantiateTaughtPairingIntoHull` gained a `force = false`
    guard that blocks a second copy of the **same** model in the same hull (different models in a
    multi-model hull remain allowed).
- `Pm4CurationAndSpatialMatchingService.IsHullOccupied` now uses the same AABB test, so the
  workbench's "unoccupied hulls" counter matches the actual spawn behaviour.

## Files changed

- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4CurationAndSpatialMatchingService.cs`
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4DoodadDuplicationTests.cs`

## Verification

| Command | Result |
| --- | --- |
| `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --nologo -v m` | exit 0, 0 errors (455 pre-existing warnings) |
| `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "FullyQualifiedName~Pm4DoodadDuplicationTests"` | exit 0, 6/6 passed |

New regression tests:

- `IsTargetOccupied_DetectsPairedHullWhenModelOriginIsOffsetFromCentroid` — WMO origin 15 yds
  from the hull centroid, bounds overlapping → occupied (old test would have said free).
- `FindUnmatchedTargets_BoundsOverload_RejectsOccupiedHullAndKeepsFreeHull`.
- `DuplicateAcrossTargets_BoundsOverload_DoesNotDuplicateOnPairedHull`.

## Criterion → evidence

| Criterion | Evidence |
| --- | --- |
| A hull that already has a paired model is not duplicated | `IsTargetOccupied_*` and `DuplicateAcrossTargets_BoundsOverload_*` tests pass |
| Transposed tile keys are considered | `BuildPlacementBounds` collects both keys; covered by code review (no unit harness for `IPm4OverlayHost`) |
| Explicit spawn cannot create a second copy of the same model | `HullHasSameModel` guard in both spawn methods; build 0 errors |
| UI "unoccupied hulls" count matches spawn behaviour | `IsHullOccupied` now calls the same engine helper |

## Owed (operator-owned)

Visual confirmation on real client data: run "Verify & Propagate" and "Fill Unmatched PM4 Doodad
Hulls" on a tile that already has paired models and confirm no duplicate copies appear.
