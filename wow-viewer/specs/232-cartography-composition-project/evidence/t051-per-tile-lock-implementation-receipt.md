# Receipt — Spec 232 T051: per-tile lock implementation

**Date**: 2026-09-08 · **Spec**: [232](../spec.md) · **Task**: T051 / FR-14

## Files changed

| File | Change |
|---|---|
| [PhaseComposition.cs](../../../src/core/WowViewer.Core/Maps/PhaseComposition.cs) | Adds an optional `Locked` member to an explicit `PhaseTilePlacement`, plus I/O-free predicates for a placement lock and an earlier enabled layer's lock. Invalid placements and offset-only coverage cannot lock a target. |
| [PhaseLayerProjectFile.cs](../../../src/core/WowViewer.Core/Maps/PhaseLayerProjectFile.cs) | Round-trips each placement's lock bit; older projects default it to false. |
| [AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs), [StandardTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs) | In stack order, a locked placement claims its target before donor resolution; later phase layers are skipped at that target in both terrain routes. |
| [MinimapHelpers.cs](../../../src/viewer/WoWViewer/MinimapHelpers.cs) | Later-layer minimap texture and footprint rendering honor preceding locks; the lock owner draws a distinct `L` badge. |
| [ViewerApp_PhaseLayers.cs](../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | Adds one lock toggle per explicit donor-to-target placement, with persistence and minimap behavior explained in its tooltip. |
| [PhaseTileSourceTests.cs](../../../tests/WowViewer.Core.Tests/Maps/PhaseTileSourceTests.cs), [PhaseLayerProjectFileTests.cs](../../../tests/WowViewer.Core.Tests/Maps/PhaseLayerProjectFileTests.cs) | Cover valid-target scope, earlier-layer exclusion, disabled-owner behavior, clone preservation, and JSON round-trip. |
| [tasks.md](../tasks.md), [activeContext.md](../../../memory-bank/activeContext.md), [progress.md](../../../memory-bank/progress.md) | Records the implementation proof and the remaining visual witness. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 1 | The first integration build found `targetLockedByEarlierLayer` declared in the wrong Alpha adapter loop. No artifact was accepted from this result. |
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**; 552 existing warnings, including NU1903 advisories. |
| `dotnet test tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~PhaseTileSourceTests\|FullyQualifiedName~PhaseLayerProjectFileTests\|FullyQualifiedName~MapFootprintTests" --logger "console;verbosity=minimal"` | 0 | **39/39 passed**. |
| `git diff --check` | 0 | No whitespace errors. Git emitted unrelated user-config access and CRLF notices. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| A lock belongs to one explicitly mapped target | `IsTargetLockedByLayer` accepts only a valid locked placement, and `LockedPlacement_ClaimsOnlyItsExplicitTarget` checks the target boundary. | Pass (39/39 focused tests) |
| Later contributing layers cannot replace a lock owner | Both adapters set the target claim while iterating the stack; `EarlierLockedPlacement_BlocksOnlyLaterContributingLayers` checks the shared ordering predicate, including a disabled owner. | Pass (source + 39/39 focused tests + solution build) |
| Lock state persists in the layer project | `PhaseLayerProjectFileTests.RoundTrip_PreservesAllFields` verifies the placement lock after save/load. | Pass (39/39 focused tests) |
| Operator can set a tile lock | The expanded layer card lists each placement with a `Lock target (x, y)` control. | Implemented (viewer build); live interaction not run |
| Minimap makes the lock visible and does not portray skipped later coverage | The owner draws an `L` badge; later texture and footprint paths ask the shared earlier-lock predicate before drawing. | Implemented (viewer build); operator visual witness open |

## Remaining operator witness

Load two enabled phase layers with different donor content targeting the same base tile. Lock the
earlier layer's placement, then capture the minimap showing its `L` badge and the composed tile
remaining owned by that earlier layer. The later layer must not replace the target. This is the
remaining visual acceptance evidence, so T051 stays unchecked.
