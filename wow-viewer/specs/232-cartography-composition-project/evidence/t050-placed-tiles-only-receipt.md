# Receipt — Spec 232 T050: placed tiles only

**Date**: 2026-09-08 · **Spec**: [232](../spec.md) · **Task**: T050 / FR-13

## Files changed

| File | Change |
|---|---|
| [PhaseComposition.cs](../../../src/core/WowViewer.Core/Maps/PhaseComposition.cs) | Adds the backward-compatible `UsePlacedTilesOnly` layer mode. `ResolveTileSource` now returns no source for an unplaced target in that mode, while the legacy default still falls through to the whole-layer offset. |
| [PhaseLayerProjectFile.cs](../../../src/core/WowViewer.Core/Maps/PhaseLayerProjectFile.cs) | Persists the mode with the rest of the layer state; absent values from existing projects remain `false`. |
| [ViewerApp_PhaseLayers.cs](../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | Adds the layer toggle and changes the donor-tile picker to create an explicit `PhaseTilePlacement` in placed-only mode. It retains the computed offset so turning the mode off restores the prior whole-map offset behavior. |
| [WorldScene.cs](../../../src/viewer/WoWViewer/Terrain/WorldScene.cs), [ViewerApp_MinimapAndStatus.cs](../../../src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs) | Shows only placed targets in the minimap footprint and prevents the whole-layer offset drag from misleadingly editing a placed-only layer. |
| [PhaseTileSourceTests.cs](../../../tests/WowViewer.Core.Tests/Maps/PhaseTileSourceTests.cs), [PhaseLayerProjectFileTests.cs](../../../tests/WowViewer.Core.Tests/Maps/PhaseLayerProjectFileTests.cs) | Cover exclusive routing, empty placement sets, clone state, and project round-trip persistence. |
| [tasks.md](../tasks.md), [activeContext.md](../../../memory-bank/activeContext.md), [progress.md](../../../memory-bank/progress.md) | Marks T050 with this receipt and records the next bounded task. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet test tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~PhaseTileSourceTests\|FullyQualifiedName~PhaseLayerProjectFileTests\|FullyQualifiedName~MapFootprintTests" --logger "console;verbosity=minimal"` | 0 | **37/37 passed**. |
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**; 306 warnings, including existing NU1903 advisories. |
| `git diff --check` | 0 | No whitespace errors. Git emitted unrelated user-config access and CRLF notices. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| The donor picker places only the requested donor/target tile | The picker creates one `PhaseTilePlacement` and enables `UsePlacedTilesOnly`; `PlacedTileOnlyMode_UnmappedTarget_IsEmptyInsteadOfUsingOffset` proves an otherwise valid offset source cannot fill a different target. | Pass (source + 37/37 focused tests) |
| A placed-only layer does not compose a map when no valid target is placed | `PlacedTileOnlyMode_WithoutPlacements_IsEmpty` proves the policy returns empty rather than the offset route. | Pass (37/37 focused tests) |
| Operator can return a layer to offset mode | The Layers panel has `Compose placed tiles only`; disabling it restores the existing offset fallback, retained as `UnmappedTarget_WithPerTileMappings_FallsBackToOffset`. | Pass (source + 37/37 focused tests) |
| The mode survives a layer-project save/load | `PhaseLayerProjectFileTests.RoundTrip_PreservesAllFields` asserts the true value after JSON round-trip; false is the DTO default for prior project files. | Pass (37/37 focused tests) |
| Minimaps reflect placed-only composition | `WorldScene.GetLayerFootprints` returns only valid placement targets, and the minimap drag rejects placed-only footprints because tile offsets do not control those targets. | Pass (source + full solution build) |

No live viewer or real-client screenshot was run in this slice. T015d remains a separate operator-owned
visual gate.
