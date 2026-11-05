# Receipt — 2026-11-04 — PM4 hull alignment defect fixes (Phase 1 correction)

## Follow-up: X-axis still off (operator re-test)

Operator re-tested after the first pass: alignment improved but still off on X. Audit of the
placement math found the transform solve itself is exact — `position = centroid − rotCenter`
with `rotCenter` computed via the same `WorldPlacementTransform.Build` (incl. the `RotZ(PI)`
half-turn) that renders the model, and `WorldScene.TransformBounds` uses the same row-vector
convention. The residual X/Y bias was in the **target point**:
`Pm4FootprintExtruder.ComputePrincipalOrientation` returned the **mean of the hull vertices**
as the footprint center. On irregular footprints (L-shaped buildings, wings, bridges) the
vertex mean is pulled toward vertex clusters, so every placement solved against
`prism.Centroid` sat off the hull's visual center on X/Y. Symmetric test rectangles
(mean == center) never caught it.

Fix: `ComputePrincipalOrientation` now returns the **oriented bounding box center** — the
midpoint of the principal-axis projections (already computed for length/width). Identical to
the vertex mean for symmetric shapes; unbiased for everything else. Regression test added
(`ComputePrincipalOrientation_CenterIsOrientedBoundingBoxCenterNotVertexMean`, L-shape:
vertex mean (4,4) vs OBB center (3,3), which also matches the polygon area centroid).
Consumers verified: `AlignPlacementToPm4Hull`, `MatchesTaughtShape`, bulk magnetize
(`BuildUnifiedPm4Candidates` uses the unbiased AABB center), duplication engine, and the
generator/patcher tools all flow through this single center.

Validation: test-project build 0 errors (full-solution build blocked only by the running
viewer process holding `WowViewer.Core.PM4.dll` — MSB3027 file lock, not a compile error);
11/11 focused PM4 duplication/matcher/extrusion tests pass. Visual confirmation remains
operator-owned.

Operator report: matched/placed models (magnetized/duplicated) do not line up with PM4 hulls.
Investigation found four concrete defects in the placement-solving paths; all are corrections to
the Phase 1 (T001/T002) contract, which was checked but not fully implemented.

## 1. Files changed

| File | Change |
|---|---|
| `src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs` | `FitModelToTarget`: (B) Z solve now uses hull **top** `target.BoundsMax.Z` per the T001 formula `P_Z = Z_pm4_top − H_local_top·s` (was hull vertical center `WorldCentroid.Z`, sinking models by half the hull height); WMO templates base-align to `BoundsMin.Z` (origin authored at placement Z); (C) model long-axis +90° correction (same 1.15 rule as `AlignPlacementToPm4Hull`/`Pm4DirectModelMatcher`) so Y-long models stop duplicating crosswise; (D) WMO templates forced to scale 1.0 inside the engine (caller already placed at 1.0 but the position solve used the fitted ratio); informational world bounds now reflect the solved placement Z. |
| `src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs` | `DuplicateModelToUnmatchedPm4Doodads`: `Pm4DoodadTarget` now receives **world-space** bounds (`TransformBounds` by `BaseTransform`) and world triangle vertices, so the engine's top/floor Z solve and PCA yaw operate in the same frame as `WorldCentroid` (was mixing local bounds with a world centroid). `MatchesTaughtShape`: (A) position is re-solved for the final taught yaw (`prism.YawDegrees + taught.YawOffset`) via the same `rotCenter` math as `AlignPlacementToPm4Hull`; previously the position came from `MatchPrism` solved for the +90° heuristic yaw while the applied rotation used the taught yaw, so taught placements drifted off-hull whenever the two yaws differed. WMO Z now prefers the Ck24 placement Z with `prism.BaseZ` fallback, mirroring `AlignPlacementToPm4Hull`. |
| `src/core/WowViewer.Core.PM4/Caching/pm4PerFileCacheService.cs` | (E) stale doc comment corrected: PM4F/PM4C are both version 9 (Spec 130 bumped them together); comment still said 8/7. No behavior change; no cache bump needed — these fixes change placement solving, not cached hull geometry. |
| `tests/WowViewer.Core.PM4.Tests/Pm4DoodadDuplicationTests.cs` | `FitModelToTarget_ComputesScaleRatioAndCentroidCorrectly` updated to the T001 contract: target bounds are world-space (top 104 / floor 96), `Position.Z` asserted 100 (top of model at hull top), `WorldBoundsMax.Z` asserted 104. The old assertions encoded the defect (top at hull center Z=100 → position 96). |

Unchanged by design: `AlignPlacementToPm4Hull` and `MagnetizeTilePlacementsToPm4` were verified
internally consistent (position and rotation solved from the same yaw; top-Z from hull top), so no
change was needed there. `MagnetizeTilePlacementsToPm4` passes null MPRL/placementZ to the extruder
but recomputes Z from Ck24/bounds directly, so its placements are unaffected.

## 2. Verification commands & exit status

```
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --nologo -v q
  → exit 0 (0 errors, 372 warnings — all pre-existing)

dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj `
  -c Debug --no-build `
  --filter "FullyQualifiedName~Pm4DoodadDuplication|FullyQualifiedName~Pm4DirectModelMatcher|FullyQualifiedName~Pm4FootprintExtrusion" `
  --nologo
  → exit 0 — Passed! Failed: 0, Passed: 10, Skipped: 0, Total: 10
```

## 3. Criterion → evidence table

| Criterion (T001/T002 contract) | Evidence |
|---|---|
| Doodad duplication places model top at PM4 collision top: `P_Z = Z_pm4_top − H_local_top·s` | `FitModelToTarget` now computes `targetZ = target.BoundsMax.Z − modelTopZ` with world-space `BoundsMax`; unit test asserts `Position.Z = 104 − 2·2 = 100` and `WorldBoundsMax.Z = 104` (hull top) — real test output, 10/10 pass. |
| Magnetization keeps top-aligned placement (no regression) | `AlignPlacementToPm4Hull` / `MagnetizeTilePlacementsToPm4` code inspected: both already use hull top (`prism.TopZ` / candidate `Max.Z`) and solve position+rotation from one yaw; no change required. |
| Taught-pairing automation lands on hull centroid | `MatchesTaughtShape` re-solves position for the taught yaw with the same `rotCenter` math as `AlignPlacementToPm4Hull`; drift term `(R(yawB) − R(yawA))·(center·scale)` is now zero by construction. |
| Duplicated elongated models align long axis to hull principal axis | Same 1.15 span rule applied in `FitModelToTarget`; PCA yaw path now fed world vertices by the production caller. |
| WMO duplication at authentic 1:1 scale | Engine forces `fittedScale = 1.0` for `.wmo` templates and base-aligns Z; caller's 1.0 override no longer contradicts the position solve. |

## 4. Remaining operator-owned verification

Visual confirmation in the viewer (real client data): duplicate a template into PM4 doodad hulls
and magnetize a tile, then confirm duplicated/magnetized models sit flush on the hulls. Compilation
and unit tests do not constitute runtime/visual proof (§9.2).
