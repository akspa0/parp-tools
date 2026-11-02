# Phase 5 Receipt: Generator/Patcher Tooling Upgrade & Legend Tally

- **Date**: 2026-11-01
- **Spec**: Spec 289 (PM4 3D Footprint Extrusion & Direct MPQ Matching)
- **Phase**: Phase 5 - Generator/Patcher Tooling Upgrade & Legend Tally

## Summary of Changes
1. **Semantic Palette Legend Live Counts (T013)**:
   - Resolved the `[0]` count bug across `Pm4ColorCodingService.cs` and `Pm4OverlayScene.Reports.cs`.
   - Populated `Pm4LegendEntry` with dynamic counts of visible objects matching `0x03` (Doodad), `0x10` (Lowest Floor), `0x11` (Standard Floor), `0x13` (Upper Floor), and `0x14` (Angled/Roof).
2. **Placement Generator PCA Orientation & Unrotated Dimensions (T014)**:
   - Upgraded `Pm4PlacementGeneratorSupport.cs` from naive AABB `ShapeOf(min, max)` to 2D convex hull PCA orientation via `Pm4FootprintExtruder.ComputePrincipalOrientation`.
   - Computed rotation-invariant unrotated shape `(Math.Min(L, W), Math.Max(L, W), H)` for both learning and prediction.
   - Preserved authentic planar Yaw in `Pm4GeneratedPlacement` and exported `rotYaw` in `pm4-generated-placements.csv`.
3. **ADT Patching Support Authentic Yaw & Centroid Output (T015)**:
   - Upgraded `Pm4AdtPatchSupport.cs` to calculate 2D footprint convex hulls and PCA principal axes.
   - Emitted authentic PCA Yaw rotations into `MODF` (`AdtWorldModelPlacement.Rotation.Z`) rather than writing zero.
   - Set placement position to the authentic 2D footprint centroid rather than the distorted AABB midpoint.

## Verification Commands & Output
1. `dotnet build I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug`
   - Exit Code: 0 (0 Errors, 10 Warnings)
2. `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
   - Exit Code: 0 (0 Errors, 361 Warnings)
3. `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug`
   - Exit Code: 0 (183 Passed, 0 Failed, 0 Skipped)
4. `pwsh -File "I:\parp\parp-tools\wow-viewer\eng\ui\Verify-UiReference.ps1"`
   - Exit Code: 0 (UI_REFERENCE_CHECK PASSED errors=0 warnings=0)

