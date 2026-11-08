# Tasks 302: PM4 Semantic Color-Coding, Single-Object Disambiguation & Null-Safety Fix

## Phase 1: Null-Safety & Propagation Crash Fix
- [ ] T001: Make `WorldAssetPathResolver.NormalizeKey(string? path)` safely return `string.Empty` when input is null or whitespace.
- [ ] T002: Add null/empty guards in `WorldAssetManager.cs` for `QueueMdxLoad`, `PrioritizeMdxLoad`, `QueueWmoLoad`, `PrioritizeWmoLoad`.
- [ ] T003: Set explicit `ModelKey` in `Pm4MapWidePropagationService.cs` when constructing synthetic `ObjectInstance` from taught rules and donor placements, and guard `FlushPendingScenePlacements`.

## Phase 2: Semantic Color-Coding Engine & Viewport Rendering Fix
- [ ] T004: Update `Pm4ColorCodingEngine.ComputeSemanticCategoryColor` to evaluate `dominantGroupKey` (`MSUR.0x08` SurfaceClass) first, mapping `0x03` to `ColorDoodad`, `0x10` to `ColorFloorLowest`, `0x11`/`0x12` to `ColorFloorCommon`, `0x13` to `ColorFloorUpper`, `0x14` to `ColorFloorAngled`, `0x01` to `ColorTerrainLinker`, `0x02` to `ColorConnector`.
- [ ] T005: Update `Pm4OverlayScene.Reports.cs` `GetPm4LegendColor` for `SemanticCategory` to pass `(byte)value` as `dominantGroupKey`.
- [ ] T006: In `Pm4OverlayMeshRenderer.cs` `RebuildTileMeshGpuBuffer`, preserve semantic surface colors when `SemanticCategory` is active by subtly blending with selection yellow (`Vector3.Lerp(color, Pm4ColorSelection, 0.25f)`) rather than clobbering 100% of the mesh into yellow.

## Phase 3: Single-Object Semantic Disambiguation & Seam Strictness
- [ ] T007: Extend `Pm4SpatialClustering.ClusterItems` with `Func<T, int, byte>? surfaceClassSelector` to enforce doodad isolation (`0x03` never clusters with `0x10..0x16`), separate doodad independence, and 2D horizontal footprint overlap ($overlapX > 0.5\text{ yd}$, $overlapY > 0.5\text{ yd}$) for vertically stacked floor slices.
- [ ] T008: In `Pm4SpatialClustering.ClusterCrossTileCompounds`, remove the blind fallback that merged all compounds lacking MSCN points across seams.
- [ ] T009: In `Pm4CompoundObjectService.AssembleTileCompoundObjects`, pass `surfaceClassSelector: (h, idx) => h.DominantGroupKey`.

## Phase 4: Verification, Automated Testing & Governance
- [ ] T010: Add unit tests in `Pm4SpatialClusteringTests.cs` verifying doodad isolation, separate doodads non-merging, side-by-side structures non-merging, and cross-tile compounds non-merging without MSCN.
- [ ] T011: Add unit tests in `Pm4ColorCodingEngineTests.cs` verifying semantic color assignments for all surface classes.
- [ ] T012: Build solution cleanly with 0 errors (`dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -m:1`).
- [ ] T013: Run test suites and ensure all tests pass (`dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj`).
- [ ] T014: Verify UI reference with 0 errors/warnings (`pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1`).
- [ ] T015: Write compliant receipt in `evidence/receipt-spec302.md`, update `STATUS.md` and `activeContext.md`.
