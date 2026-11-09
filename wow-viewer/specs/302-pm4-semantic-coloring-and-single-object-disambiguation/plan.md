# Plan 302: PM4 Semantic Color-Coding, Single-Object Disambiguation & Null-Safety Fix

## Technical Architecture & Design

### Phase 1: Null-Safety & Crash Prevention
1. Modify `WorldAssetPathResolver.NormalizeKey(string? path)`:
   - Check `string.IsNullOrWhiteSpace(path) ? string.Empty : path.Replace('/', '\\').ToLowerInvariant();`.
2. Modify `WorldAssetManager.cs`:
   - Guard `QueueMdxLoad`, `PrioritizeMdxLoad`, `QueueWmoLoad`, `PrioritizeWmoLoad` with `if (string.IsNullOrWhiteSpace(normalizedKey)) return;`.
3. Modify `Pm4MapWidePropagationService.cs`:
   - Set `ModelKey = !string.IsNullOrWhiteSpace(taught.ModelPath) ? taught.ModelPath : taught.ModelName` when creating `newInst`.
   - Set `ModelKey = !string.IsNullOrWhiteSpace(donor.ModelKey) ? donor.ModelKey : (!string.IsNullOrWhiteSpace(donor.ModelPath) ? donor.ModelPath : donor.ModelName)` when creating donor instances.
   - In `FlushPendingScenePlacements`, check `if (!string.IsNullOrWhiteSpace(inst.ModelKey))`.

### Phase 2: Semantic Color-Coding Engine & Viewport Rendering
1. In `WowViewer.Core.PM4.Services.Pm4ColorCodingEngine`:
   - Update `ComputeSemanticCategoryColor(byte ck24Type, byte dominantGroupKey)`:
     - Derive `byte surfaceClass = dominantGroupKey != 0 ? dominantGroupKey : ck24Type;`.
     - Switch on `surfaceClass` for `0x03` (Doodad), `0x10` (Lowest floor), `0x11`/`0x12` (Common floor), `0x13` (Upper floor), `0x14` (Roof), `0x01` (Terrain linker), `0x02` (Connector).
2. In `WoWViewer.Terrain.Pm4.Pm4OverlayScene.Reports.cs`:
   - In `GetPm4LegendColor`, pass `(byte)value` as `dominantGroupKey`:
     `Pm4OverlayColorMode.SemanticCategory => Pm4ColorCodingService.Instance.GetSemanticCategoryColor(0, (byte)value)`.
3. In `WoWViewer.Terrain.Pm4.Services.Pm4OverlayMeshRenderer`:
   - In `RebuildTileMeshGpuBuffer`:
     When `overlay._pm4ColorMode == Pm4OverlayColorMode.SemanticCategory`, do not replace `color` with `Pm4ColorSelection`. Blend 25% with selection yellow (`Vector3.Lerp(color, Pm4ColorSelection, 0.25f)`), preserving the semantic surface color readability while highlighting selected objects via the yellow bounding box.

### Phase 3: Single-Object Semantic Disambiguation & Cross-Tile Seam Strictness
1. In `WowViewer.Core.PM4.Services.Pm4SpatialClustering`:
   - Extend `ClusterItems<T>` to accept `Func<T, int, byte>? surfaceClassSelector = null`.
   - Implement disambiguation rules:
     - Non-zero `LinkGroupObjectId` equality always clusters.
     - Doodad isolation: Doodads (`0x03`) NEVER cluster with non-doodads (`0x10..0x16`).
     - Separate doodads only cluster if sharing `LinkGroupObjectId` or having near-identical center ($\le 0.25\text{ yd}$).
     - WMO structures only cluster if they have substantial $(X, Y)$ 2D horizontal overlap ($overlapX > 0.5\text{ yd}$ and $overlapY > 0.5\text{ yd}$) and vertical elevation gap $\le tolerance$. Side-by-side structures touching along edges or separated horizontally do not cluster.
   - In `ClusterCrossTileCompounds`:
     - Eliminate the blind fallback that merged all compounds lacking MSCN points across seams.
2. In `WoWViewer.Terrain.Pm4.Services.Pm4CompoundObjectService`:
   - Pass `surfaceClassSelector: (h, idx) => h.DominantGroupKey` to `Pm4SpatialClustering.ClusterItems`.

### Phase 4: Unit Testing, Verification & Governance
1. Add unit tests in `Pm4SpatialClusteringTests.cs`:
   - Test doodad isolation from WMO floors.
   - Test separate doodads non-merging.
   - Test side-by-side structures non-merging.
   - Test cross-tile compounds non-merging without MSCN.
2. Add unit tests for `ComputeSemanticCategoryColor` in `Pm4ColorCodingEngineTests.cs`.
3. Run test suites, compile solution, verify UI reference.
4. Record receipt in `evidence/receipt-spec302.md`.

