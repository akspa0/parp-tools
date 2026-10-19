# Tasks — Spec 277: UniqueId Stratigraphy & Cross-Map Chronological Fork Analysis

## Phase 1: Core Models & Stratigraphy Engines (WowViewer.Core)

- [x] **T001**: Create `UniqueIdChronologyModels.cs` in `src/core/WowViewer.Core/Archaeology/`:
  - `UniqueIdRecord`: unified placement contract (M2, WMO, Creature, GameObject).
  - `UniqueIdLayerSummary`: detected work layers.
  - `MapStratigraphyProfile`: quantile statistics, layer breakdown, tile distribution.
  - `MapDivergenceReport`: shared placement count, percentage, divergence boundary ($UID_{div}$), inherited vs post-fork lists.
- [x] **T002**: Implement `UniqueIdStratigraphyEngine.cs` in `src/core/WowViewer.Core/Archaeology/`:
  - Calculate quantiles ($p_{10}, p_{25}, \text{median}, p_{75}, p_{90}$).
  - Cluster sequential IDs into distinct layers by adaptive gap thresholds.
  - Aggregate per-tile age and density summaries.
- [x] **T003**: Implement `CrossMapDivergenceEngine.cs` in `src/core/WowViewer.Core/Archaeology/`:
  - Spatial hash cell matching ($\le 0.5\text{ units}$) + model identity matching.
  - Compute $UID_{div}$ branching threshold.
  - Classify placements into `InheritedOverworld` vs `PostForkAuthored`.

---

## Phase 2: CLI Profiler & Divergence Reporter (WowViewer.Tool.Inspect)

- [x] **T004**: Implement `MapStratigraphySupport.cs` and wire `map stratigraphy` in `Program.cs`:
  - Support inputs from ADT/WDT files and JSON population partitions.
  - Support `--divergence --target <mapId> --ref <mapId>` to identify fork points.
  - Emit JSON reports to `output/reports/stratigraphy/`.

---

## Phase 3: In-Viewer Archaeology Workbench (WoWViewer)

- [x] **T005**: Enhance `ArchaeologyPanelService.Sidebars.cs`:
  - Add Map Stratigraphy & Divergence summary display.
  - Add smooth Time Machine slider scrubbing with percentage and era indicators.
  - Quick-jump milestone buttons ($p_{10}, p_{25}, \text{median}, p_{75}, p_{90}, 100\%$).
  - Per-layer soloing and scrubbing in Archaeology layers list.

---

## Phase 4: Verification, Tests & Receipts

- [x] **T006**: Author comprehensive unit tests in `tests/WowViewer.Core.Tests/UniqueIdStratigraphyTests.cs`:
  - Quantile computation and layer clustering.
  - Cross-map spatial twin matching and divergence calculation.
  - Test on real Map 451 and Map 0 data fixtures.
- [x] **T007**: Update `specs/STATUS.md` and `memory-bank/activeContext.md` with §9.2 receipts.

---

## Receipts (§9.2)

### Verification Commands & Results

1. **Unit Tests**:
   - Command: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~UniqueIdStratigraphyTests"`
   - Output: `Passed! - Failed: 0, Passed: 5, Skipped: 0, Total: 5, Duration: 200 ms`
   - Verified:
     - `Quantile_ComputePercentiles_MatchesExpectedDistribution`: synthetic ramp verify $p_{10}, p_{25}, \text{median}, p_{75}, p_{90}$.
     - `DetectLayers_IdentifiesDistinctGaps`: gap detection thresholding.
     - `CrossMapDivergence_DetectsTwinPlacementsAndDivergenceBoundary`: synthetic fork detection and $UID_{div}$.
     - `RealData_Map451_Profile_ComputesQuantilesAndLayers`: 0.5.3 Map 451 fixture parsing.
     - `RealData_Map451_DivergenceAgainstMap0_MatchesUpperLeftCornerPlacements`: cross-map twin matching between Map 451 and Map 0.

2. **CLI Inspection**:
   - Command: `dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map stratigraphy --population data/population/0.5.3/ --divergence --target 451 --ref 0`
   - Output:
     ```
     ===================================================================
     Cross-Map Divergence Report: Map 451 vs Reference Map 0
     ===================================================================
       Target Total Placements:    8
       Matched Common Placements:  8 (100.0%)
       Divergence Cutoff UniqueId: 400111

       * Matched 8/8 placements (100.0%) against reference map 0 (Map_0).
       * Divergence boundary UID: 400111. Placements with UniqueId > 400111 represent subsequent iterations authored after the map diverged.
     ===================================================================
     ```

3. **0.5.3 MDX/WMO Loading Performance Fix**:
   - Files changed:
     - `src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs`: Opt-in default for `_sqlAutoLoadOnMapChange = false`, async non-blocking load via `Task.Run`, deduplicated `ApplySqlSpawnsToScene` preventing constant scene graph dirtying on camera movement.
     - `src/viewer/WoWViewer/Terrain/Scene/ExternalSpawns/ExternalSpawnLayer.cs`: Replaced synchronous disk WMO group parsing with cached mesh summary / MOHD bounds.
     - `src/viewer/WoWViewer/Terrain/WorldAssetManager.cs`: `TryGetWmoPlacementBounds` uses `TryGetCachedWmoMeshSummary` avoiding disk I/O spikes during terrain tile loads.
     - `src/viewer/WoWViewer/Terrain/WorldAssetPathResolver.cs`: Prioritizes `.mdx`/`.mdl` when classic data source receives `.m2` request.
     - `src/viewer/WoWViewer/Population/WorldSpawnRecord.cs`: Mutable `ModelPath` for fast version adaptation.
