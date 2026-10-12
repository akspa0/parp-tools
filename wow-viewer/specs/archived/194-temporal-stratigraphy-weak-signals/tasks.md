# Tasks: Temporal Stratigraphy & Weak Signal Development Mesh Restoration

## Phase 1: Core Stratigraphy Analysis & Classification (US1)

- [x] T001 Define `TemporalStratum` enum and `StratigraphyAnalysisResult` records in `WowViewer.Core.Runtime/World/Terrain/Stratigraphy/TemporalStratum.cs`
- [x] T002 Implement `StratigraphyLevelAnalyzer.cs` computing `surviving_height_levels`, raw MCVT delta distributions, and amplitude metrics across $16 \times 16$ chunk grids without altitude floor bias
- [x] T003 Implement `SeamDiscontinuityProfiler.cs` evaluating C0 step and C1 slope gradient across 15 internal MCNK boundaries, identifying 2x2 and 4x4 sub-tile merge spikes
- [x] T004 Implement `StratigraphyLevelAnalyzer.ClassifyStratum` classifying tiles and chunks into discrete strata (`Active_1x`, `LateRevision_4x_8x`, `ClassicErasure_33x`, `DeepProto_64x_512x`, `Holed_DevMesh_1x`, `Submerged_OceanFloor`, `BitExact_Flat`)
- [x] T005 Author comprehensive unit tests in `WowViewer.Core.Tests/StratigraphyLevelAnalyzerTests.cs` for level counts, seam spikes, and stratum classification (Gate 1)

---

## Phase 2: High-Performance SIMD Restoration & Normal Solver (US2)

- [x] T006 Implement `FastTerrainNormalSolver.cs` utilizing `System.Numerics.Vector<float>` SIMD operations for rapid normal generation across 257x257 height lattices
- [x] T007 Implement `TemporalMeshRestorer.cs` providing in-place SIMD height transformation, negative floor preservation, and C0/C1 boundary slope stitching to adjacent active terrain
- [x] T008 Author unit tests in `WowViewer.Core.Tests/TemporalMeshRestorerTests.cs` verifying exact factor scaling ($\times 33.334$, $\times 16$, $\times 64$), seam stitching continuity, and SIMD normal accuracy (Gate 2)

---

## Phase 3: Interactive In-Viewer Stratigraphy Workbench (US3)

- [x] T009 Replace legacy render-thread `RefreshTerrainWeakSignalRestoreForLoadedTiles()` and `ReplaceTileChunksAndRebuild()` in `ViewerApp.cs` with zero-allocation `TemporalMeshRestorer` and `StratigraphyTileExporter` workflows
- [x] T010 Implement the "Temporal Stratigraphy Workbench" in `ViewerApp_Sidebars.cs` (Inspect > Archeology > Stratigraphy & Terrain Lab) with continuous gradient slider ($1.0\times \to 512.0\times$), preset snap buttons ($33.334\times$, $16\times$, $64\times$), "Unhide Dev Meshes" hole-mask bypass, and save/export modal
- [x] T011 Implement live stratum statistics and badges (`DominantStratum`, `SurvivingLevels`, `ChunkCounts`, `MergeSpikeOrigin`)
- [x] T012 Verify in-viewer interaction and export mechanisms (Gate 3)

---

## Phase 4: Corpus-Wide Batch Scanner & Dual-Era Offline Patcher (US4)

- [x] T013 Implement `TerrainStratigraphyScanCommand.cs` in `WowViewer.Tool.Inspect` emitting structured `stratigraphy_manifest.json` and `stratigraphy_summary.csv` with per-chunk level counts, seam spikes, and dev mesh counts
- [x] T014 Implement `TerrainStratigraphyPatchCommand.cs` in `WowViewer.Tool.Converter` exporting pre-computed loose LK ADTs and Alpha monolithic `.wdt` files
- [x] T015 Register CLI commands in `WowViewer.Tool.Inspect/Program.cs` and `WowViewer.Tool.Converter/Program.cs`
- [x] T016 Execute validation test sweep and update `specs/STATUS.md` and `memory-bank/activeContext.md` (Gate 4)
