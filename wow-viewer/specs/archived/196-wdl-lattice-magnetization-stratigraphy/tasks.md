# Tasks: Spec 196 WDL Lattice Magnetization, Polarity Inversion & Stratigraphy Restoration Engine

## Phase 1: Polarity Inversion & Multi-Anchor Geometry Solver
- [x] T001: Create `StratigraphyAnchorMode.cs` enum (`Floor`, `Ceiling`, `Mean`, `NeighborEdge`, `WdlLattice`) in `WowViewer.Core.Runtime.World.Terrain.Stratigraphy`.
- [x] T002: Add polarity inversion flag (`PolarityInverted`), `AnchorMode`, and custom anchor height properties to `TemporalStratigraphyOptions.cs`.
- [x] T003: Update `TemporalMeshRestorer.cs` to apply polarity-directed deformation and datum anchor offsets without height clipping.

## Phase 2: Neighboring Mesh Auto-Fit Solver
- [x] T004: Create `NeighborMeshHeightSolver.cs` with 1–3 chunk spatial radius adjacency search.
- [x] T005: Implement boundary RMSE evaluation over candidate scale bands ($S \in \{1, 3.33, 10, 16, 33.334, 64, 80, 128, 256, 512\}$) and polarities ($\pm 1$).
- [x] T006: Calculate best-fit vertical delta $\Delta Z$ to seamlessly stitch restored chunk borders to active neighbor terrain.

## Phase 3: WDL Macro-Lattice Magnetization & WDL Writer
- [x] T007: Create `WdlLatticeMagnetizer.cs` implementing bilinear interpolation over WDL $17\times 17$ tile vertices and $16\times 16$ chunk center heights.
- [x] T008: Implement micro-relief high-frequency displacement addition onto WDL macro-topography surface.
- [x] T009: Create `WdlFileWriter.cs` supporting Blizzard-standard `.wdl` file format serialization (`MWMO`, `MWID`, `MODF`, `MAOF`, `MARE`, `MAHO`).

## Phase 4: Asynchronous Pipeline & In-Viewer UI Controls
- [x] T010: Refactor `ViewerApp.cs` tile restoration loop to execute off the render thread via background worker tasks and double-buffered mesh swaps.
- [x] T011: Add Polarity Inversion checkbox, Anchor Mode selector, Auto-Fit Neighbor button, and WDL Magnetization controls to `DrawTemporalStratigraphySubTab` and `DrawTerrainControlsAdjustmentWeakSignalContent`.
- [x] T012: Add "Export Modified WDL..." button to the stratigraphy tile export dialog.

## Phase 5: Verification & Unit Tests
- [x] T013: Author unit tests in `WowViewer.Core.Tests` for `NeighborMeshHeightSolver`, `WdlLatticeMagnetizer`, and `WdlFileWriter`.
- [x] T014: Validate full solution build with zero errors across Windows and CrossPlatform configurations.
