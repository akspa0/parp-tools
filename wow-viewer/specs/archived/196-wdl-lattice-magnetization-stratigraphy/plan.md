# Spec 196 Technical Plan: WDL Lattice Magnetization & Stratigraphy Restoration Engine

## Architectural Overview

The engine adds advanced stratigraphy restoration modules into `WowViewer.Core.Runtime.World.Terrain.Stratigraphy` and integrates them into `ViewerApp` and `Archeology` UI:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Stratigraphy Restoration Engine                      │
├────────────────────────┬────────────────────────┬──────────────────────┤
│  Polarity & Anchors    │ Neighbor Mesh Auto-Fit │  WDL Magnetization   │
│  - Floor / Ceiling     │ - 1-3 Chunk Radius     │  - 17x17 Tile Grid   │
│  - Mean / Median       │ - Boundary RMSE Solver │  - Micro-Relief Add  │
│  - Positive / Negative │ - Scale + Delta Z Fit  │  - WDL Writer/Export │
└────────────────────────┴────────────────────────┴──────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Asynchronous Background Mesh Dispatcher                  │
│       - ThreadPool SIMD Deformation & Fast Normal Generation          │
│       - Double-Buffered GPU Chunk Mesh Uploads (<1ms main thread)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## File Changes & New Components

### 1. `WowViewer.Core.Runtime` (`World/Terrain/Stratigraphy/`)
- **[NEW] `StratigraphyAnchorMode.cs`**: Enum for `LowestZ_Floor`, `HighestZ_Ceiling`, `MeanZ`, `NeighborMeshBorder`, `WdlLattice`.
- **[NEW] `NeighborMeshHeightSolver.cs`**: Samples adjacent active chunks within 1–3 chunk radii, tests discrete scale candidate bands and polarities, and calculates best-fit vertical delta $\Delta Z$ and scale factor minimizing boundary RMSE.
- **[NEW] `WdlLatticeMagnetizer.cs`**: Performs bilinear interpolation of WDL $17\times 17$ and $16\times 16$ heights and applies high-frequency ADT micro-relief displacement onto the macro lattice surface.
- **[NEW] `WdlFileWriter.cs`**: Synthesizes Blizzard-format `.wdl` chunks (`MWMO`, `MWID`, `MODF`, `MAOF`, `MARE`, `MAHO`) from current terrain data.
- **[MODIFY] `TemporalStratum.cs` / `TemporalStratigraphyOptions.cs`**: Add `PolarityInverted`, `AnchorMode`, `CustomAnchorHeight`, `UseNeighborAutoFit`, `UseWdlMagnetization`, and `WdlMagnetizationStrength`.
- **[MODIFY] `TemporalMeshRestorer.cs`**: Support multi-anchor and inverted polarity scaling, neighbor offset adjustments, and WDL surface blending.

### 2. `WoWViewer` (`src/viewer/WoWViewer/`)
- **[MODIFY] `ViewerApp.cs` / `ViewerApp_WeakSignals.cs`**:
  - Replace synchronous whole-tile rebuilds with asynchronous `Task.Run` pipeline.
  - Integrate neighbor mesh height solver and WDL lattice magnetization.
- **[MODIFY] `ViewerApp_Sidebars.cs`**:
  - Add Polarity Inversion checkbox (`[ ] Invert Polarity (Negative Scaling)`).
  - Add Anchor Datum dropdown (`Floor`, `Ceiling`, `Mean`, `Neighbor Auto-Fit`, `WDL Magnetization`).
  - Add "Auto-Fit to Neighbor Heights" and "Magnetize to WDL Lattice" buttons.
  - Add "Export Modified WDL" button in the stratigraphy save dialog.

---

## Phase Roadmap

- **Phase 1: Polarity Inversion & Multi-Anchor Geometry Solver**
  - Implement `StratigraphyAnchorMode`, update `TemporalStratigraphyOptions`, and enhance `TemporalMeshRestorer` with polarity and custom datum anchors.
- **Phase 2: Neighboring Mesh Auto-Fit Solver**
  - Implement `NeighborMeshHeightSolver` with 1–3 chunk radius search and boundary RMSE optimization.
- **Phase 3: WDL Lattice Magnetization & WDL Exporter**
  - Implement `WdlLatticeMagnetizer` and `WdlFileWriter` supporting WDL macro-guidance and `.wdl` file export.
- **Phase 4: Asynchronous Worker Pipeline & In-Viewer UI Controls**
  - Integrate background worker task queue into `ViewerApp` and add full UI controls to the Archeology > Stratigraphy workbench.
- **Phase 5: Verification & Unit Tests**
  - Write test suite covering polarity inversion, boundary auto-fitting, WDL lattice interpolation, and WDL file serialization.
