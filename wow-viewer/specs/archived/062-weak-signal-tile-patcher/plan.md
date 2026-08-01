# Implementation Plan: Weak Signal Tile Patcher Export

**Feature**: 062-weak-signal-tile-patcher

**Created**: 2026-06-13

**Status**: Draft

## Phase 1 — Extract Weak Signal Detection into Shared Library (P1)

Move the detection logic from `ViewerApp.cs` into a shared location so both the viewer and the converter tool can use it.

### Steps

1. **Create `WeakSignalDetector.cs`** in `wow-viewer/src/core/WowViewer.Core.IO/Terrain/` (or `WowViewer.Core.Runtime/Terrain/`). Extract the detection methods from `ViewerApp.cs` (lines ~4192-4649): `IsWeakSignalCandidate()`, `EstimateAmplificationFactor()`, and supporting helper methods. Make them static and stateless.

2. **Create `WeakSignalTileAnalysis.cs`** data class in the same directory. Fields: `TileX`, `TileY`, `IsWeakSignal`, `HeightRange`, `NormalRelief`, `NormalCoverage`, `AmplificationFactor`, `AnchorHeight`, `Severity`.

3. **Refactor `ViewerApp.cs`** to call the shared `WeakSignalDetector` methods instead of its private implementations. Verify the viewer still works identically (no behavioral change).

4. **Write unit tests** for `WeakSignalDetector` in `wow-viewer/tests/WowViewer.Core.Tests/`: test detection on synthetic heightmaps (flat terrain, ridged terrain, ocean tile), test amplification factor estimation.

### Validation

- `dotnet build wow-viewer/WowViewer.slnx` succeeds
- `dotnet test wow-viewer/WowViewer.slnx` passes
- Viewer loads a map and weak signal detection still works (manual check)

---

## Phase 2 — Build the Tile Patcher Command (P1)

Create the `terrain-weak-signal-patch` command in the converter tool.

### Steps

5. **Create `TerrainWeakSignalPatchCommand.cs`** in `wow-viewer/tools/converter/WowViewer.Tool.Converter/`. Wire up CLI args: `--map-path`, `--output-dir`, `--format`, `--client-root`, detection threshold overrides, amplification factor override.

6. **Implement map loading**: Use `AlphaTerrainAdapter` (for Alpha maps) or `StandardTerrainAdapter` (for LK maps) to load every tile in the map. Iterate all 256 possible tile coordinates, attempt to load each.

7. **Implement detection pass**: For each loaded tile, call `WeakSignalDetector.Analyze()` to produce a `WeakSignalTileAnalysis`. Collect all analyses.

8. **Implement amplification pass**: For each weak-signal tile, apply height amplification using the shared logic. Produce a corrected 257x257 heightmap per tile.

9. **Implement LK ADT output**: For each patched tile, locate the original `_root.adt` file, copy it to the output directory, then call `AdtTerrainWriter.Write()` to patch heights and normals. Also copy `_tex0.adt`, `_obj0.adt`, `_lod.adt` for full tile family.

10. **Implement Alpha WDT output**: For each patched tile, build a corrected `AlphaTileData` in memory. After processing all tiles, call `AlphaWdtWriter.Build()` to produce the complete WDT.

11. **Write the patch report**: Serialize `WeakSignalPatchReport` to `weak_signal_patch_report.json` in the output root.

### Validation

- `dotnet build wow-viewer/WowViewer.slnx` succeeds
- Command runs against the development test map and produces output files
- `WowViewer.Tool.Inspect` validates the output ADTs

---

## Phase 3 — Copy Unpatched Tiles for Full Overlay (P2)

Ensure the output directory is a complete map overlay, not just patched tiles.

### Steps

12. **Copy unpatched tile families**: For every tile that exists in the source map but was NOT patched, copy its `_root.adt`, `_tex0.adt`, `_obj0.adt`, `_lod.adt` to the output directory. This ensures the viewer can load the complete map from the overlay root.

13. **Copy WDT**: For Alpha format, the WDT is already complete (contains all tiles). For LK format, copy the original WDT to the output directory.

14. **Copy WDL**: Copy the WDL file to the output directory so the viewer's WDL preview works.

### Validation

- Output directory contains a complete map that the viewer can load via `--loose-overlay-root`
- All tiles load correctly, not just the patched ones

---

## Phase 4 — Viewer Integration and End-to-End Test (P2)

Verify the full pipeline: export → overlay → viewer loads corrected terrain.

### Steps

15. **End-to-end test**: Run the patcher on a known map with weak-signal tiles. Launch the viewer with `--loose-overlay-root <output>`. Verify the corrected terrain renders (no real-time weak signal computation needed).

16. **Performance comparison**: Time the patcher command vs. the viewer's real-time weak signal computation for the same map. Document the speedup.

### Validation

- Viewer loads patched map from overlay without real-time weak signal computation
- Corrected terrain visually matches the viewer's real-time output

---

## Execution Guardrails

- Phase 1 must complete before Phase 2 (shared library is prerequisite).
- Phase 2 must complete before Phase 3 (output format is prerequisite for full overlay).
- Phase 3 must complete before Phase 4 (full overlay is prerequisite for end-to-end test).
- Each phase produces an independently committable diff.
- `AlphaWdtWriter` is frozen (Rule 10) — do not modify it, only call `AlphaWdtWriter.Build()`.
- `AdtTerrainWriter` is a shared library — use its public API, do not modify it.
