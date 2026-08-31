# Spec 194: Temporal Stratigraphy & Weak Signal Development Mesh Restoration

## 1. Overview & Vision

In early World of Warcraft terrain builds (specifically Alpha 0.5.3 build 3368, but also persisting into 1.12.1 Vanilla, 3.3.5 WotLK, and 4.0.0 Cataclysm), certain map tiles and chunks contain compressed, sub-meter elevation profiles that represent **temporal stratigraphy** — historical geological layers of world development that were wiped, compressed, or flagged invisible during level editing passes.

Archaeological analysis reveals that when developers erased or altered terrain using early WoWEdit tools:
1. **Mathematical Scale Degradation**: Height data was scaled down by $\approx 0.03$ ($\frac{1}{0.0333...} \approx 30\text{x}-33.334\text{x}$, aligned with the $33.334\text{m}$ MCNK chunk metric) into the sub-meter noise floor rather than truly deleted.
2. **Spatial Continuity**: This sub-millimeter relief ($5.19 \times 10^{-4}$ amplitude) continues across chunk and tile boundaries, proving it is genuine historical landscape geometry.
3. **Preserved Development Meshes**: Geometry behind `HoleMask` (offset 0x40 in MCNK headers) was authored at full $1\times$ fidelity and retains complete subterranean caves, ramps, and Outland development blockouts.
4. **Sub-Tile Merge Boundaries**: Seam discontinuity spikes at MCNK chunks 4, 8, and 12 reveal fragments of earlier $128 \times 128$ MCNK cell worlds from 1999–2001.

Existing viewer "weak signal" tooling is brittle and unoptimized: it runs synchronous 257x257 array allocations and full GPU buffer recreations on the render thread (inducing 100ms–400ms hitches), blindly rejects non-sea-level tiles via a hardcoded $|Z| < 50\text{m}$ filter, and applies flat multipliers without boundary stitching or stratum awareness.

Spec 194 delivers a high-performance **Temporal Stratigraphy & Development Mesh Restoration Architecture**:
1. **Stratigraphic Analysis Engine (`StratigraphyEngine`)**: Level-count profiling (`surviving_height_levels`), raw MCVT delta analysis, MCNK seam discontinuity metrics, and stratum classification.
2. **SIMD In-Place Mesh Restoration (`TemporalMeshRestorer`)**: Zero-allocation `Vector<float>` height amplification with C0/C1 slope boundary stitching and fast SIMD normal solvers.
3. **Interactive Viewer Stratigraphy Workbench**: Real-time false-color stratigraphy heatmaps, non-hitching GPU buffer updates, continuous gradient sliders ($1\times \to 512\times$), and "Unhide Dev Mesh" hole-mask toggles.
4. **Dual-Era Offline Patcher & Manifest Exporter (`terrain-stratigraphy-scan` / `terrain-stratigraphy-patch`)**: Corpus-wide batch scanning and pre-computed loose LK ADT and monolithic Alpha WDT export.

---

## 2. User Stories

- **US1 (Stratigraphic Level & Gradient Analysis)**: As a terrain researcher or tool developer, I want to compute `surviving_height_levels`, raw MCVT delta distributions, and seam discontinuity profiles across map tiles without altitude limits, so that I can automatically classify terrain chunks into distinct temporal strata (Active, Squeezed-33x, Deep-Proto, Holed-DevMesh, Submerged-Ocean, Flat-Void).
- **US2 (High-Performance SIMD Restoration Engine)**: As a viewer developer, I want an in-place SIMD-accelerated amplification engine with boundary slope stitching to adjacent active terrain, so that weak signal terrain can be restored seamlessly without tearing down GPU tile scene nodes or stalling the render loop.
- **US3 (Interactive In-Viewer Stratigraphy Workbench)**: As a viewer user, I want an interactive "Temporal Stratigraphy" workbench panel with false-color heatmaps (Level Count, Stratum Class, Relief Amplitude), continuous gradient scaling ($1.0\times \to 512.0\times$), preset snap points ($33.334\times$, $16\times$, $64\times$), and an "Unhide Dev Meshes" toggle to visualize hidden historical geometry in real time.
- **US4 (Corpus-Wide Batch Scanner & Dual-Era Patcher CLI)**: As a researcher or server operator, I want CLI commands (`terrain-stratigraphy-scan` and `terrain-stratigraphy-patch`) to scan entire client builds, generate `stratigraphy_manifest.json` reports, and export pre-patched loose LK ADTs and Alpha monolithic `.wdt` files.

---

## 3. Requirements & Acceptance Criteria

### Functional Requirements (FR)
- **FR-001**: Implement `StratigraphyLevelAnalyzer` in `WowViewer.Core.Runtime.World.Terrain.Stratigraphy` computing `surviving_height_levels` ($|\{h\}|$), raw MCVT delta entropy, min/max bounds, and relief amplitude across $16 \times 16$ chunk grids without any hardcoded $|Z| < 50\text{m}$ altitude floor.
- **FR-002**: Implement `TemporalStratumClassifier` classifying chunks and tiles into `TemporalStratum` enums:
  - `Active_1x`: Full-scale active game terrain.
  - `LateRevision_4x_8x`: Moderate compression from late editing passes.
  - `ClassicErasure_33x`: Standard $\frac{1}{0.0333...} \approx 33.334\times$ WoWEdit compression.
  - `DeepProto_64x_512x`: Sub-millimeter early prototypes or micro-traces.
  - `Holed_DevMesh_1x`: Preserved full-scale geometry hidden by `HoleMask`.
  - `Submerged_OceanFloor`: Underwater contours beneath water planes.
  - `BitExact_Flat`: Zero authored relief.
- **FR-003**: Implement `SeamDiscontinuityProfiler` measuring C0 (step) and C1 (gradient) discontinuity across all 15 internal MCNK chunk boundaries, reporting merge spike indices (Spike at 8 = 2x2 merge, Spikes at 4, 8, 12 = 4x4 merge).
- **FR-004**: Implement `TemporalMeshRestorer` utilizing `System.Numerics.Vector<float>` SIMD operations for:
  - In-place proportional height transformation: $Z_{\text{new}} = Z_{\text{anchor}} + (Z_{\text{orig}} - Z_{\text{anchor}}) \times S$.
  - Preserving negative floors when $Z_{\text{min}} < 0$.
  - C0/C1 boundary slope blending towards adjacent non-weak tiles.
  - Fast SIMD normal recalculation (`FastTerrainNormalSolver`).
- **FR-005**: Integrate the Stratigraphy Workbench into `ViewerApp_Sidebars.cs` (Inspect > Terrain > Stratigraphy):
  - False-color heatmap overlay rendering modes: `LevelCount`, `StratumClass`, `ReliefAmplitude`.
  - Gradient slider ($1.0\times \to 512.0\times$) with preset snap buttons ($33.334\times$, $16\times$, $64\times$).
  - `Unhide Dev Meshes` checkbox (bypasses `HoleMask` quads).
  - Background task evaluation without main-thread stalls.
- **FR-006**: Create CLI command `terrain-stratigraphy-scan` in `WowViewer.Tool.Inspect` producing structured `stratigraphy_manifest.json`.
- **FR-007**: Create CLI command `terrain-stratigraphy-patch` in `WowViewer.Tool.Converter` supporting `--format lk|alpha|both`, pre-computing amplified geometry into loose `.adt` and monolithic `.wdt` files.

### Non-Functional & Safety Requirements (NFR)
- **NFR-001 (Zero-Allocation Render Path)**: Live stratum amplification and slider adjustments must perform zero array re-allocations on the render thread and complete in $< 2\text{ms}$ per tile.
- **NFR-002 (Format Integrity)**: Format readers (`LkAdtReader`, `AlphaWdtReader`, `NativeMpqService`) and writers (`AlphaWdtWriter`) remain strictly untouched and protected.
- **NFR-003 (Read-Only Client Safety)**: Client source directories (`H:\CLIENTS`) are strictly read-only; all outputs write to specified target output paths.
- **NFR-004 (Core Separation)**: Analysis and restoration logic resides entirely in `WowViewer.Core.Runtime` and `WowViewer.Core.IO`; UI is isolated in `WoWViewer`.
