# Spec 196: WDL Lattice Magnetization, Multi-Anchor Polarity Inversion & Neighbor-Mesh Auto-Fitting Stratigraphy Engine

## Executive Summary
Provides next-generation historical terrain stratigraphy restoration for compressed, scaled, and inverted development data (such as Dragon Isles, 0.5.3 prototype maps, and subterranean dev geometry). Unifies four core capabilities:
1. **Polarity Inversion & Multi-Anchor Geometry Reconstruction**: Allows negative polarity scaling and flexible anchor baselines (Anchor to Floor, Anchor to Ceiling, Anchor to Mean, Anchor to Neighbor Mesh Boundary) to eliminate vertical spike artifacts and correct inverted compressed relief.
2. **Neighboring Mesh Height Auto-Fitting**: Detects uncompressed or residual geometry within 1–3 chunks (on the current tile or adjacent resident tiles), automatically inferring the optimal scale band, vertical offset $\Delta Z$, and polarity to seamlessly bridge terrain boundaries.
3. **WDL Lattice Magnetization & Micro-Relief Displacement**: Leverages low-frequency $17 \times 17 + 16 \times 16$ WDL elevation lattices as macro guides, "magnetizing" micro-relief ADT weak signal displacements onto the macro-topography surface without boundary tearing.
4. **Asynchronous Non-Blocking Tile Restoration Pipeline**: Eliminates the 2–4 second UI/render freezes by running tile analysis, SIMD deformation, and normal synthesis on background worker threads with double-buffered GPU mesh swaps.
5. **Modified WDL Generation & Dual-Era Export**: Enables generating and exporting modified `.wdl` files alongside loose LK ADTs and monolithic Alpha WDT maps.

---

## User Stories

### US1: Inverted Polarity & Custom Baseline Anchor Controls
As an archaeologist/researcher, I want to toggle negative polarity scaling and choose the anchor baseline (Floor, Ceiling, Mean, Neighbor) so that inverted historical terrain (e.g. Dragon Isles) reconstructs downward or upward into coherent landscape rather than exploding into skyward wall spikes.

### US2: Neighboring Mesh Auto-Fit & Spatial Scale/Offset Solver
As a viewer user, I want the system to sample adjoining active terrain within 1–3 chunks and automatically calculate the best-fitting scale factor and vertical offset $\Delta Z$ along shared edges so that restored prototype areas seamlessly match surrounding landmasses.

### US3: WDL Macro-Lattice Magnetization & WDL Exporter
As a terrain researcher, I want to use the map's low-frequency WDL lattice (or synthesize one) as a topographical anchor to guide high-frequency ADT relief, and export the resulting `.wdl` file alongside restored ADT/WDT terrain files.

### US4: Zero-Freeze Asynchronous Tile Restoration
As an operator navigating large maps with restored stratigraphy enabled, I want tiles to reconstruct and update asynchronously without 2–4 second main render thread hitches or stuttering.

---

## Acceptance Criteria

- **AC-001 (Polarity Inversion)**: Negative polarity toggle ($-\text{factor}$) and anchor modes (Lowest Z / Floor, Highest Z / Ceiling, Mean Z, Neighbor Edge) prevent upward spike artifacts on inverted terrain.
- **AC-002 (Neighbor Auto-Fit Solver)**: Boundary RMSE minimizer correctly matches adjoining active chunk edges within $\le 0.05\text{m}$ error when valid neighbor mesh vertices are present.
- **AC-003 (WDL Lattice Magnetization)**: Bilinear/bicubic interpolation over $17\times 17$ WDL heights anchors $9\times 9 + 8\times 8$ MCNK vertices to the macro terrain surface.
- **AC-004 (WDL File Export)**: Standalone and in-viewer export pipelines can write valid Blizzard-standard `.wdl` files containing modified tile and chunk heights.
- **AC-005 (Async Background Pipeline)**: Moving the camera into weak-signal tiles maintains $\ge 60\text{ FPS}$ with zero blocking calls $>16\text{ms}$ on the UI thread.
