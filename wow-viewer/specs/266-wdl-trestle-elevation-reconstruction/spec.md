# Spec 266: WDL Trestle Elevation Reconstruction & Cross-Map Lineage Synthesis

## 1. Overview & Problem Statement

In previous iterations (Spec 264 and Spec 265), deep neural elevation prediction from 256x256 minimap crops successfully captured local relative surface slopes ($r > 0.90$ locally). However, predicting absolute world-space elevation directly from RGB imagery without macro-context suffers from severe **regression to the mean**:
- Authentic world tiles span $-656$ yards to $+1,889$ yards in vertical placement across regions.
- Isolated RGB crops lack absolute elevation anchors, causing the models to compress dynamic vertical spans to $\approx 35\text{--}39$ yards, squashing high-relief alpine massifs down to gentle mounds.
- Furthermore, the authentic development map (`test_data/original_development`) has missing ADT geometry and missing WDL data for hundreds of tiles whose minimap tiles exist.

### Key Discovery: Cross-Map Prototype Lineage
Analysis of the development map revealed that its minimap tiles are direct prototypes of Wrath of the Lich King's Northrend continent (`H:\CLIENTS\Wrath\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft`). Cross-map visual matching (`scripts/match_development_to_northrend.py`) identified **2,117 matched tiles**, with over 1,740 matches having visual similarity $\ge 0.90$. The authentic Northrend 3.0.1.8303 WDL file contains coarse 17x17 macro-elevation lattices spanning up to 985 yards of relief.

### The Solution: Lean Modern Successor to the February 2026 v7 Trestle Model
In February 2026, the repository's historical `MultiChannelUNetV7` utilized a coarse WDL lattice as an elevation "trestle" (`wdl_base`) to anchor global height, allowing the model to predict residual deltas ($\Delta Z$) rather than struggling with unconstrained absolute elevations. While the original v7 model had over 100M parameters, 13 complex input channels, and high-frequency spiky artifacts, its core architectural intuition—**using the coarse WDL lattice as a structural trestle**—is sound.

Spec 266 realizes a lightweight, modern successor:
1. **Cross-Map Lineage WDL Synthesizer**: Transposes authentic Northrend 3.0.1 macro-elevation lattices onto missing development tiles using the high-confidence visual match ledger.
2. **`TrestleElevationUNet`**: A lean 6-channel U-Net (~6M parameters, base channels 32) accepting Minimap RGB (3), Upsampled WDL Trestle (1), and Photometric Normals (2).
3. **Dual Residual & Bounds Head**: Directly predicts high-frequency surface carving $\Delta Z$ such that $Z_{\text{final}} = Z_{\text{trestle}} + \Delta Z$, supervised by an auxiliary $[Z_{\min}, Z_{\max}]$ bounds head.
4. **Upright Watertight 3D Mesh Export**: Generates OBJ and GLB terrain models with counter-clockwise face winding, guaranteeing surface normals face upward (+Z / +Y).

---

## 2. User Stories

### User Story 1 - Lineage-Derived WDL Prior Lattice (Priority: P1)
As a map reconstruction engineer, I want missing WDL data on the development map to be reconstructed by mapping prototype minimaps to authentic Northrend 3.0.1 WDL lattices, so that every development tile possesses an authentic macro-elevation prior.

### User Story 2 - Lean Modern Trestle Architecture (Priority: P1)
As an ML researcher, I want a lean, modern PyTorch elevation model (~6M parameters) that accepts minimap RGB, photometric surface normals, and the coarse WDL trestle to predict dense residual elevation $\Delta Z$, avoiding the parameter bloat and artifacts of the historical v7 model.

### User Story 3 - Full Vertical Mountain Amplitude Recovery (Priority: P1)
As a 3D world builder, I want reconstructed alpine tiles like `development_16_33` and `development_16_32` to span their authentic vertical height ($\ge 250$ to $600+$ yards) instead of squashing to 35 yards, perfectly capturing ridges, valleys, and mountain massifs.

### User Story 4 - Upright 3D Mesh Rendering (Priority: P2)
As a technical artist, I want exported OBJ and GLB 3D meshes to have correct counter-clockwise face winding and upward-pointing normals, eliminating backface culling inversions and dark shaded artifacts in 3D viewers.

---

## 3. Acceptance Criteria

| ID | Criterion | Requirement | Target Metric |
|---|---|---|---|
| **AC-001** | Lineage WDL Synthesis | Synthesize missing development WDL lattices from Northrend 3.0.1 matches | Complete 64x64 WDL grid cached in `output/development_synthesized_wdl.npz` |
| **AC-002** | Lean Trestle Architecture | Implement `TrestleElevationUNet` in `harvester.v60.trestle_elevation_model` | Parameter count $\le 10$M, 6 input channels, dual residual/bounds heads |
| **AC-003** | Mountain Relief Recovery | Restore vertical amplitude on high-relief mountain tiles | Predicted elevation span on `development_16_33` $\ge 250$ yards (was 39 yards) |
| **AC-004** | Error & Correlation Performance | Combined elevation accuracy on held-out authentic tiles | Validation MAE $\le 25.0$ yards, Pearson correlation $r \ge 0.75$ |
| **AC-005** | Upward 3D Mesh Normal Fidelity | Watertight OBJ/GLB meshes with CCW winding | Normals point upward (+Z in OBJ, +Y in GLB), no backface camera flips |
| **AC-006** | Integrated Reconstruction CLI | Update `scripts/v60_reconstruct_minimap.py` to support `--trestle-model` | Single-command automated reconstruction from minimap PNG to 3D OBJ/GLB |

---

## 4. Constraints

- Zero hardcoded machine-local client paths in source code or portable docs (AGENTS.md §5). Local client `H:\CLIENTS\Wrath\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft` passed via runtime arguments/config.
- Pure PowerShell 7 syntax for all operator commands (AGENTS.md §5).
- God-class freeze: Zero member additions to `WorldScene.cs` or `ViewerApp.cs` (AGENTS.md §10).
- All Python work lives under `wow-viewer/data-harvester/` using `uv` (AGENTS.md §5).
- Respect scope freeze and receipts requirement (AGENTS.md §9.1, §9.2).
