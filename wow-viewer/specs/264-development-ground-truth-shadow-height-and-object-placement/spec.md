# Spec 264: Authentic Development Map Shadow-to-Height Model & Rosetta Object Placement

## 1. Context & Background

In Spec 262 and Spec 263, we established photometric minimap residual shadow extraction, ComfyUI SAM 3.1 object sieving, and cross-validation against 1.60 MCCV vertex colors. However, testing on structured environments (such as Stormwind Harbor, Goldshire, or settlements) revealed critical failure modes identified by the operator:
1. **Flat Pancake Terrain Under Buildings**: Objects and structures were excised by SAM and inpainted flat with Laplacian smoothing, causing the pipeline to sculpt entire towns and structures as flat pancakes rather than identifying building foundations and placements.
2. **Missing Shadow-to-World-Space Z Elevation Model**: The height recovery relied on uncalibrated Frankot-Chellappa shape-from-shading integration with hardcoded synthetic scaling (`height_scale = 45.0`). It lacked a model mapping bare shadow signals to true world-space elevations in yards ($Z \in [Z_{\min}, Z_{\max}]$).
3. **Synthetic Whiteplate Fallacy**: Prior models evaluated on synthetic whiteplates ($\rho = 1.0$) with zero texture splats, ignoring real alpha-splatted terrain with multi-layer textures (`MCLY`/`MCAL`).
4. **Authentic Non-Museum Dataset Mandate**: The repository contains authentic, unedited development assets under `wow-viewer/test_data/original_development/World/Maps/development` (273 sculpted ADTs, 138 tiles with WMO buildings, 110 rich 4-way tiles with root ADT, `_obj0.adt`, `_tex0.adt`, `.pm4`, and matching 256x256 minimap PNGs in `World/Textures/Minimap/`). The operator explicitly directed to use this authentic corpus for training, calibration, and validation.

---

## 2. Requirements & User Stories

### User Story 1 (US-1): Authentic Development Corpus Ground-Truth Extraction
As a terrain reconstruction pipeline developer, I need an automated extractor that reads authentic `original_development` ADTs (`development_XX_YY.adt`), object files (`_obj0.adt`), texture files (`_tex0.adt`), and matching minimap PNGs (`development_XX_YY.png`), producing paired arrays of:
- Exact world-space ground truth elevation `height_257` in yards ($Z = \text{baseHeight} + \text{MCVT}$).
- Matching 256x256 RGB minimap crop.
- Placed M2 doodad entries (`MDDF`) and WMO building entries (`MODF`) with 3D bounds and world positions.
- Terrain liquid / hole masks.

### User Story 2 (US-2): Calibrated Shadow-to-Height Elevation Model
As an operator, I want the reconstruction pipeline to map bare terrain residual shadow signals directly to calibrated world-space elevation in yards, replacing uncalibrated Frankot-Chellappa multipliers with a model trained/calibrated against authentic `original_development` ground truth:
- Evaluated against true relief spans (tiles with $\Delta Z$ up to 784 yards).
- Target validation Mean Absolute Error (MAE) $\le 12.0$ yards on authentic non-flat development terrain.
- Retains macroscopic terrain slope and ridge morphology ($R^2 \ge 0.70$).

### User Story 3 (US-3): Rosetta Object Library Placement & Foundation Plateau Carving
As an operator reconstructing terrain from minimaps containing structures, I want detected buildings (e.g., Stormwind Harbor, Goldshire Inn, Blacksmith, Farm):
- Matched against the Rosetta vision catalog (`RosettaVisionCatalog`) or placed from candidate object proposals.
- Positioned in 3D world space (`MDDF`/`MODF` placements) rather than thrown away.
- Foundation terrain under building footprints sculpted as level foundation plateaus matched to entry perimeter elevations, rather than sunken or flattened pancake artifacts.

### User Story 4 (US-4): 3D Mesh Materialization & Side-by-Side Verification
As an operator, I want exported 3D Wavefront OBJ and binary glTF 2.0 (`.glb`) meshes of the reconstructed terrain, ground-truth ADT terrain, and placed object bounding boxes, along with side-by-side elevation heatmaps, to visually and numerically verify true elevation fidelity.

---

## 3. Acceptance Criteria

| ID | Category | Requirement | Target Metric |
|---|---|---|---|
| **AC-001** | Data Pipeline | Extract all valid non-flat tiles from `test_data/original_development` with paired `height_257`, minimap RGB, and `_obj0` placements | $\ge 100$ valid paired tiles |
| **AC-002** | Height Accuracy | Reconstructed terrain elevation vs ground-truth ADT `height_257` across held-out development tiles | Validation MAE $\le 12.0$ yards |
| **AC-003** | Relief Correlation | Elevation correlation between reconstructed mesh and ground truth | Pearson $r \ge 0.75$, $R^2 \ge 0.60$ |
| **AC-004** | Object Foundation | Terrain under placed WMO building footprints forms flat/leveled foundation plateaus matching building base height | Foundation slope variance $\le 0.05$ |
| **AC-005** | Rosetta Integration | Export reconstructed 3D `.glb` and `.obj` scenes including sculpted terrain and placed 3D object bounding boxes/models | Valid `.glb` & `.obj` loadable in 3D Viewer/Blender |
| **AC-006** | Test Coverage | Unit and integration tests for dataset extraction, height calibration, and mesh export | 100% test pass rate |
