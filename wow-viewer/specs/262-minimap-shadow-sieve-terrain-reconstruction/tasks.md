# Tasks: Spec 262 — Minimap Residual Model, Automated Lighting Calibration & 3D Fractal Editor Brush Reconstruction

- [x] **Phase 1: Automated Lighting Calibration Engine (Synthetic vs 0.5.3 Whiteplates)**
  - [x] T001: Implement `MinimapLightingOptimizer.cs` in `WowViewer.Core.IO/Maps/` to solve for optimal solar azimuth, zenith, ambient/diffuse balance, and texture specular reflectance ($k_{s, k}, p_k$) against DXT1 synthetic shadow renders.
  - [x] T002: Add unit tests in `MinimapLightingOptimizerTests.cs` validating parameter recovery on synthetic test fixtures with known solar angles and specular lobes.
  - [x] T003: Implement `minimap_lighting_solver.py` in `data-harvester/src/harvester/v60/` utilizing SciPy L-BFGS-B / PyTorch optimization against real 0.5.3 whiteplate tiles.
  - [x] T004: Add test coverage in `data-harvester/tests/v60/test_minimap_lighting_solver.py`.
  - [x] T005: Run calibration on 0.5.3 test tiles and produce `lighting_calibration_0_5_3.json` satisfying AC-001 ($< 5\%$ photometric MAE on unoccluded whiteplates).

- [x] **Phase 2: Rosetta Overhead Object Asset Library & Vision Integration**
  - [x] T010: Implement `RosettaOverheadCatalogExporter.cs` in `WowViewer.Core.IO/Maps/` to export 2D top-down silhouettes, bounding extents, and orthographic textures from Rosetta datastores.
  - [x] T011: Implement `rosetta_vision_catalog.py` in `data-harvester/src/harvester/v60/` to index Rosetta overhead object captures into the canonical Zarr v3 datastore and integrate candidate prompting.
  - [x] T012: Add test coverage in `data-harvester/tests/v60/test_rosetta_vision_catalog.py` verifying catalog loading and query operations.

- [x] **Phase 3: SAM 3.1 Minimap Sieve & ComfyUI Orchestration Layer**
  - [x] T019: Implement `comfyui_orchestrator.py` in `data-harvester/src/harvester/v60/` connecting to `http://127.0.0.1:8199` (RTX 4070 Ti SUPER) for SAM 3.1 & Mothership VLM workflow execution.
  - [x] T020: Implement `sam_minimap_sieve.py` in `data-harvester/src/harvester/v60/` utilizing the ComfyUI orchestrator with promptable grid points and Rosetta/Mothership box prompts.
  - [x] T021: Implement `minimap_shadow_stripper.py` in `data-harvester/src/harvester/v60/` performing multi-scale inpainting and diffuse albedo normalization to extract `stripped_residual_shadow_256`.
  - [x] T022: Add test coverage in `data-harvester/tests/v60/test_comfyui_orchestrator.py`, `test_sam_minimap_sieve.py`, and `test_minimap_shadow_stripper.py`.
  - [x] T023: Verify SAM 3.1 achieves $\ge 0.85$ IoU against control contamination masks (AC-003) and attenuates object energy by $\ge 90\%$ (AC-004).

- [x] **Phase 4: Closed-Loop Shadow Difference & Ridge Residual Synthesis**
  - [x] T030: Implement `shadow_difference_refiner.py` in `data-harvester/src/harvester/v60/` computing $\Delta S(x, y) = S_{\text{real}} - S_{\text{synth}}$ and extracting directional ridge/crest contours.
  - [x] T031: Add test coverage in `data-harvester/tests/v60/test_shadow_difference_refiner.py`.
  - [x] T032: Integrate difference feedback into synthetic shadow generation, verifying ridge alignment matches ground truth $\ge 80\%$ (AC-005).

- [x] **Phase 5: 3D Fractal Editor Brush Discovery & Fitting Engine**
  - [x] T040: Implement `fractal_brush_extractor.py` in `data-harvester/src/harvester/v60/` extracting discrete 3D brushes coupling spatial mesh displacement $\Delta Z(u, v)$ with texture alpha footprints $\alpha_k(u, v)$.
  - [x] T041: Implement `fractal_brush_fitter.py` in `data-harvester/src/harvester/v60/` fitting parameterized stamp invocations $(\mathcal{B}_j, \mathbf{p}_i, \sigma_i, \theta_i, A_i)$ to recognized residual patterns.
  - [x] T042: Add test coverage in `data-harvester/tests/v60/test_fractal_brush_engine.py` validating brush extraction and fitting accuracy (AC-006).

- [x] **Phase 6: Terrain Mesh Reconstruction Pipeline & $\ge 75\%$ Parity Benchmark**
  - [x] T050: Implement `reconstruction_benchmark.py` in `data-harvester/src/harvester/v60/` combining continuous residual height prediction with discrete 3D fractal brush stamping.
  - [x] T051: Add test coverage in `data-harvester/tests/v60/test_reconstruction_benchmark.py`.
  - [x] T052: Run evaluation across held-out 0.5.3 tiles to verify RelMAE $\le 25\%$ ($\ge 75\%$ accuracy), normal similarity $\ge 0.88$, and ridge F1 $\ge 0.75$ (AC-007).

- [x] **Phase 7: Governance, Receipts & Documentation**
  - [x] T060: Verify registration of Spec 262 in `wow-viewer/specs/STATUS.md`.
  - [x] T061: Compile Phase 1-6 verification evidence into `specs/262-minimap-shadow-sieve-terrain-reconstruction/evidence/receipt-spec262.md` per `AGENTS.md` §9.2.
  - [x] T062: Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.
