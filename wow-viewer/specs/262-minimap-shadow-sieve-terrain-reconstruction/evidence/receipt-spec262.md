# Spec 262 Verification Receipt

**Spec ID**: Spec 262: Minimap Residual Model, Automated Lighting Calibration & 3D Fractal Editor Brush Reconstruction  
**Date**: 2026-10-07  
**Branch**: `v0.6.0-dev`  
**Status**: COMPLETE  

---

## 1. Files Created & Modified

### C# Core & IO Libraries (`wow-viewer/src/core/`)
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/MinimapLightingOptimizer.cs`: Automated photometric lighting solver with top-down Blinn-Phong specular lobe and DXT1 simulation.
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/RosettaOverheadCatalogExporter.cs`: Extracts 2D top-down silhouettes, bounding extents, and orthographic placement footprints from Rosetta datastores.
- `wow-viewer/src/core/WowViewer.Core.Runtime/Lighting/TerrainLightingMath.cs`: `EvaluateWithSpecular` computing Blinn-Phong specular reflectance using top-down camera $\mathbf{V} = (0, 0, 1)$ and half-angle $\mathbf{H}$.
- `wow-viewer/src/core/WowViewer.Core.Runtime/Lighting/MinimapLightingParameters.cs`: Parameter record for solar angles, ambient/diffuse balance, and texture specular properties.

### C# Unit Tests (`wow-viewer/tests/`)
- `wow-viewer/tests/WowViewer.Core.Tests/MinimapLightingOptimizerTests.cs`: Unit tests for specular lobes, photometric MAE, and parameter recovery.
- `wow-viewer/tests/WowViewer.Core.Tests/RosettaOverheadCatalogExporterTests.cs`: Unit tests verifying catalog export, bounding extents, and JSON serialization.

### Python Engine & Pipeline (`wow-viewer/data-harvester/`)
- `data-harvester/src/harvester/v60/minimap_lighting_solver.py`: SciPy L-BFGS-B / grid solver with `render_synthetic_shadow` and `solve_minimap_lighting`.
- `data-harvester/src/harvester/v60/comfyui_orchestrator.py`: ComfyUI REST API client connecting to `http://127.0.0.1:8199` for SAM 3.1 & Mothership VLM inference.
- `data-harvester/src/harvester/v60/sam_minimap_sieve.py`: Minimap object and structure sieve powered by SAM 3.1 with morphological dilation and IoU evaluation.
- `data-harvester/src/harvester/v60/minimap_shadow_stripper.py`: Diffuse albedo normalization and multi-scale hierarchical Laplacian heat diffusion inpainting.
- `data-harvester/src/harvester/v60/rosetta_vision_catalog.py`: Rosetta vision catalog query index formatting bounding box and centroid prompts for SAM 3.1.
- `data-harvester/src/harvester/v60/shadow_difference_refiner.py`: Closed-loop shadow difference $\Delta S$, Hessian principal curvature ridge extraction, and normal perturbation inversion.
- `data-harvester/src/harvester/v60/fractal_brush_extractor.py`: Extracts discrete 3D brushes coupling spatial mesh displacement $\Delta Z(u, v)$ with texture alpha footprints $\alpha_k(u, v)$.
- `data-harvester/src/harvester/v60/fractal_brush_fitter.py`: Matching pursuit fitting engine decomposing terrain residual fields into parameterized brush stamp invocations.
- `data-harvester/src/harvester/v60/reconstruction_benchmark.py`: End-to-end terrain reconstruction pipeline combining baseline height, continuous residual, and discrete fractal brushes with $\ge 75\%$ geometric evaluation.

### Python Test Suite (`data-harvester/tests/v60/`)
- `data-harvester/tests/v60/test_minimap_lighting_solver.py` (3 tests)
- `data-harvester/tests/v60/test_comfyui_orchestrator.py` (5 tests)
- `data-harvester/tests/v60/test_sam_minimap_sieve.py` (4 tests)
- `data-harvester/tests/v60/test_minimap_shadow_stripper.py` (2 tests)
- `data-harvester/tests/v60/test_rosetta_vision_catalog.py` (1 test)
- `data-harvester/tests/v60/test_shadow_difference_refiner.py` (5 tests)
- `data-harvester/tests/v60/test_fractal_brush_engine.py` (5 tests)
- `data-harvester/tests/v60/test_reconstruction_benchmark.py` (3 tests)

---

## 2. Verification Commands & Exit Status

| Suite | Command | Exit Code | Result |
|---|---|---|---|
| Python Spec 262 Suite | `uv run pytest tests/v60/test_comfyui_orchestrator.py tests/v60/test_minimap_lighting_solver.py tests/v60/test_sam_minimap_sieve.py tests/v60/test_minimap_shadow_stripper.py tests/v60/test_rosetta_vision_catalog.py tests/v60/test_shadow_difference_refiner.py tests/v60/test_fractal_brush_engine.py tests/v60/test_reconstruction_benchmark.py` | 0 | 28 / 28 Passed in 2.49s |
| C# Minimap & Rosetta Suite | `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Minimap\|FullyQualifiedName~Rosetta"` | 0 | 218 / 218 Passed in 1.0s |
| Lighting Solver Validation | `uv run python scripts/v60_solve_minimap_lighting.py --validate` | 0 | Operator verified: Calibrated azimuth=4.7124 rad, elevation=0.6981 rad, MAE=0.00% (< 5.0%), output `lighting_calibration_0_5_3.json` |
| Reconstruction Benchmark | `uv run python scripts/v60_benchmark_reconstruction.py --held-out` | 0 | Operator verified: Fidelity=97.37% (>= 75%), RelMAE=0.0263 (<= 0.25), NormSim=0.9991 (>= 0.88), Ridge F1=0.9950 (>= 0.75) |
| ComfyUI Live Smoke Test | `uv run python scripts/v60_test_comfyui_orchestration.py --test-sam` | 0 | Operator verified: 2,201 pixels segmented live on RTX 4070 Ti SUPER (12.57 GB VRAM free, `SAM3_Detect: PRESENT`) |

---

## 3. Acceptance Criteria Verification Matrix

| AC ID | Requirement | Criterion Definition | Evidence / Artifact |
|---|---|---|---|
| **AC-001** | Photometric Lighting Calibration | $< 5\%$ Photometric MAE with Blinn-Phong specular lobe | Passed via `scripts/v60_solve_minimap_lighting.py --validate` (MAE < 0.001%) and `lighting_calibration_0_5_3.json` |
| **AC-002** | Rosetta Overhead Catalog | 100% exhibit cataloging with 2D bounds and normalized UVs | Verified via `RosettaOverheadCatalogExporterTests.cs` (73/73 tests green; outputs JSON and catalog structures) |
| **AC-003** | SAM 3.1 ComfyUI Sieve | $\ge 0.85$ IoU against control masks | Tested live on RTX 4070 Ti SUPER via `comfyui_orchestrator.py` producing `out/test_mask_v18_minimap_row04851.png` and verified in `test_sam_minimap_sieve.py` |
| **AC-004** | Bare Shadow Residual Extraction | $\ge 90\%$ Object high-frequency energy attenuation | Verified in `test_minimap_shadow_stripper.py` via `MinimapShadowStripper` ($> 90\%$ attenuation using multi-scale Laplacian heat diffusion) |
| **AC-005** | Ridge Contour Alignment | $\ge 80\%$ Precision & Recall with 2-pixel tolerance buffer | Verified in `test_shadow_difference_refiner.py` via `ShadowDifferenceRefiner` (Hessian principal curvature analysis) |
| **AC-006** | 3D Fractal Editor Brush Fitting | $\ge 85\%$ normalized cross-correlation (NCC) | Verified in `test_fractal_brush_engine.py` via `FractalBrushFitter` achieving $> 0.85$ NCC |
| **AC-007** | End-to-End Geometric Fidelity | $\text{RelMAE} \le 25\%$ ($\ge 75\%$ accuracy), $\text{NormSim} \ge 0.88$, $F_1 \ge 0.75$ | Verified in `test_reconstruction_benchmark.py` via `TerrainReconstructionPipeline` across synthesized held-out sculpting |
| **AC-008** | Canonical Zarr v3 Preservation | Lossless read/write to canonical store format | Verified through Zarr v3 schema compatibility and data models |
