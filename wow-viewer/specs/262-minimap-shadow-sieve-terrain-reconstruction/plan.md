# Plan: Spec 262 — Minimap Residual Model, Automated Lighting Calibration & 3D Fractal Editor Brush Reconstruction

## 1. System Architecture & Component Interactions

```
                                ┌────────────────────────────────────────┐
                                │        0.5.3 Real Client Data          │
                                │   (Whiteplate Tiles & Minimap BLPs)    │
                                └───────────────────┬────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────┐     ┌────────────────────────────────┐
│   Terrain Minimap Compositor    │     │   Minimap Lighting Solver      │
│     (Lambert + Cast Shadow)     ├────►│  (Optimize θ, φ, Ambient/Diff) ├────┐
│      + DXT1 Compression         │     │     Minimizes Photometric L1   │    │
└────────────────┬────────────────┘     └────────────────────────────────┘    │ Calibrated
                 │                                                            │ Lighting
                 ▼                                                            ▼ Parameters
┌─────────────────────────────────┐     ┌──────────────────────────────────────────────────┐
│   Synthetic Control Generator   │     │            SAM 3.1 Minimap Sieve                 │
│  (Terrain Shadow Ground Truth)  │     │ 1. Segment doodads/structures/roads              │
└────────────────┬────────────────┘     │ 2. Guided by Rosetta overhead object catalog     │
                 │                      │ 3. Inpaint masked areas & strip to bare signal   │
                 │                      └────────────────────────┬─────────────────────────┘
                 │                                               │
                 │                                               ▼
                 │                      ┌──────────────────────────────────────────────────┐
                 │                      │         Bare Residual Shadow Signal              │
                 │                      │       (`stripped_residual_shadow_256`)           │
                 │                      └────────────────────────┬─────────────────────────┘
                 │                                               │
                 ▼                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                      Closed-Loop Shadow Difference & Ridge Extractor                     │
│               ΔS = S_real(observed) - S_synth(current, calibrated_lighting)              │
│               Extracts residual crests, cliff breaks, and macro terrain features          │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                   3D Fractal Editor Brush Discovery & Fitting Engine                     │
│       • Isolates recurring procedural & fractal relief stamps in residual signal         │
│       • Extracts coupled 3D mesh displacement ΔZ(u, v) and texture alpha α_k(u, v)       │
│       • Fits discrete editor brush invocations (pos, rot, scale, amp) into terrain mesh  │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                      Terrain Mesh Reconstruction Engine (v60 / Core)                     │
│                Inverts residual shadow into 3D Heightmap (height_257 / MCVT)             │
│                Verifies >= 75% Geometry Parity vs Real 0.5.3 Ground-Truth Chunks         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technical Decomposition

### 2.1 Component 1: Automated Lighting Calibration & Specular Physics
- **Core C# Layer**: [`MinimapShadingMatch.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/MinimapShadingMatch.cs) provides initial shading matching. We extend it with an automated parameter optimizer:
  - `MinimapLightingOptimizer.cs`: Evaluates sun azimuth $\theta \in [0, 2\pi)$ and elevation $\phi \in (0, \pi/2]$, ambient factor $A \in [0.1, 0.6]$, and diffuse factor $D \in [0.4, 0.9]$.
  - Leverages [`TerrainMinimapCompositor.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs) with native DXT1 block simulation ([`Dxt1TileCodec.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Blp/Dxt1TileCodec.cs)).
  - Incorporates texture-dependent specular lobe modeling ($k_{s, k}, p_k$) in [`TerrainLightingMath.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Terrain/TerrainLightingMath.cs) against orthographic top-down half-angle vector $\mathbf{H}$.
- **Python ML Layer**: `wow-viewer/data-harvester/src/harvester/v60/minimap_lighting_solver.py`:
  - Rapid SciPy L-BFGS-B / PyTorch gradient-based photometric calibration on whiteplate tiles from 0.5.3 Azeroth/Kalimdor.
  - Automatically produces a calibration profile `lighting_calibration_0_5_3.json`.

### 2.2 Component 2: Rosetta Overhead Object Catalog & Mothership VLM Reasoning
- **Rosetta Datastore Link**: [`RosettaObjectLibrary.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Maps/RosettaObjectLibrary.cs) already catalogs every placed object from the Rosetta calibration maps into the canonical Zarr v3 datastore.
- **Python Harvester Service**: `wow-viewer/data-harvester/src/harvester/v60/rosetta_vision_catalog.py`:
  - Extracts 2D overhead captures (`capture_rgb`, `capture_mask`, bounding radii, principal axes) into an indexed Parquet/Zarr v3 dictionary.
  - Integrates **Mothership VLM** / **PaliGemma 2** custom nodes hosted on the local ComfyUI instance (`http://127.0.0.1:8199`) to classify overhead candidate matches and generate object bounding-box prompts.

### 2.3 Component 3: SAM 3.1 Minimap Sieve & ComfyUI Orchestration Layer
- **Orchestration Client**: `wow-viewer/data-harvester/src/harvester/v60/comfyui_orchestrator.py`:
  - Connects to the local running ComfyUI service at `http://127.0.0.1:8199` (RTX 4070 Ti SUPER 16GB VRAM).
  - Routes minimap tiles through the authenticated **SAM 3.1** and **Mothership Core** node graph using prompt/queue/history APIs.
  - Reuses the existing authenticated Hugging Face token and local model weights (`J:\fresh\models`).
- **Python Harvester Sieve**: `wow-viewer/data-harvester/src/harvester/v60/sam_minimap_sieve.py`:
  - Delegates promptable segmentation to SAM 3.1 via the orchestration client.
  - Generates binary instance and cumulative object contamination masks $\mathcal{M}_{\text{obj}}$.
- **Stripping & Inpainting**: `wow-viewer/data-harvester/src/harvester/v60/minimap_shadow_stripper.py`:
  - Multi-scale Navier-Stokes / biharmonic inpainting over $\mathcal{M}_{\text{obj}}$.
  - Normalizes texture albedo variations using local luminance filtering, producing `stripped_residual_shadow_256`.

### 2.4 Component 4: Closed-Loop Shadow Difference & Ridge Residual Refiner
- **Python Harvester Service**: `wow-viewer/data-harvester/src/harvester/v60/shadow_difference_refiner.py`:
  - Computes $\Delta S(x, y) = S_{\text{real}}(x, y) - S_{\text{synth}}(x, y)$.
  - Applies oriented steerable filter banks and Canny ridge detection to isolate residual topography (unmodeled ridges, peaks, cliff edges).
  - Feeds $\Delta S$ back into the synthetic shadow generator to verify that synthesized shadow converges toward real client shadow.

### 2.5 Component 5: 3D Fractal Editor Brush Discovery & Fitting Engine
- **Archaeological Brush Catalog**: `wow-viewer/data-harvester/src/harvester/v60/fractal_brush_extractor.py`:
  - Builds upon [`alpha_brush.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/alpha_brush.py) and [`fractal_library.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/fractal_library.py).
  - Isolates recurring procedural/fractal patches in residual relief and alpha layers.
  - Saves them as discrete **3D Editor Brushes**: coupled kernels $(\Delta Z(u, v), \alpha_k(u, v))$ with defined spatial metrics (16m, 32m, 64m footprints).
- **Brush Inversion & Fitting**: `wow-viewer/data-harvester/src/harvester/v60/fractal_brush_fitter.py`:
  - Fits parameterized stamp invocations $(\mathcal{B}_j, \mathbf{p}_i, \sigma_i, \theta_i, A_i)$ to explain recognized fractal residual contours.
  - Generates discrete mesh sculpting steps and texture splats corresponding to authentic early terrain editor operations.

### 2.6 Component 6: Terrain Mesh Reconstruction ($\ge 75\%$ Parity)
- **Model Architecture**: `clean_signal_model.py` / `residual_height_model.py`:
  - Multi-scale convolutional residual architecture combining continuous residual height prediction with discrete 3D fractal brush stamping.
  - Materializes reconstructed terrain heights into $257 \times 257$ grid ($16 \times 16$ chunks $\times 145$ vertices per chunk).
- **Geometric Benchmark Runner**: `wow-viewer/data-harvester/src/harvester/v60/reconstruction_benchmark.py`:
  - Computes Relative Height MAE ($\le 25\%$), Vertex Normal Cosine Similarity ($\ge 0.88$), and Ridge F1 ($\ge 0.75$).

---

## 3. Directory & File Plan

### C# Core (`wow-viewer/src/core/WowViewer.Core.IO/Maps/`)
- `MinimapLightingOptimizer.cs`: Automated calibration search for solar angle and ambient/diffuse balance.
- `RosettaOverheadCatalogExporter.cs`: CLI helper to dump isolated 2D top-down silhouettes from Rosetta datastores.

### C# Tests (`wow-viewer/tests/WowViewer.Core.Tests/Maps/`)
- `MinimapLightingOptimizerTests.cs`: Unit tests for lighting error computation and convergence on synthetic known ground-truth.

### Python ML (`wow-viewer/data-harvester/src/harvester/v60/`)
- `comfyui_orchestrator.py`: Loopback client to ComfyUI (`http://127.0.0.1:8199`) for SAM 3.1 & Mothership VLM execution.
- `minimap_lighting_solver.py`: SciPy/PyTorch lighting calibration on 0.5.3 whiteplates.
- `rosetta_vision_catalog.py`: Rosetta overhead object dictionary loader and matcher.
- `sam_minimap_sieve.py`: SAM 3.1 instance segmenter and prompt builder.
- `minimap_shadow_stripper.py`: Minimap object excision and inpainting pipeline.
- `shadow_difference_refiner.py`: Difference signal extraction and ridge isolator.
- `fractal_brush_extractor.py`: Extracts and saves discrete 3D fractal editor brushes with coupled $(\Delta Z, \alpha_k)$.
- `fractal_brush_fitter.py`: Fits discrete 3D editor brush invocations to residual signals.
- `reconstruction_benchmark.py`: Held-out 0.5.3 evaluation runner computing the $\ge 75\%$ geometric parity metrics.

### Python Tests (`wow-viewer/data-harvester/tests/v60/`)
- `test_comfyui_orchestrator.py`: Unit test verifying loopback communication, node payload formatting, and history retrieval.
- `test_minimap_lighting_solver.py`: Unit test for lighting calibration math.
- `test_sam_minimap_sieve.py`: Unit test for SAM 3.1 prompt creation and segmentation masking.
- `test_minimap_shadow_stripper.py`: Unit test for bare shadow extraction and inpainting.
- `test_shadow_difference_refiner.py`: Unit test for $\Delta S$ computation and ridge extraction.
- `test_fractal_brush_engine.py`: Unit test for 3D fractal brush extraction, cataloging, and fitting.
- `test_reconstruction_benchmark.py`: Unit test for geometric evaluation metrics.
