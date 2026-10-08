# Technical Plan: Spec 266 WDL Trestle Elevation Reconstruction

## 1. System Architecture

```
                       [ Development Minimap RGB (256x256) ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     [ Lineage WDL Match Lookup ]                 [ Minimap Normal Estimator ]
     (development_X_Y -> Northrend_A_B)           (Photometric Normals Nx, Ny)
                 │                                               │
                 ▼                                               │
     [ Coarse WDL Lattice 17x17 ]                                │
     (Authentic or Synthesized Lineage)                          │
                 │                                               │
                 ▼                                               │
     [ Bicubic Upsample to 257x257 ]                             │
     (Z_trestle in physical yards)                               │
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                         [ 6-Channel Feature Tensor ]
                         - Channel 0..2: Minimap RGB (3)
                         - Channel 3: Normalized Z_trestle (1)
                         - Channel 4..5: Normals (Nx, Ny) (2)
                                         │
                                         ▼
                           [ TrestleElevationUNet ]
                       (Base ch 32 -> 64 -> 128 -> 256)
                       (~6.2M Parameters vs 100M+ in v7)
                                  │             │
                ┌─────────────────┘             └─────────────────┐
                ▼                                                 ▼
     [ Dense Residual Head ]                           [ Global Bounds Head ]
     Predicts ΔZ (yards)                               Predicts [Z_min, Z_max]
                │
                ▼
     [ Z_final = Z_trestle + ΔZ ]
                │
                ▼
     [ Upright Mesh Generation ]
     - CCW Face Winding
     - Normals Point +Z (OBJ) / +Y (GLB)
     - Full Vertical Relief (200-600+ yards)
```

---

## 2. Key Modules & Design

### 2.1 `harvester.v60.trestle_wdl_synthesizer`
- Loads `output/development_to_northrend_matches.json` (2,117 matched pairs) and `output/northrend_wdl_301.npz` (736 WDL tiles).
- For each development tile $(x, y) \in [0..63] \times [0..63]$:
  - If development map has native WDL data in `original_development/development.wdl`, load native 17x17.
  - Else if matched to a Northrend tile with similarity $\ge 0.85$, transfer Northrend's 17x17 outer lattice.
  - Else interpolate from neighboring valid tiles.
- Caches the complete synthesized development lattice map to `output/development_synthesized_wdl.npz`.

### 2.2 `harvester.v60.trestle_elevation_model`
- PyTorch implementation of `TrestleElevationUNet`:
  - 6 input channels: RGB (3) + Normalized $Z_{\text{trestle}}$ (1) + Photometric Normals (2).
  - Encoder: 4 stages with residual convolutions (32, 64, 128, 256 channels).
  - Decoder: 4 stages with transposed convolutions and skip connections.
  - Dense Output: $\Delta Z$ (residual in physical yards).
  - Auxiliary Head: Global Average Pooling $\to$ Linear layers predicting $[Z_{\min}, Z_{\max}]$.
- Loss function:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{1}(Z_{\text{pred}}, Z_{\text{gt}}) + \lambda_{\text{grad}} \mathcal{L}_{\text{grad}} + \lambda_{\text{bounds}} \mathcal{L}_{\text{bounds}}$$
  where $Z_{\text{pred}} = Z_{\text{trestle}} + \Delta Z$.

### 2.3 `harvester.v60.trestle_dataset`
- Assembles dataset pairs:
  - Minimap PNG (256x256).
  - Coarse WDL trestle (17x17 upsampled to 257x257).
  - Photometric normals $(N_x, N_y)$.
  - Ground truth ADT elevation $Z_{\text{gt}}$ (257x257).
  - Target residual $\Delta Z = Z_{\text{gt}} - Z_{\text{trestle}}$.
- Augmentation: Horizontal and vertical flips, random 90-degree rotations (with normal coordinate transformation), subtle color jitter.

### 2.4 `scripts/v60_train_trestle_model.py`
- Training script using PyTorch, AMP (mixed precision), and AdamW optimizer.
- Evaluates validation MAE, Pearson $r$, and relief span restoration on held-out tiles.
- Saves model checkpoint to `output/models/trestle_elevation_v1.pt`.

### 2.5 Pipeline Integration in `scripts/v60_reconstruct_minimap.py`
- Injects `--trestle-model` and `--synthesized-wdl`.
- Automatically loads the matched/synthesized WDL trestle for any target development tile.
- Reconstructs $Z_{\text{final}} = Z_{\text{trestle}} + \Delta Z$.
- Exports OBJ and GLB meshes with verified upward-facing CCW face winding.

---

## 3. Implementation Phases

- **Phase 1: Lineage WDL Prior Synthesis (AC-001)**
  - Implement `trestle_wdl_synthesizer.py`.
  - Generate and verify `output/development_synthesized_wdl.npz`.
- **Phase 2: Lean Trestle Model Architecture & Unit Testing (AC-002, AC-005)**
  - Implement `TrestleElevationUNet` in `trestle_elevation_model.py`.
  - Author unit tests in `tests/v60/test_trestle_elevation.py` validating shapes, gradient flow, bounds head, and CCW face winding.
- **Phase 3: Dataset Builder & Model Training (AC-003, AC-004)**
  - Build `trestle_dataset.py` caching training tensors.
  - Implement `scripts/v60_train_trestle_model.py`.
  - Train model and record convergence receipts.
- **Phase 4: Full Pipeline Reconstruction & Mesh Validation (AC-003, AC-006)**
  - Integrate into `scripts/v60_reconstruct_minimap.py`.
  - Reconstruct mountain tiles (`development_16_33`, `development_16_32`, `development_0_0`).
  - Verify vertical amplitude ($\ge 250$ yards) and CCW upward mesh normals.
