# Technical Plan: Spec 265 Two-Stage Cascaded Residual Elevation Refinement

## 1. System Architecture

```
                                  [ Minimap RGB (256x256) ]
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │                                             │
                       ▼                                             ▼
          [ Stage 1: Base U-Net ]                       [ Minimap Context Features ]
          (Trained in Spec 264)                         (Luminance, Rock Mask, Edge Gradient)
                       │                                             │
                       ▼                                             │
               Z_initial (257x257)                                   │
                       │                                             │
                       ├──────────────────────┐                      │
                       ▼                      ▼                      │
                 Surface Normals        Height Gradient              │
                 (257x257x3)             (257x257x2)                 │
                       │                      │                      │
                       └──────────────┬───────┴──────────────────────┘
                                      │
                                      ▼
                        [ 6-Channel Feature Tensor ]
                        (RGB [3] + Z_initial [1] + Normals [2])
                                      │
                                      ▼
                   [ Stage 2: ResidualElevationRefiner ]
                   (Predicts ΔZ = Z_gt - Z_initial in physical yards)
                                      │
                                      ▼
                                 ΔZ_pred (257x257)
                                      │
                                      ▼
               [ Z_final = Z_initial + ΔZ_pred ]
                                      │
                                      ▼
                   [ Boundary Seam & Water Level Clamp ]
                                      │
                                      ▼
                        [ Export OBJ & GLB (Yards) ]
```

---

## 2. Key Modules & File Structure

1. **`harvester.v60.residual_elevation_dataset`**:
   - Generates and caches `(x_features, delta_z)` pairs from the 270 authentic development tiles.
   - Computes $Z_{\text{initial}}$ using the frozen Stage 1 checkpoint (`supervised_elevation_development.pt`).
   - Targets: $\Delta Z = Z_{\text{gt}} - Z_{\text{initial}}$.

2. **`harvester.v60.residual_elevation_model`**:
   - Multi-scale residual U-Net (`ResidualElevationRefiner`).
   - 6 input channels: RGB (3) + Normalized $Z_{\text{initial}}$ (1) + Unit Surface Normals (2).
   - Direct output: $\widehat{\Delta Z}$ in yards.
   - Composite loss: L1 error on residual + Normal alignment + Boundary edge penalty.

3. **`scripts/v60_train_residual_refiner.py`**:
   - Operator-owned PyTorch training CLI for Stage 2.
   - Evaluates combined $Z_{\text{final}} = Z_{\text{initial}} + \widehat{\Delta Z}$ on the held-out validation set.

4. **`harvester.v60.seam_boundary_calibrator`**:
   - Enforces border vertex continuity between adjacent tiles $(T_x, T_y)$ and $(T_x+1, T_y)$.
   - Clamps water vertices to $Z = 0.000$ sea level.

5. **`scripts/v60_reconstruct_minimap.py`**:
   - Integrates cascaded two-stage execution (`--stage1-model` + `--refiner-model`).

---

## 3. Phase Roadmap

- **Phase 1: Residual Dataset Generation & Error Analysis**
  - Extract $Z_{\text{initial}}$ predictions on all 270 tiles.
  - Calculate residual $\Delta Z$ and cache to `output/datasets/residual_elevation_corpus.npz`.
- **Phase 2: Stage 2 Model Implementation & Unit Tests**
  - Implement `ResidualElevationRefiner` and composite residual loss.
  - Unit tests covering forward pass, gradient flow, and shape checks.
- **Phase 3: Operator GPU Training & Validation**
  - Train Stage 2 model on RTX 4070 Ti SUPER.
  - Verify validation MAE reduction and mountain relief recovery.
- **Phase 4: Seam Boundary Continuity & Pipeline Integration**
  - Wire into `v60_reconstruct_minimap.py`.
  - Re-export verified 3D meshes for `development_16_33` and `development_0_0`.
