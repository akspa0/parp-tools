# Specification Quality & Requirements Checklist: Spec 262

**Feature**: [Spec 262: Minimap Residual Model & 3D Fractal Editor Brush Reconstruction](../spec.md)  
**Created**: 2026-10-06  

---

## 1. Specification Quality

- [x] **No Ambiguous Markers**: No `[TBD]` or `[NEEDS CLARIFICATION]` markers remain.
- [x] **User Value Centered**: Clearly defines the path from raw 2D minimaps to high-fidelity 3D terrain meshes.
- [x] **Separation of Concerns**: Physical lighting calibration, object sieving, brush cataloging, and mesh reconstruction are decoupled into isolated services.
- [x] **Technology-Agnostic Success Criteria**: Targets (MAE, Normal Cosine Similarity, F1 Score) are mathematically defined and independent of framework specifics.
- [x] **Strict Architectural Compliance**:
  - [x] Zero additions to god-classes (`WorldScene.cs`, `ViewerApp.cs`) per `AGENTS.md` §10.
  - [x] Strict adherence to Core Library First per `AGENTS.md` §4.
  - [x] Python tools contained in `wow-viewer/data-harvester/` using `uv` per `AGENTS.md` §5.
  - [x] Operator-owned training and client data harvest gates per `AGENTS.md` §5.

---

## 2. Acceptance Criteria Verification Matrix

| AC ID | Description | Target Metric | Verification Method / Command |
|---|---|---|---|
| **AC-001** | Lighting Calibration Convergence | $< 5\%$ Photometric MAE | `uv run python scripts/v60_solve_minimap_lighting.py --validate` |
| **AC-002** | Rosetta Overhead Catalog Completeness | 100% M2/WMO exhibits cataloged | `dotnet test --filter "RosettaOverheadCatalogExporterTests"` |
| **AC-003** | SAM 2.1 Object Segmentation IoU | $\ge 0.85$ IoU against control masks | `uv run pytest tests/v60/test_sam_minimap_sieve.py -k test_iou` |
| **AC-004** | Bare Shadow Residual Extraction | $\ge 90\%$ Object energy attenuation | `uv run pytest tests/v60/test_minimap_shadow_stripper.py -k test_attenuation` |
| **AC-005** | Ridge/Residual Fidelity | $\ge 80\%$ Precision & Recall on ridges | `uv run pytest tests/v60/test_shadow_difference_refiner.py -k test_ridges` |
| **AC-006** | 3D Fractal Brush Catalog | $\ge 85\%$ cross-correlation on 0.5.3 motifs | `uv run pytest tests/v60/test_fractal_brush_engine.py -k test_correlation` |
| **AC-007** | 75% Geometric Accuracy Target | $\text{RelMAE} \le 25\%$, $\text{NormSim} \ge 0.88$, $F_1 \ge 0.75$ | `uv run python scripts/v60_benchmark_reconstruction.py --held-out` |
| **AC-008** | Governance & Receipts | Complete receipt in `evidence/receipt-spec262.md` | Audit inspection per `AGENTS.md` §9.2 |

---

## 3. Edge Cases & Mitigation Strategies

| Edge Case | Failure Mode | Mitigation Strategy |
|---|---|---|
| **Flat Water / Deep Ocean Tiles** | Zero relief causes division by zero in $\text{RelMAE}$. | Gate evaluation on `terrain_amplitude > 1.0` meters; classify liquid tiles using liquid flags. |
| **Extreme Specular Saturation (Snow/Sand)** | High specular reflection clips RGB to 255/255/255, destroying shadow gradients. | Tone-mapping compression in `TerrainLightingMath.cs` and tone-curve optimization during photometric solve. |
| **Whiteplates with Sub-Millimeter Relief** | Weak signal tiles (relief $< 1\,\text{cm}$) lost under DXT1 quantization noise floor. | Integrate `WeakSignalDetector.cs` auto-amplification and level-counting before calibration. |
| **Boundary-Crossing Fractal Brushes** | Stamps cut off by tile borders produce seam tears in reconstructed mesh. | Evaluate $17 \times 17$ chunk neighborhood with 1-chunk border padding during brush fitting. |
| **Unrecognized Novel Objects** | Obscure doodads absent from Rosetta cause inpainting hole artifacts. | SAM 2.1 background prior fills mask using surrounding diffuse albedo textures. |
