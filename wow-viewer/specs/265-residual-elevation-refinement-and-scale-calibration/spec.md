# Spec 265: Two-Stage Cascaded Residual Elevation Refinement & Scale Calibration

## 1. Overview & Problem Statement

In Spec 264, the single-stage U-Net successfully captured high relative terrain correlation ($r = 0.9029$ on `development_16_33`, $r = 0.9310$ on `development_16_32`). However, when predicting absolute world-space elevation directly from isolated 256x256 minimap crops, the model suffered from **regression to the mean**:
- Across the authentic development map, base elevation spans from $-656.87$ to $+813.10$ yards (over a 1,470-yard vertical offset across tiles).
- Without macro-elevation context, the base network compressed predicted relief spans to an almost constant $\approx 35$ yards across all tiles, squashing a 312-yard mountain down to 39 yards.
- Global validation MAE hovered at 81.07 yards because subterranean tiles (e.g. `development_15_34` at $Z = -650$) and high alpine peaks were penalized heavily.

This spec addresses the problem by adopting a **Two-Stage Cascaded Residual Architecture**:
1. **Stage 1 (Shape Prior)**: Generates the base terrain shape and surface slopes $Z_{\text{initial}}$.
2. **Stage 2 (Residual Refinement Model)**: Directly consumes the initial prediction, surface normals, and minimap RGB to predict the explicit difference metric:
   $$\Delta Z(x, y) = Z_{\text{gt}}(x, y) - Z_{\text{initial}}(x, y)$$
3. **Seam Boundary Continuity**: Calibrates base elevation and scale using shared border vertices between adjacent tiles ($T_{x}, T_{y}$).

---

## 2. User Stories

### User Story 1 - Macro Amplitude Restoration (Priority: P1)
As a map reconstruction engineer, I want the mountain relief on tiles like `development_16_33` and `development_16_32` to span their authentic vertical height ($>250$ yards) rather than being squashed to 35 yards, so that 3D meshes match the scale of the real World of Warcraft game world.

### User Story 2 - Residual Error Attribution & Learning (Priority: P1)
As an ML researcher, I want a dedicated residual model that learns the difference metric $\Delta Z$ between reconstructed and ground-truth elevation, correcting systematic under-prediction and texture splat artifacts.

### User Story 3 - Tile Boundary Seam Consistency (Priority: P2)
As a world builder, I want adjacent reconstructed tiles to share continuous elevation along shared borders (Column 256 of Tile $X$ matching Column 0 of Tile $X+1$), eliminating inter-tile cliffs and floating boundaries.

---

## 3. Acceptance Criteria

| ID | Criterion | Requirement | Target Metric |
|---|---|---|---|
| **AC-001** | Residual Dataset Generation | Cache pairs of `(RGB, Z_initial, Normals)` mapped to `ΔZ = Z_gt - Z_initial` | 270 tiles cached with $\Delta Z$ fields |
| **AC-002** | Relief Span Recovery | Restores vertical amplitude on high-relief mountain tiles | Predicted span on `development_16_33` $\ge 200$ yds (was 39 yds) |
| **AC-003** | Error Reduction | Cascaded model $Z_{\text{final}} = Z_{\text{initial}} + \widehat{\Delta Z}$ beats baseline | Validation MAE $\le 30.0$ yards (down from 81.07 yds) |
| **AC-004** | Land Correlation | Land-only Pearson correlation on held-out validation tiles | Land Pearson $r \ge 0.70$ across validation split |
| **AC-005** | Seamless Seam Continuity | Boundary vertex delta between adjacent tiles | Mean edge gap $\le 5.0$ yards |
| **AC-006** | Pure Terrain Export | Clean OBJ/GLB meshes without black boxes or wave artifacts | $o \text{ Terrain}$ only, loadable in MeshLab |

---

## 4. Constraints

- Zero hardcoded machine-local client paths (AGENTS.md §5).
- All training and dataset tooling lives under `wow-viewer/data-harvester/` (AGENTS.md §5).
- God-class freeze: Zero member additions to `WorldScene.cs` or `ViewerApp.cs` (AGENTS.md §10).
- Pure PowerShell 7 syntax for all operator commands (AGENTS.md §5).
