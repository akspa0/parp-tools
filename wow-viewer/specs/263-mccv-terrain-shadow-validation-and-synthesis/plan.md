# Plan: Spec 263 — 1.60 MCCV Terrain Shadow Ground-Truth Validation & Bidirectional Renderer Synthesis

## 1. System Architecture

```
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│      1.12.1 Authored Minimap Image     │       │        1.60 Modern Client ADT          │
│        (64x64 Continent Grid)          │       │        (WoW: Forever 1.60.1)           │
└───────────────────┬────────────────────┘       └───────────────────┬────────────────────┘
                    │                                                │
                    ▼                                                ▼
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│     Spec 262 Minimap Shadow Sieve      │       │          MCNK MCCV Extractor           │
│   `stripped_residual_shadow_256`       │       │    (145 Vertices BGRA per Chunk)       │
└───────────────────┬────────────────────┘       └───────────────────┬────────────────────┘
                    │                                                │
                    ▼                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       MCCV Terrain Shadow Cross-Correlation Engine                      │
│            1. Rasterize 145-vertex lattice to continuous 256x256 shadow map             │
│            2. Compute Normalized Cross-Correlation (NCC >= 0.70) & MAE                  │
│            3. Align directional Hessian ridges with MCCV crease/valley minima           │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Bidirectional Synthesis & Injection Bridge (C# / Py)                 │
│      • Forward: Sample minimap residual shadow onto 145 vertices -> Write 1.60 MCCV     │
│      • Reverse: Use 1.60 MCCV as analytical supervisor for 3D terrain reconstruction    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technical Decomposition

### 2.1 Component 1: 1.60 ADT MCCV Extractor & Grid Interpolator
- **C# Core / IO**:
  - `MccvExtractorService.cs` in `WowViewer.Core.IO/Maps/`:
    - Reads MCNK `MCCV` sub-chunks (580 bytes: 145 BGRA vertex colors).
    - Checks MCNK header flag `0x40` (`has_mccv`) or non-null chunk payload.
    - Exports normalized float arrays or serialized JSON for ML pipeline.
- **Python Harvester**:
  - `mccv_shadow_comparator.py` in `data-harvester/src/harvester/v60/`:
    - Interpolates 145 vertices (81 outer $9 \times 9$ + 64 inner $8 \times 8$) using `LinearNDInterpolator` onto a $256 \times 256$ continuous terrain shadow modulation grid.

### 2.2 Component 2: 1.12.1 vs 1.60 Comparative Correlation Engine
- Evaluates:
  $$\text{NCC} = \frac{\sum (S_{1.12} - \bar{S}_{1.12})(S_{1.60} - \bar{S}_{1.60})}{\|S_{1.12} - \bar{S}_{1.12}\| \cdot \|S_{1.60} - \bar{S}_{1.60}\|}$$
- Extracts crease and valley minimums in 1.60 `MCCV` and compares with directional Hessian ridges from `ShadowDifferenceRefiner`.
- CLI script `scripts/v60_compare_mccv_residuals.py` produces visual comparison quilts `[1.12 Minimap | Bare Residual Shadow | 1.60 MCCV Shadow | Overlay Difference]`.

### 2.3 Component 3: 1.60 MCCV Synthesis & Injection Engine
- Takes any 2D residual shadow field ($256 \times 256$) from 0.5.3 or 1.12.1 maps.
- Evaluates `synthesize_mccv_from_residual()`:
  - Maps chunk coordinates into $(u, v)$ for the 145 vertices.
  - Samples pixel luminance using bilinear interpolation.
  - Generates valid 580-byte BGRA byte streams per chunk.
- Integrates with `LkAdtWriter.cs` / `AdtChunkWriter.cs` in `WowViewer.Core.IO` to populate `MCCV` chunks into modern ADT exports.

---

## 3. Directory & File Plan

### C# Core (`wow-viewer/src/core/WowViewer.Core.IO/Maps/`)
- `MccvTerrainShadowService.cs`: Service to extract and inject 145-vertex MCCV terrain shadow arrays.

### C# Tests (`wow-viewer/tests/WowViewer.Core.Tests/Maps/`)
- `MccvTerrainShadowServiceTests.cs`: Unit tests for vertex lattice indexing, 580-byte BGRA packing, and roundtrip serialization.

### Python Engine (`wow-viewer/data-harvester/`)
- `src/harvester/v60/mccv_shadow_comparator.py`: Comparator and synthesis bridge.
- `scripts/v60_compare_mccv_residuals.py`: CLI comparison runner producing correlation metrics and visual sheets.
- `tests/v60/test_mccv_shadow_comparator.py`: Unit test coverage.

---

## 4. Phase Roadmap

- **Phase 1**: C# & Python 1.60 MCCV Extraction & 145-Vertex Lattice Geometry.
- **Phase 2**: Comparative Correlation Engine & Residual Cross-Validation (AC-001, AC-002, AC-003).
- **Phase 3**: 1.60 MCCV Synthesis & ADT Injection Bridge (AC-004).
- **Phase 4**: Renderer Modulation Verification in WoW: Forever (AC-005).
- **Phase 5**: Governance, Receipts & Documentation (AC-006).
