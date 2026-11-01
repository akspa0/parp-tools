# Spec 289 Phase 2 Receipt: Core 3D Footprint & Synthetic Wall Extrusion Engine

**Date**: 2026-11-01  
**Owner**: Spec 289 (PM4 3D Footprint Extrusion & Direct MPQ Matching)  
**Status**: COMPLETE / VERIFIED  

---

## 1. Summary of Changes

Phase 2 implemented the core pure geometric algorithms in `WowViewer.Core.PM4`:
1. **2D Convex Hull Extraction (`ComputeConvexHull2D`)**: Implemented Andrew's monotone chain algorithm ($O(n \log n)$) to compute the convex hull / boundary perimeter from surface vertices.
2. **Base Elevation Resolution (`ResolveBaseElevation`)**: Formulated $Z_{\text{base}}$ derivation prioritizing explicit Placement-Z for WMO structures (`MSUR.0x1C`), mean terrain contact elevations from MPRL points for doodads, and bounding minimum fallback.
3. **2D Principal Component Analysis (`ComputePrincipalOrientation`)**: Covariance-matrix PCA recovering the true unrotated Length, Width, Centroid, and primary Yaw angle $\theta$ independent of rotation.
4. **Synthetic Vertical Wall Extrusion (`ExtrudeFootprint`)**: Generates vertical quad walls connecting each top perimeter edge $[v_i, v_{i+1}]$ at $Z_{\text{top}}$ down to $[v_i, v_{i+1}]$ at $Z_{\text{base}}$, providing 3D collision envelope quads and 6-vertex CCW triangles.

### Files Created:
- `wow-viewer/src/core/WowViewer.Core.PM4/Footprint/Pm4FootprintExtruder.cs`: Algorithmic engine and data models (`Pm4ExtrudedWall`, `Pm4FootprintPrism`).
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4FootprintExtrusionTests.cs`: 4 unit tests validating convex hull extraction, PCA orientation/dimension recovery, base elevation resolution, and 3D prism generation.

---

## 2. Verification Receipts

| Command | Exit Code | Result |
|---|---|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/ -c Debug --filter "FullyQualifiedName~Pm4FootprintExtrusionTests"` | 0 | **Passed: 4, Failed: 0, Skipped: 0** |

### Criterion Mapping:
- **AC-002**: Pure algorithm in `WowViewer.Core.PM4` extracts 2D convex hull of top surfaces, resolves $Z_{\text{base}}$ from MPRL/Placement-Z, and extrudes vertical wall quad geometry. PASS.

