# Spec 289 Phase 3 Receipt: Scale-Invariant Geometric Signature & Zero-Donor MPQ Matcher

**Date**: 2026-11-01  
**Owner**: Spec 289 (PM4 3D Footprint Extrusion & Direct MPQ Matching)  
**Status**: COMPLETE / VERIFIED  

---

## 1. Summary of Changes

Phase 3 implemented the scale-invariant geometric matching engine that replaces legacy Rosetta / cross-map heuristic guesswork with direct shape comparison:
1. **Geometric Shape Descriptor (`Pm4GeometricShapeDescriptor`)**:
   - Calculates dimensionless aspect ratios: $\text{Aspect}_{XY} = \text{Length} / \text{Width} \ge 1.0$ and $\text{Aspect}_{ZH} = \text{Height} / \text{Length}$.
   - Calculates 2D footprint area via the Shoelace formula on the convex hull.
   - Computes the local origin offset $\vec{\Delta}_{\text{origin}} = -\vec{C}_{\text{local}}$ to capture the WC3-heritage asymmetric pivot offset (where model origin $(0,0,0)$ resides at the entrance doorstep / stairs / bridge anchor rather than the bounding box center).
2. **Direct Zero-Donor Model Matcher (`Pm4DirectModelMatcher`)**:
   - Compares PM4 3D footprint prisms directly against candidate model bounds.
   - Computes scale-invariant scoring based on normalized aspect differences: $\text{Score} = 1.0 - (0.5 \cdot \text{Diff}_{XY} + 0.5 \cdot \text{Diff}_{ZH})$.
   - Solves the model scale factor: $s = \text{PrismPlanarSpan} / \text{CandidatePlanarSpan}$.
   - Solves the Yaw angle by aligning candidate principal axes to the PM4 footprint's principal axis.
   - Reconstructs authentic placement origin by rotating $\vec{\Delta}_{\text{origin}} \cdot s$ into world space: $\vec{P}_{XY} = \vec{C}_{\text{prism}} + \text{Rotate}(\vec{\Delta}_{\text{origin}} \cdot s, \theta)$.
   - Solves vertical placement $P_Z$ using top-surface alignment for doodads and base alignment for WMOs.

### Files Created:
- `wow-viewer/src/core/WowViewer.Core.PM4/Footprint/Pm4GeometricShapeDescriptor.cs`: Shape signature model and Shoelace area calculation.
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DirectModelMatcher.cs`: Direct MPQ asset matching engine.
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4DirectModelMatcherTests.cs`: 2 unit tests validating aspect ratio ranking, scale solving, top-alignment, and asymmetric pivot offset mapping.

---

## 2. Verification Receipts

| Command | Exit Code | Result |
|---|---|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/ -c Debug --filter "FullyQualifiedName~Pm4DirectModelMatcherTests"` | 0 | **Passed: 2, Failed: 0, Skipped: 0** |

### Criterion Mapping:
- **AC-003**: Geometric descriptors compute $W, D, H$, aspect ratio $W/D$, height ratio $H/\max(W,D)$, and footprint polygon area. PASS.
- **AC-004**: Matcher ranks candidates by shape distance and solves placement position using WC3-heritage asymmetric pivot offset without donor map dependencies. PASS.

