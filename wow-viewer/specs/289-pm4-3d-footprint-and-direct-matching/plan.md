# Spec 289 Plan: PM4 3D Footprint Extrusion & Direct MPQ Matching

## 1. Architectural Strategy & Boundaries

In accordance with `AGENTS.md`:
- **Core Library First**: All pure geometric algorithms (footprint extraction, wall extrusion, shape signatures, and asset matching) reside in `WowViewer.Core.PM4`.
- **God-Class Freeze (§10)**: No new members in `WorldScene.cs` or `ViewerApp.cs`. UI logic is hosted exclusively in `Pm4WorkbenchSidebarService.cs` and rendering additions in `Pm4OverlayMeshRenderer.cs`.
- **UI Standardization (§11)**: All controls use `SharedUiWidgets` and pass `Verify-UiReference.ps1`.
- **Do Not Touch Working Format Readers (§4)**: We consume existing `Pm4OverlayObject`, `MSUR`, and `MPRL` data models without modifying format parsers.

---

## 2. Phase Breakdown

```
Phase 1: Top-Alignment Fix in Doodad Magnetization & Duplication
   ├── Fix Pm4DoodadDuplicationService (single & tile magnetization)
   ├── Fix Pm4DoodadDuplicationEngine & AttractPositionToPm4Mesh
   └── Unit test verification in Pm4DoodadDuplicationTests
         │
Phase 2: Core 3D Footprint & Synthetic Wall Extrusion Engine
   ├── Pm4FootprintExtruder.cs in WowViewer.Core.PM4
   │     ├── 2D convex hull / perimeter polygon extraction
   │     ├── Z_base resolution (MPRL terrain contact vs Placement-Z)
   │     └── Wall quad generation (top edge -> base edge)
   └── Unit tests for synthetic wall extrusion & bounds calculation
         │
Phase 3: Scale-Invariant Geometric Signature & MPQ Matcher
   ├── Pm4GeometricShapeDescriptor.cs (W, D, H, AspectXY, AspectZH)
   ├── Pm4DirectModelMatcher.cs (zero-donor model scoring)
   └── Unit tests validating matching against sample M2/WMO bounds
         │
Phase 4: Viewer Workbench Integration & Overlay Visualization
   ├── Extruded wall rendering in Pm4OverlayMeshRenderer.cs
   ├── Display toggle & matching UI in Pm4WorkbenchSidebarService.cs
   └── UI Reference verification & documentation sync
```

---

## 3. File Changes

### Core Library (`WowViewer.Core.PM4`)
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4UnauthoredPlacementResolver.cs`: Fix `AttractPositionToPm4Mesh` for top-aligned doodad geometry.
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`: Pass top-offset when computing doodad placement positions.
- `wow-viewer/src/core/WowViewer.Core.PM4/Footprint/Pm4FootprintExtruder.cs` *(NEW)*: Pure 3D footprint extractor and wall generator.
- `wow-viewer/src/core/WowViewer.Core.PM4/Footprint/Pm4GeometricShapeDescriptor.cs` *(NEW)*: Computes aspect ratios and scale-invariant descriptors.
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DirectModelMatcher.cs` *(NEW)*: Matches PM4 3D footprints against candidate model signatures.

### Viewer UI & Rendering (`WoWViewer`)
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`: Compute top-relative offset in `MagnetizePlacementToPm4` and `MagnetizeTilePlacementsToPm4`.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4OverlayMeshRenderer.cs`: Render extruded 3D footprint walls.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs`: Add toggle for extruded footprint walls and direct matching UI.

### Tests
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4DoodadDuplicationTests.cs`: Update and add tests verifying top-alignment and synthetic wall extrusion.

