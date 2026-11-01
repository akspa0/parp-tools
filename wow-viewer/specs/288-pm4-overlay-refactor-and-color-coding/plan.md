# Implementation Plan: Spec 288 — PM4 Overlay Modernization, Color-Coding & Sidebar Refactor

## 1. Architectural Strategy & Boundaries

### 1.1 Decomposition of `Pm4OverlayScene`
In accordance with AGENTS.md §10 (God-Class Freeze), `Pm4OverlayScene` will delegate its major responsibilities to focused, testable service classes located in `WoWViewer/Terrain/Pm4/`:

```
wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/
├── Services/
│   ├── Pm4ColorCodingService.cs      # Palette management, elevation colormaps, instance hashing, semantic roles
│   ├── Pm4SpatialIndex.cs            # Tile & object AABB trees, distance culling, fast multi-stage ray picking
│   ├── Pm4OverlayMeshRenderer.cs     # OpenGL VAO/VBO buffer management, shader compilation, GPU uploads
│   └── Pm4SelectionCoordinator.cs    # Selection state, 3x3 neighborhood caching, telemetry aggregation
├── Pm4OverlayScene.cs                # Lightweight orchestrator implementing IPm4OverlayScene (~800 lines)
├── Pm4OverlayScene.Reports.cs        # Export/reporting helpers (<1,200 lines)
└── UI/
    └── Pm4WorkbenchSidebarService.cs # Modernized SharedUiWidgets workbench with interactive legend
```

### 1.2 Color-Coding Engine (`Pm4ColorCodingService`)
The service provides dedicated coloring strategies:
1. **Instance Clustered**: Hashes `(tileKey, objectPartId, centroid)` to produce a distinct hue via golden-ratio spacing. Prevents adjacent doodads from sharing colors.
2. **Semantic Category**:
   - `0x03` / Doodad collision: Soft warm coral `(0.95f, 0.45f, 0.25f)`
   - `0x10` / Lowest floor: Vibrant green `(0.20f, 0.85f, 0.40f)`
   - `0x11` / Common floor: Bright cyan `(0.15f, 0.75f, 0.90f)`
   - `0x13` / Upper floor: Deep sky blue `(0.25f, 0.50f, 0.95f)`
   - `0x14` / Angled/Roof: Violet `(0.70f, 0.35f, 0.90f)`
   - Walls (`MSPV/MSPI`): Amber `(0.95f, 0.75f, 0.15f)`
   - Connectors (`MSCN`): Turquoise `(0.10f, 0.95f, 0.85f)`
3. **Continuous Elevation Ramp**:
   - Computes world $Z$ relative to min/max elevations.
   - Maps $t \in [0, 1]$ across a high-contrast perceptual colormap (deep navy $\to$ cyan $\to$ chartreuse $\to$ orange $\to$ white).
4. **Curation / Match Status**:
   - Matched with ADT placement: Emerald `(0.20f, 0.85f, 0.30f)`
   - Unmatched PM4 hull: Coral `(0.95f, 0.40f, 0.30f)`
   - Live duplicated: Gold `(1.00f, 0.84f, 0.00f)`
   - Ignored/Terrain contact: Muted steel `(0.55f, 0.60f, 0.65f)`

### 1.3 Fast Spatial Indexing & Picking (`Pm4SpatialIndex`)
- Maintains per-tile object bounding boxes `(tileMin, tileMax)`.
- Ray picking executes in three strict stages:
  1. Ray-AABB intersection with tile bounding box. Skip tile entirely if ray misses.
  2. Ray-AABB intersection with object bounding box. Skip object if ray misses.
  3. Triangle-level Möller–Trumbore test only on surviving candidates.
- Returns hits in $\le 0.5$ ms without allocating heap memory or scanning non-visible tiles.

---

## 2. Phased Implementation Roadmap

### Phase 1: Color-Coding Service & Palette Engine
- Create `Pm4ColorCodingService` with:
  - Instance clustering hash.
  - Semantic category classification.
  - Perceptual elevation colormap.
  - Match status evaluation.
- Wire into `Pm4OverlayScene.GetPm4ObjectColor()`.
- Add unit tests validating palette separation and gradient monotonicity.

### Phase 2: Spatial Index & Fast Ray-Cast Decomposition
- Create `Pm4SpatialIndex`:
  - Tile AABB and object AABB indexing.
  - Hierarchical ray-cast picking with early-exit bounds tests.
  - 3x3 tile neighborhood queries.
- Refactor `Pm4OverlayScene.Selection.cs` to delegate to `Pm4SpatialIndex`.
- Validate zero regressions in ray-picking precision.

### Phase 3: Mesh Renderer & Buffer Extraction
- Extract OpenGL buffer allocation and shader rendering from `Pm4OverlayScene.cs` into `Pm4OverlayMeshRenderer`.
- Implement per-tile buffer caching to avoid CPU vertex copies on static frames.
- Ensure all source files remain below 2,000 lines.

### Phase 4: Modernized Sidebar Workbench (`Pm4WorkbenchSidebarService`)
- Refactor PM4 sidebar into `Pm4WorkbenchSidebarService` using `SharedUiWidgets`.
- Implement:
  - Master overlay toggles (Show, Wireframe, Fill, X-Ray).
  - Palette selector dropdown and dynamic color legend with interactive swatches.
  - Multi-select surface class filter.
  - Curated duplication actions & live status telemetry.
- Register controls in `UI-REFERENCE.md` and pass `Verify-UiReference.ps1`.
