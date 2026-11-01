# Feature Specification: Spec 288 — PM4 Overlay Modernization, Color-Coding & Sidebar Refactor

**Owner**: Epic 251 (Viewer UX, Shell & Code Health) & Epic 253 (PM4/PD4 Navmesh Research)  
**Origin**: Operator prompt (2026-10-31) regarding:
1. Fixing PM4 color-coding: `MSUR._0x1C` is the producing placement's Z height, but currently hashing it via `ColorFromSeed` collapses all doodads (`MSUR._0x1C == 0`) into an identical bright red hue (`ColorFromSeed(0) = HSV(0, 0.75, 0.95)`), making adjacent doodads completely indistinguishable.
2. Modernizing the PM4 sidebar UI: Replace fragmented and dense accordion trees with a clean, standardized `SharedUiWidgets` workbench panel featuring an interactive color legend, palette switcher, and structured inspection.
3. Architecture rewrite & decomposition: `Pm4OverlayScene` dates back to early viewer iterations (Spec 056) and has grown past 4,300 lines across 3 files with per-frame CPU dynamic buffer rebuilds and unindexed linear scans. Refactor it into modular service classes (rendering, spatial indexing, color coding, selection) conforming to AGENTS.md §10.

**Status**: Draft / Specified (2026-10-31)

---

## 1. Executive Summary & Problem Analysis

### 1.1 The "Sea of Red" Color-Coding Defect
In PM4 navmesh collision files:
- `MSUR._0x1C` is a 32-bit IEEE 754 float holding the producing placement's world Z coordinate (or 0 for doodad collision surfaces).
- The viewer previously hashed `obj.Ck24` (`MSUR._0x1C >> 8`) into an RGB color via `ColorFromSeed(uint seed)`:
  ```csharp
  uint golden = seed * 2654435761u;
  float hue = (golden & 0x00FFFFFF) / 16777215f;
  return HsvToRgb(hue, 0.75f, 0.95f);
  ```
- Because all doodad collision surfaces carry `MSUR._0x1C = 0.0f`, `seed = 0` produces `hue = 0.0f` (bright red: RGB `0.95, 0.24, 0.24`).
- On tiles such as `development.wdt` `35_38`, dozens of adjacent rock/cliff doodads (`DRAGONBLIGHT_CLIFF_01.M2`) rendered as an indivisible red blob. The operator could not distinguish where one doodad ended and the next began.
- Furthermore, for WMO placements, hashing the float bits gave arbitrary pseudo-random rainbow colors that bore no relation to actual altitude or spatial grouping.

### 1.2 Inefficient Legacy Architecture (Spec 056 Debt)
- `Pm4OverlayScene.cs` was originally written during the early GPU LOD experiments (Spec 056) and accumulated features over >230 specs.
- It currently spans 4,298 lines:
  - `Pm4OverlayScene.cs` (2,077 lines)
  - `Pm4OverlayScene.Reports.cs` (1,834 lines)
  - `Pm4OverlayScene.Selection.cs` (387 lines)
- Violates AGENTS.md §10 (God-Class Freeze: files must remain $\le 2,000$ lines, cohesive features extracted into owned service classes).
- Performance bottlenecks:
  - Rebuilding dynamic vertex buffers and re-uploading thousands of triangles per frame.
  - Linear iteration over all 616 tiles in memory for picking and queries instead of a spatial grid or bounding quadtree.
  - Mixing UI ImGui layout, OpenGL buffer management, ray picking, and format decoding in a single partial class.

### 1.3 Outdated Sidebar UI
- The PM4 workbench sidebar contained nested collapsible sections with inconsistent header styling.
- Controls lacked a dedicated visual palette legend explaining what colors currently mean.
- The user had to hunt across multiple panels to toggle surface classes, inspect selected components, or duplicate models.

---

## 2. User Stories & Acceptance Criteria

### User Story 1: Intuitive PM4 Color-Coding & Palettes (Priority: P1)
As an archaeologist or level designer inspecting PM4 navmeshes, I want meaningful, configurable color-coding modes so that adjacent doodads have distinct colors and objects clearly convey their semantic role and placement height.

- **AC-101**: The viewer provides distinct color-coding modes:
  1. **Instance Identity (Clustered)**: Each distinct object/doodad instance receives a unique distinguishable color based on its spatial centroid / instance hash, ensuring adjacent doodads never share the same color.
  2. **Semantic Surface Category**: Colors differentiate WMO Floor, WMO Wall/Sloped, Doodad Collision Hull, Terrain Linker (`MPRL`), and Connector Anchor (`MSCN`).
  3. **Elevation Height Gradient**: Continuous perceptual elevation ramp (e.g. Turbo/Viridis from minimum to maximum tile Z) showing real altitude rather than hashed float bits.
  4. **Curation & Match Status**: Highlights matched ADT placements (green/cyan), unmatched PM4 hulls (coral/amber), and newly duplicated live objects (gold).
- **AC-102**: Selecting any color mode immediately updates both the 3D viewport rendering and the sidebar's interactive visual legend.
- **AC-103**: Doodad collision hulls (`Ck24 == 0`) are never rendered as a monolithic red blob when in Instance or Category mode.

### User Story 2: PM4 Overlay Source Decomposition (Priority: P1)
As a maintainer of the viewer codebase, I want `Pm4OverlayScene` decomposed into focused, decoupled service classes conforming to AGENTS.md §10 so that no file exceeds 2,000 lines and rendering/indexing/picking concerns are clean.

- **AC-201**: `Pm4OverlayScene` is decomposed into:
  - `Pm4OverlayMeshRenderer`: Owns OpenGL shaders, VAOs, VBOs, and batched/instanced drawing.
  - `Pm4SpatialIndex`: Owns tile-based spatial hashing, bounding box hierarchies, distance culling, and ray-cast picking.
  - `Pm4ColorCodingService`: Owns color palette generation, elevation ramps, instance hashing, and semantic classification.
  - `Pm4SelectionCoordinator`: Owns active selection, hover tracking, multi-object grouping, and cache invalidation.
- **AC-202**: Every decomposed file is $\le 2,000$ lines.
- **AC-203**: Zero regressions in existing PM4 features (wireframe, solid fill, X-ray, ray-picking, live doodad duplication, Zarr export).

### User Story 3: Modernized PM4 Sidebar Workbench (Priority: P2)
As a user working with PM4 collision data, I want a standardized, cohesive sidebar workbench utilizing `SharedUiWidgets` with an interactive color legend and streamlined controls.

- **AC-301**: The sidebar header displays active tile and total loaded object/surface counts with one-click master toggles (Show Overlay, Wireframe, Solid Fill, X-Ray).
- **AC-302**: An interactive **Color & Legend** widget allows switching color modes and shows active color swatches with labels and object counts.
- **AC-303**: A dedicated **Surface Filters** section allows toggling individual surface classes (Doodad 0x03, Lowest floors 0x10, Upper floors 0x13, Walls MSPV/MSPI).
- **AC-304**: An **Inspector & Live Actions** section displays selected object telemetry (dimensions, centroid, surface count) with live duplicate and undo buttons.
- **AC-305**: All controls are registered in `UI-REFERENCE.md` and pass `Verify-UiReference.ps1`.

---

## 3. Technical Constraints & Non-Goals

- **Preserve Core IO Readers**: Do not modify raw PM4 chunk decoders in `WowViewer.Core.PM4.IO`. All refactoring is confined to viewer presentation and service layers.
- **Performance Budget**: Ray picking must execute in $\le 1$ ms; frame rendering with 100k PM4 surfaces must achieve $\ge 60$ FPS on modern GPUs without UI-thread hitches.
- **Backwards Compatibility**: All existing PM4 CLI tools and dataset exporters continue functioning without modification.
