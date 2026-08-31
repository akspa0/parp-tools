# Spec 195: Overhead Chunk Manipulator & Multi-Tile Sub-Cell Transposition Engine

## 1. Executive Summary & Vision

The **Overhead Chunk Manipulator** unifies fragmented tile selection and chunk clipboard tools into a single, high-performance, precision terrain manipulation workstation in Parp Tools WoWViewer. 

Users can inspect their world map in an overhead 2D / top-down perspective (with minimap texture backdrop, tile borders at $533.334\text{m}$, and chunk lattices at $33.334\text{m}$), select arbitrary multi-tile rectangles, contiguous strips, or discrete chunk clusters, and transpose (cut / copy / move / paste / rotate / mirror) full-fidelity terrain data across tile boundaries with sub-cell chunk precision.

The manipulator transposes all terrain attributes simultaneously:
- **Geometry**: 145-vertex MCVT height arrays with relative or absolute floor anchoring and normal vector recalculation.
- **Textures & Splats**: MCLY layer tables and 64x64 MCAL alpha splats with intelligent target tile palette re-indexing and 4-layer chunk allocation enforcement.
- **Features & Flags**: HoleMask dev-mesh flags, sound/area IDs, and liquid instances (MCLQ / MCLH).
- **Embedded Placements**: M2 doodads (MDDF) and WMO objects (MODF) intersecting the selected chunk footprint, transformed by the exact 3D displacement vector $(\Delta X, \Delta Y, \Delta Z)$.
- **Undo / Redo Architecture**: Fully reversible operations integrated with `EditorSession`.

---

## 2. Mathematical Model: Global Chunk Space

World coordinates $(X, Y, Z)$ map to standard ADT tiles ($0 \le T_x, T_y < 64$) and sub-tile MCNK chunks ($0 \le C_x, C_y < 16$):

$$\text{MapOrigin} = 17066.666\text{m}, \quad \text{TileSize} = 533.33333\text{m}, \quad \text{ChunkSize} = 33.33333\text{m}$$

We define the unified **Global Chunk Coordinate** $(G_x, G_y) \in [0, 1023] \times [0, 1023]$:

$$G_x = T_x \times 16 + C_x, \quad G_y = T_y \times 16 + C_y$$

$$T_x = \lfloor G_x / 16 \rfloor, \quad C_x = G_x \pmod{16}$$
$$T_y = \lfloor G_y / 16 \rfloor, \quad C_y = G_y \pmod{16}$$

This continuous 2D integer lattice eliminates tile-boundary discontinuities. A selection bounding box from $(G_{x\text{min}}, G_{y\text{min}})$ to $(G_{x\text{max}}, G_{y\text{max}})$ seamlessly encompasses arbitrary sub-regions spanning $1, 2, 4,$ or $N$ tiles.

A transposition offset $(\Delta G_x, \Delta G_y)$ moves every chunk in the payload from $(G_x, G_y) \to (G_x + \Delta G_x, G_y + \Delta G_y)$, automatically re-bucketed into the destination tile $T'_x = \lfloor (G_x + \Delta G_x) / 16 \rfloor, T'_y = \lfloor (G_y + \Delta G_y) / 16 \rfloor$.

---

## 3. User Stories

### US1: Unified Selection Model & Bounding Region
As an editor user, I want a single unified selection model where I can select individual chunks, drag rectangular bounding boxes, or click whole tiles, so that I can manipulate arbitrary regions of the map regardless of tile borders.

### US2: Overhead 2D Minimap & Viewport Canvas Overlay
As a map designer, I want an interactive top-down / overhead view of the map with zoom and pan, showing tile borders ($16\text{ chunks}$) and chunk grid lines ($1\text{ chunk}$), so that I can see the exact terrain layout and drag-select regions visually.

### US3: Multi-Layer Chunk Transposition & Shift Engine
As a world builder, I want to cut/copy a selection of chunks and paste them at an arbitrary target chunk location, shifting heights, normals, texture alpha splats, holes, and doodad/WMO placements accurately without clipping or corrupted texture indices.

### US4: In-Viewport 3D Gizmo & Manipulation
As an editor user, I want a 3D wireframe bounding box in the perspective viewport showing my current chunk selection, with interactive drag controls, rotation ($90^\circ, 180^\circ, 270^\circ$), horizontal/vertical mirroring, and relative height adjustment.

### US5: Undo/Redo & Multi-Tile Persistence
As a map developer, I want all chunk manipulation operations to be fully undoable via `EditorSession`, and saved directly to the active terrain manager and project outputs with zero data loss.

---

## 4. Acceptance Criteria

1. **Precision Transposition**: Moving $N$ chunks by $(\Delta G_x, \Delta G_y)$ correctly updates all affected destination tiles ($T'_x, T'_y$) and recalculates boundary normals along adjoining unmoved chunks.
2. **Texture Layer Harmonization**: When pasting chunks into a target tile that already has 4 texture layers in its palette, the engine harmonizes the layer definitions and enforces the $\le 4$ layer hardware limit per MCNK chunk.
3. **Placements Transposition**: M2 doodad and WMO placements positioned within the source bounding box are cleanly extracted, shifted by $(\Delta X, \Delta Y, \Delta Z) = (-\Delta G_x \times 33.334, -\Delta G_y \times 33.334, \Delta Z)$, and inserted into the destination tile's placement tables.
4. **Interactive Overhead Canvas**: The overhead selection tool supports click-drag box selection, Shift+click multi-selection, Ctrl+click toggle, and whole-tile double-click selection.
5. **No Allocation Hitching**: In-viewport selection rendering and transposition previews run at $\ge 60\text{ FPS}$ with zero GC pressure during hover and drag.
