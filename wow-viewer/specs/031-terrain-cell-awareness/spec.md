# Feature Specification: Terrain Cell Awareness and Sub-Chunk Addressing

**Feature Branch**: `031-terrain-cell-awareness`

**Created**: 2026-05-30

**Status**: Research complete — consumed by spec 056

**Input**: Ghidra RE of wowclient.exe build 3368 reveals the complete terrain chunk vertex layout, sub-chunk (cell) grid, face plane system, hole mask, and collision addressing scheme. The current tooling treats terrain chunks as flat 9x9 vertex grids without inner vertices, has no cell-level structure, applies hole masks as a viewer overlay rather than at the cell granularity the native client uses, and uses a CPU-bound renderer that processes all chunks at full detail. Understanding and implementing the native cell system is key to efficient terrain rendering and correct spatial queries.

## Problem Statement

The current terrain tooling has several fundamental gaps relative to the native client:

1. **No sub-chunk (cell) structure**: The native client divides each MCNK into an 8x8 grid of cells, each with 4 triangles (256 face planes per chunk). Our tooling treats the whole chunk as a flat heightfield.

2. **Missing inner vertices**: The native vertex layout is 9x9 outer + 8x8 inner = 145 vertices per chunk. Inner vertices sit at the center of each cell, enabling diagonal splits and providing more detail at mid-cell positions. Our tooling only handles the 9x9 outer grid (81 vertices).

3. **Hole mask applied at wrong granularity**: The native client uses a 16-bit `holes` field with a 4x4 grouping pattern (`holeMask[(subX>>1) + (subY>>1)*4]`), where each bit covers a 2x2 block of cells. Our tooling already renders holes as a toggleable overlay mask (for reconstructive viewing — we show holed terrain rather than cutting it away, because some terrain data exists under hole masks and research viewing needs to see it), but the overlay is not cell-aware and may not match the native client's exact 2x2-cell-block grouping.

4. **No distance-based LOD**: The native client sorts chunks into 26 distance buckets, reduces texture layers at distance, and uses a low-detail 17x17 vertex area for far terrain. Our CPU renderer processes everything at full detail.

5. **No cell-level addressing**: The native collision system uses packed 13-bit coordinates (sub-chunk 3 bits, chunk 4 bits, ADT 6 bits) to efficiently address any cell in the world. Our tooling has no equivalent.

These gaps mean our terrain rendering is both incorrect (missing inner vertices, wrong triangle topology, hole overlay not cell-granular) and inefficient (no LOD, no cell-level culling).

## Ghidra RE Findings (Build 3368)

### Vertex Layout (`CMapChunk::CreateVertices` at 0x006997e0)

- **9x9 outer grid** (81 vertices): evenly spaced at `chunkSize/8` intervals
- **8x8 inner vertices** (64 vertices): offset by `chunkSize/16` from each outer grid intersection, sitting at the center of each cell
- Total: 81 + 64 = **145 vertices** (0x91 hex)
- Heights from MCVT: first 81 entries are the outer grid (row-major, 9 per row), then 64 entries are the inner grid (row-major, 8 per row)
- World position: computed from `cOffset` (chunk tile indices), then subtracted by the chunk's world origin for camera-relative rendering

### Normal Layout (`CMapChunk::CreateNormals` at 0x00699b60)

- Same 9x9 + 8x8 layout as vertices (145 normals)
- Normals are packed as 3 signed bytes per vertex, unpacked with factor `0.251388` (hex `3c010204` / `bc010204`)
- Normal order: (Y, Z, X) in the packed format (note: not X,Y,Z)

### Face Plane System (`CMapChunk::CreateFacePlanes` at 0x00699c50)

- **256 planes** per chunk (8x8 cells × 4 triangles per cell)
- Each cell has 2 quads split diagonally using the inner vertex
- Two triangulation variants per cell, selected from lookup tables at `0x008a10b8` and `0x008a10bc`
- Plane equation: computed from cross product of two triangle edge vectors, normalized, with signed distance

### Hole Mask (`CMapChunk::Create` at 0x00698e10)

- `this->holes = *(ushort *)(param_1 + 0x48)` — 16-bit field from MCNK header
- Test: `(g_holeMask[(subX>>1) + (subY>>1)*4] & chunk->holes) == 0` for "not holed"
- `g_holeMask` is a 16-entry lookup: each entry corresponds to one of the 16 2x2 cell blocks
- Bits map to 4x4 groups of 2x2 cell blocks within the 8x8 cell grid

### Cell Addressing (Collision System)

- **World → sub-chunk**: pack coordinates as uint16 pairs in `scCollideList`
- Format within chunk: `subX = coord & 7`, `subY = coord & 7`
- Chunk within ADT: `chunkIdx = coord >> 3 & 0xF` (4 bits)
- ADT within continent: `areaIdx = coord >> 7 & 0x3F` (6 bits)
- Ray-cast: `GetFacetTerrain` converts world segment to cell coordinates, builds `scCollideList`, then `GetFacetSubchunks` iterates and tests each cell's 4 face planes

### Rendering Pipeline

- **Distance sort**: chunks placed in 26 distance buckets (`sortTable.table[0..25]`)
- **Texture LOD**: `nLayersTest` drops to 1 when `dist - textureLodDist >= 256.0`; alpha fade below that
- **Two render paths**: `RenderLayers` (static GxVBO) vs `RenderLayersDyn` (dynamic buffer for animated terrain)
- **Per-layer setup**: texgen matrices for detail texture (Tex0) and alpha mask (Tex1), then `GxBufRender` the full 145-vertex triangle list
- **Shadow pass**: separate `shadowGxTexture` blended on top when `CWorld::enables & 0x40`
- **Low-detail area**: `CMapAreaLow` uses 17x17 grid with fog-colored vertices for far terrain (no textures)
- **Neighbor linking**: `CreateChunkNeighborPtrs` builds a 4-neighbor ring (up, right, down, left) for LOD seam stitching

## Scope

### In Scope

- Adding sub-chunk (cell) awareness to `WowViewer.Core.IO` terrain readers — parsing the 145-vertex layout, 8x8 cell grid, and diagonal split
- Implementing hole mask support — reading and applying the 16-bit `holes` field with the 4x4 grouping pattern
- Documenting the cell addressing scheme in `wow-viewer/docs/architecture/`
- Adding the 145-vertex triangle topology (outer + inner vertices, per-cell diagonal split) to the terrain mesh builder
- Exposing cell-level spatial queries (world coords → cell → chunk → ADT)

### Out of Scope

- GPU-level LOD implementation (this requires the Vulkan/WebGL backend, separate from data structure work)
- MdxViewer terrain rendering fixes (those go in `gillijimproject_refactor` which is READ-ONLY)
- MCAL alpha mask rendering fixes (separate spec 014)
- Minimap BLP harvesting (separate spec 029)
- WMO rendering (separate spec 030)

## User Scenarios & Testing

### User Story 1 — Terrain cell structure is documented and parsed (Priority: P1)

An engine developer can read the terrain cell architecture doc and use the `WowViewer.Core.IO` terrain readers to access the full 145-vertex layout, 8x8 cell grid with diagonal splits, and hole masks for any MCNK chunk, without needing to run Ghidra.

**Why this priority**: Documentation and data structure are the foundation. Without the correct vertex topology and cell grid, all downstream rendering and queries are wrong.

**Independent Test**: Load an MCNK from a staged client, read 145 vertices and 256 face planes, verify they match the native client's structure by checking vertex count and plane count.

**Acceptance Scenarios**:

1. **Given** an MCNK chunk from a staged client, **When** the terrain reader parses it, **Then** it exposes 145 vertices (9x9 outer + 8x8 inner) with correct world positions.
2. **Given** the 145 vertices, **When** face planes are computed, **Then** 256 planes exist (8x8 cells × 4 triangles).
3. **Given** an MCNK with non-zero `holes` field, **When** the terrain reader processes it, **Then** the holed cells are identified and flagged at the correct 2x2-cell-block granularity (4x4 grouping, each bit covers a 2x2 block), matching the native client's `holeMask` pattern.
4. **Given** the architecture doc, **When** a developer reads it, **Then** they understand the cell addressing scheme (13-bit packed coordinates: sub-chunk 3 bits, chunk 4 bits, ADT 6 bits).

---

### User Story 2 — Cell-level spatial queries work (Priority: P2)

A tool can query which cell a world-space point falls in, which chunk owns that cell, and which ADT tile contains that chunk, using the same packed-coordinate addressing the native client uses.

**Why this priority**: Spatial queries are essential for the minimap sieve, data harvester, and any future collision/raycast system. P2 because the data structures (P1) must exist first.

**Independent Test**: For a known world position (e.g., the Deadmines entrance), verify the query returns the correct ADT index, chunk index, and cell index.

**Acceptance Scenarios**:

1. **Given** a world-space (X, Y, Z) position, **When** the cell query runs, **Then** it returns the ADT tile index, chunk index within the ADT, and cell index within the chunk.
2. **Given** a cell coordinate, **When** the query reverses it, **Then** it produces the correct world-space bounding box for that cell.
3. **Given** a holed cell, **When** a spatial query tests it, **Then** the cell is reported as holed (empty).

---

### User Story 3 — Terrain mesh uses correct 145-vertex topology (Priority: P2)

The terrain mesh builder generates triangles using the full 145-vertex layout with per-cell diagonal splits, matching the native client's topology, instead of the current 9x9 flat grid.

**Why this priority**: Correct mesh topology is needed for accurate rendering and for the terrain signal to match ground truth. P2 because it depends on the data structures from P1.

**Independent Test**: Render a chunk with the new topology and compare against the current 9x9 rendering. Verify inner vertices produce visible mid-cell detail and correct diagonal splits.

**Acceptance Scenarios**:

1. **Given** an MCNK chunk, **When** the mesh builder generates triangles, **Then** each of the 64 cells produces 4 triangles using the correct diagonal split variant.
2. **Given** an MCNK with holes, **When** the mesh builder generates triangles, **Then** holed cells can be either excluded (native-accurate mode) or rendered with a hole overlay mask (reconstructive viewing mode).
3. **Given** the 145-vertex mesh, **When** rendered, **Then** the mid-cell inner vertices are visible as additional height detail compared to the flat 9x9 grid.

---

### Edge Cases

- Some chunks may have all 16 hole bits set (fully holed) — these should produce zero triangles.
- The inner vertex heights in MCVT are at offsets 81..144 — if the MCVT chunk is truncated (fewer than 145 floats), the inner vertices should default to interpolated values from the outer grid.
- Alpha-era terrain (pre-1.12) may have different MCNK layouts — the cell structure should be gated on the build version.
- The diagonal split variant (which of the two triangulation patterns) can vary per-cell in the native client — the lookup tables at `0x008a10b8`/`0x008a10bc` encode this. Our tooling needs to determine which variant to use or support both.

## Requirements

### Functional Requirements

- **FR-001**: The terrain cell architecture MUST be documented in `wow-viewer/docs/architecture/terrain-cell-architecture-2026-05-30.md` covering vertex layout, cell grid, face planes, hole masks, cell addressing, and rendering pipeline.
- **FR-002**: The `WowViewer.Core.IO` terrain reader MUST parse the full 145-vertex layout (9x9 outer + 8x8 inner) from MCVT data.
- **FR-003**: The terrain reader MUST expose the 8x8 cell grid with per-cell diagonal split information and 4 triangles per cell.
- **FR-004**: The terrain reader MUST parse and expose the 16-bit `holes` field from MCNK, with the correct 4x4 grouping pattern (each bit covers a 2x2 cell block, matching `holeMask[(subX>>1) + (subY>>1)*4]`). The existing toggleable hole overlay MUST be upgraded to use this cell-granular mapping.
- **FR-005**: The terrain reader MUST provide a cell-level spatial query: given world (X, Y), return (ADT index, chunk index, cell index, is_holed).
- **FR-006**: The terrain reader MUST provide a reverse query: given (ADT, chunk, cell), return the world-space AABB for that cell.
- **FR-007**: The terrain mesh builder MUST generate triangles using the 145-vertex topology with per-cell diagonal splits. Hole mask handling MUST support two modes: native-accurate (exclude holed cells) and reconstructive (render all cells with toggleable overlay, preserving terrain data under hole masks for research viewing).
- **FR-008**: All code MUST live under `wow-viewer/`.
- **FR-009**: The architecture doc MUST include the Ghidra function addresses as references for future RE work.
- **FR-010**: The normal unpacking MUST use the correct byte order (Y, Z, X) and scaling factor (~0.251388) as confirmed by Ghidra.

### Key Entities

- **Terrain Cell (Sub-chunk)**: An 8x8 grid subdivision within a single MCNK chunk. Each cell has 4 triangles (2 quads split diagonally), 4 face planes, and a corner vertex from the 9x9 outer grid plus a center vertex from the 8x8 inner grid.
- **Cell Address**: A packed coordinate that identifies a cell within the world: sub-chunk (3 bits within chunk) + chunk (4 bits within ADT) + ADT (6 bits within continent) = 13 bits per axis.
- **Hole Mask**: A 16-bit value where each bit covers a 2x2 block of cells. Bit N corresponds to `g_holeMask[N]` where N = `(subX>>1) + (subY>>1)*4`. Our tooling applies holes as a toggleable overlay rather than cutting terrain away, because reconstructive viewing requires seeing terrain data that exists under hole masks.
- **145-Vertex Layout**: 81 outer vertices (9x9 grid at chunk corners/edges) + 64 inner vertices (8x8 grid at cell centers). Heights from MCVT: entries 0-80 are outer, entries 81-144 are inner.

## Success Criteria

- **SC-001**: The architecture doc covers vertex layout, cell grid, face planes, hole masks, cell addressing, and rendering pipeline with Ghidra addresses.
- **SC-002**: Loading an MCNK from a staged client produces 145 vertices and 256 face planes.
- **SC-003**: A spatial query for a known world position returns the correct ADT/chunk/cell indices.
- **SC-004**: A chunk with holes correctly identifies which cells are holed at the 2x2-cell-block granularity. The hole overlay mask can be toggled on/off, and in native-accurate mode, exactly the holed cells × 4 triangles are excluded.
- **SC-005**: The 145-vertex mesh produces visible mid-cell detail compared to the flat 9x9 grid when rendered.

## Assumptions

- The 145-vertex layout and 8x8 cell grid are consistent across all target builds (0.5.3 through 4.0.0). The core structure hasn't changed — only the LOD and rendering paths differ.
- The diagonal split variant can be determined from the vertex positions (the native client uses fixed lookup tables, but we may need to infer the split from heightfield gradient or use a fixed default).
- The existing `WowViewer.Core.IO` ADT reader already parses MCNK headers and MCVT data — this spec adds the inner vertex extraction and cell grid construction on top of existing parsing.
- The `g_holeMask` lookup table is constant across all builds (16 entries, each mapping a hole bit index to a cell block mask).

## Relationship to Other Specs

- **Complements**: `030-wmo-render-pass-architecture` — terrain and WMO are the two primary world rendering subsystems.
- **Informs**: `029-wmo-minimap-signal` — cell-level addressing enables correlating WMO footprint positions with terrain minimap tiles.
- **Informs**: `025-object-roof-mask-library-and-minimap-sieve` — cell awareness improves the minimap sieve's spatial precision.
- **Informs**: `020-renderer-culling-and-tile-capture` — cell-level culling is a prerequisite for efficient terrain capture.
- **Extends**: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (viewer-first + UE bridge)
