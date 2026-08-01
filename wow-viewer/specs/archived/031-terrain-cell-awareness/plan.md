# Implementation Plan: Terrain Cell Awareness and Sub-Chunk Addressing
**Branch**: `031-terrain-cell-awareness` | **Date**: 2026-05-30 | **Spec**: [spec.md](./spec.md)

**Convergence Note**: Active renderer-owner planning now lives in [specs/036-renderer-improvements/plan.md](../036-renderer-improvements/plan.md). This document remains a source-slice reference for terrain topology and cell-awareness details.

## Summary

Implement the terrain cell (sub-chunk) awareness system decompiled from wowclient.exe build 3368 in `WowViewer.Core.IO` and `WowViewer.Core.Runtime`. The native client uses a 145-vertex layout (9x9 outer + 8x8 inner), 8x8 cell grid with per-cell diagonal splits (256 face planes), 16-bit hole mask with 4x4 grouping, and 13-bit packed cell addressing. This plan covers porting the data structures from MdxViewer's `StandardTerrainAdapter` / `AlphaTerrainAdapter` into the wow-viewer libraries.

## Technical Context

**Language/Version**: C# / .NET 10
**Primary Dependencies**: WowViewer.Core.IO (ADT/MCNK readers), WowViewer.Core.Runtime (mesh builder, spatial queries)
**Storage**: No persistent storage changes (data structures only)
**Testing**: `dotnet test wow-viewer/WowViewer.slnx -c Debug`; validate vertex count, face planes, hole mask, cell addressing
**Target Platform**: Windows x64
**Performance Goals**: Cell address lookup < 1us; mesh build < 1ms per chunk
**Constraints**: No code in `gillijimproject_refactor/`; terrain data stays in wow-viewer; one phase at a time

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All code in `wow-viewer/` |
| II. Library-First | PASS | Logic in Core.IO and Core.Runtime |
| III. Real-Data Validation | PASS | Validates against staged clients |
| IV. Residual Model Chain | N/A | No ML work |
| V. Streaming-First | N/A | Not a dataset pipeline |
| VI. No Game Client Assumptions | PASS | Uses staged clients only |
| Read-Only Reference | PASS | MdxViewer adapters are reference only |
| One Phase at a Time | PASS | Phases ordered by dependency |

## Project Structure

### Documentation (this feature)
```
specs/031-terrain-cell-awareness/
├── spec.md          # Feature specification (194 lines)
├── plan.md          # This file
└── tasks.md         # Task breakdown
```

### Source Code
```
wow-viewer/src/core/WowViewer.Core.IO/
├── Maps/
│   ├── AdtChunkReader.cs        # EXISTING — extend for 145-vertex layout
│   ├── AdtTerrainData.cs         # EXISTING — extend for cell grid
│   ├── AdtCellGrid.cs             # NEW — 8x8 cell grid, diagonal splits
│   ├── AdtFacePlaneBuilder.cs    # NEW — 256 face planes per chunk
│   ├── AdtHoleMask.cs            # NEW — 16-bit holes, 4x4 grouping
│   └── AdtCellAddress.cs          # NEW — 13-bit packed (sub-chunk 3 + chunk 4 + ADT 6)
└── Terrain/
    └── TerrainCellSnapshot.cs     # NEW — immutable cell data for runtime

wow-viewer/src/core/WowViewer.Core.Runtime/
├── World/
│   ├── Terrain/
│   │   ├── WorldTerrainChunkData.cs    # EXISTING — extend for 145 vertices + cells
│   │   ├── WorldTerrainHeightmapData.cs # EXISTING — extend for inner vertices
│   │   ├── WorldTerrainCellGrid.cs      # NEW — cell grid + diagonal split
│   │   ├── WorldTerrainFacePlanes.cs  # NEW — 256 face planes
│   │   ├── WorldTerrainHoleMask.cs    # NEW — hole mask, 2x2 block granularity
│   │   └── WorldTerrainLodBuilder.cs   # EXISTING — extend for cell-aware LOD
│   └── Spatial/
│       ├── WorldCellAddress.cs        # NEW — (ADT, chunk, cell) ↔ world-space
│       └── WorldCellRaycast.cs       # NEW — ray vs cell face planes
└── Rendering/
    └── TerrainRenderPipeline.cs    # EXISTING — extend for 145-vertex topology
```

## Implementation Phases

### Phase 1 — Terrain Data Structures in Core.IO (P1, US1)
**Goal**: Parse 145-vertex layout (9x9 outer + 8x8 inner from MCVT), build 8x8 cell grid with diagonal splits, compute 256 face planes, parse 16-bit hole mask.

**Dependencies**: None. This phase is purely I/O — no rendering required.

**Approach**:
1. Extend `AdtChunkReader` to expose 145 vertices (81 outer + 64 inner in interleaved order)
2. Build `AdtCellGrid` — 8x8 cells, per-cell diagonal split variant (from Ghidra lookup tables `0x008a10b8`/`0x008a10bc`)
3. Build `AdtFacePlaneBuilder` — 256 face planes (8x8 cells × 4 triangles per cell)
4. Build `AdtHoleMask` — 16-bit holes field, test per 2x2 cell block: `(g_holeMask[(subX>>1) + (subY>>1)*4] & chunk->holes) == 0`
5. Validate: Load MCNK from staged client, verify 145 vertices, 256 face planes, correct hole mask

**Steps** (max 10):
1. Extend `AdtChunkReader` to read 145 vertices from MCVT (entries 0-80 outer, 81-144 inner)
2. Build `AdtCellGrid` with 64 cells, assign diagonal split variant per cell (from height gradient or fixed table)
3. Build `AdtFacePlaneBuilder` — compute 4 face planes per cell from triangle vertices
4. Build `AdtHoleMask` — 16-bit field, 4x4 grouping, per-cell test method
5. Build `AdtCellAddress` — 13-bit packed: sub-chunk (3 bits) + chunk (4 bits) + ADT (6 bits)
6. Extend `AdtTerrainData` to carry cell grid, face planes, hole mask
7. Add unit tests: vertex count (145), face plane count (256), hole mask test
8. Validate against staged client `3_3_5_12340` — load MCNK, verify data
9. Port normal unpacking: Y/Z/X order, scale ~0.251388
10. Validate normal count (145), correct byte order

---

### Phase 2 — Runtime Cell Awareness (P1, US1 + US2)
**Goal**: `WorldTerrainChunkData` carries cell grid + 145-vertex mesh. Spatial queries work: world → (ADT, chunk, cell, is_holed).

**Dependencies**: Phase 1 (Core.IO data structures must exist).

**Approach**:
1. Extend `WorldTerrainChunkData` for 145-vertex layout + cell grid
2. Build `WorldTerrainCellGrid` — runtime cell structure with diagonal splits
3. Build `WorldTerrainFacePlanes` — 256 planes for collision/raycast
4. Build `WorldTerrainHoleMask` — hole mask with 2x2 block granularity
5. Build `WorldCellAddress` — spatial queries: world (X,Y) → (ADT, chunk, cell, is_holed)
6. Build reverse query: (ADT, chunk, cell) → world-space AABB
7. Validate: spatial query for known position returns correct indices

**Steps** (max 10):
1. Extend `WorldTerrainChunkData` for 145 vertices + cell grid + face planes
2. Build `WorldTerrainCellGrid` — 64 cells with diagonal split, 4 triangles each
3. Build `WorldTerrainFacePlanes` — store 256 planes, per-cell access
4. Build `WorldTerrainHoleMask` — hole test per cell, toggleable overlay mode
5. Build `WorldCellAddress` — world→cell query, cell→world AABB reverse
6. Wire into `WorldTerrainLodBuilder` — LOD respects cell boundaries
7. Add unit tests: spatial query (known position → indices), reverse query (indices → AABB)
8. Validate against native client: query Deadmines entrance, verify ADT/chunk/cell
9. Validate hole mask: holed cell returns `is_holed=true`
10. Validate cell AABB: reverse query produces correct world-space box

---

### Phase 3 — Mesh Builder with 145-Vertex Topology (P2, US3)
**Goal**: `WorldTerrainTileBuilder` generates triangles using 145-vertex layout with per-cell diagonal splits.

**Dependencies**: Phase 2 (runtime cell awareness must exist).

**Approach**:
1. Extend `WorldTerrainTileBuilder` for 145-vertex mesh
2. Per-cell: 4 triangles using diagonal split (use `AdtCellGrid` split variant)
3. Handle hole mask: native mode (exclude holed cells) vs reconstructive mode (render all + overlay)
4. Validate: render chunk, verify mid-cell inner vertices visible, correct diagonal splits

**Steps** (max 8):
1. Extend `WorldTerrainTileBuilder` — 145-vertex index buffer (256 triangles)
2. Implement per-cell diagonal split using `WorldTerrainCellGrid`
3. Implement hole mask handling: native mode (skip holed cells) vs reconstructive (overlay)
4. Add unit tests: triangle count (256), diagonal split correctness
5. Validate mesh: render chunk, compare against 9x9 flat grid — inner vertices add detail
6. Validate hole mask: holed cells excluded or overlaid correctly
7. Validate against native client screenshot: terrain topology matches
8. Performance test: mesh build < 1ms per chunk

---

### Phase 4 — Cell-Level Raycast/Collision (P2, US2)
**Goal**: Ray vs terrain collision uses cell face planes for efficient hit testing.

**Dependencies**: Phase 2 (face planes must exist).

**Approach**:
1. Build `WorldCellRaycast` — ray vs 256 face planes, early out on first hit
2. Use cell address to narrow search: only test cells along ray path
3. Validate: cast ray at known terrain position, verify hit cell is correct

**Steps** (max 6):
1. Build `WorldCellRaycast` — ray vs face planes, return hit cell + distance
2. Optimize: use cell address to test only relevant cells
3. Add unit tests: ray hits correct cell, miss returns null
4. Validate against native client: raycast at known position, verify cell index
5. Performance test: raycast < 10us per ray
6. Wire into world scene collision system

---

## Complexity Tracking
No constitution violations. All phases are data structure + rendering logic in wow-viewer.
