# Tile Connectivity & Link Welding Detailed Plan

**Author:** Cascade AI
**Last Updated:** 2025-07-09 22:45

---

## Goals
1. Produce a stitched, watertight terrain OBJ directly from PM4 tiles.
2. Weld vertices across tile boundaries using real inter-tile link data (MSLK, MPRR, MPRL).
3. Keep exporter simple and single-command (`terrain-stitch`).
4. Maintain 100 % reliance on chunk data – no synthetic geometry.

---

## Phase A – MSLK Discovery & Diagnostics

| Step | Task | Output |
|------|------|--------|
| A-1 | Parse MSLK chunk structure into POCO (`MslkEntry`) | New model class |
| A-2 | Implement `MslkInterTileAnalyzer` service | `TileLinkGraph` object |
| A-3 | CLI command `mslk-inspect <pm4-root>` | CSV/JSON of links |
| A-4 | Unit test with real dataset | Passing tests |

### Expected Fields (from empirical investigation)
* `LinkIdHex` – unique 32-bit hash
* `TileX / TileY` – origin tile coordinates
* `EdgeIndex` – which edge of the tile (0-3 clockwise)
* `NeighbourTileOffset` – Δx, Δy

---

## Phase B – Link-Aware Export

| Step | Task | Implementation |
|------|------|----------------|
| B-1 | Extend `TerrainMeshExporter` with `--with-links` flag | Option parsing |
| B-2 | During vertex collection, snap vertices shared by linked edges | Epsilon-based merge |
| B-3 | Where a tile lacks MSVI, but neighbour provides indices, replicate faces | Face cloning |
| B-4 | Integration test – export with/without links & compare | MeshLab visual + vertex / face counts |

---

## Phase C – MPRR / MPRL Pathfinding Graph (Future)
1. Parse MPRR (reference) & MPRL (layout) chunks.
2. Build nav-mesh graph for higher-level tooling.

---

## CLI Consolidation Roadmap
1. Deprecate legacy `terrain-mesh` & `terrain-stamp`.
2. Introduce single `terrain-stitch`:
   ```bash
   dotnet run --project src/Pm4BatchTool -- terrain-stitch <pm4-root> [--out <obj>] [--with-links]
   ```
3. Route relative `--out` through `ProjectOutput` as today.

---

## Testing & Validation
* **Visual QA:** Blender / MeshLab continuous surface, no cracks.
* **Numeric QA:** Vertex dedup ratios, shared-edge vertex count parity.
* **Regression:** Ensure no change in per-tile vertex positions.

---

*End of plan.*
