# PM4 Assembly Relationships

_Last updated: 2025-08-08_

This document captures the confirmed relationships between core PM4 chunks and how they are used to reconstruct coherent game objects. Keep it up-to-date whenever we discover new fields.

---

## 1. Key Chunks & Fields

| Chunk | Field | Size | Purpose |
|-------|-------|------|---------|
| **MSUR** | `IndexCount` (byte @0x01) | indices/triangles in MSVI | **Primary object-id** used by legacy tools, good baseline grouping but still yields sub-objects (floor slices, roofs, etc.). |
| **MSUR** | `CompositeKey` (`SurfaceKey`) | uint32 @0x1C | **High-16 bits (`SurfaceKeyHigh16`) correspond to a whole placed object** (building, terrain piece). Low-16 bits often subdivide by ornaments or broken windows, etc. |
| **MSUR** | `GroupKey` (`FlagsOrUnknown_0x00`) | byte @0x00 | Broad object category (exterior, interior, terrain, etc.). |
| **MSUR** | `AttributeMask` (`Unknown_0x02`) | byte @0x02 | Surface subtype, bit `0x80` marks liquid candidates. |
| **MSLK** | `ParentIndex` (uint32 @0x04) | Link to placement (`MPRL.Unknown4`). Container nodes have `MspiFirstIndex = -1`. |
| **MSLK** | `MspiFirstIndex` / `MspiIndexCount` | Geometry slice inside global index buffer. |
| **MPRL** | `Unknown4` | Placement → geometry mapping (`ParentIndex`). |
| **MPRR** | `Value1` | Sentinel `65535` marks property separators. |

---

## 2. Proven Grouping Strategies

### 2.1 Surface-Key Strategy (recommended)
* Group by `MSUR.SurfaceKeyHigh16` (upper 16 bits of `CompositeKey`).
* Produces **one OBJ per placed object** (building, terrain section).
* Merge surfaces within the same key; ignore `SurfaceKeyLow16` unless we need finer splits.
* Robust against interior/exterior mixing; retains ~38k-650k triangles per building.

### 2.2 Index-Count Strategy (legacy research)
* Group by `MSUR.IndexCount`.
* Useful for visualising large vs small slices; not a full object.

### 2.3 Parent-Index Hierarchy
* Start with `MSLK.ParentIndex` → `MPRL.Unknown4` mapping.
* Recursively include child `MSLK` where `MspiFirstIndex != -1`.
* Produces acceptable object shapes but misses cross-tile data and yields more fragments than Surface-Key.

---

## 3. Cross-Tile Vertex Resolution
* PM4 tiles reference vertices outside their own MSVT buffer.
* Load region with `Pm4GlobalTileLoader` → rebases each surface’s `MsviFirstIndex` by per-tile `IndexOffset`.
* Without this step ~64 % of vertices are out-of-bounds (see _Critical PM4 Data Loss Discovery_ memory).

---

## 4. Implementation Notes
* `SurfaceKeyAssembler` (new): implements the Surface-Key strategy, re-uses global vertex validation/remap logic.
* `MsurIndexCountAssembler`: keeps IndexCount grouping for diagnostic purposes.
* All assemblers **skip triangles containing invalid vertex indices** to avoid (0,0,0) phantom vertices.
* OBJ exporter inverts X by default; use `--legacy-obj-parity` to disable.

---

## 5. CLI Cheatsheet
```
# Per-object (recommended)
dotnet run --project src/PM4NextExporter -- <mytile.pm4> --assembly surface-key --include-adjacent

# IndexCount diagnostic view
dotnet run --project src/PM4NextExporter -- <mytile.pm4> --assembly msur-indexcount
```

---

## 6. TODO
* Evaluate whether we need to merge multiple `SurfaceKeyHigh16` values that belong to the same building via MPRL placements.
* Snapshot tests for Surface-Key grouping vs legacy exporter SHA.
* Document MSLK container-node detection logic once implemented.
