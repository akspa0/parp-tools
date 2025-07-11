# PM4 Chunk List

Confirmed chunk identifiers observed in PM4 files.  Names are taken directly from file headers; no speculative purposes are listed.

- MDBH
- MDSF
- MSUR
- MSPV
- MSVT
- MSVI
- MSPI
- MSLK
- MSCN
- MPRL
- MDOS

> For linkage rules see `mslk_linkage_reference.md`.

Quick cheat-sheet of every chunk relevant to current work.  Keep this file ≤2 KB.

| Chunk | Purpose | Key Confirmed Fields |
|-------|---------|----------------------|
| MDBH  | Directory header of sub-chunks | Entry count (but padding often wrong) |
| MDSF  | Surface flags                 | Flags per surface (navigation) |
| MSUR  | Render surface header         | `IndexCount`, `Unknown_1C` (object key), normals & height |
| MSPV  | Navigation vertex buffer      | `C3Vector` (X Y Z) |
| MSVT  | Render vertex buffer (LOD0)   | (Y X Z) ordering; use `Pm4CoordinateTransforms.FromMsvtVertex` |
| MSVI  | Unified index buffer          | 32-bit indices into MSPV/MSVT |
| MSPI  | Navigation indices            | Per-triangle indices – reference by `MSLK.MspiFirstIndex`|
| MSLK  | Object metadata / links       | `Flag` (0x00), `RefHigh/Low`, `LinkSubHigh/Low`, `GroupObjectId` |
| MSCN  | Collision mesh                | Vertex positions & AABB |
| MPRL  | Placement positions           | World coordinates of placed props |
| MDOS  | Destructible object states    | Index per MSUR surface |

> See `mslk_linkage_reference.md` for full linkage rules.
