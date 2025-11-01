# System Patterns

Architecture:
- Parser (v14):
  - Read top-level chunks, slice each MOGP region by absolute offsets.
  - Inside region: skip 0x40 header, realign to next valid subchunk.
  - MOVT: x,z,-y axis mapping; MOPY: (flags, materialId) pairs; MOVI: indices.
  - MOVI-absent fallback: sequential triples using MOPY face count.
- Converter:
  - Duplicate 3 vertices per triangle.
  - Faces use Type=3 with MeshVertices (0,1,2) per face.
- BSP writer (IBSP v46):
  - Header: magic IBSP + version 46.
  - Lump order (17): Entities, Textures, Planes, Nodes, Leaves, LeafFaces, LeafBrushes, Models, Brushes, BrushSides, Vertices, MeshVertices, Effects, Faces, Lightmaps, LightGrid, VisData.
  - Entities is null-terminated text.
  - Node/Leaf AABB are int32.

Key guarantees:
- No cross-map assumptions; geometry only from current WMO file.
- Prefer correctness over features; add UVs/materials later.
