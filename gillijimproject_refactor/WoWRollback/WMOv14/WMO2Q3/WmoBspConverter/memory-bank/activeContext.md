# Active Context

Current focus:
- Geometry correctness: MOVI fallback, MeshVertices, proper lump order, entities terminator, int AABB.

Recent changes:
- Implemented MOVI fallback from MOPY face count.
- Faces Type=3 with meshverts 0,1,2 per triangle.
- Entities null-terminated; LightGrid/VisData order fixed; Node size=36.

Next steps:
- Verify castle01 produces non-zero faces/verts and larger .bsp.
- Add UVs from MOTV; map MOPY→MOMT→MOTX for texture selection.
- Improve bounds from geometry rather than fixed cube.

Risks:
- Some v14 samples may have padding/odd subchunk alignment; scanner realigns up to 16 bytes.
