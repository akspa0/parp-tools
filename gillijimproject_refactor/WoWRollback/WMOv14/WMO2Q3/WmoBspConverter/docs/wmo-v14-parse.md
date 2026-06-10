# WMO v14 Parse Notes

Monolithic layout:
- Top-level chunks include MVER, MOMO, MOGP blocks.
- Record MOMO subchunks with absolute file offsets.
- Slice each MOGP region: from MOGP header start to next MOGP/EOF.

Inside MOGP region:
- Skip 0x40-byte header.
- Realign to first valid subchunk; realign up to 16 bytes between subchunks.
- MOVT: vertices with axis map x,z,-y.
- MOPY: 2 bytes per face (flags, materialId). Face count = len/2.
- MOVI: uint16 indices; if absent, fallback to sequential triples using MOPY face count.

Conversion:
- Duplicate 3 vertices per triangle.
- Emit Face Type=3 with MeshVertices (0,1,2).
