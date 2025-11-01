# Progress

What works:
- v14 chunk parse with MOGP slicing and MOVI fallback.
- IBSP writer emits correct header/lump order; entities terminator.
- Mesh face emission (Type=3) with MeshVertices.

Pending:
- UVs (MOTV) and materials (MOMT/MOTX) mapping.
- Better model bounds from geometry.
- Quake 3 load path docs and PK3 helper (optional).

Status:
- Ready to regenerate castle01; expect non-trivial BSP size and >0 faces.
