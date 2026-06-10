# Conversion Pipeline

1) Parse WMO v14
- Read top-level chunks, slice MOGP regions by absolute offsets.
- Parse MOVT/MOVI/MOPY; apply MOVI fallback.

2) Build BSP
- Add textures (names only initially).
- For each triangle: push 3 vertices; push meshverts 0,1,2; create Type=3 face.
- Generate planes from faces; minimal nodes/leaves/models; entities worldspawn.

3) Write IBSP v46
- Prepare 17 lumps in correct order; 4-byte align.
- Null-terminate entities; Node/Leaf AABB as int32.

4) Optional outputs
- .map generation from triangles (prism brushes per tri).
- Textures and shader script.
