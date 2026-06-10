# Tech Context

- Entry: `src/PM4FacesTool/Program.cs`.
- Key Method: `AssembleAndWrite()` builds local verts/tris and writes OBJ/GLTF.
- Snapping: `TryMapProjected(...)` applies plane projection when `--snap-to-plane`.
- Projection Formula (correct): v' = v − N * (dot(N, v) − H') / |N|^2, where H' = Height * heightScale.
- Flags:
  - `--snap-to-plane` (bool)
  - `--height-scale <float>` (default 1.0)
  - `--group-by`, `--batch`, `--ck-*`, `--gltf`, `--glb`.
