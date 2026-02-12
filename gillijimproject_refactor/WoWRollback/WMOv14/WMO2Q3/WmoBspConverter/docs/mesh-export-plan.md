# WMO → Quake 3 Mesh Export Plan

## Goal
Replace triangle brush emission with ASE mesh exports per WMO group, referenced by `misc_model` entities in the generated Quake 3 `.map` so complex geometry compiles without plane errors.

## Mesh Format
- **Format:** ASE (ASCII Scene Export) for ease of authoring and compatibility with GtkRadiant/ioquake3.
- **Per-file scope:** One WMO group per ASE file. File naming: `models/wmo/<wmoName>/group_<index>.ase`.
- **Coordinate system:** Apply existing WMO→Q3 transform `(x, y, z) → (x, -z, y)` then subtract geometry offset so meshes sit centered in the sealed room.
- **Units:** Keep source units (Q3 expects 1 unit = 1 game unit). No additional scaling.

## ASE Structure
```
*3DSMAX_ASCIIEXPORT 200
*MATERIAL_LIST ...
*GEOMOBJECT "group_<index>"
  *NODE_NAME "group_<index>"
  *NODE_TM (identity)
  *MESH
    *MESH_NUMVERTEX <N>
    *MESH_NUMFACES <M>
    *MESH_VERTEX_LIST ...
    *MESH_FACE_LIST ...
    *MESH_NUMTVERTEX <Nuv>
    *MESH_TVERTLIST ... (if UVs)
    *MESH_NUMTVFACES <M>
    *MESH_TFACELIST ...
```
- Triangles follow MOVI order.
- Materials map to shader names (texture paths) referenced via `*MATERIAL_REF`.

## Texture/Shader Mapping
- Reuse `TextureProcessor` outputs: shader names already generated under `textures/wmo/...`.
- ASE `*MAP_DIFFUSE` uses the shader path (without extension) so Quake 3 resolves via shader scripts.

## Map Integration
- Keep sealed worldspawn + spawn placement.
- Remove `func_group` brush emission.
- For each group:
  - Create `misc_model` entity:
    ```
    {
      "classname" "misc_model"
      "model" "models/wmo/<wmoName>/group_<index>.ase"
      "origin" "<offset.x> <offset.y> <offset.z>"
    }
    ```
  - Origin derived from geometry offset (negated because mesh is offset into room).
  - Optionally add `_wmo_group` key for debugging.

## Output Layout
```
output/
  castle01.map
  castle01.bsp (optional)
  models/
    wmo/
      castle01/
        group_000.ase
        group_001.ase
  wmo/
    shaders/
      wmo_textures.shader
  textures/
    wmo/<exported tgas>
```

## Validation Steps
1. Run converter on `castle01.wmo` → ensure ASE files + map produced.
2. Open `.map` in GtkRadiant → verify models render, origin aligned.
3. Compile with q3map2 → check for absence of `bad float plane` errors.
4. Inspect `.bsp` in game/editor.

## Coordinate Transform and Origin Details
- **Vertex transform (per-vertex):** `q3 = (x, -z, y)`
- **Group-local recentering:** compute `groupCenter = average(transformed vertices)` and subtract from all vertices so each ASE is centered around its own origin.
- **Model origin in .map:** set `origin = -groupCenter` on the `misc_model` so the placed model lands at the intended position in the sealed room.
- **Example:** if `groupCenter = (128.0, 64.0, -32.0)`, write `origin "-128 -64 32"`.
- **Material path convention:** ASE `*MAP_DIFFUSE` should use shader path like `textures/wmo/<wmoName>/<material_name>` (no extension). The corresponding shader is provided in `wmo/shaders/wmo_textures.shader`.

## Implementation Checklist
- [ ] Add ASE writer with MOVT/MOVI/MOTV support and optional normals.
- [ ] Implement group-local recentering and write `origin` = negative center in `.map`.
- [ ] Emit one ASE per group to `models/wmo/<wmoName>/group_<index>.ase`.
- [ ] Keep sealed brushes and player spawn; remove triangle wedge brushes.
- [ ] Map materials to shader names; write a minimal shader file if missing.
- [ ] Export textures into `textures/wmo/<wmoName>/...` (TGA/PNG as currently produced).
- [ ] Verify in GtkRadiant and compile with q3map2.

## Acceptance Criteria
- **Editor load:** `.map` opens in GtkRadiant with all groups visible as models.
- **Placement:** Each model aligns correctly; no visible offsets between groups.
- **Compile:** q3map2 completes without `bad float plane`/brush convexity errors.
- **In-game:** Resulting `.bsp` renders WMO geometry with correct materials.
