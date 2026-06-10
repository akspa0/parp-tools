# MDX/M2 0.9.0 Render Path — Part 04 (ModelRender Function Lock)

## Scope
Lock concrete renderer-side function addresses tied to `ModelRender.cpp` and geoset/material assembly behavior.

## Direct function evidence (decompiled)
### `FUN_0043cb90`
- Binds to `ModelRender.cpp` via assertions:
  - `s_C__build_buildWoW_ENGINE_Source__008225bc` @ lines `0xaf7`, `0xaf8`, `0xb07`
- Behavior:
  - validates `model` and `shared`
  - grows arrays/capacities for render-side structures
  - creates/attaches material via `FUN_00428a80`
  - calls `FUN_0043cea0(...)` to populate geoset shared payload
- Assessment: **high-confidence AddGeoset pipeline entry candidate**.

### `FUN_0043cea0`
- Binds to `ModelRender.cpp` via assertion:
  - `s_C__build_buildWoW_ENGINE_Source__008225bc` @ line `0xacd`
  - null-check token: `s_geoShared_00821de0`
- Behavior:
  - writes geoset-shared output struct fields
  - copies vertex/normal/index-related streams with strict index checks
  - uses repeated `index < array_size` assert helper pattern
- Assessment: **high-confidence geoset shared-payload materialization**.

### `FUN_0043a680`
- Binds to `ModelRender.cpp` via assertion:
  - `s_C__build_buildWoW_ENGINE_Source__008225bc` @ line `0xfbb`
- Behavior:
  - iterates per-geoset-ish entries, branching on mode values (`0/1/2`)
  - delegates to `FUN_0043ae70`, `FUN_0043b010`, and intersection-like routines
- Assessment: renderer-side per-element processing path, likely culling/intersection/gating adjacent to scene-add flow.

### `FUN_0043ae70`
- Binds to `ModelRender.cpp` via assertion:
  - `s_C__build_buildWoW_ENGINE_Source__008225bc` @ line `0xf4d`
  - assert token: `s_CMath__fnotequal__cylScale_0__008227f4`
- Behavior:
  - geometric/cylindrical math check path.
- Assessment: helper in same render module, not primary material-id gate.

## Current status of target literal
- Exact literal not yet observed in pseudocode this pass:
  - `AddGeosetToScene: geoShared->materialId (%d) >= numMaterials (%d)` (`00822668`)
- However, function-level binding to `ModelRender.cpp` is now concrete, and geoset shared + material creation flow is localized.

## Promotion decision
- Promote `FUN_0043cb90` + `FUN_0043cea0` as primary render-path ground-truth anchors for 0.9.0 contract addendum.
- Keep literal-level material-id branch as pending micro-closure item.
