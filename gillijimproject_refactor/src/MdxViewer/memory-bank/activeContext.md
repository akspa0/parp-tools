# Active Context — MdxViewer Renderer Reimplementation

## Current Focus

**BLOCKED: WMO rotation/facing is wrong.** Models sit in their bounding boxes correctly but face the wrong direction. Multiple approaches tried and failed. Need fresh approach next session.

## Immediate Next Steps

1. **SOLVE WMO rotation** — the critical blocker. See "WMO Rotation Problem" section below.
2. MDX texturing could use more work but is lower priority.
3. Standalone WMO/MDX viewer has black screen (medium priority).

## WMO Rotation Problem (UNSOLVED)

**Symptoms**: WMO models sit correctly in their MODF bounding boxes but face ~180° wrong direction. The bounding box position is always correct.

**Known-good baseline** (current code state):
- WMO vertices: raw pass-through from file (X, Y, Z) — no modification
- WMO rotation: `CreateRotationX(rx) * CreateRotationY(ry) * CreateRotationZ(-rz) * CreateTranslation(p.Position)`
- Where `rx = p.Rotation.X`, `ry = p.Rotation.Y`, `rz = p.Rotation.Z`
- Models sit in bounding boxes but face wrong direction

**What was tried and failed**:
- Vertex swap X↔Y → displaced model from BB
- Vertex negate Y → closer but still wrong
- Vertex negate X → flipped wrong way
- Vertex swap+negate (-wowY, -wowX, wowZ) → model went sideways/flat
- Basis change matrix in transform → model flipped upside down
- Adding 180° to heading → displaced from BB
- Adding 90° to heading → displaced from BB
- Noggit's eulerAngleYZX formula adapted → various wrong results

**Key reference: noggit-red SceneObject::updateTransformMatrix()**:
```cpp
// In noggit's Y-up rendering space:
matrix = translate(pos);
matrix *= eulerAngleYZX(dir.y - 90°, -dir.x, dir.z);
matrix = scale(matrix, vec3(scale));
// where dir = (rot[0], rot[1], rot[2]) from MODF file bytes
```

**Coordinate system facts**:
- WoW server: X=north, Y=west, Z=up
- Noggit rendering: pos=(rawX, rawZ, rawY) from file, Y=up
- Our renderer: rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX, rendererZ = wowZ (Z=up)
- MODF adapter reads: rotX=rot[0] (off+20), rotZ=rot[1] (off+24), rotY=rot[2] (off+28)
- Adapter stores: `Rotation = (rotX, rotY, rotZ)` — note rotY and rotZ are swapped vs file order
- So: `p.Rotation.X = rot[0]`, `p.Rotation.Y = rot[2]`, `p.Rotation.Z = rot[1]`
- Noggit's dir: `dir.x = rot[0] = p.Rotation.X`, `dir.y = rot[1] = p.Rotation.Z`, `dir.z = rot[2] = p.Rotation.Y`

**Fresh approach ideas for next session**:
- Study how noggit renders WMO vertices (does it also swap/convert them?)
- Study wow.export's WMO rendering pipeline end-to-end
- Try rendering WMO in noggit's coordinate system (Y-up) and convert the camera instead
- Check if the WMO v14 parser itself does any coordinate conversion during parsing

## Phase 4 — World Scene ✅ MOSTLY COMPLETE

Working features:
- Terrain rendering with AOI-based lazy tile loading (radius=2)
- MDDF/MODF placement loading from ADT chunks
- MDX doodad rendering (backface culling disabled, blend mode fixes, depth mask for transparency)
- WMO rendering with BLP textures per-batch
- WMO doodad sets loaded and rendered (transforms combined with WMO modelMatrix)
- Bounding box visualization (actual MODF extents)
- Object visibility toggles (terrain, WMOs, doodads)
- Live minimap with click-to-teleport
- AreaPOI system (DBC loading, 3D markers, minimap markers, UI list with teleport)

## What Works

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI loading | ✅ |
| MDX model rendering | ✅ (textures, no culling, blend modes) |
| WMO rendering + textures | ✅ (BLP per-batch) |
| WMO doodad sets | ✅ |
| WMO rotation/facing | ❌ WRONG — models face ~180° off |
| MDDF/MODF placements | ✅ (position correct, rotation TBD) |
| Bounding boxes | ✅ (actual MODF extents) |
| Live minimap + click-to-teleport | ✅ |
| AreaPOI system | ✅ |
| Object picking/selection | ✅ |
| Standalone WMO/MDX viewer | ❌ Black screen |

## Key Files

- `Terrain/WorldScene.cs` — Object instance building, rotation transforms, rendering loop
- `Terrain/AlphaTerrainAdapter.cs` — MDDF/MODF parsing, coordinate conversion
- `Rendering/WmoRenderer.cs` — WMO geometry, textures, doodad sets
- `Rendering/ModelRenderer.cs` — MDX rendering, blend modes, textures
- `ViewerApp.cs` — Main app, UI, DBC loading, minimap

## Dependencies (all already integrated)

- `MdxLTool` — MDX file parser
- `WoWMapConverter.Core` → `gillijimproject-csharp` — Alpha WDT/ADT/MCNK parsers, WMO v14 parser
- `SereniaBLPLib` — BLP texture loading
- `Silk.NET` — OpenGL + windowing + input
- `ImGuiNET` — UI overlay
- `DBCD` — DBC database access
