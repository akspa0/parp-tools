# Project Brief — WmoBspConverter

Goal: Parse World of Warcraft WMO v14 and export geometry to OBJ+MTL for visual validation. BSP/.map remain secondary outputs.

Success criteria:
- Build on .NET 9 without legacy dependencies.
- Parse v14 WMO monoliths (MOGP regions) reliably.
- Export OBJ+MTL with grouped materials (usemtl) and UVs (vt) that match the source.
- Emit valid IBSP v46 with correct header/lump directory and non-empty geometry.
- Basic textures/shaders optional; geometry correctness first.
- Produce a loadable .map for editors.

Out of scope (initial):
- Lightmaps baking.
- Portal/vis generation beyond a stub.
- Full material systems. 
