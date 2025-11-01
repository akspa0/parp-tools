# Project Brief — WmoBspConverter

Goal: Convert World of Warcraft WMO v14 files to Quake 3 BSP (IBSP v46) and .map for editing.

Success criteria:
- Build on .NET 9 without legacy dependencies.
- Parse v14 WMO monoliths (MOGP regions) reliably.
- Emit valid IBSP v46 with correct header/lump directory and non-empty geometry.
- Basic textures/shaders optional; geometry correctness first.
- Produce a loadable .map for editors.

Out of scope (initial):
- Lightmaps baking.
- Portal/vis generation beyond a stub.
- Full material systems. 
