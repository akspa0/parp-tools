# Tech Context

Runtime:
- .NET 9, C# modern features enabled.

Dependencies:
- Internal BSP writer; avoid LibBSP at runtime.
- WoWFormatLib present for reference; v14 reader is custom.

Specs / References:
- Quake 3 IBSP v46 format (entities, lumps, faces, meshverts).
- WMO v14 chunk IDs: MVER, MOMO, MOTX, MOMT, MOGN, MOGI, MOGP, MOVT, MOVI, MOPY, MOTV.

Paths:
- Output under provided -d. Q3 search uses fs_basepath and fs_homepath; place in baseq3/maps or pack PK3.
