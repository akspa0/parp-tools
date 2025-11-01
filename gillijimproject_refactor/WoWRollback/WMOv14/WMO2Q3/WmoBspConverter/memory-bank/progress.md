# Progress

## What works:
- ✅ v14 chunk parse with MOGP slicing and MOVI fallback
- ✅ IBSP writer emits correct header/lump order; entities terminator
- ✅ Mesh face emission (Type=3) with MeshVertices
- ✅ **Q3Map2-compliant brush generation** (6-plane rectangular boxes)
- ✅ Axis-aligned bounding box generation from triangle geometry
- ✅ Strict CCW plane winding for Q3Map2 compiler compatibility
- ✅ Texture extraction to 24-bit TGA with lowercase paths
- ✅ Degenerate triangle culling with logging

## Pending:
- Test in GtkRadiant to confirm "brush plane with no normal" is resolved
- UVs (MOTV) and materials (MOMT/MOTX) mapping
- Better model bounds from geometry
- Quake 3 load path docs and PK3 helper (optional)
- BSP node/leaf structure refinement for complex geometry

## Status:
- 🚨 **ROOT CAUSE IDENTIFIED**: WMO data extraction is broken
  - **POC uses MOVI for indices, current code doesn't**
  - MOVI = vertex indices (ushort, 3 per triangle)
  - MOPY = face metadata (flags + material ID per triangle)
  - Current fallback assembles triangles sequentially instead of using MOVI indices
- ✅ **COORDINATE TRANSFORMS**: Working correctly (100x scale, XYZ→XZY mapping, Z inversion)
- ✅ **BRUSH GENERATION**: 5-plane thin slab brushes render correctly in GtkRadiant
- 🔧 **URGENT FIX NEEDED**: Update WmoV14Parser to read MOVI chunk
  - Reference: FullV14Converter.AnalyzeAndBuildGroup() lines 165-294
  - Replace fallback sequential assembly with proper MOVI index reading
  - Ensure MOPY metadata is correctly associated with triangles
