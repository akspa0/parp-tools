# Active Context

Current focus:
- 🚨 ROOT CAUSE FOUND: WMO data extraction is broken
  - **POC uses MOVI for indices, current code doesn't**
  - MOVI = vertex indices (ushort, 3 per triangle)
  - MOPY = face metadata (flags + material ID)
  - Current fallback assembles triangles sequentially instead of using MOVI
- ✅ SOLVED: WMO to Q3 coordinate transformation (for valid data)
  1. Coordinate mapping: WMO(X,Y,Z) → Q3(X,Z,-Y)
  2. Scaling: Uniform 100x for all axes, with Z inverted
  3. Plane winding: Normals point OUTWARD from solid volume
- 🔧 NEXT: Fix WmoV14Parser to read MOVI chunk correctly
  - Compare with FullV14Converter.AnalyzeAndBuildGroup() (lines 165-294)
  - Implement proper MOVI index reading instead of fallback
  - Ensure MOPY metadata is associated with correct triangles

## Critical Implementation Notes

### POC Reference Code Location
`old_sources/src/WoWToolbox/WoWToolbox.Core.v2/Services/WMO/Legacy/FullV14Converter.cs`
- Lines 165-294: `AnalyzeAndBuildGroup()` method
- Shows correct MOVI/MOPY/MOVT/MONR/MOTV parsing
- Reference for fixing current implementation

### Chunk Reading Order (from POC)
1. MOPY - Face metadata (flags + material ID)
2. MOVI - Vertex indices (ushort array)
3. MOVT - Vertex positions (Vector3 array)
4. MONR - Vertex normals (Vector3 array)
5. MOTV - Texture UVs (Vector2 array)
6. MOBA - Render batches (optional)

### Key Parsers Needed
- MOVIParser - Reads ushort indices
- MOPYParser - Reads flags + material ID per triangle
- MOVTParser - Reads Vector3 positions
- MONRParser - Reads Vector3 normals
- MOTVParser - Reads Vector2 UVs

Recent changes (2025-10-31):
- **Q3MAP2 COMPLIANCE FIX**: Switched from triangular prisms (5 planes) to rectangular boxes (6 planes)
  - Q3Map2 was reporting "mirrored plane" and "degenerate plane" errors on complex prisms
  - Root cause: colinear/duplicate points in plane definitions → zero-length cross products
  - Solution: emit simple axis-aligned bounding boxes with proper 3-point plane definitions
  - Each brush now has 6 standard planes (bottom, top, front, back, left, right)
  - All planes follow strict CCW winding convention for Q3Map2 compiler
  - Added minimum dimension enforcement (0.1 units) to prevent degenerate brushes
  - Ensured all 3 points per plane are non-colinear and distinct
- **WORKFLOW OPTIMIZATION**: Disabled BSP file writing
  - Converter now outputs .map only (primary format for GtkRadiant)
  - Users compile .map → BSP using GtkRadiant's Q3Map2 (Bsp → Compile)
  - Avoids writing potentially invalid BSP during development
- Added degenerate triangle culling with logging

Previous work:
- Implemented MOVI fallback from MOPY face count
- Faces Type=3 with meshverts 0,1,2 per triangle
- Entities null-terminated; LightGrid/VisData order fixed; Node size=36
- Texture export: 24-bit uncompressed TGA, lowercase names; .map uses lowercase texture paths
- .map generator extrudes triangular prisms along +Z; per-face texture names wired

Next steps:
- Test in GtkRadiant: load test.map and verify no "brush plane with no normal" errors
- If valid: test with larger WMO files (castle01, etc.)
- Wire UVs (MOTV) and per-face materials (MOMT/MOTX) end-to-end
- Add optional BSP validation tool to diagnose plane issues

Risks:
- Some v14 samples may have padding/odd subchunk alignment; scanner realigns up to 16 bytes
- BSP node/leaf structure still minimal; may need refinement for complex geometry
