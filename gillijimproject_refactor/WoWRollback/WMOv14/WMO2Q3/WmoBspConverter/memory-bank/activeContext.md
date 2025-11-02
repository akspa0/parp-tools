# Active Context

Current focus:
- ✅ COMPLETE: WMO v14 parsing and material assignment
  - MOVI/MOVT/MOTV chunks parsed correctly
  - MOMT materials: 44 bytes per entry (v14 has version field, no shader field)
  - MOBA batches: 24 bytes per entry (lightMap, texture, boundingBox, indices)
  - Material assignment: MOBA preferred over MOPY (MOPY often has incorrect data)
  - Geometry exports successfully (test.wmo: 36 verts, castle01.wmo: 7113 verts)
- ✅ VERIFIED: WMO to Q3 coordinate transformation working
  1. Coordinate mapping: WMO(X,Y,Z) → Q3(X,Z,-Y)
  2. Scaling: Uniform 100x for all axes, with Z inverted
  3. Plane winding: Normals point OUTWARD from solid volume
- ✅ VERIFIED: Material and texture mapping working correctly
  - MOBA `texture` field contains material ID in v14
  - Multiple materials per group correctly assigned (e.g., stone, wood, roof tiles)
  - Textures correctly mapped to surfaces (castle01.wmo verified in MeshLab)
  - BLP → TGA texture conversion working
- 🔧 NEXT: Test .map files in GtkRadiant and compile to BSP for Quake 3

## Critical Implementation Notes

### V14 Format Specifications (VERIFIED)

**MOMT Structure (44 bytes):**
```
0x00: version (uint32) - V14 only!
0x04: flags (uint32)
0x08: blendMode (uint32)
0x0C: texture1Offset (uint32) - offset into MOTX
0x10: sidnColor (uint32) - emissive
0x14: frameSidnColor (uint32) - runtime
0x18: texture2Offset (uint32)
0x1C: diffColor (uint32)
0x20: groundType (uint32)
0x24: padding (8 bytes)
```
NO shader field, NO texture3/color2/flags2/runTimeData in v14!

**MOBA Structure (24 bytes):**
```
0x00: lightMap (byte)
0x01: texture (byte) - MATERIAL ID in v14!
0x02: boundingBox (12 bytes) - 6 int16 values
0x0E: startIndex (uint16) - index into MOVI
0x10: count (uint16) - number of MOVI indices
0x12: minIndex (uint16) - first vertex
0x14: maxIndex (uint16) - last vertex
0x16: flags (byte)
0x17: padding (byte)
```
NO materialId field in v14 - use `texture` field instead!

**MOPY Structure (2 bytes per face):**
```
0x00: flags (byte) - render flags
0x01: materialId (byte) - often incorrect in v14 (all zeros)
```
In v14, MOPY material IDs are unreliable - use MOBA instead!

### Reference Implementations
- `WoWFormatParser/Structures/WMO/MOBA.cs` - Correct v14 structure
- `mirrormachine/src/WMO_exporter.cpp` - MOPY writes material per face
- `old_sources/.../FullV14Converter.cs` - POC chunk reading

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
