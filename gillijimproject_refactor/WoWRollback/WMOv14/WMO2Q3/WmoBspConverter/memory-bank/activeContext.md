# Active Context - WMO to Quake 3 Conversion (2025-11-02)

## Current Status: PARTIALLY WORKING

### ✅ What Works
1. **WMO v14 Parsing** - Correctly reads MOVI/MOVT/MOTV/MOMT/MOBA chunks
2. **Geometry Export** - WMO → OBJ works perfectly with all textures
3. **.map File Generation** - Creates valid Q3 .map files that load in GtkRadiant
4. **Q3Map2 Compilation** - Maps compile successfully (no leak errors with sealed room)
5. **Coordinate Transform** - WMO(X,Y,Z) → Q3(X,Y,Z) passthrough (WMO already in correct format)
6. **Sealed Worldspawn** - Automatic 6-sided room generation around WMO geometry
7. **Group Splitting** - `--split-groups` option for large WMOs like Ironforge

### ❌ Current Issues (BLOCKING Q3 LOADING)
1. **Mirrored Planes** - ~5 brushes per map have inverted normals (Q3Map2 warnings)
2. **Single Texture Only** - Only one texture renders despite 11 textures in WMO
3. **Q3 Won't Load** - Compiled BSP files fail to load in Quake 3 engine
4. **Inside-Out Geometry** - Brushes render with inverted faces in GtkRadiant

### 🔧 Critical Fixes Needed
1. **Fix Brush Winding Order** - Swap v1/v2 vertices to correct plane normals
2. **Fix Texture Assignment** - Map WMO material IDs to correct brush textures
3. **Validate BSP Output** - Ensure compiled BSP meets Q3 engine requirements

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

## Key Discoveries (2025-11-02 Session)

### Coordinate Transform
- **CRITICAL**: WMO files store vertices as (X, Z, -Y) already
- Wiki states: "coordinates are in (X,Z,-Y) order"
- **Current transform**: Pass-through (no transformation needed)
- **Previous wrong transform**: WMO(X,Y,Z) → Q3(X,Z,-Y) caused double-transform

### Brush Generation
- **Working approach**: 6-plane axis-aligned bounding boxes around each triangle
- **Format**: `( x y z ) ( x y z ) ( x y z ) TEXTURE offsetX offsetY rotation scaleX scaleY contentFlags surfaceFlags value`
- **Required**: 8 texture parameters (not 5!)
- **Test cube format** (working): Uses min/max corners with proper winding

### Map Structure
- **Worldspawn**: Sealed 6-sided room (128 unit padding around WMO)
- **WMO Geometry**: func_group entity (detail brushes)
- **Player Spawn**: info_player_deathmatch positioned above WMO geometry

### Texture Issues
- WMO has 11 textures, but only 1 shows in compiled map
- Texture assignment from MOBA/MOMT might not be reaching brush generation
- Need to verify material ID → texture name mapping in brush creation

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
