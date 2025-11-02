# WMO v14 Format Fix - Material/Texture Mapping

## Problem Summary

The WMO v14 converter was correctly extracting geometry but applying incorrect materials/textures. Castle models showed all wood texture instead of the correct stone/wood/tile combination.

## Root Cause

**Incorrect structure sizes and material assignment logic:**

1. **MOMT Structure**: Using 64 bytes (v17 format) instead of 44 bytes (v14 format)
2. **MOBA Structure**: Using 12 bytes instead of 24 bytes (missing bounding box)
3. **Material Assignment**: Using MOPY data which is unreliable in v14 files (all zeros)

## Solution

### 1. Fixed MOMT Structure (44 bytes for v14)

**Before (incorrect - 64 bytes):**
```csharp
// Reading v17 structure with shader field
mat.Shader = BitConverter.ToUInt32(momtData, off + 0x04);
```

**After (correct - 44 bytes):**
```csharp
// V14 structure: version(4), flags(4), blendMode(4), texture_1(4), sidnColor(4), 
// frameSidnColor(4), texture_2(4), diffColor(4), ground_type(4), padding(8)
uint version = BitConverter.ToUInt32(momtData, off + 0x00);  // V14 only!
mat.Flags = BitConverter.ToUInt32(momtData, off + 0x04);
mat.Shader = 0;  // NO shader field in v14
mat.BlendMode = BitConverter.ToUInt32(momtData, off + 0x08);
mat.Texture1Offset = BitConverter.ToUInt32(momtData, off + 0x0C);
// ... (no texture3, color2, flags2, runTimeData in v14)
```

### 2. Fixed MOBA Structure (24 bytes for v14)

**Before (incorrect - 12 bytes):**
```csharp
const int ENTRY_SIZE = 12;
byte lightMap = mobaData[o + 0];
byte texture = mobaData[o + 1];
ushort startIndex = BitConverter.ToUInt16(mobaData, o + 2);  // WRONG OFFSET!
```

**After (correct - 24 bytes):**
```csharp
const int ENTRY_SIZE = 24;
byte lightMap = mobaData[o + 0];
byte texture = mobaData[o + 1];           // Material ID in v14
// Skip bounding box (12 bytes at offset 2-13)
ushort startIndex = BitConverter.ToUInt16(mobaData, o + 14);  // Correct offset
ushort numIndices = BitConverter.ToUInt16(mobaData, o + 16);
ushort minIndex = BitConverter.ToUInt16(mobaData, o + 18);
ushort maxIndex = BitConverter.ToUInt16(mobaData, o + 20);
byte flags = mobaData[o + 22];
```

### 3. Fixed Material Assignment Logic

**Before (incorrect):**
```csharp
// Always preferred MOPY over MOBA
if (g.Mopy.Count >= triCount * 2) {
    // Use MOPY (which has all zeros in v14!)
}
```

**After (correct):**
```csharp
// V14: Try MOBA first if available
if (g.Batches != null && g.Batches.Count > 0) {
    // Check if MOBA has multiple materials
    bool hasMultipleMaterials = false;
    for (int i = 1; i < g.Batches.Count; i++) {
        if (g.Batches[i].MaterialId != g.Batches[0].MaterialId) {
            hasMultipleMaterials = true;
            break;
        }
    }
    if (hasMultipleMaterials) {
        // Use MOBA - it has correct material IDs
        var moba = BuildFaceMaterialsFromMoba(g, triCount);
        return moba;
    }
}
// Fall back to MOPY only if MOBA doesn't have useful data
```

## Verification

### Test Results (castle01.wmo)

**Before Fix:**
- All surfaces: Material 0 (wood trim texture)
- Material histogram: `[0:444]` (all triangles using material 0)

**After Fix:**
- Group 0 (interior): 8 materials correctly assigned
  - Material 0: 68 triangles (trim)
  - Material 1: 64 triangles (misc)
  - Material 2: 128 triangles (brick)
  - Material 6: 10 triangles (floor)
  - Material 7: 16 triangles (ceiling)
  - Material 8: 10 triangles
  - Material 9: 70 triangles
  - Material 10: 78 triangles
- Group 1 (exterior): 6 materials correctly assigned
  - Material 3: 145 triangles (stone walls)
  - Material 4: 120 triangles (wood)
  - Material 5: 665 triangles (stone)
  - Material 7: 96 triangles
  - Material 9: 247 triangles
  - Material 10: 341 triangles (roof tiles)

**Visual Verification:**
- ✅ Stone walls show stone texture
- ✅ Wooden roof shows wood texture
- ✅ Green roof tiles show tile texture
- ✅ Interior shows correct brick/floor/ceiling textures

## Reference Implementations

The fix was based on three reference implementations:

1. **WoWFormatParser** (`libs/WoWFormatParser/Structures/WMO/MOBA.cs`)
   - Showed correct 24-byte MOBA structure with bounding box
   - Confirmed `texture` field is material ID in v14
   - Confirmed `StartIndex` is uint16 in v14, not uint32

2. **mirrormachine** (`libs/mirrormachine/src/WMO_exporter.cpp`)
   - Showed MOPY writes material per face (2 bytes)
   - Confirmed MOBA batches group faces by material

3. **wowdev.wiki** (`z_wowdev.wiki/WMO.md`)
   - Documented v14-specific structure differences
   - Confirmed build 0.5.5.3494 uses v14 format

## Key Learnings

1. **Version-specific structures**: v14 and v17 have significantly different structure sizes
2. **MOPY unreliability**: In v14 files, MOPY often has incorrect material IDs (all zeros)
3. **MOBA is source of truth**: For v14, MOBA `texture` field contains correct material IDs
4. **Bounding box matters**: The 12-byte bounding box in MOBA shifts all subsequent fields

## Files Modified

- `Wmo/WmoV14Parser.cs` - Fixed MOMT (44 bytes) and MOBA (24 bytes) parsing
- `Wmo/WmoObjExporter.cs` - Changed material assignment to prefer MOBA over MOPY
- `memory-bank/activeContext.md` - Updated with verified v14 format specifications
- `README.md` - Updated with November 2025 fixes and test results

## Date

November 2, 2025
