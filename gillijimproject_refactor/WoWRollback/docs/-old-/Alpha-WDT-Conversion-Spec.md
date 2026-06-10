# Alpha WDT Conversion Technical Specification

## Overview

This document specifies the requirements for converting WoW 3.3.5 (LK) WDT/ADT files to Alpha 0.5.3 monolithic WDT format. This specification is implementation-agnostic and can be used to build converters in any language.

**Status**: ✅ Verified working in 0.5.3 Alpha client (build 3368)

---

## File Format Differences

### LK Format (3.3.5)
- **Separate chunks**: Terrain, objects, and textures in different files
- **Modern features**: MH2O liquids, high-res holes, vertex lighting

### Alpha Format (0.5.3)
- **Monolithic file**: Single `map.wdt` containing all tile data
- **Embedded tiles**: Each tile has MHDR + MCIN + 256 MCNKs inline
- **Legacy features**: MCLQ liquids, simple holes, no vertex lighting
- **Reversed FourCC**: Chunk tokens stored backwards on disk

---

## Critical: Sequential Chunk Reading (Ghidra Verified)

**Reverse engineered from WoWClient.exe (0.5.3.3368) with PDB symbols:**

The Alpha client reads WDT chunks **SEQUENTIALLY** - it does NOT seek to MPHD offsets!

```c
// From CMap::LoadWdt decompilation:
SFile::Read(wdtFile, &iffChunk, 8, ...);  // MVER header
SFile::Read(wdtFile, &version, 4, ...);    // MVER data
SFile::Read(wdtFile, &iffChunk, 8, ...);  // MPHD header
SFile::Read(wdtFile, &header, 0x80, ...); // MPHD data (128 bytes)
SFile::Read(wdtFile, &iffChunk, 8, ...);  // MAIN header
SFile::Read(wdtFile, &areaInfo, 0x10000, ...); // MAIN data (65536 bytes)
LoadDoodadNames();  // Reads MDNM sequentially
LoadMapObjNames();  // Reads MONM sequentially
SFile::Read(wdtFile, &iffChunk, 8, ...);  // Check for MODF
```

**Implications:**
- MPHD offsets (`offsDoodadNames`, `offsMapObjNames`) are metadata for indexing names, NOT used for seeking
- **ANY padding bytes between chunks will cause assertion failures**
- Chunks must be written in exact order with NO gaps

---

## Top-Level WDT Structure

### Required Chunks (in order)

```
MVER (4 bytes)
  - Version = 18 (uint32)

MPHD (128 bytes)
  - nDoodadNames: uint32      // Count of M2 model names
  - offsDoodadNames: uint32   // Absolute offset to MDNM chunk
  - nMapObjNames: uint32      // Count of WMO names (see counting rules below)
  - offsMapObjNames: uint32   // Absolute offset to MONM chunk
  - padding: byte[112]        // Must be zeros

MAIN (65536 bytes = 4096 tiles * 16 bytes)
  - Array of SMAreaInfo entries (64x64 grid)
  
MDNM (variable)
  - M2 model names (null-terminated strings)
  - Usually empty for terrain-only conversions
  
MONM (variable)
  - WMO names (null-terminated strings)
  - CRITICAL: Must contain actual WMO names from source
  - CRITICAL: NO padding byte after MONM even if size is odd
  
[Tile Data]
  - Embedded MHDR + tile data immediately after MONM
  - NOTE: Original Alpha format has NO MODF chunk here
  - Embedded MHDR + tile data for each non-zero tile
```

### SMAreaInfo Structure (16 bytes)

```c
struct SMAreaInfo {
    uint32 offset;  // Absolute offset to tile MHDR letters
    uint32 size;    // Size from MHDR letters to first MCNK (NOT total tile size)
    uint32 flags;   // Runtime flag (FLAG_LOADED = 0x1), write as 0
    uint32 pad;     // Padding, write as 0
};
```

**CRITICAL**: The `size` field is the distance from MHDR letters to the first MCNK, NOT the total tile size!

---

## MPHD.nMapObjNames Counting Rules

**CRITICAL IMPLEMENTATION DETAIL**

The Alpha client counts WMO names by **splitting on null terminators**, which includes the trailing empty string after the final null.

### Example

```
MONM data: "world\wmo\building.wmo\0"
Split by \0: ["world\wmo\building.wmo", ""]
Count: 2 (not 1!)
```

### Implementation

```
if (wmoNames.Count > 0)
    nMapObjNames = wmoNames.Count + 1;  // Add 1 for trailing empty
else
    nMapObjNames = 0;
```

**Why this matters**: If the count is wrong, the client will read garbage data when trying to access WMO definitions, causing a crash in `CMap::CreateMapObjDef`.

---

## Per-Tile Structure

Each tile in the MAIN grid that has `offset != 0` must have embedded data:

```
MHDR (64 bytes)
  - Tile header with offsets to sub-chunks
  
MCIN (4096 bytes)
  - Array of 256 MCNK offset/size entries
  
MTEX (variable)
  - Texture filenames (null-terminated strings)
  
MDDF (variable)
  - M2 doodad placements (usually empty)
  
MODF (variable)
  - WMO placements (usually empty)
  
[256 MCNK chunks]
  - Terrain data for each 16x16 grid of chunks
```

### MHDR Structure (64 bytes)

```c
struct MHDR {
    uint32 offsInfo;    // Offset to MCIN (relative to MHDR.data start)
    uint32 offsTex;     // Offset to MTEX
    uint32 sizeTex;     // Size of MTEX data
    uint32 offsDoo;     // Offset to MDDF
    uint32 sizeDoo;     // Size of MDDF data
    uint32 offsMob;     // Offset to MODF
    uint32 sizeMob;     // Size of MODF data
    uint32 pad[7];      // Padding, write as zeros
};
```

**CRITICAL**: All offsets are **relative to MHDR.data start** (8 bytes after MHDR letters).

**BUG FIX**: `offsInfo` was missing in initial implementation. Must point to MCIN!

---

## MCNK Structure

Each MCNK chunk represents a 33.33x33.33 unit terrain patch.

### MCNK Header (128 bytes)

```c
struct McnkAlphaHeader {
    uint32 flags;
    uint32 indexX;          // Chunk X coordinate (0-15)
    uint32 indexY;          // Chunk Y coordinate (0-15)
    float  radius;          // Bounding sphere radius (CRITICAL!)
    uint32 nLayers;         // Number of texture layers
    uint32 nDoodadRefs;     // Number of M2 references
    uint32 mcvtOffset;      // Offset to MCVT (usually 0 = immediately after header)
    uint32 mcnrOffset;      // Offset to MCNR
    uint32 mclyOffset;      // Offset to MCLY
    uint32 mcrfOffset;      // Offset to MCRF
    uint32 mcalOffset;      // Offset to MCAL
    uint32 mcalSize;        // Size of MCAL data
    uint32 mcshOffset;      // Offset to MCSH
    uint32 mcshSize;        // Size of MCSH data
    uint32 areaId;          // Area ID (zone + subzone)
    uint32 nMapObjRefs;     // Number of WMO references
    uint16 holes;           // Hole bitmap
    uint16 predTex[8];      // Predicted textures (unused)
    uint64 noEffectDoodad;  // Doodad effect flags
    uint32 mcseOffset;      // Offset to MCSE
    uint32 nSndEmitters;    // Number of sound emitters
    uint32 mclqOffset;      // Offset to MCLQ (CRITICAL!)
    uint32 unused[6];       // Padding
};
```

### CRITICAL MCNK Fields

#### 1. Radius (float)

**BUG**: Was hardcoded to 0, causing potential culling issues.

**FIX**: Calculate from MCVT height data:

```
1. Find min/max heights from 145 MCVT floats
2. Calculate height range = max - min
3. Horizontal radius ≈ 23.57 (chunk diagonal / 2)
4. radius = sqrt(horizontalRadius² + (heightRange/2)²)
```

Typical value: ~23.57 for flat terrain

#### 2. MclqOffset (uint32)

**BUG**: Was hardcoded to 0, causing ACCESS_VIOLATION when client tried to read liquid data.

**FIX**: Point to end of sub-chunks:

```
mclqOffset = mcvtSize + mcnrSize + mclySize + mcrfSize + mcshSize + mcalSize
```

For chunks without liquids, this points to where liquid data *would* be (after all sub-chunks).

### MCNK Sub-Chunks

```
MCVT (580 bytes)
  - 145 floats: height values for 9x9 outer + 8x8 inner grid
  - NO chunk header, raw data only
  
MCNR (448 bytes)
  - 145 normals (3 bytes each) + 13 bytes padding
  - NO chunk header, raw data only
  
MCLY (16 bytes per layer)
  - Texture layer definitions
  - HAS chunk header (letters + size)
  
MCRF (variable)
  - M2/WMO reference indices
  - HAS chunk header (usually empty)
  
MCSH (variable)
  - Shadow map
  - HAS chunk header (usually empty)
  
MCAL (variable)
  - Alpha maps for texture blending
  - **NO chunk header in Alpha output**; write raw 4-bit maps back-to-back
```

**CRITICAL**: MCVT and MCNR have NO chunk headers, just raw data!

---

## Conversion Algorithm

### Step 1: Read Source Data

```
1. Read LK WDT MAIN chunk to find which tiles exist
2. Read WMO names from LK WDT MWMO chunk
3. For each tile:
   a. Read root ADT (map_xx_yy.adt)
   b. Extract MCNK terrain data
   c. Convert MCVT heights (LK uses relative, Alpha uses absolute)
```

### Step 2: Build Top-Level Structure

```
1. Write MVER (version = 18)
2. Write MPHD placeholder (will patch later)
3. Write MAIN placeholder (will patch later)
4. Write MDNM (usually empty)
5. Write MONM with WMO names from source
6. Write MODF (empty but must exist)
7. Remember offsets for patching
```

### Step 3: Write Tile Data

```
For each tile in MAIN:
  1. Write MHDR placeholder
  2. Write MCIN (256 entries)
  3. Write MTEX (texture names)
  4. Write MDDF (empty)
  5. Write MODF (empty)
  6. For each of 256 MCNKs:
     a. Build MCNK header with correct offsets
     b. Write MCVT (raw 580 bytes)
     c. Write MCNR (raw 448 bytes)
     d. Write MCLY (with chunk header)
     e. Write MCRF (empty, with header)
     f. Write MCSH (empty, with header)
     g. Write MCAL (empty, with header)
  7. Patch MHDR with actual offsets
  8. Patch MCIN with actual MCNK offsets
```

### Step 4: Patch Headers

```
1. Patch MPHD:
   - Set nMapObjNames (count + 1 if any names)
   - Set offsets to MDNM/MONM
   
2. Patch MAIN:
   - For each tile, set offset to MHDR letters
   - Set size = distance from MHDR letters to first MCNK
```

---

## Common Pitfalls

### 1. ❌ Wrong MPHD.nMapObjNames Count

**Symptom**: Crash in `CMap::CreateMapObjDef` with error `index (0x4D484452)`

**Cause**: Count doesn't match client's split-by-null logic

**Fix**: Add 1 to count if any WMO names exist

### 2. ❌ Missing MHDR.offsInfo

**Symptom**: Client can't find MCIN chunk

**Cause**: offsInfo field not set

**Fix**: Set to 0 (MCIN immediately follows MHDR.data)

### 3. ❌ MCNK.Radius = 0

**Symptom**: Terrain may not render or cull incorrectly

**Cause**: Radius not calculated

**Fix**: Calculate from MCVT heights

### 4. ❌ MCNK.MclqOffset = 0

**Symptom**: ACCESS_VIOLATION when client reads liquid data

**Cause**: Offset points to start of file instead of end of sub-chunks

**Fix**: Set to total size of all sub-chunks

### 5. ❌ MAIN.size is Total Tile Size

**Symptom**: Client reads wrong data

**Cause**: size field should be MHDR-to-first-MCNK, not total tile size

**Fix**: Calculate distance from MHDR letters to first MCNK letters

### 6. ❌ Empty MONM Chunk

**Symptom**: Crash when client tries to read WMO definitions

**Cause**: Not reading WMO names from source

**Fix**: Extract MWMO data from source LK WDT

### 7. ❌ MDNM/MONM with Zero Bytes When Empty

**Symptom**: `iffChunk.token=='MONM'` assertion failure in `CMap::LoadMapObjNames`

**Cause**: When no M2/WMO names exist, MDNM/MONM chunks written with 0 bytes of data

**Fix**: Always write at least a single null byte (0x00) as the string list terminator, even when no names exist:
```csharp
if (names.Count == 0)
    return new byte[] { 0 };  // Trailing null terminator
```

### 8. ❌ Extra MODF Chunk or Padding After MONM

**Symptom**: `iffChunk.token=='MONM'` assertion failure

**Cause**: Writing MODF chunk or padding bytes after MONM - original Alpha format has neither

**Fix**: Tile data (MHDR) starts immediately after MONM with NO intervening chunks or padding:
```
MVER -> MPHD -> MAIN -> MDNM -> MONM -> [Tile Data (MHDR...)]
```

---

## Verification Checklist

✅ MVER version = 18  
✅ MPHD offsets point to correct chunks  
✅ MPHD.nMapObjNames uses split-by-null counting  
✅ MONM contains actual WMO names from source  
✅ MDNM/MONM have at least 1 byte (trailing null) even when empty  
✅ NO padding bytes between MDNM and MONM chunks  
✅ NO MODF chunk after MONM - tile data starts immediately  
✅ MAIN entries have correct offsets and sizes  
✅ MHDR.offsInfo points to MCIN  
✅ MHDR offsets are relative to MHDR.data  
✅ MCNK.radius calculated from heights  
✅ MCNK.mclqOffset points to end of sub-chunks  
✅ MCVT/MCNR have NO chunk headers  
✅ MCLY/MCRF/MCSH/MCAL HAVE chunk headers  
✅ All FourCC tokens reversed on disk  

---

## Testing

### Minimal Test Case

1. Convert a small map (e.g., RazorfenDowns with ~24 tiles)
2. Load in 0.5.3 Alpha client
3. Verify:
   - No crash on load
   - Terrain renders
   - Can move around
   - No ACCESS_VIOLATION errors

### Byte-by-Byte Comparison

Compare first 100KB of converted file with real Alpha WDT:
- MPHD should match exactly
- MAIN structure should match (offsets may differ if tile layout changed)
- MONM should contain same WMO names

---

## Implementation Notes

### C++ Implementation

```cpp
// WMO name counting (critical!)
int nMapObjNames = wmoNames.empty() ? 0 : wmoNames.size() + 1;

// Radius calculation
float CalculateRadius(const float* mcvtHeights, int count) {
    float minH = FLT_MAX, maxH = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        minH = std::min(minH, mcvtHeights[i]);
        maxH = std::max(maxH, mcvtHeights[i]);
    }
    float heightRange = maxH - minH;
    float horizontalRadius = 23.57f;
    return std::sqrt(horizontalRadius * horizontalRadius + 
                     (heightRange / 2.0f) * (heightRange / 2.0f));
}
```

### Python Implementation

```python
# WMO name counting
n_map_obj_names = len(wmo_names) + 1 if wmo_names else 0

# Radius calculation
def calculate_radius(mcvt_heights):
    min_h = min(mcvt_heights)
    max_h = max(mcvt_heights)
    height_range = max_h - min_h
    horizontal_radius = 23.57
    return math.sqrt(horizontal_radius**2 + (height_range/2)**2)
```

---

## References

- **Alpha.md**: WoWDev wiki documentation for Alpha format
- **ADT_v18.md**: Detailed ADT chunk specifications
- **CRITICAL-Next-Fix.md**: Debugging session that found the WMO name bug

---

## Version History

- **2025-10-15**: Initial specification based on successful C# implementation
- Verified working in 0.5.3 Alpha client (build 3368)
- All critical bugs identified and fixed
