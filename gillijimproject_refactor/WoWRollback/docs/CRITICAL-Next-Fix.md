# CRITICAL FIX: WMO Name Data (RESOLVED ✅)

## Original Crash

```
ERROR #128 (0x85100080)
index (0x4D484452) = 'RDHM'
array size (0x00000000) = 0

CMap::CreateMapObjDef
TSBaseArray::CheckArrayBounds
```

## What This Means

The client is trying to:
1. Read MODF (WMO definitions)
2. Use an offset to index into MONM string table
3. Getting 'RDHM' (MHDR token) instead of a valid offset
4. Crashing because array size is 0

## Root Cause

The client is reading from the **wrong location** when looking for WMO data.

This is happening at the **top-level WDT structure**, not inside tiles.

## What to Check

### 1. MPHD Structure
```
Real Alpha:
- nDoodadNames: 0
- offsDoodadNames: 65692 → MDNM
- nMapObjNames: 0
- offsMapObjNames: 65700 → MONM

Our Packed:
- Need to verify these are identical
```

### 2. Top-Level Chunk Order
```
Real Alpha:
MVER @ 0
MPHD @ 12
MAIN @ 148
MDNM @ 65692 (empty, size=0)
MONM @ 65700 (size=57)
MODF @ 65766 (empty, size=0)
[End marker] @ 65774
[First tile MHDR] @ 65837

Our Packed:
Need to verify exact same structure
```

### 3. MONM Content
Real Alpha has 57 bytes of MONM data.
Our packed should have 0 bytes (empty).

**If MONM is empty but client tries to read it, crash!**

## Hypothesis

The client is reading MPHD.offsMapObjNames, jumping to MONM, and trying to parse WMO names.

If MONM is empty (size=0) but the client expects data, it might read past the chunk into the next chunk (MODF or first tile MHDR), getting 'RDHM'.

## Root Cause (FOUND ✅)

**Missing WMO name data in MPHD/MONM chunks**

The converter was:
1. Writing `nMapObjNames = 0` in MPHD (hardcoded)
2. Writing empty MONM chunk (no WMO names)
3. Not reading WMO names from source LK WDT

The Alpha client **requires** valid WMO name data even for terrain-only conversions. When it tried to read WMO definitions, it got garbage data and crashed.

## The Fix (IMPLEMENTED ✅)

### 1. Read WMO Names from Source
**File**: `LkWdtReader.cs`

Added `ReadWmoNames()` method to extract WMO names from LK WDT MWMO chunk:
```csharp
public List<string> ReadWmoNames(string wdtPath)
{
    // Find MWMO chunk (reversed as 'OMWM' on disk)
    // Parse null-terminated strings
    // Return list of WMO paths
}
```

### 2. Write WMO Names to MONM
**File**: `AlphaWdtMonolithicWriter.cs`

```csharp
// Read WMO names from source
var wmoNames = wdtReader.ReadWmoNames(lkWdtPath);

// Build MONM data (null-terminated strings)
byte[] monmData = BuildMonmData(wmoNames);
var monm = new Chunk("MONM", monmData.Length, monmData);
ms.Write(monm.GetWholeChunk());
```

### 3. Fix nMapObjNames Count
**Critical Detail**: Alpha client counts by **splitting on null terminators**, which includes the trailing empty string!

Example: `"name\0"` splits to `["name", ""]` = **2 parts**

```csharp
// Alpha client counts split parts, not just names
// "name\0" = 2 parts (name + empty string after null)
int wmoCount = wmoNames.Count > 0 ? wmoNames.Count + 1 : 0;
BitConverter.GetBytes(wmoCount).CopyTo(mphdData.Slice(8));
```

## Verification

**Byte-by-byte comparison** with real 0.5.3 Alpha:

**Before Fix:**
```
Offset 0x1C (MPHD.nMapObjNames):
  Reference: 02 00 00 00  (2)
  Our File:  00 00 00 00  (0)  ❌
```

**After Fix:**
```
Offset 0x1C (MPHD.nMapObjNames):
  Reference: 02 00 00 00  (2)
  Our File:  02 00 00 00  (2)  ✅
```

## Result

✅ **Client loads without crashing**  
✅ **Terrain displays correctly**  
✅ **No more ACCESS_VIOLATION**

## Additional Fixes in Same Session

1. **MHDR.offsInfo** - Was missing, now points to MCIN
2. **MCNK.Radius** - Was 0, now calculated from MCVT heights
3. **MCNK.MclqOffset** - Was 0, now points to end of sub-chunks

All fixes verified working in 0.5.3 Alpha client.
