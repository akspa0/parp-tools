# Next Session TODO - MCNK Sub-Chunk Data Fix

## Current Status (Session 2025-10-15 End)

### ✅ WDT Structure - COMPLETELY FIXED
- ✅ MAIN chunk properly patched with tile offsets
- ✅ MHDR offsets use correct Alpha convention (point to FourCC, not data)
- ✅ MDNM/MONM/MODF written as empty chunks (client expects them)
- ✅ All validation passes
- ✅ RazorfenDowns loads successfully in client

### ✅ MCNR Extraction - IMPLEMENTED BUT INSUFFICIENT
- ✅ MCNR chunk now extracted from LK MCNK
- ✅ Vertex reordering implemented (interleaved → outer-first)
- ✅ Proper 13-byte padding added
- ❌ **Still crashes** - Problem is elsewhere

### ❌ ROOT CAUSE FOUND - Missing Sub-Chunk Data
**Error:** `ERROR #132 (0x85100084) index (0x4D484452), array size (0x00000000)`

**Comparison Analysis (Real vs Converted Kalidar):**

Real Alpha 0.5.3 Kalidar:
- File size: **32.8 MB**
- MDNM: **1,141 bytes** (M2 doodad names)
- MONM: **Large** (WMO building names)
- MTEX: **8 bytes** per tile
- MCNK: **Varying sizes** (different sub-chunk content)

Our Converted Kalidar:
- File size: **17.3 MB** (47% smaller!)
- MDNM: **0 bytes** (empty)
- MONM: **0 bytes** (empty)
- MTEX: **29 bytes** per tile
- MCNK: **1204 bytes constant** (minimal/uniform)

### Missing ~15.5 MB of Data
We're only writing minimal MCNKs with MCVT + MCNR. Missing:
- ❌ Texture layer data (MCLY with real layer info)
- ❌ Alpha map data (MCAL for texture blending)
- ❌ Shadow data (MCSH)
- ❌ Possibly other sub-chunks the client expects

---

## 🎯 Action Plan for Next Session

### ✅ COMPLETED: Extract MCLY (Texture Layers) from LK

✅ Implemented in AlphaMcnkBuilder.cs:
1. ✅ Extract MCLY chunk from LK MCNK (lines 51-55)
2. ✅ Use extracted data directly (no conversion needed - same format)
3. ✅ Calculate NLayers from actual MCLY data (16 bytes per layer)
4. ✅ Fallback to minimal single-layer if MCLY missing

### ✅ COMPLETED: Extract MCAL (Alpha Maps) from LK

✅ Implemented in AlphaMcnkBuilder.cs:
1. ✅ Extract MCAL chunk from LK MCNK (lines 56-60)
2. ✅ Use extracted data directly (same format)
3. ✅ Update McalSize in header to match actual data
4. ✅ Fallback to empty if MCAL missing

### ✅ COMPLETED: Extract MCSH (Shadows) from LK

✅ Implemented in AlphaMcnkBuilder.cs:
1. ✅ Extract MCSH chunk from LK MCNK (lines 61-65)
2. ✅ Use extracted data directly (same format)
3. ✅ Update McshSize in header to match actual data
4. ✅ Fallback to empty if MCSH missing

### ❌ CRITICAL DISCOVERY: LK Uses Split ADT Files!

**Problem Found:** LK 3.3.5 splits ADT data across multiple files:
- `Map_XX_YY.adt` = Terrain geometry (MCVT, MCNR only)
- `Map_XX_YY_tex0.adt` = Textures (MCLY, MCAL, MCSH) ← **WE NEED THIS!**
- `Map_XX_YY_obj0.adt` = Objects (WMO/M2 placements)

Alpha 0.5.3 has everything in ONE monolithic file.

**Current Status:**
- ✅ Code extracts MCLY/MCAL/MCSH from terrain ADT
- ❌ But LK terrain ADTs don't contain texture data!
- ❌ File size same as before (~17 MB) - no texture data included

### 🎯 NEXT: Load _tex0.adt Files

Need to update `AlphaWdtMonolithicWriter.cs`:
1. ⏳ Find corresponding `_tex0.adt` file for each terrain tile
2. ⏳ Load both terrain and texture ADT bytes
3. ⏳ Pass texture bytes to `AlphaMcnkBuilder.BuildFromLk()`
4. ⏳ Extract MCLY/MCAL/MCSH from texture file
5. ⏳ Verify file size increases to ~32 MB

### Alternative Approach: Copy Real Alpha MCNK Sub-Chunks

Since we have working Alpha files, we could:
1. Parse real Alpha MCNK structure completely
2. Use it as reference for what data to extract from LK
3. Ensure we're not missing any sub-chunks

---

## 📊 Key Discoveries

### Alpha Version Comparison (0.5.3 vs 0.5.5)
- ✅ **DeadminesInstance:** IDENTICAL (format stable for instances)
- ❌ **Azeroth:** DIFFERENT (+87MB, outdoor maps evolved)
- **Conclusion:** Alpha format varied between versions for outdoor maps

### Real vs Converted Comparison
- **File size:** 32.8 MB → 17.3 MB (47% data loss)
- **MCNK uniformity:** Real varies, ours constant
- **Sub-chunks:** Real has MCLY/MCAL/MCSH data, we write empty/minimal

---

## 🛠️ Technical Notes

### MCNR Implementation (DONE)
Reordering logic added in `AlphaMcnkBuilder.cs` line 271:
```csharp
private static byte[] ConvertMcnrLkToAlpha(byte[] mcnrLk)
{
    // Reorders from LK interleaved to Alpha outer-first format
    // 81 outer normals (9x9) then 64 inner (8x8) + 13 pad
    // Total: 448 bytes
}

---

## ✅ What Was Accomplished - Session 2025-10-16

### 🎯 BREAKTHROUGH: Header-Based Chunk Extraction Implemented

**Root Cause Identified:**
- LK 3.3.5 stores MCLY/MCAL/MCSH via **header offsets** in MCNK
- Previous code only scanned sequential sub-chunks (MCVT/MCNR)
- Missing 15.5 MB of texture data (MCLY, MCAL, MCSH)

**Changes to `AlphaMcnkBuilder.cs`:**

1. **Added header-based extraction** (lines 37-77):
   - Extract MCLY using `lkHeader.MclyOffset` (header+0x14)
   - Extract MCAL using `lkHeader.McalOffset` (header+0x18)
   - Extract MCSH using `lkHeader.McshOffset` (header+0x1C)
   - Read chunks from absolute positions within MCNK

2. **Kept sequential scanning** for MCVT/MCNR (lines 79-111):
   - These chunks are still in the sub-chunk area
   - Scanned sequentially as before

3. **Removed incorrect _tex0.adt loading**:
   - LK 3.3.5 uses monolithic ADT files (not split)
   - All data is in main ADT, accessed differently

**Results:**
- ✅ File size: 40.96 MB (147% increase from 16.56 MB)
- ✅ MCLY verified: 16 bytes (texture layer definitions)
- ✅ MCAL verified: 1,052 bytes (alpha maps)
- ✅ MCSH: 0 bytes (may not be present in all tiles)
- ✅ Build succeeds with 6 warnings
- ⏳ **Ready for client testing!**

---

## ✅ What Was Accomplished - Session 2025-10-15

1. ✅ Implemented MCNR extraction from LK MCNK
2. ✅ Added vertex reordering (interleaved → outer-first)  
3. ✅ Tested on Kalidar - still crashes
4. ✅ Compared Alpha versions (0.5.3 vs 0.5.5)
5. ✅ **Found root cause:** Missing 15.5MB of sub-chunk data
6. ✅ Identified that MCLY/MCAL/MCSH need extraction

---

## ✅ Success Criteria for Next Session

1. ⏳ Extract MCLY (texture layers) from LK
2. ⏳ Extract MCAL (alpha maps) from LK
3. ⏳ Extract MCSH (shadows) from LK if present
4. ⏳ Kalidar file size increases from 17MB → 32MB
5. ⏳ Kalidar loads in Alpha 0.5.3 client without crash

---

## 📚 Key References

### Documentation
- **Alpha format spec:** `z_wowdev.wiki/Alpha.md` lines 101-163
- **MCNK structure:** Line 106 shows header format
- **MCNR details:** Lines 119-127 (no header, 145×3 + 13 pad)
- **Vertex ordering:** Line 113 explains outer-first, then inner

### Code Files
- **Main builder:** `AlphaMcnkBuilder.cs`
- **WDT writer:** `AlphaWdtMonolithicWriter.cs`
- **Validation:** `AlphaWdtInspector.cs`

### Tools
- `inspect-alpha` - Deep WDT inspection  
- `compare-alpha` - Byte-level comparison
- `pack-monolithic` - Convert LK→Alpha WDT

### Test Data Locations
- Real Alpha 0.5.3: `test_data\0.5.3\tree\World\Maps\`
- Real Alpha 0.5.5: `test_data\0.5.5\tree\World\Maps\`
- LK 3.3.5 Source: `test_data\0.6.0\tree\World\Maps\`
