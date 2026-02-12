# Next Steps for Debugging Client Crash

## Current Status

✅ **Fixed**: MHDR.offsInfo field is now written (was missing)
❌ **Still Crashing**: Client now crashes with ACCESS_VIOLATION

## Error Analysis

### Error 1
```
index (0x4D484452) = 'RDHM' 
array size (0x00000000) = 0
```

### Error 2
```
ACCESS_VIOLATION at 0x006815CB
Referenced memory at 0x62F11148
Memory could not be read
```

## What We've Verified

✅ MAIN.offset points to MHDR letters (correct)
✅ MAIN.size = distance to first MCNK (correct)
✅ MHDR.offsInfo = 64 (now fixed)
✅ MHDR.offsTex, offsDoo, offsMob calculated correctly
✅ MCIN offsets are absolute file positions (correct)
✅ Chunk order is correct (MHDR, MCIN, MTEX, MDDF, MODF, MCNKs)
✅ All chunks tightly packed (no gaps)

## Possible Remaining Issues

### 1. MCNK Header Fields
The client might be reading MCNK header fields and finding unexpected values.

**To Check**:
- Are IndexX/IndexY correct?
- Are sub-chunk offsets (MCVT, MCNR, etc.) correct?
- Is the AreaID field correct?
- Are flags correct?

### 2. Sub-Chunk Data
Alpha MCNKs have unnamed sub-chunks (no FourCC headers).

**To Check**:
- Is MCVT data in correct Alpha format (outer 81, then inner 64)?
- Are heights absolute (not relative)?
- Is MCNR in correct format?
- Are MCLY/MCRF/MCAL/MCSH present and correct?

### 3. Byte Order / Alignment
**To Check**:
- Are all multi-byte values in little-endian?
- Is there proper padding after odd-sized chunks?
- Are chunk sizes including padding correctly?

### 4. MCIN Size Field
The MCIN size field should be the **whole MCNK chunk** size.

**Current**: We write `alphaMcnkBytes[i].Length`
**Verify**: This includes letters + size field + data + padding

### 5. Compare Byte-for-Byte
**Action Needed**: 
1. Export first MCNK from real RazorfenDowns
2. Export first MCNK from our packed Kalidar
3. Hex compare them field-by-field

## Recommended Next Steps

### Step 1: Detailed MCNK Inspection
Add to inspector to dump first MCNK header fields:
- All 128 bytes of header
- Sub-chunk offsets
- Verify they point to correct data

### Step 2: Hex Dump Comparison
```powershell
# Extract first tile from real Alpha WDT
# Extract first tile from our packed WDT
# Compare hex dumps
```

### Step 3: Add More Validation
Check in our code:
- MCNK IndexX/IndexY match tile position
- Sub-chunk offsets are relative to header end
- All offsets are within bounds

### Step 4: Test with Minimal Data
Try packing a single-tile WDT with:
- Flat terrain (all heights = 0)
- Single texture
- No doodads/WMOs
- Minimal MCNK data

This will help isolate whether the issue is with:
- WDT structure (unlikely now)
- MCNK structure (likely)
- MCNK data (possible)

## Questions to Answer

1. **Does the client crash immediately on map load, or after some processing?**
   - Immediate = WDT/MAIN/MHDR issue
   - Delayed = MCNK data issue

2. **Can we get a more detailed crash dump?**
   - Memory address being accessed
   - Call stack
   - Register values

3. **Does a minimal test case crash the same way?**
   - Single tile
   - Flat terrain
   - No objects

## Tools Needed

1. **Hex editor** to compare real vs packed WDTs
2. **Debugger** attached to client (if possible)
3. **Enhanced inspector** to dump MCNK headers in detail
