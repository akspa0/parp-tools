# Session 2025-10-16 Summary: MCLY/MCSH Extraction Success (But Client Still Crashes)

## 🎯 Objective
Fix the Alpha 0.5.3 client crash by extracting missing sub-chunk data (MCLY, MCAL, MCSH) from LK ADT files.

## ✅ Major Accomplishment: File Size Increased 147%!

### Before
- **File size**: 16.56 MB (17,373,710 bytes)
- **Missing data**: MCLY (texture layers), MCAL (alpha maps), MCSH (shadows)
- **MCNK chunks**: Uniform size (~1204 bytes each)

### After
- **File size**: 40.96 MB (42,952,206 bytes) ✅
- **Increase**: +24.4 MB (+147%)
- **MCNK chunks**: Varying sizes (correct!)
- **Extracted data**: MCLY (24 bytes), MCSH (520 bytes), MCVT (588 bytes), MCNR (444 bytes)

## 🔧 Technical Implementation

### Key Discovery: LK Stores Chunks via Header Offsets
The breakthrough was realizing that in LK 3.3.5:
- MCVT and MCNR are scanned sequentially in the sub-chunk area
- **MCLY, MCAL, MCSH are accessed via offsets in the MCNK header**
  - `MclyOffset` (0x49C in test data)
  - `McalOffset` (0x6C4 in test data)
  - `McshOffset` (0x4BC in test data)

### Code Changes in `AlphaMcnkBuilder.cs`

1. **Added header-based extraction** (lines 37-85):
```csharp
// Extract MCLY, MCAL, MCSH using header offsets
if (lkHeader.MclyOffset > 0)
{
    int mclyPos = mcNkOffset + lkHeader.MclyOffset;
    // Read chunk from absolute position
}
```

2. **Removed sequential scanning** for MCLY/MCAL/MCSH (they're not in the sub-chunk area)

3. **Kept sequential scanning** for MCVT/MCNR (they are in the sub-chunk area)

### Debug Process
- Added extensive logging to track chunk extraction
- Discovered FourCC reversal issue (on-disk is reversed, we read as ASCII)
- Found that Select-String was filtering out non-prefixed debug lines
- Confirmed extraction working: MCLY (24 bytes), MCSH (520 bytes) per MCNK

## ❌ Remaining Issue: Client Still Crashes

### Error Details
```
ERROR #132 (0x85100084)
index (0x4D484452) = "MHDR" (reversed)
array size (0x00000000)
```

### Analysis
- The index `0x4D484452` decodes to "MHDR" (reversed FourCC)
- Array size is 0 - client expects MHDR data but finds empty array
- This suggests a structural issue with how offsets are calculated

### Possible Root Causes

1. **Two-Pass Problem**: We might be calculating offsets before knowing final chunk sizes
   - Need to build all data first, then calculate offsets
   - Then patch the headers with correct offsets

2. **Missing MCAL Data**: Debug output showed 0 bytes for alpha maps
   - MCAL might be required even if empty
   - Need to verify MCAL extraction logic

3. **Offset Calculation Error**: 
   - Alpha MCNK header offsets should be relative to MCNK start
   - Need to verify we're not mixing absolute and relative offsets
   - Chunk headers vs data offsets confusion

4. **Chunk Header Confusion**:
   - In Alpha: MCVT/MCNR are raw (no headers)
   - In Alpha: MCLY/MCSH/MCAL have chunk headers
   - Offsets might need to point to data, not chunk start

## 📊 Statistics

- **Build time**: <1 second
- **Pack time**: ~30 seconds for 55 tiles
- **Tiles processed**: 55
- **Total MCNK chunks**: 55 × 256 = 14,080 (but only ~256 have data)
- **Data extracted per MCNK**: ~1576 bytes (MCVT + MCNR + MCLY + MCSH)

## 🎯 Next Steps

1. **Verify offset calculations** in `AlphaMcnkBuilder.cs`
   - Check that all offsets are relative to MCNK start
   - Verify chunk header vs data offset usage

2. **Fix MCAL extraction** (currently showing 0 bytes)
   - Debug why MCAL isn't being extracted
   - Check if MCAL is required even when empty

3. **Compare with real Alpha files**
   - Use `inspect-alpha` on real 0.5.3 Kalidar
   - Compare MCNK structure byte-by-byte
   - Identify what's different

4. **Consider two-pass approach**
   - Build all MCNK data first
   - Calculate offsets based on actual sizes
   - Patch headers with correct offsets

5. **Test with simpler map**
   - Try a smaller map with fewer chunks
   - Easier to debug and compare

## 🔍 Debug Commands Used

```bash
# Build
dotnet build WoWRollback.LkToAlphaModule\WoWRollback.LkToAlphaModule.csproj

# Pack with debug output
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic \
  --lk-dir ..\test_data\0.6.0\tree\World\Maps\Kalidar\ \
  --lk-wdt ..\test_data\0.6.0\tree\World\Maps\Kalidar\Kalidar.wdt \
  --map Kalidar

# Check file size
Get-ChildItem "project_output\Kalidar_*\Kalidar.wdt" | Select-Object Name, Length

# Inspect structure
dotnet run --project WoWRollback.AdtConverter -- inspect-alpha \
  --wdt project_output\Kalidar_20251016_003856\Kalidar.wdt --tiles 3
```

## 📝 Files Modified

1. `AlphaMcnkBuilder.cs`
   - Added MCLY/MCAL/MCSH extraction via header offsets
   - Removed sequential scanning for these chunks
   - Added bounds checking to prevent crashes

2. `AlphaWdtMonolithicWriter.cs`
   - Added error handling for tile processing
   - Improved FourCC validation in FindFourCC

## 💡 Key Learnings

1. **LK chunk storage is hybrid**:
   - Some chunks (MCVT, MCNR) are sequential
   - Others (MCLY, MCAL, MCSH) are accessed via header offsets

2. **FourCC handling is tricky**:
   - On-disk: reversed byte order
   - In memory: forward byte order
   - Reading as ASCII gives reversed string

3. **File size is a good indicator**:
   - 147% increase confirms data is being extracted
   - But size alone doesn't guarantee correctness

4. **Client errors are cryptic**:
   - ERROR #132 with index as FourCC is helpful
   - Array size 0 indicates missing/empty data
   - Need to decode hex values to understand errors

## 🎉 Success Metrics

- ✅ MCLY extraction working (24 bytes per MCNK)
- ✅ MCSH extraction working (520 bytes per MCNK)
- ✅ File size increased 147% (16.56 MB → 40.96 MB)
- ✅ Build succeeds with no errors
- ❌ Client still crashes (but we're much closer!)

The foundation is solid - we're successfully extracting the data. Now we need to fix the structural/offset issues to make the client happy.
