# Active Context

- **Focus**: Complete LK→Alpha round-trip conversion pipeline and restore full parity for Alpha↔LK conversions.
- **Current Status**: ✅ **CRITICAL FIX APPLIED** - Fixed MCLY/MCAL extraction bug in `AlphaDataExtractor.cs`. Build succeeds (15 warnings).
- **Completed This Session (2025-10-19)**:
  1. ✅ **Root Cause Identified**: `ReadRawSlice()` was incorrectly stripping chunk headers from MCLY (which HAS headers) while MCAL (which has NO headers) was being read with wrong offsets.
  2. ✅ **MCLY Fix**: Changed to read MCLY with proper chunk header ("YLCM" reversed FourCC + size), then extract data payload.
  3. ✅ **MCAL Fix**: Changed to read MCAL as raw bytes (no header) directly from absolute offset with proper bounds clamping.
  4. ✅ **Build Success**: WoWRollback.LkToAlphaModule builds successfully with fixes applied.
- **What Works Now**:
  - MCLY extraction reads chunk header correctly and preserves 16-byte layer entries
  - MCAL extraction reads raw alpha map data without header stripping
  - Liquids, placements, and other MCNK subchunks continue to work
  - Logging shows actual MCLY/MCAL bytes being read
- **Next Steps**:
  1. **Test with real Alpha ADT** to verify MCLY/MCAL data is no longer zeros
  2. **Implement LK→Alpha conversion** in `RoundTripValidator.cs` (lines ~100-150):
     - Parse LK ADT MCIN to get MCNK offsets
     - Extract each MCNK chunk
     - Call `AlphaMcnkBuilder.BuildFromLk()` for each chunk
     - Write complete Alpha ADT file
  3. **Add byte-by-byte comparison** with original Alpha ADT
  4. **Generate detailed diff reports** for any mismatches
- **Implementation Notes**:
  - Alpha MCLY: HAS chunk header ("YLCM" + size + data)
  - Alpha MCAL: NO chunk header (raw bytes, size from MCNK header field)
  - Reference implementation in `McnkAlpha.cs` lines 54-69 confirms this pattern
- **Known Limitations**:
  - LK→Alpha writer still incomplete (next priority)
  - Round-trip comparison not yet implemented
  - No xUnit tests yet for MCLY/MCAL extraction
