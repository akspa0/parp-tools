# Task 1: ADT Loading — Confirm Unified Format

## Summary

**Status**: ✅ **CONFIRMED** - This build uses unified ADT format (no split files)

## Findings

### ADT Filename Format

**Function Address**: `0x006a13b0` (`FUN_006a13b0`)

The ADT loading uses the unified format string:
```
"%s\%s_%d_%d.adt"
```

**Location**: String at `0x00a23e94`

**Referenced from**: `0x006a13d0` in `FUN_006a13b0`

### No Split ADT Support

**Search Results**:
- `_tex0.adt` strings: **NOT FOUND**
- `_obj0.adt` strings: **NOT FOUND**
- `MAID` chunk strings: **NOT FOUND**

**Conclusion**: The Cataclysm Alpha 4.0.0.11927 build does NOT contain any code for loading split ADT files (`_tex0`/`_obj0`). The split ADT feature was added in later Cataclysm builds.

### WDT Loading Analysis

**Function Address**: `0x0069c590` - WDT initialization

**WDT Path Format**: `World\Maps\%s\%s.wdt`

**WDT Chunk Handling** (from `0x0069b410`):
- Checks for chunk `0x4d574d4f` ("MWMO" = Map WMO)
- Checks for chunk `0x4d41484f` (likely "MAHO" or MAIN variant)
- **NO MAID chunk handling found** - confirms split ADT support doesn't exist

### WDT MAIN Structure

The code iterates through a 64x64 grid (0x40 x 0x40), checking entries in what appears to be the MAIN chunk. The iteration pattern confirms **row-major** order (same as 3.3.5):

```c
// Outer loop: row (Y coordinate)
for (row = 0; row < 64; row++) {
    // Inner loop: column (X coordinate)  
    for (col = 0; col < 64; col++) {
        // Check if tile exists at (col, row)
    }
}
```

## Comparison with 3.3.5

| Aspect | 3.3.5 | 4.0.0.11927 | Status |
|--------|-------|-------------|--------|
| ADT Format | Unified | Unified | **Unchanged** |
| File Extension | `.adt` | `.adt` | **Unchanged** |
| Split ADT Support | No | No | **Unchanged** |
| WDT MAIN Order | Row-major | Row-major | **Unchanged** |
| MAID Chunk | No | No | **Unchanged** |

## Confidence Level

**HIGH** - No references to split ADT files found in binary. Only unified ADT format string present.

## Notes

- This is consistent with the prompt's statement that 4.0.0.11927 uses "3.3.5-style data file formats"
- Split ADTs were introduced in later Cataclysm builds (4.0.3+ approximately)
- The MAID chunk (for file data IDs in WDT) was also added in later builds
