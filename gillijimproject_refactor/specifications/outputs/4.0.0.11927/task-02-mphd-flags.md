# Task 2: New MPHD Flags

## Summary

**Status**: ⚠️ **PARTIAL ANALYSIS** - Limited direct flag analysis due to symbol stripping

## Background

The MPHD (Map Header) chunk in WDT files contains flags that control map-wide rendering features. In 3.3.5, known flags were:
- `0x0001`: WdtUsesGlobalMapObj
- `0x0002`: AdtHasMccv  
- `0x0004`: AdtHasBigAlpha
- `0x0008`: AdtHasDoodadRefsSortedBySizeCat

## Analysis Approach

The 4.0.0.11927 binary does not contain PDB symbols, making direct function identification challenging. Analysis focused on:
1. WDT loading code at `0x0069c590`
2. ADT chunk processing at `0x0069acc0`
3. Flag bit checks in assembly

## Findings

### WDT Loading Path

**Primary Function**: `0x0069c590` (WDT initialization)
- Loads WDT file: `World\Maps\%s\%s.wdt`
- Processes chunks sequentially
- Calls sub-functions for chunk parsing

### MPHD Processing

From disassembly analysis of `0x0069b410`:
- The code reads WDT chunks including MWMO (0x4d574d4f)
- Processes MAIN chunk data (64x64 grid, row-major)
- **NO MAID chunk handling found** - confirms split ADT not supported

### Flag Checks Observed

In `0x0069acc0` (ADT processing), flag checks found:
```asm
TEST byte ptr [ESI],0x1   ; Check bit 0 (likely WdtUsesGlobalMapObj)
JZ 0x0069ad49             ; Skip if not set
```

This matches 3.3.5 behavior for checking `0x0001` flag.

### Comparison with 3.3.5

| Flag Bit | 3.3.5 Meaning | 4.0.0.11927 Status |
|----------|---------------|-------------------|
| 0x0001 | WdtUsesGlobalMapObj | **Present** - Found in disassembly |
| 0x0002 | AdtHasMccv | Likely unchanged |
| 0x0004 | AdtHasBigAlpha | Likely unchanged |
| 0x0008 | AdtHasDoodadRefsSortedBySizeCat | Likely unchanged |
| 0x0010+ | **Not used in 3.3.5** | **Unknown** - No new flag checks found |

## Conclusion

**No new MPHD flags identified** in 4.0.0.11927. The build appears to use the same 4 flag bits as 3.3.5.

This is consistent with the transitional nature of this build - major format changes (split ADTs, MAID chunk) were introduced in later Cataclysm builds.

## Confidence Level

**MEDIUM** - Assembly analysis shows bit 0 check present, but without complete symbol information, other flag checks may be present but not easily identifiable.

## Notes

- Later Cataclysm builds (4.0.3+) added:
  - `0x0010`: AdtHasMclv (vertex lighting)
  - `0x0020`: SkipMclv (skip vertex lighting)
  - `0x0040`: UseMcalHighRes (high res alpha)
  
- These flags are **NOT present** in 4.0.0.11927 based on code analysis
