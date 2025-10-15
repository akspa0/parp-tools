# Alpha WDT Specification - Verified Against Real Data

**Source**: Analysis of real Alpha 0.5.3.3368 WDT files (RazorfenDowns.wdt)
**Date**: 2025-10-15
**Status**: Verified with inspector tool against authentic client data

This document provides a verified specification of the Alpha WDT format based on direct analysis of real game files.

## Critical Verified Findings

### MAIN Chunk Behavior

**VERIFIED**: MAIN.offset points to MHDR **letters** (FourCC), not data

From RazorfenDowns tile 26_27:
- offset = 65837 (0x1010D) points to 'RDHM'
- size = 4200 (0x1068)
- offset + size = 70037 points to 'KNCM' (first MCNK)

### MHDR Field Values

**VERIFIED**: Consistent across all tiles in RazorfenDowns

```
offsInfo = 64    (0x40)   - MCIN immediately follows MHDR.data
offsTex  = 4168  (0x1048) - MTEX after MCIN
sizeTex  = 8              - Minimal texture table
offsDoo  = 4176  (0x1050) - MDDF after MTEX
sizeDoo  = 0              - Empty
offsMob  = 4184  (0x1058) - MODF after MDDF
sizeMob  = 0              - Empty
```

All offsets are relative to MHDR.data start (MHDR letters + 8 bytes).

### Chunk Layout

Verified from tile 26_27:

```
MHDR letters: 0x1010D (65837)
MHDR.data:    0x10115 (65845) = letters + 8
MCIN:         0x10155 (65909) = data + 64
MTEX:         0x1118D (70013) = data + 4168
MDDF:         0x11195 (70021) = data + 4176
MODF:         0x1119D (70029) = data + 4184
First MCNK:   0x111A5 (70037) = data + 4192
```

### Gap Analysis

All chunks are tightly packed with ZERO gaps:
- MHDR to MCIN: 0 bytes
- MCIN to MTEX: 0 bytes
- MTEX to MDDF: 0 bytes
- MDDF to MODF: 0 bytes
- MODF to first MCNK: 0 bytes

## Implementation Requirements

For correct Alpha WDT writing:

1. MAIN.offset = absolute position of MHDR letters
2. MAIN.size = (first MCNK absolute) - (MHDR letters absolute)
3. MHDR.offsInfo = 64 (fixed)
4. MHDR.offsTex = 64 + 8 + 4096 = 4168 (for minimal MTEX)
5. All chunks must be tightly packed (no gaps)
6. Chunk order: MHDR, MCIN, MTEX, MDDF, MODF, MCNKs

## Differences from wowdev.wiki

### Corrections

1. MAIN.offset behavior: VERIFIED points to letters, not data
2. MHDR offset base: VERIFIED all offsets relative to MHDR.data
3. MHDR.offsInfo value: VERIFIED always 64
4. Chunk packing: VERIFIED zero gaps between chunks

### New Information

- Concrete offset calculations with real values
- Gap analysis showing tight packing
- Absolute file positions for reference
- MHDR field consistency across tiles
