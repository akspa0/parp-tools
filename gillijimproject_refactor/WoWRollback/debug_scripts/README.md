# Debug Scripts for Alpha WDT Development

This folder contains PowerShell scripts for debugging and verifying Alpha 0.5.3 WDT file generation.

## Scripts

### Verify-AlphaWDT.ps1
Verifies the structure of an Alpha WDT file, checking:
- Top-level chunk order and sizes
- MPHD structure (doodad/WMO name pointers)
- MHDR structure (per-tile headers)
- Chunk offset validity

**Usage:**
```powershell
.\Verify-AlphaWDT.ps1 -WdtPath "project_output\Kalidar_20251016_012513\Kalidar.wdt"
```

### Compare-AlphaWDTs.ps1
Compares two Alpha WDT files to find structural differences:
- File size comparison
- Chunk-by-chunk comparison
- First byte difference location
- Context around differences

**Usage:**
```powershell
.\Compare-AlphaWDTs.ps1 `
    -OriginalPath "test_data\0.5.3\tree\World\Maps\Kalidar\Kalidar.wdt" `
    -GeneratedPath "project_output\Kalidar_20251016_012513\Kalidar.wdt"
```

## Common Issues Found

### Issue 1: MPHD Null Pointers
**Symptom:** Client ERROR #132 with "index (0x4D484452), array size (0x00000000)"
**Cause:** MPHD pointing to empty chunks when count=0
**Fix:** Set offsDoodadNames and offsMapObjNames to 0 when counts are 0

### Issue 2: MHDR Missing sizeInfo
**Symptom:** All MHDR offsets shifted by 4 bytes
**Cause:** Added non-existent sizeInfo field to Alpha MHDR structure
**Fix:** Remove sizeInfo field, Alpha structure is different from LK

### Issue 3: MCNK Offset Order
**Symptom:** Texture data at wrong positions
**Cause:** MCSH/MCAL offsets didn't match write order
**Fix:** Write order must be MCLY → MCRF → MCSH → MCAL

### Issue 4: MHDR Field Positions
**Symptom:** Chunks not found at expected offsets
**Cause:** All offset/size pairs shifted by 4 bytes
**Fix:** Correct field order per Alpha.md documentation

## Alpha Format Reference

### MPHD Structure (128 bytes)
```c
struct SMMapHeader {
    uint32_t nDoodadNames;      // [0x00] Count of doodad names
    uint32_t offsDoodadNames;   // [0x04] Offset to MDNM (0 if count=0)
    uint32_t nMapObjNames;      // [0x08] Count of WMO names
    uint32_t offsMapObjNames;   // [0x0C] Offset to MONM (0 if count=0)
    uint8_t pad[112];           // [0x10-0x7F] Padding
};
```

### MHDR Structure (64 bytes)
```c
struct SMAreaHeader {
    uint32_t offsInfo;  // [0x00] Offset to MCIN
    uint32_t offsTex;   // [0x04] Offset to MTEX
    uint32_t sizeTex;   // [0x08] Size of MTEX data
    uint32_t offsDoo;   // [0x0C] Offset to MDDF
    uint32_t sizeDoo;   // [0x10] Size of MDDF data
    uint32_t offsMob;   // [0x14] Offset to MODF
    uint32_t sizeMob;   // [0x18] Size of MODF data
    uint8_t pad[36];    // [0x1C-0x3F] Padding
};
```

**CRITICAL:** Alpha MHDR does NOT have a sizeInfo field after offsInfo!

## Session Notes

### 2025-10-16: Four Critical Bugs Fixed
1. MCNK offset order corrected
2. MHDR field positions fixed (removed 4-byte shift)
3. MHDR structure corrected (removed sizeInfo)
4. MPHD null pointers (set to 0 when count=0)

File size increased from 16.56 MB to 40.96 MB (147% increase) with proper texture data extraction.
Client still crashes - needs deeper investigation into CreateMapObjDef function.
