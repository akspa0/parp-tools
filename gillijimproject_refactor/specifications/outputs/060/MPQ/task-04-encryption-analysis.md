# Task 4: Encryption Analysis for Small Files

## Overview

This document analyzes whether small file blocks use encryption in WoW 0.6.0 MPQ archives.

## Block Flags Analysis

From the problem statement:
- Observed block flags: `0x80000200`
- Breakdown:
  - `0x80000000` = FLAG_EXISTS - File exists
  - `0x00000200` = FLAG_COMPRESSED - File is compressed
  - **NOT** `0x00010000` = FLAG_ENCRYPTED - File is NOT encrypted

## Code Analysis

### File Read Function - [`FUN_00658c20`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md)

The main file read function checks block flags before processing:

```c
// Pseudocode from analysis of FUN_00658c20
undefined4 FUN_00658c20(int param_1, undefined4 param_2, uint param_3, uint *param_4, undefined4 param_5)
{
    // ... validation code ...

    // Get block entry flags
    uint blockFlags = *(uint *)(fileEntry + 0x10);
    
    // Check if encrypted
    if (blockFlags & 0x00010000) {
        // Decrypt sector data
        uint sectorOffset = 0;
        if (blockFlags & 0x01000000) {
            sectorOffset = *(uint *)(param_1 + 0x20);  // File key
        }
        // Call decryption routine
        FUN_00659140(sectorData, sectorSize, sectorOffset);
    }
    
    // Check if compressed
    if (blockFlags & 0x00000200) {
        // Decompress sector
        FUN_006525d0(outputBuffer, &outputSize, sectorData, sectorSize, blockFlags);
    }
    
    // ... rest of function ...
}
```

### Encryption Check in Decompression Dispatch - [`FUN_006525d0`](specifications/outputs/060/MPQ/task-02-decompression-dispatch.md)

The decompression dispatch function does NOT perform any encryption checks. It only handles compression:

```c
undefined4 FUN_006525d0(byte *param_1, uint *param_2, byte *param_3, byte *param_4, undefined4 param_5)
{
    // ... validation ...

    // Read compression type byte
    uVar7 = (uint)*param_3;  // First byte is compression type bitmask
    pbVar9 = param_3 + 1;     // Skip compression type byte
    
    // No encryption check here - just compression dispatch
    // ...
}
```

### Encryption Flag Constants

From the MPQ format specification:

| Flag | Value | Description |
|------|-------|-------------|
| FLAG_EXISTS | 0x80000000 | File exists in archive |
| FLAG_COMPRESSED | 0x00000200 | File is compressed |
| FLAG_ENCRYPTED | 0x00010000 | File is encrypted |
| FLAG_FIX_KEY | 0x00020000 | File uses fixed key |
| FLAG_PATCH_FILE | 0x00100000 | File is a patch |
| FLAG_SINGLE_UNIT | 0x01000000 | File is stored as single unit (no sectors) |
| FLAG_DELETE_MARKER | 0x02000000 | File is marked for deletion |
| FLAG_SECTOR_CRC | 0x04000000 | Sector CRCs present |

## Verification

### Test Case: WMO Root Files

From the problem statement:
- Block: offset=435912, size=318 (compressed), fileSize=472 (decompressed), flags=0x80000200
- First byte of compressed data = `0x08` (PKWARE DCL compression type)

**Analysis**:
1. Flags `0x80000200` = EXISTS | COMPRESSED
2. No `0x00010000` (ENCRYPTED) flag set
3. First byte `0x08` is the compression type, not encrypted data
4. If encrypted, the first byte would be XOR'd with the key stream

### Encryption Detection

If the file were encrypted:
1. The `0x00010000` flag would be set in block entry
2. The file key would be calculated from filename hash
3. Each sector would be decrypted before decompression
4. The first byte would NOT be `0x08` (it would be `0x08 XOR key_byte`)

## Conclusion

**Small files in WoW 0.6.0 MPQs are NOT encrypted.**

The block flags `0x80000200` clearly indicate:
- File exists (`0x80000000`)
- File is compressed (`0x00000200`)
- **No encryption** (`0x00010000` is NOT set)

The decompression failure is NOT due to encryption - it's due to the non-standard PKWARE DCL header handling documented in [Task 3](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md).

## Recommendation

No encryption handling is needed for these files. Focus on fixing the PKWARE DCL decompression by calculating dictionary size from compressed data length instead of reading header bytes.
