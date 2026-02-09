# Task 5: Sector Size and Single-Unit Handling

## Overview

This document analyzes how WoW 0.6.0 handles sector sizes and single-sector files in MPQ archives.

## Sector Size Configuration

### MPQ Header Field

The MPQ v1 header contains a `wBlockSize` field at offset 12 (2 bytes):
- This is the **sector size shift value**, not the actual sector size
- Actual sector size = `512 << wBlockSize`
- Typical value: 3 (resulting in 4096 byte sectors)

### Default Sector Size

From analysis of the 0.6.0 MPQ archives:
- **Sector size shift**: 3
- **Sector size**: 512 << 3 = **4096 bytes**

## Sector Offset Table

### Structure

For files with more than one sector:
- A sector offset table is stored before the file data
- Each entry is a 4-byte offset from the start of the file data
- The last entry contains the total compressed size (serving as end marker)

### Single-Sector Files

For files smaller than or equal to one sector:
- **No sector offset table** is stored
- File data is read directly
- The block entry's `FileSize` and `BlockSize` fields determine read sizes

## Code Analysis

### File Read Function - [`FUN_00658c20`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md)

The file read function handles single-sector files specially:

```c
// Pseudocode from analysis
undefined4 FUN_00658c20(int param_1, undefined4 param_2, uint param_3, uint *param_4, undefined4 param_5)
{
    uint fileSize = *(uint *)(fileEntry + 0x08);
    uint blockSize = *(uint *)(fileEntry + 0x0c);
    uint blockFlags = *(uint *)(fileEntry + 0x10);
    
    // Get sector size from archive
    uint sectorSize = 512 << *(ushort *)(archive + 0x12);  // wBlockSize shift
    
    // Check if single-sector file
    if (fileSize <= sectorSize) {
        // Single sector - no sector offset table
        // Read directly from block offset
        uint readSize = blockSize;
        if (blockFlags & 0x00000200) {
            // Compressed - read compressed size
            readSize = blockSize;
        } else {
            // Uncompressed - read file size
            readSize = fileSize;
        }
        
        // Read sector data
        FUN_00658920(archive, sectorBuffer, blockOffset, readSize);
        
        // Decompress if needed
        if (blockFlags & 0x00000200) {
            FUN_006525d0(outputBuffer, &outputSize, sectorBuffer, readSize, blockFlags);
        } else {
            memcpy(outputBuffer, sectorBuffer, readSize);
        }
    } else {
        // Multi-sector file - use sector offset table
        // ...
    }
}
```

### Sector Size in PKWARE Decompression

The PKWARE decompressor uses the compressed data size to determine dictionary size:

```c
// From FUN_00651550
if (param_4 < 0xc00) {
    // For small compressed data (< 3072 bytes)
    local_8 = (-(uint)(param_4 < 0x600) & 0xfffffc00) + 0x800;
    // Results in:
    // - If compressed < 1536 bytes: dictSize = 1024
    // - If compressed < 3072 bytes: dictSize = 2048
} else {
    // For larger compressed data (>= 3072 bytes)
    local_8 = 0x1000;  // 4096 bytes
}
```

## Test Case Analysis

### WMO Root File Example

From the problem statement:
- Block: offset=435912, size=318 (compressed), fileSize=472 (decompressed), flags=0x80000200

**Analysis**:
1. File size (472) < sector size (4096) → **Single-sector file**
2. No sector offset table stored
3. Compressed data (318 bytes) read directly from block offset
4. Compressed data format:
   - Byte 0: `0x08` (compression type = PKWARE DCL)
   - Bytes 1-317: PKWARE compressed data (NO header bytes!)

### Dictionary Size Calculation

For this file:
- Compressed size = 318 bytes
- 318 < 0x600 (1536) → dictionary size = **1024 bytes** (0x400)

## Special Handling for Very Small Files

### Minimum Read Size

The client may have minimum read size requirements:
- Typically reads in sector-sized chunks
- For very small files, reads only the necessary bytes

### Compression Threshold

Files are only compressed if compression reduces size:
- If compressed size >= file size, file is stored uncompressed
- Block flags will NOT have `0x00000200` set

## Summary

| Scenario | Sector Offset Table | Read Size | Notes |
|----------|---------------------|-----------|-------|
| fileSize > sectorSize | Yes | Per-sector | Multiple sectors |
| fileSize <= sectorSize, compressed | No | blockSize | Single sector, compressed |
| fileSize <= sectorSize, uncompressed | No | fileSize | Single sector, raw |

## Recommendation

For single-sector compressed files:
1. Read `blockSize` bytes from the block offset
2. First byte is compression type (e.g., `0x08` for PKWARE)
3. Remaining bytes are compressed data (NO PKWARE header)
4. Calculate dictionary size from compressed data length
5. Decompress using PKWARE DCL algorithm
