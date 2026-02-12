# Summary and Recommendations

## Problem Statement

The implementation fails to decompress small files from WoW 0.6.0 MPQ archives that use compression type 0x08 (PKWARE DCL implode). The error message is "Invalid dict size bits: 0" because the data after the 0x08 type byte doesn't match standard PKWARE DCL format.

## Root Cause

**The WoW 0.6.0 client does NOT include PKWARE DCL header bytes in the compressed data stream.**

Standard PKWARE DCL expects:
- Byte 0: Compression type (0=binary, 1=ASCII)
- Byte 1: Dictionary size bits (4=1KB, 5=2KB, 6=4KB)

WoW 0.6.0 format:
- **No PKWARE header bytes** - dictionary size is calculated from compressed data length
- Compressed data starts immediately after the MPQ compression type byte (0x08)

## Key Findings

### 1. Dictionary Size Calculation

From [`FUN_00651550`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md):

```c
if (compressedSize < 0x600) {       // < 1536 bytes
    dictSize = 0x400;               // 1024 bytes
} else if (compressedSize < 0xC00) { // < 3072 bytes
    dictSize = 0x800;               // 2048 bytes
} else {
    dictSize = 0x1000;              // 4096 bytes
}
```

### 2. Compression Mode

- Always use **binary mode (0)** for MPQ files
- ASCII mode (1) is for text data with special character encoding

### 3. Data Format

After the MPQ compression type byte (0x08):
- **No header bytes** - start PKWARE decompression immediately
- The "dictShift=0" we observed is actually the first byte of PKWARE compressed data, not a header

### 4. No Encryption

Block flags `0x80000200` indicate:
- File exists (`0x80000000`)
- File is compressed (`0x00000200`)
- **No encryption** (`0x00010000` is NOT set)

## Implementation Fix

### Current (Broken) Code

```c
// This is WRONG for WoW 0.6.0
uint8_t compType = compressedData[0];  // Actually PKWARE data, not header!
uint8_t dictBits = compressedData[1];   // Also PKWARE data!
uint32_t dictSize = 1 << dictBits;      // Results in dictSize=1 (invalid!)
```

### Fixed Code

```c
// Calculate dictionary size from compressed length
uint32_t dictSize;
size_t compressedPayloadSize = compressedSize - 1;  // Minus MPQ compression type byte

if (compressedPayloadSize < 0x600) {
    dictSize = 1024;  // 0x400
} else if (compressedPayloadSize < 0xC00) {
    dictSize = 2048;  // 0x800
} else {
    dictSize = 4096;  // 0x1000
}

// Use binary mode
uint8_t compType = 0;  // Binary mode

// Start PKWARE decompression from byte 1 (skip MPQ compression type byte)
pkware_explode(output, outputSize, compressedData + 1, compressedPayloadSize, compType, dictSize);
```

## Function Reference

| Function | Address | Purpose |
|----------|---------|---------|
| SFileOpenArchive | 0x00647380 | Open MPQ archive |
| SFileOpenFileEx | 0x00646f70 | Open file from MPQ |
| SFileReadFile | 0x00658c20 | Read file data from MPQ |
| SCompDecompress | 0x006525d0 | Main decompression dispatch |
| PKWARE Wrapper | 0x00651550 | PKWARE DCL wrapper - calculates dict size |
| PKWARE Init | 0x006646d0 | PKWARE initialization |
| PKWARE Decompress | 0x00664870 | PKWARE main decompression loop |

## Compression Type Mapping

| Mask | Handler Address | Description |
|------|-----------------|-------------|
| 0x01 | 0x006517e0 | Huffman encoding |
| 0x02 | 0x00661930 | zlib/deflate |
| 0x08 | 0x00651550 | PKWARE DCL implode |
| 0x10 | 0x00651a90 | BZip2 |

## Testing

### Test Case: WMO Root File

- Block: offset=435912, size=318, fileSize=472, flags=0x80000200
- Compressed payload: 317 bytes (318 - 1 for compression type byte)
- Dictionary size: 1024 bytes (317 < 1536)
- Compression mode: Binary (0)

### Expected Behavior

1. Read 318 bytes from offset 435912
2. First byte (0x08) indicates PKWARE compression
3. Calculate dictSize = 1024 (since 317 < 1536)
4. Decompress bytes 1-317 using PKWARE DCL with dictSize=1024, mode=0
5. Output 472 bytes of decompressed data

## Files Generated

- [README.md](specifications/outputs/060/MPQ/README.md) - Overview and key discovery
- [task-01-mpq-file-read-function.md](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md) - MPQ file reading functions
- [task-02-decompression-dispatch.md](specifications/outputs/060/MPQ/task-02-decompression-dispatch.md) - Compression dispatch analysis
- [task-03-pkware-decompressor.md](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md) - PKWARE DCL algorithm details
- [task-04-encryption-analysis.md](specifications/outputs/060/MPQ/task-04-encryption-analysis.md) - Encryption verification
- [task-05-sector-size-handling.md](specifications/outputs/060/MPQ/task-05-sector-size-handling.md) - Sector size handling
