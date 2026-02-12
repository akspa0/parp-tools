# MPQ Decompression Analysis - WoWClient.exe 0.6.0.3592

This folder contains the Ghidra reverse engineering analysis of the MPQ decompression system in WoW Alpha 0.6.0 (build 3592).

## Files

- **task-01-mpq-file-read-function.md** - MPQ file reading and archive handling functions
- **task-02-decompression-dispatch.md** - Compression type dispatch and handler mapping
- **task-03-pkware-decompressor.md** - PKWARE DCL implode algorithm analysis (compression type 0x08)
- **task-04-encryption-analysis.md** - Encryption handling in small files
- **task-05-sector-size-handling.md** - Sector size and single-unit file handling
- **summary-and-recommendations.md** - Key findings and implementation recommendations

## Key Discovery

**CRITICAL FINDING**: The WoW 0.6.0 client does NOT read the PKWARE DCL header bytes (compType and dictShift) from the compressed data stream. Instead, it **calculates the dictionary size based on the compressed data size**:

```c
// From FUN_00651550 (PKWARE wrapper)
if (param_4 < 0xc00) {
    local_8 = (-(uint)(param_4 < 0x600) & 0xfffffc00) + 0x800;
    // If size < 0x600 (1536 bytes): dictSize = 0x400 (1024)
    // If size < 0xC00 (3072 bytes): dictSize = 0x800 (2048)
} else {
    local_8 = 0x1000;  // dictSize = 4096
}
```

This explains why our implementation fails - we were trying to read header bytes that don't exist in the MPQ compressed data!

## Compression Type Mapping

| Mask | Handler Function | Description |
|------|-----------------|-------------|
| 0x01 | FUN_006517e0 | Huffman encoding |
| 0x02 | FUN_00661930 | zlib/deflate |
| 0x08 | FUN_00651550 | PKWARE DCL implode |
| 0x10 | FUN_00651a90 | BZip2 |

## Function Addresses

| Function | Address | Purpose |
|----------|---------|---------|
| SFileOpenArchive | 0x00647380 | Open MPQ archive |
| SFileOpenFileEx | 0x00646f70 | Open file from MPQ |
| SFileReadFile | 0x00658c20 | Read file data from MPQ |
| SCompDecompress | 0x006525d0 | Main decompression dispatch |
| PKWARE Explode | 0x00651550 | PKWARE DCL wrapper |
| PKWARE Init | 0x006646d0 | PKWARE initialization |
| PKWARE Decompress | 0x00664870 | PKWARE main decompression loop |
