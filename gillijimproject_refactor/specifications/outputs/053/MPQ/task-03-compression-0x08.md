# Task 3: Compression 0x08 (PKWARE DCL)

## Overview

The compression type `0x08` corresponds to the PKWARE Data Compression Library (DCL), specifically the `implode`/`explode` algorithms. This is a dictionary-based compression algorithm (LZ77 variant) with Shannon-Fano trees.

## Handler Analysis

The handler function is `PkwareDecompress` (`00648d50`).

### PkwareDecompress

```c
void __fastcall PkwareDecompress(void *dest, ulong *destSizePtr, void *source, ulong sourceSize)
{
  _FREEBLOCK **workBuffer;
  _PKWAREINFO info;
  
  // Allocate work buffer (approx 12KB)
  workBuffer = _SMemAlloc_16(0x3134, ...);
  
  // Setup info structure
  info.destsize = *destSizePtr;
  info.destpos = 0;
  info.source = source;
  info.sourcepos = 0;
  info.sourcesize = sourceSize;
  info.dest = dest;
  
  // Call explode
  explode(PkwareBufferRead, PkwareBufferWrite, (int)workBuffer, &info);
  
  // Free work buffer
  _SMemFree_16(workBuffer, ...);
  
  // Update destination size
  *destSizePtr = info.destpos;
}
```

### Explode Function

The `explode` function (`00658a50`) implements the decompression logic. It reads a small header from the compressed stream to determine parameters:

1.  **Compression Type** (Byte 0): Determines if the stream is ASCII or Binary (affects tree construction).
2.  **Dictionary Size** (Byte 1): Log2 of the dictionary size (4, 5, or 6 -> 1KB, 2KB, 4KB).
3.  **Flags** (Byte 2): Additional flags.

The function then constructs the necessary Shannon-Fano trees and decodes the stream.

### Callbacks

The `explode` function uses callbacks for I/O:
*   `PkwareBufferRead` (`00648cc0`): Reads bytes from the source buffer.
*   `PkwareBufferWrite` (`00648d10`): Writes bytes to the destination buffer.

## Conclusion

The `0x08` compression is a standard implementation of PKWARE DCL `explode`. It requires a work buffer of approximately 12KB. The compressed data includes a 3-byte header specifying the compression parameters.
