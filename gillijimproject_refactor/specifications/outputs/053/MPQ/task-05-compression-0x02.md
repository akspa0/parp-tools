# Task 5: Compression 0x02 (Zlib)

## Overview

The compression type `0x02` corresponds to the Zlib compression algorithm (Deflate).

## Handler Analysis

The handler function is `ZlibDecompress` (`00649060`).

### ZlibDecompress

```c
void __fastcall ZlibDecompress(uchar *dest, ulong *destSizePtr, uchar *source, uint sourceSize, char *extra)
{
  byte *result;
  ulong size;
  
  size = *destSizePtr;
  
  // Call standard zlib uncompress
  result = zlib_uncompress(dest, &size, source, sourceSize);
  
  if (result != 0) {
    // Error handling
    _SErrDisplayError_24(0x85100083, extra, -4, 0, 0, 1);
    *destSizePtr = size; // Update size even on error?
    return;
  }
  
  *destSizePtr = size;
}
```

### Implementation Details

The function is a thin wrapper around the standard `zlib_uncompress` function (`00648fa0`). It handles the return code and reports errors using the Storm error handling mechanism if decompression fails. The `extra` parameter passed from the dispatch loop is used here for error context.
