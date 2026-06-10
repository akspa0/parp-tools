# Task 2: Decompression Dispatch Analysis

## Overview

The decompression dispatch logic is handled by `SCompDecompress2` (00649b90). This function iterates through a table of registered decompression algorithms and applies them sequentially if the corresponding bit is set in the compression mask.

## Dispatch Logic

The function `SCompDecompress2` performs the following steps:

1.  **Validation**: Checks if the destination buffer size is sufficient and if input pointers are valid.
2.  **Direct Copy**: If the compression mask is 0 or the compressed size equals the decompressed size (and flags indicate no compression), it performs a direct memory copy.
3.  **Decompression Loop**:
    *   It iterates through a static table of decompression handlers.
    *   The iteration is **backwards**, starting from the end of the table (`0x0080f8f0`) down to the beginning (`0x0080f8d0`).
    *   For each entry, it checks if the handler's mask bit is set in the input compression mask.
    *   If set, it calls the handler function.
    *   The output of one decompressor becomes the input for the next (if multiple compressions are applied).

## Decompression Table

The table is located at `0x0080f8d0` in the `.rdata` section. Each entry is 8 bytes:
*   Offset 0: Compression Mask (4 bytes)
*   Offset 4: Function Pointer (4 bytes)

Based on analysis, the table entries are:

| Address | Mask | Function | Algorithm |
|---|---|---|---|
| `0x0080f8f0` | `0x08` | `PkwareDecompress` (`00648d50`) | PKWARE DCL (Implode) |
| `0x0080f8e8` | `0x02` | `ZlibDecompress` (`00649060`) | Zlib (Deflate) |
| `0x0080f8e0` | `0x01` | *Unknown (Likely Huffman)* | Huffman |
| `0x0080f8d8` | `0x40` | *Unknown (Likely IMA ADPCM Mono)* | IMA ADPCM (Mono) |
| `0x0080f8d0` | `0x80` | *Unknown (Likely IMA ADPCM Stereo)* | IMA ADPCM (Stereo) |

## Handler Signature

The decompression handlers follow this signature (based on `PkwareDecompress`):

```c
void __fastcall DecompressHandler(void *dest, ulong *destSizePtr, void *source, ulong sourceSize);
```

*   `dest`: Destination buffer.
*   `destSizePtr`: Pointer to the size of the destination buffer (updated with actual decompressed size).
*   `source`: Source buffer.
*   `sourceSize`: Size of the source data.

Note: `SCompDecompress2` passes an extra 5th argument (a pointer to a temporary buffer or similar), but `PkwareDecompress` ignores it.

## Compression Order

Since the decompression loop runs backwards (Pkware -> Zlib -> ...), and decompression must reverse the compression order, the **compression** order must have been:
... -> Zlib -> Pkware.

This implies that if multiple compressions are used, Pkware is applied *last* (outermost layer), and Zlib is applied *before* it.
