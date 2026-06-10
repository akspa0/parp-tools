# Task 6: MPQ Read Pipeline

## Overview

The MPQ file reading pipeline involves several layers of abstraction, handling caching, decryption, and decompression. The primary entry point is `SFileReadFile`, which delegates to internal functions.

## Call Stack

1.  `SFileReadFile` (`00652460`)
2.  `SFileReadFileEx2` (`006524c0`)
3.  `InternalReadUnaligned` (`0064d790`)
4.  `InternalReadAligned` (`0064da90`)

## Pipeline Stages

### 1. Request Handling (`SFileReadFileEx2`)
*   Handles asynchronous requests (if applicable).
*   Manages the read-ahead buffer (caching).
*   Determines if the read request is aligned to sector boundaries.
*   Calls `InternalReadUnaligned` to perform the actual read.

### 2. Sector Alignment (`InternalReadUnaligned`)
*   Handles unaligned reads by reading partial sectors if necessary.
*   Calls `InternalReadAligned` for the bulk of the data.

### 3. Sector Processing (`InternalReadAligned`)
This function iterates over the requested sectors and processes them:

#### A. Disk Read
*   Reads compressed/encrypted sector data from the archive file using `ReadFileChecked`.

#### B. Decryption
*   Checks the `MPQ_FILE_ENCRYPTED` flag (`0x00010000`).
*   If set, calls `Decrypt` (`0063d290` - inferred) on the sector data.
*   The decryption key is derived from the file key + sector index.

#### C. Decompression
*   Checks compression flags in the file's block table entry.
*   **Flag `0x00000100` (Implode/LZW?)**:
    *   Calls `DecompressLzw` (`0064dfa0`).
    *   *Note: Standard MPQ specs usually associate 0x100 with PKWARE Implode. The function name `DecompressLzw` suggests a different algorithm or a misnomer in the symbol/analysis.*
*   **Flag `0x00000200` (Compressed/Multi)**:
    *   Calls `SCompDecompress2` (`00649b90`).
    *   This handles the multi-compression dispatch (Pkware, Zlib, Huffman, ADPCM) as documented in Task 2.

## Summary

The read pipeline is robust, supporting encryption and two distinct compression modes: a legacy mode (0x100) and the modern multi-compression mode (0x200). The multi-compression mode delegates to the `SComp` layer.
