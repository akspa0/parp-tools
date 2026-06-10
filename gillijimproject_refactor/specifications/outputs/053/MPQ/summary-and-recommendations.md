# MPQ Analysis Summary

This directory contains the analysis of the MPQ handling functions in the provided binary.

## Findings

1.  **Function Identification**: Identified key Storm/MPQ functions including `SFile`, `SComp`, and compression handlers.
2.  **Decompression Dispatch**: Documented the `SCompDecompress2` function which dispatches to specific decompressors based on a bitmask.
3.  **Compression Algorithms**:
    *   **0x08 (PKWARE DCL)**: Implemented via `explode`.
    *   **0x02 (Zlib)**: Implemented via `zlib_uncompress`.
    *   **0x01 (Huffman)**: Implemented via `CHuffmanDecoder` (Adaptive Huffman).
    *   **0x40/0x80 (IMA ADPCM)**: Likely present but not fully analyzed in this pass.
4.  **Read Pipeline**: Traced the read path from `SFileReadFile` -> `SFileReadFileEx2` -> `InternalReadUnaligned` -> `InternalReadAligned`.
5.  **Encryption**: Confirmed the standard MPQ encryption algorithm and key derivation logic.

## Recommendations

*   **Decompression**: Implement handlers for PKWARE DCL, Zlib, and Adaptive Huffman to support reading all file types.
*   **Encryption**: Implement the standard MPQ decryption logic with support for sector-based keys.
*   **Read Pipeline**: Replicate the sector alignment and caching logic for optimal performance.

## File Structure

*   `task-01-function-list.md`: List of identified functions.
*   `task-02-decompression-dispatch.md`: Analysis of the decompression dispatch logic.
*   `task-03-compression-0x08.md`: Analysis of PKWARE DCL compression.
*   `task-04-compression-0x01.md`: Analysis of Huffman compression.
*   `task-05-compression-0x02.md`: Analysis of Zlib compression.
*   `task-06-read-pipeline.md`: Documentation of the file reading process.
*   `task-07-encryption.md`: Analysis of encryption and key derivation.
