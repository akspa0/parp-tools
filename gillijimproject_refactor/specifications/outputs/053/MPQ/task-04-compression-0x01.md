# Task 4: Compression 0x01 (Huffman)

## Overview

The compression type `0x01` corresponds to a Huffman compression algorithm. The implementation in `CHuffmanDecoder::Decompress` (`00648790`) suggests an **Adaptive Huffman** coding scheme.

## Handler Analysis

The handler uses a `CBitInput` class to read the compressed stream bit by bit.

### Decompression Logic

1.  **Initialization**:
    *   Reads the first 8 bits from the stream.
    *   Calls `CHuffman::BuildTree` with this initial value.
    *   Sets a flag (likely indicating if the tree is initialized).

2.  **Main Loop**:
    *   Calls `CHuffmanDecoder::DecodeSymbol` to read the next symbol from the stream using the current Huffman tree.
    *   **Symbol Handling**:
        *   **0x100 (EOF)**: End of stream. The loop terminates.
        *   **0x101 (Escape/New Symbol)**:
            *   Reads the next 8 bits as a literal byte.
            *   Calls `CHuffman::AddSymbol` to add this new symbol to the tree.
            *   If the tree was empty/uninitialized, it calls `CHuffman::IncrementWeight` for the new symbol.
        *   **0x00 - 0xFF (Literal)**:
            *   Writes the byte to the output buffer.
            *   Calls `CHuffman::IncrementWeight` to update the frequency of the symbol in the tree.

### Adaptive Nature

The calls to `AddSymbol` and `IncrementWeight` during the decompression process confirm that the Huffman tree is dynamic. It adapts to the data stream as it is processed, which is characteristic of Adaptive Huffman coding (likely Vitter's algorithm or similar).

## Data Structures

*   **CHuffmanDecoder**: Wrapper for the decompression state.
*   **CHuffman**: Manages the Huffman tree, weights, and symbol mappings.
*   **CBitInput**: Handles bit-level reading from the source buffer.

## Conclusion

The `0x01` compression is an Adaptive Huffman algorithm. It does not require a pre-defined frequency table; instead, it builds the tree dynamically as it encounters new symbols (marked by an escape code `0x101`) and updates weights for existing symbols.
