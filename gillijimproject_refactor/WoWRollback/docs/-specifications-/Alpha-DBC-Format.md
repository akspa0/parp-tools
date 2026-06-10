# Alpha 0.5.3 DBC Format & Loading

**Source**: Ghidra reverse engineering of WoWClient.exe (0.5.3.3368)
**Date**: 2025-12-28
**Status**: Verified Ground-Truth

---

## 1. File Format

The Alpha client uses the standard **WDBC** format found in later WoW versions (1.x - 3.x).

**Header (20 bytes)**:
*   `Magic` (4 bytes): `WDBC` (0x43424457)
*   `RecordCount` (4 bytes): Number of records.
*   `FieldCount` (4 bytes): Number of fields per record.
*   `RecordSize` (4 bytes): Size of a record in bytes.
*   `StringBlockSize` (4 bytes): Size of the string block in bytes.

**Data**:
*   **Records**: Raw data array (`RecordCount * RecordSize`).
*   **String Block**: Raw character data (`StringBlockSize`).

---

## 2. Loading Mechanism

**Function**: `WowClientDB<T>::Load` (Generic Template)

The client uses a specific strategy to load these files efficiently:

1.  **Single Allocation**:
    *   It reads the header to calculate total size.
    *   Allocates a **contiguous memory block** for both Records and the String Block:
        `Buffer = new byte[(RecordCount * RecordSize) + StringBlockSize]`

2.  **Row Parsing & Swizzling**:
    *   Iterates through `RecordCount`.
    *   Calls a type-specific `Read` function for each row (e.g., `CreatureDisplayInfoRec::Read`).
    *   **String Fixup**: String fields in the file are offsets (integers). The `Read` function converts them to absolute pointers in memory:
        `Row.StringPtr = StringBufferBase + FileStringOffset`
    *   *Note*: This happens *before* the string block is actually read from disk, which is safe because the pointer address is valid even if the data isn't there yet.

3.  **String Block Read**:
    *   After processing all rows, the client reads the String Block from the file directly into the end of the allocated buffer.

4.  **Indexing**:
    *   Allocates a separate Lookup Table (`int* Index`).
    *   Maps `ID -> RecordPointer` for O(1) access.
    *   `Index[Row.ID] = &Row`

---

## 3. Implications

*   **Format Stability**: The file format is identical to 1.12.1. Tools handling Vanilla DBCs should work on Alpha DBCs without modification, provided the **Column Schema** (number and types of fields) matches.
*   **Schema dependency**: The client has hardcoded `Read` functions for each DBC, implying strict schema expectations. To edit these files, you must use the exact column layout expected by Alpha 0.5.3.

---
*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) on 2025-12-28.*
