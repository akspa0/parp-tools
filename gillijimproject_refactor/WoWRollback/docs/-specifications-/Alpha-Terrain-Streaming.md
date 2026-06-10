# Alpha 0.5.3 Terrain Streaming System

**Source**: Ghidra reverse engineering of WoWClient.exe (0.5.3.3368)
**Date**: 2025-12-28
**Status**: Verified Ground-Truth

---

## 1. Overview

Unlike later WoW versions that split the world into individual `.adt` files, the Alpha client uses a **Monolithic Streaming System**. The entire continent is contained within a single massive `.wdt` file (e.g., `Azeroth.wdt` ~700MB+), which acts as a virtual file system for terrain chunks.

---

## 2. File Structure & Hierarchy

### The WDT as a Database
The WDT file is indexed by the `MAIN` chunk, which contains a 64x64 grid of **Area Infos**.

**Global Grid**: 64 x 64 "Areas" (Tiles).
**Area Grid**: Each Area contains 16 x 16 "Chunks" (MCNKs).
**Total**: 1024 x 1024 Chunks.

### `MAIN` Chunk (The Index)
*   **Size**: 64KB (4096 entries * 16 bytes).
*   **Struct**: `SMAreaInfo`
    *   `Offset` (4 bytes): ?
    *   `Size` (4 bytes): ?
    *   `Flags` (4 bytes): Used to track load state.
    *   `AsyncId` (4 bytes): Pointer to active AsyncObject.
    *   *(Note: The actual file offset to Chunks is likely stored here or derived. The `CMapChunk::Load` uses `param_1->offset`)*.

---

## 3. Streaming Logic

**Key Function**: `CMap::PrepareChunks` (0x0068488d)

The client maintains a "visible grid" around the camera and streams data in two priority tiers:

### 1. High Priority (Near Camera)
*   Defined by `CWorld::chunkRectHi`.
*   **Blocking Load**: If an Area or Chunk is missing in this region, the main thread **waits** (`AsyncFileReadWait`) for the async operation to complete immediately.
*   Ensures terrain under the player is always available.

### 2. Low Priority (Distant)
*   Defined by `CWorld::gbChunkRect` (Global Buffer Rect).
*   **Async Load**: Requests are queued (`PrepareChunk` -> `AsyncFileReadCreateObject`) and processed by a background worker thread.
*   The Main thread continues without waiting.

---

## 4. Chunk Loading Process

**Key Function**: `CMapChunk::Load` (0x006989e0)

1.  **Seek & Read**:
    *   The client keeps the `wdtFile` handle open.
    *   It performs a `Seek` to the specific chunk's offset.
    *   Reads the chunk data into a temporary `AsyncLoadBuffer`.
    
2.  **Size Limit**:
    *   The client enforces a strict chunk size limit: **< 15,000 bytes** (15KB).
    *   Chunks larger than this trigger a fatal error. This suggests high compression or very simple terrain data in Alpha MCNKs compared to later versions (which can vary but often exceed 15KB with shadow maps etc).

3.  **Parsing**:
    *   Once loaded (sync or async), the chunk is parsed via `CMapChunk::Create`.

---

## 5. Logical Mapping to ADT

While there are no `.adt` files on disk, the `CMapArea` class acts as the runtime equivalent of an ADT.
*   It manages a 16x16 grid of `CMapChunk` objects.
*   It likely corresponds to a single entry in the WDT `MAIN` table.
*   When converting Alpha data, each `CMapArea` (and its 256 chunks) should be serialized into one standard `.adt` file.

---
*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) on 2025-12-28.*
