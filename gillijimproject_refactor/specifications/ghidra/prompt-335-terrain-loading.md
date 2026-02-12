# Ghidra LLM Prompt — Wow.exe 3.3.5.12340 — Terrain Loading Pipeline

**Binary**: Wow.exe (WotLK 3.3.5a build 12340)
**PDB**: NOT available.
**Architecture**: x86 (32-bit)
**Focus**: Complete terrain loading pipeline — WDT, ADT, MCNK, coordinate mapping, and rendering setup.

---

## Why This Analysis Is Needed

We are building a terrain viewer that loads 3.3.5 ADT files. Terrain loads but renders incorrectly:
- **Tile positions are mirrored** — camera appears in Arathi Highlands instead of Wetlands
- **Chunks within tiles may have seams** — not forming continuous mesh
- **Subchunk offset convention is unclear** — chunk-relative vs data-relative

We need the **ground truth** from the binary for every step of the terrain loading pipeline.

---

## Critical Background

### What We Know
- FourCCs are reversed on disk (e.g., "MCNK" stored as bytes `4B 4E 43 4D`)
- Coordinate system: X=North, Y=West, Z=Up
- MapOrigin = 17066.66666 (32 × 533.33333)
- ChunkSize (tile size) = 533.33333
- Each tile has 16×16 MCNK chunks
- Each MCNK chunk has 145 vertices (9-8-9-8... interleaved rows)

### What We DON'T Know (and need from Ghidra)
1. WDT MAIN chunk: row-major `[row][col]` or column-major `[col][row]`?
2. ADT filename construction: which index is XX and which is YY in `MapName_XX_YY.adt`?
3. MCNK subchunk offsets: relative to MCNK chunk start (including 8-byte header) or MCNK data start (after header)?
4. MCNK header Position field: exact data offset, and component order (Z,X,Y or X,Z,Y or something else)?
5. MCVT heights: absolute or relative to MCNK Position Z?
6. MCNK IndexX/IndexY: which axis does each correspond to? Row or column within the 16×16 grid?
7. How does the client compute world position from tile index + chunk index + vertex position?

---

## Task 1: WDT MAIN Chunk — Tile Enumeration and Indexing

**Goal**: Determine the exact memory layout and indexing convention of the WDT MAIN chunk.

**Method**:
1. Find the WDT loading function. Search for reversed "MAIN" FourCC = `4E 49 41 4D` (0x4E49414D as uint32).
2. Find where the client iterates over MAIN entries to determine which tiles exist.
3. Determine: for flat index `i` in the 4096-entry array, how does the client decompose `i` into two coordinates?
   - `x = i / 64, y = i % 64` (column-major) — OR —
   - `y = i / 64, x = i % 64` (row-major)?
4. Trace how these two coordinates are used to construct the ADT filename.

**Specific questions**:
- What is the entry size? (Expected: 8 bytes — uint32 flags + uint32 asyncId)
- For entry at flat index `i`, what are the two grid coordinates?
- Which coordinate appears first in the filename `MapName_XX_YY.adt`?
- Is there any coordinate swapping between the MAIN index and the filename?

**What to look for**:
- String formatting calls that build `"%s\\%s_%d_%d.adt"` or similar
- Division/modulo by 64 near the MAIN chunk handler
- The relationship between the two extracted coordinates and the `sprintf`/string format arguments

**This is the single most important question.** Everything else depends on getting this right.

---

## Task 2: ADT Filename Construction

**Goal**: Find the exact code that builds the ADT file path from tile coordinates.

**Method**:
1. Search for the string pattern `"_%d_%d"` or `"_%02d_%02d"` or similar in the binary's string table.
2. Find all xrefs to this format string.
3. For each xref, trace back to determine which variables are passed as the two `%d` arguments.
4. Determine: are these the raw MAIN indices, or are they transformed (swapped, offset, etc.)?

**Specific output needed**:
- The exact format string used
- The function address where the filename is constructed
- Which coordinate (from MAIN decomposition) maps to the first `%d` and which to the second
- Whether any transformation is applied between MAIN index and filename coordinates

---

## Task 3: MCNK Header — Complete 128-Byte Layout

**Goal**: Verify every field of the MCNK header by tracing actual field accesses in the binary.

**Method**:
1. Find the MCNK chunk handler. Search for reversed "MCNK" = `4B 4E 43 4D` (0x4B4E434D).
2. The handler receives a pointer to the MCNK data (after the 8-byte chunk header, or including it — determine which!).
3. Trace every access to `base + offset` where `base` is the MCNK pointer.
4. Document each field with its offset, size, and semantic meaning.

**Critical fields to verify** (offsets shown are from MCNK chunk start including 8-byte header, per wowdev.wiki):

| Chunk Offset | Data Offset | Field | Type | Notes |
|---|---|---|---|---|
| 0x08 | 0x00 | flags | uint32 | |
| 0x0C | 0x04 | IndexX | uint32 | **Which axis? Row or column?** |
| 0x10 | 0x08 | IndexY | uint32 | **Which axis?** |
| 0x14 | 0x0C | nLayers | uint32 | |
| 0x18 | 0x10 | nDoodadRefs | uint32 | |
| 0x1C | 0x14 | ofsHeight | uint32 | **Relative to chunk start or data start?** |
| 0x20 | 0x18 | ofsNormal | uint32 | Same question |
| 0x24 | 0x1C | ofsLayer | uint32 | |
| 0x28 | 0x20 | ofsRefs | uint32 | |
| 0x2C | 0x24 | ofsAlpha | uint32 | |
| 0x30 | 0x28 | sizeAlpha | uint32 | |
| 0x34 | 0x2C | ofsShadow | uint32 | |
| 0x38 | 0x30 | sizeShadow | uint32 | |
| 0x3C | 0x34 | areaId | uint32 | |
| 0x40 | 0x38 | nMapObjRefs | uint32 | |
| 0x44 | 0x3C | holes | uint16 | **Or uint32?** |
| 0x70 | 0x68 | position[3] | float×3 | **Exact offset? Component order?** |

**The offset relativity question is critical**:
- When the client reads `ofsHeight` and seeks to it, does it seek from:
  - (A) The start of the MCNK chunk (including the 8-byte FourCC+size header), or
  - (B) The start of the MCNK data (after the 8-byte header)?
- Look for the actual seek/pointer arithmetic: `base + ofsHeight` — what is `base`?

**The position field is critical**:
- What is the exact byte offset of the position field from the MCNK data start?
- What is the component order? `(Z, X, Y)` per wiki, but verify.
- Are these world coordinates or local coordinates?

---

## Task 4: MCNK Subchunk Offset Convention

**Goal**: Definitively determine whether MCNK subchunk offsets (ofsHeight, ofsNormal, etc.) are relative to the MCNK chunk start or the MCNK data start.

**Method**:
1. Find where the client uses `ofsHeight` to locate the MCVT subchunk.
2. Trace the pointer arithmetic:
   ```
   // Option A: chunk-relative (includes 8-byte header)
   mcvt_ptr = mcnk_chunk_start + ofsHeight;
   
   // Option B: data-relative (after 8-byte header)  
   mcvt_ptr = mcnk_data_start + ofsHeight;
   // where mcnk_data_start = mcnk_chunk_start + 8
   ```
3. Check if the client adds 8 to the offset, subtracts 8, or uses it as-is.
4. Verify with at least two different subchunk offsets (e.g., ofsHeight AND ofsNormal).

**What to look for**:
- The base pointer used in the addition
- Whether the base pointer points to the FourCC bytes or to the first data byte after the size field
- Any `+ 8` or `- 8` adjustments

---

## Task 5: MCVT Height Interpretation

**Goal**: Determine if MCVT heights are absolute world Z values or deltas from the MCNK base height.

**Method**:
1. Find the MCVT reader function (processes 145 floats = 580 bytes).
2. Trace how the height values are used in the rendering pipeline.
3. Look for any addition of the MCNK Position Z to the MCVT values.
4. Check: `final_z = mcvt_height` or `final_z = mcvt_height + mcnk_position_z`?

**This directly affects whether chunks form a continuous surface or appear as disconnected flat patches.**

---

## Task 6: World Position Computation

**Goal**: Document the complete pipeline from tile index → chunk index → vertex → world position.

**Method**:
1. Find where the client computes the world-space position of a terrain vertex.
2. Trace the full chain:
   - Tile grid coordinates (from WDT MAIN) → tile world origin
   - Chunk indices (IndexX, IndexY from MCNK header) → chunk world origin within tile
   - Vertex local position (from MCVT height + grid position) → final world position
3. Document every multiplication, addition, and constant used.

**Expected formula** (verify each part):
```
// Tile origin
tile_world_x = 17066.666 - tileRow * 533.333
tile_world_y = 17066.666 - tileCol * 533.333

// Chunk origin within tile  
chunk_world_x = tile_world_x - chunkRow * 33.333
chunk_world_y = tile_world_y - chunkCol * 33.333

// Vertex position
vertex_world_x = chunk_world_x - vertexRow * 4.1666
vertex_world_y = chunk_world_y - vertexCol * 4.1666
vertex_world_z = mcvt_height (+ mcnk_position_z ???)
```

**Key questions**:
- Which MAIN coordinate maps to the X (north-south) world axis?
- Which MAIN coordinate maps to the Y (east-west) world axis?
- Which MCNK index (IndexX or IndexY) maps to the X world axis?
- Is there any coordinate swapping at any stage?
- What constants are used? (533.333, 33.333, 4.1666, 17066.666)

---

## Task 7: MCNK Chunk Iteration Order

**Goal**: Verify the order in which MCNK chunks appear within an ADT file.

**Method**:
1. Find where the client iterates over the 256 MCNK chunks in an ADT.
2. Determine: are chunks stored row-by-row (Y then X) or column-by-column (X then Y)?
3. For chunk at flat index `ci` (0-255), what are IndexX and IndexY?
   - `IndexX = ci % 16, IndexY = ci / 16` — OR —
   - `IndexX = ci / 16, IndexY = ci % 16`?
4. How do IndexX and IndexY map to the north-south and east-west axes?

**What to look for**:
- Nested loops: `for (y=0; y<16; y++) for (x=0; x<16; x++)` or the reverse
- The relationship between loop variables and the MCNK header's IndexX/IndexY fields

---

## Task 8: MCNR Normal Format

**Goal**: Verify the MCNR normal data format for correct lighting.

**Method**:
1. Find the MCNR reader function.
2. Verify: 145 normals × 3 components (X, Y, Z) as signed bytes?
3. Check the component order and sign convention.
4. Is there padding after the 435 bytes of normal data? (Wiki says 3 extra bytes + optional 2 unknown bytes)
5. How are the signed bytes converted to float normals? `normal = byte / 127.0`?

---

## Task 9: Terrain Vertex Grid Layout

**Goal**: Verify the 9-8-9-8 interleaved vertex layout within each MCNK chunk.

**Method**:
1. Find where the client builds the terrain mesh from MCVT data.
2. Verify the vertex ordering:
   - Row 0: 9 outer vertices (grid corners)
   - Row 1: 8 inner vertices (cell centers)
   - Row 2: 9 outer vertices
   - ... alternating for 17 rows total = 145 vertices
3. How are these vertices positioned in local space?
   - Outer vertex (row r, col c): `x = c * cellSize, y = r * cellSize`?
   - Inner vertex (row r, col c): `x = (c + 0.5) * cellSize, y = (r + 0.5) * cellSize`?
4. What is `cellSize`? Expected: `533.333 / 8 / 16 = 4.1666`

---

## Task 10: MCIN Chunk — MCNK Offset Table

**Goal**: Verify the MCIN chunk format that provides offsets to each MCNK chunk.

**Method**:
1. Find the MCIN reader (reversed FourCC "NICM" = 0x4E49434D).
2. Verify entry format: 256 entries × 16 bytes each?
   ```
   struct SMChunkInfo {
       uint32_t offset;    // absolute file offset to MCNK chunk
       uint32_t size;      // size of MCNK chunk
       uint32_t flags;     // always 0?
       uint32_t asyncId;   // always 0?
   };
   ```
3. Is the `offset` field an absolute file offset, or relative to something?
4. Does the offset point to the MCNK FourCC bytes or to the data after the header?

---

## Tips for This Binary

- **No PDB** — but this is the most studied WoW binary. Many function addresses are known.
- The client uses class hierarchies: `CMap`, `CMapArea`, `CMapChunk`, `CMapObj`
- `CMapArea` likely handles per-tile (ADT) loading
- `CMapChunk` likely handles per-MCNK chunk processing
- Look for `CMap::Load` or `CMap::LoadTile` style functions
- String references like `"World\\Maps\\%s\\%s_%d_%d.adt"` are gold — find them
- The terrain rendering pipeline likely goes through `CMapChunk::CreateGeometry` or similar
- Error/debug strings containing "MCNK", "MCVT", "height", "terrain" are helpful
- Constants like `533.333` (0x44055555 as float), `17066.666` (0x46855555), `33.333` (0x42055555) can be searched for

---

## Output Format

For each task, provide:

1. **Function address(es)** in the binary
2. **Decompiled/pseudocode** of the relevant section
3. **Definitive answer** to the question asked
4. **Evidence**: the actual assembly or decompiled code that proves the answer
5. **Constants found**: any magic numbers, sizes, or multipliers
6. **Confidence level**: High / Medium / Low

### Priority Order

If time is limited, prioritize in this order:
1. **Task 2** (filename construction) — tells us the coordinate convention
2. **Task 1** (MAIN indexing) — tells us how indices map to coordinates  
3. **Task 4** (subchunk offset convention) — tells us how to find MCVT/MCNR
4. **Task 5** (MCVT height interpretation) — tells us if heights are absolute or relative
5. **Task 6** (world position computation) — tells us the full coordinate pipeline
6. **Task 3** (MCNK header layout) — verifies our struct
7. **Task 7** (chunk iteration order) — verifies IndexX/IndexY meaning
8. **Task 10** (MCIN format) — verifies offset table
9. **Task 9** (vertex grid layout) — verifies mesh building
10. **Task 8** (MCNR format) — verifies normals
