# Ghidra LLM Prompt — WoWClient.exe 0.6.0.3592 — Terrain Loading Pipeline

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**PDB**: NOT available. Use 0.5.3 PDB symbols (Wowae.pdb from build 3368) as reference for function identification.
**Architecture**: x86 (32-bit)
**Focus**: Complete terrain loading pipeline — WDT, ADT, MCNK, coordinate mapping, and rendering setup.

---

## Why This Analysis Is Needed

We are building a terrain viewer that loads 0.6.0+ ADT files. This is a **transitional** build between the monolithic Alpha WDT (0.5.x) and the split ADT format used in later versions. We need to determine exactly which format conventions 0.6.0 uses, because it may differ from both 0.5.3 and 3.3.5.

### Known 0.5.3 Conventions (from previous Ghidra analysis with PDB)
- WDT is **monolithic** — all ADT data embedded in a single `.wdt` file
- WDT MAIN is **column-major**: index = `x*64+y` (verified via Ghidra with PDB)
- MCNK subchunk offsets are relative to MCNK **chunk start** (including 8-byte header)
- MCVT/MCNR are **non-interleaved**: 81 outer vertices then 64 inner vertices
- FourCCs are **forward** on disk (e.g., "MCNK" stored as bytes `4D 43 4E 4B`)
- WMO format is **v14** (monolithic)
- Model format is **MDX** (MDLX header)

### What Changed Between 0.5.3 and 3.3.5
At some point between these versions, the following transitions occurred:
- Monolithic WDT → split ADT files (`MapName_XX_YY.adt`)
- Column-major MAIN → possibly row-major MAIN
- Non-interleaved MCVT/MCNR → interleaved (9-8-9-8 pattern)
- Forward FourCCs → reversed FourCCs on disk
- WMO v14 → v17
- MDX → M2

**0.6.0 is the key build to determine WHEN each transition happened.**

---

## Task 1: WDT Format — Monolithic or Split?

**Goal**: Determine if 0.6.0 uses monolithic WDT (all ADT data in one file) or split ADT files.

**Method**:
1. Find the WDT/map loading function. Use 0.5.3 PDB names as reference — look for `CMap::Load`, `CMap::LoadMap`, or similar.
2. Check if the client opens separate `MapName_XX_YY.adt` files or reads embedded ADT data from the WDT.
3. Look for file path construction with `_%d_%d` patterns.
4. Check MAIN chunk entry format:
   - **Alpha (0.5.3)**: MAIN entries contain file offsets into the monolithic WDT
   - **LK (3.3.5)**: MAIN entries contain flags (non-zero = tile exists, load separate file)

**Key question**: Does 0.6.0 use separate ADT files? If so, what is the filename format?

---

## Task 2: WDT MAIN Chunk — Indexing Convention

**Goal**: Determine the MAIN chunk indexing convention.

**Method**:
1. Find where the client iterates over MAIN entries.
2. For flat index `i` (0-4095), how does the client decompose it?
   - `x = i / 64, y = i % 64` (column-major, like 0.5.3) — OR —
   - `y = i / 64, x = i % 64` (row-major)?
3. If split ADT files exist, trace how the two coordinates map to the filename.

**0.5.3 reference**: Column-major (`x*64+y`). Has this changed?

---

## Task 3: FourCC Byte Order

**Goal**: Determine if 0.6.0 uses forward or reversed FourCCs on disk.

**Method**:
1. Find the chunk parser dispatch loop (reads FourCC + size, then dispatches).
2. Check what constant the FourCC is compared against:
   - Forward: `0x4D434E4B` ("MCNK" as big-endian uint32)
   - Reversed: `0x4B4E434D` ("KNCM" as little-endian uint32 of reversed bytes)
3. Check multiple chunk types to confirm consistency.

**0.5.3 reference**: Forward FourCCs. Has this changed?

---

## Task 4: MCNK Header Layout

**Goal**: Document the complete MCNK header for 0.6.0.

**Method**:
1. Find the MCNK chunk handler.
2. Trace all field accesses from the MCNK base pointer.
3. Document every field with offset, size, and meaning.

**Critical fields to verify**:

| Offset | Field | Question |
|---|---|---|
| 0x04 | IndexX | Which axis? Same as 0.5.3? |
| 0x08 | IndexY | Which axis? |
| 0x14 | ofsHeight | Relative to chunk start or data start? |
| 0x18 | ofsNormal | Same question |
| 0x68 | position[3] | Present? Same offset as 3.3.5? Component order? |

**0.5.3 reference**: The 0.5.3 MCNK header is shorter and has different field layout. Key differences to check:
- Does 0.6.0 have the `position[3]` field at offset 0x68?
- Is the header still the same size as 0.5.3, or has it grown to 128 bytes (like 3.3.5)?
- Are subchunk offsets still chunk-relative (including 8-byte header)?

---

## Task 5: MCNK Subchunk Offset Convention

**Goal**: Determine whether subchunk offsets are relative to MCNK chunk start or data start.

**Method**:
1. Find where the client uses `ofsHeight` to locate MCVT.
2. Trace the pointer arithmetic:
   ```
   // Option A: chunk-relative (0.5.3 convention)
   mcvt_ptr = mcnk_chunk_start + ofsHeight;
   
   // Option B: data-relative
   mcvt_ptr = mcnk_data_start + ofsHeight;
   ```
3. Verify with multiple subchunk offsets.

**0.5.3 reference**: Chunk-relative (Option A). Has this changed?

---

## Task 6: MCVT/MCNR Vertex Layout

**Goal**: Determine if heights and normals are interleaved or non-interleaved.

**Method**:
1. Find the MCVT reader function.
2. Check the vertex reading pattern:
   - **Non-interleaved (0.5.3)**: Read 81 outer vertices, then 64 inner vertices sequentially
   - **Interleaved (3.3.5)**: Read in 9-8-9-8 alternating row pattern (17 rows, 145 total)
3. Same check for MCNR normals.

**What to look for**:
- Loop structure: single loop of 145 (interleaved) vs two loops of 81+64 (non-interleaved)
- Array indexing patterns that suggest row-by-row alternation

---

## Task 7: MCVT Height Interpretation

**Goal**: Determine if MCVT heights are absolute or relative to MCNK base height.

**Method**:
1. Find where MCVT height values are used in rendering.
2. Check if the MCNK Position Z value is added to MCVT heights.
3. Look for: `final_z = mcvt[i] + position_z` vs `final_z = mcvt[i]`

**This determines whether chunks form a continuous surface or appear as disconnected flat patches.**

---

## Task 8: World Position Computation

**Goal**: Document the complete coordinate pipeline from tile → chunk → vertex → world position.

**Method**:
1. Find where the client computes world-space terrain vertex positions.
2. Document the full chain with all constants and transformations.
3. Pay special attention to which coordinate (from MAIN decomposition) maps to which world axis.

**Expected constants**: 533.333 (tile size), 33.333 (chunk size), 4.1666 (cell size), 17066.666 (map origin)

---

## Task 9: ADT Filename Construction (if split files exist)

**Goal**: If 0.6.0 uses split ADT files, find the exact filename construction code.

**Method**:
1. Search for string patterns like `"_%d_%d"` or `"_%02d_%02d"` in the string table.
2. Find xrefs and trace which coordinates are passed as arguments.
3. Document the exact format string and argument mapping.

**If 0.6.0 still uses monolithic WDT, skip this task and note that.**

---

## Task 10: MCAL Alpha Map Format

**Goal**: Determine which alpha map formats 0.6.0 supports.

**Method**:
1. Find the MCAL reader function.
2. Check for format branching:
   - 4-bit uncompressed (2048 bytes)
   - 8-bit uncompressed (4096 bytes, "bigAlpha")
   - RLE compressed
3. Check MPHD flags — does bit 0x4 control bigAlpha?
4. Check MCLY flags — does any flag control compression?

**0.5.3 reference**: Only 4-bit uncompressed. Has bigAlpha or RLE been added?

---

## Strategy for Function Identification

Since there is no PDB for 0.6.0, use these approaches:

### Byte-Pattern Matching from 0.5.3
The 0.5.3 binary has full PDB symbols. Many functions will have similar byte patterns in 0.6.0:
- `CMap::Load` / `CMap::LoadMap`
- `CMapArea::Load` (per-tile loading)
- `CMapChunk::Load` (per-MCNK loading)
- `CMapChunk::CreateGeometry` (mesh building)
- `CChunkLiquid` (liquid handling)

### String References
Search for these strings in the 0.6.0 binary:
- `"World\\Maps\\"`
- `"_%d_%d"` or `"_%02d_%02d"`
- `".adt"`
- `"MCNK"` or `"KNCM"` (depending on FourCC convention)
- Error/debug strings containing "terrain", "chunk", "height", "map"

### Constant Searches
- `0x44055555` = 533.333f (tile size)
- `0x46855555` = 17066.666f (map origin)
- `0x42055555` = 33.333f (chunk size)
- `0x40855555` = 4.1666f (cell size)
- `0x43910000` = 290.0f or similar terrain constants

### FourCC Constants
Depending on byte order convention:
- Forward: `0x4D434E4B` (MCNK), `0x4D435654` (MCVT), `0x4D434E52` (MCNR)
- Reversed: `0x4B4E434D` (KNCM), `0x54564D43` (TVCM), `0x524E434D` (RNCM)

---

## Output Format

For each task, provide:

1. **Function address(es)** in the 0.6.0 binary
2. **Corresponding 0.5.3 function** (if identifiable via PDB)
3. **Decompiled/pseudocode** of the relevant section
4. **Definitive answer** to the question asked
5. **Format determination**: Alpha-style (0.5.3) / LK-style (3.3.5) / Transitional (new)
6. **Evidence**: the actual assembly or decompiled code that proves the answer
7. **Confidence level**: High / Medium / Low

### Priority Order

If time is limited, prioritize in this order:
1. **Task 1** (monolithic vs split) — fundamental format question
2. **Task 3** (FourCC byte order) — affects all chunk identification
3. **Task 2** (MAIN indexing) — affects tile coordinate mapping
4. **Task 6** (MCVT/MCNR layout) — affects vertex data parsing
5. **Task 5** (subchunk offset convention) — affects subchunk location
6. **Task 4** (MCNK header) — affects all field reads
7. **Task 7** (height interpretation) — affects terrain continuity
8. **Task 8** (world position) — affects rendering positions
9. **Task 9** (filename construction) — only if split files
10. **Task 10** (MCAL format) — affects texture blending
