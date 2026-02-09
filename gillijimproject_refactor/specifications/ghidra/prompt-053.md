# Ghidra LLM Prompt — WoWClient.exe 0.5.3.3368

**Binary**: WoWClient.exe (Alpha 0.5.3 build 3368)
**PDB**: Available! This is the only build with symbols.
**Architecture**: x86 (32-bit)

---

## Context for the LLM

You are reverse engineering WoW Alpha 0.5.3 (build 3368) using Ghidra. This binary has a PDB file loaded, so you have function names and type information. This is the earliest known WoW client binary and uses unique file formats that differ from later versions.

### Key Format Differences from Later WoW

1. **WDT files** are monolithic — they contain all ADT terrain data embedded inline (no separate .adt files)
2. **Chunk FourCCs** are stored in **forward** byte order (e.g., "MCNK" on disk = 0x4D434E4B), unlike LK which reverses them ("KNCM")
3. **WMO files** are v14 (monolithic, single file), not v17 (split root + group files)
4. **Model files** are MDX format (WC3-like), not M2
5. **MCVT/MCNR** store vertices in **non-interleaved** order (81 outer, then 64 inner), not interleaved
6. **WDT MAIN** uses **column-major** indexing (tileX*64+tileY), not row-major

### Coordinate System

WoW uses: X=North, Y=West, Z=Up. File positions are stored as (X, Z, Y) — Z (height) in the middle.
- MapOrigin = 17066.66666 (world coord of tile 0,0)
- ChunkSize = 533.33333 (world units per tile)

---

## Research Tasks

### Task 1: MCNK Header Structure

**Goal**: Document the complete Alpha MCNK header layout.

**How to find it**:
1. Search for the string "MCNK" or the constant 0x4D434E4B
2. Find the function that reads/parses MCNK chunks
3. Trace all field accesses from the header base pointer
4. Document offset, size, and purpose of each field

**What we know** (verify these):
- Offset 0x00: IndexX (uint32)
- Offset 0x04: IndexY (uint32)
- Offset 0x08: NLayers (uint32)
- Offset 0x0C: Holes (uint32, 16-bit mask)
- Various offsets point to subchunks (MCVT, MCNR, MCLY, MCAL, MCSH)

**What we need**:
- Complete field-by-field layout with all offsets
- Which offsets are absolute vs relative
- Any flags fields and their bit meanings
- The header size (is it exactly 128 bytes like LK, or different?)

### Task 2: MCAL Alpha Map Pixel Order

**Goal**: Determine if 4-bit alpha maps are row-major or column-major.

**How to find it**:
1. Find the MCAL chunk reader (search for 0x4D43414C or "MCAL")
2. Look for the loop that reads 4-bit packed data
3. Determine which loop is outer (rows vs columns)
4. Check: does the inner loop iterate over X (column-major) or Y (row-major)?

**Expected**: Two nested loops reading 2048 bytes → 4096 values. We need to know:
```
for (y = 0; y < 64; y++)          // Option A: row-major
    for (x = 0; x < 64; x++)
        read nibble at (y*64+x)

vs

for (x = 0; x < 64; x++)          // Option B: column-major
    for (y = 0; y < 64; y++)
        read nibble at (x*64+y)
```

### Task 3: MCLQ Liquid Data Structure

**Goal**: Document the complete MCLQ chunk layout.

**How to find it**:
1. Search for "MCLQ" or 0x4D434C51
2. Find the reader function
3. Trace the structure: header, height grid, flags

**What we need**:
- Header layout (min/max height, liquid type, flags)
- Height grid dimensions (9×9? 8×8?)
- Per-vertex data format (height + flags?)
- How liquid type is determined (water vs ocean vs magma vs slime)

### Task 4: WMO v14 Rendering Path

**Goal**: Understand how WMO v14 geometry is transformed for rendering.

**How to find it**:
1. Find the WMO loading function (search for WMO-related strings or MOHD/MOGP chunks)
2. Trace to the rendering path
3. Look for vertex transformation matrices
4. Check if there's a handedness correction applied

**What we need**:
- The vertex transform pipeline (local → world)
- Whether vertices are reflected/mirrored during loading
- How rotation from MODF is applied (order of operations)
- Any coordinate swaps done during WMO loading

### Task 5: MDX Model Loading

**Goal**: Document the MDX chunk format completely.

**How to find it**:
1. Search for "MDLX" or "VERS" or "MODL" strings (MDX chunk signatures)
2. Find the MDX parser entry point
3. Trace each chunk handler

**What we need**:
- Complete chunk inventory with sizes
- Bone/skeleton structure format
- Animation sequence format
- Particle system format (if present)
- Texture reference format
- Attachment points

### Task 6: BLP Texture Loading

**Goal**: Verify Alpha BLP format matches known BLP1 spec.

**How to find it**:
1. Search for "BLP1" or 0x31504C42
2. Find the BLP loader
3. Check format detection (compression type, alpha depth)

**What we need**:
- Supported compression types in 0.5.3
- Mipmap handling
- Any Alpha-specific BLP variants

---

## Tips for This Binary

- **PDB is loaded** — use symbol names! Function names like `CMap::LoadTerrain`, `CMapObj::Load`, etc. are available.
- Look for class vtables to find method groups (e.g., `CMapChunk::*` for MCNK handling).
- The data segment may contain format constants (chunk sizes, magic numbers).
- Cross-reference string table entries — error messages often reveal parsing logic.
- Check for `assert` calls — they contain file/line info that maps to original source structure.

---

## Output Format

For each finding, please provide:
1. **Function address** and name (from PDB)
2. **Structure definition** in C-style notation
3. **Confidence level** (high/medium/low) based on evidence
4. **Cross-references** to other functions that use the same data
5. **Differences from wowdev.wiki** if any are found
