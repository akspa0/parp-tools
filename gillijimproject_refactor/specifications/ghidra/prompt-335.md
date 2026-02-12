# Ghidra LLM Prompt — Wow.exe 3.3.5.12340

**Binary**: Wow.exe (WotLK 3.3.5a build 12340)
**PDB**: NOT available.
**Architecture**: x86 (32-bit)

---

## Context for the LLM

You are reverse engineering WoW 3.3.5a (WotLK, build 12340) using Ghidra. This is the most documented WoW client version thanks to the private server community. Use wowdev.wiki as a secondary reference, but verify everything against the actual binary — the wiki has known errors.

### Key Format Characteristics (LK 3.3.5)

1. **WDT files** are small (~33KB) and reference separate `.adt` files
2. **ADT files** are split: `MapName_XX_YY.adt` (root), `_tex0.adt`, `_obj0.adt` (optional Cata-split, NOT in 3.3.5)
3. **Chunk FourCCs** are **reversed** on disk (e.g., "MCNK" stored as bytes "KNCM" = 0x4B4E434D)
4. **WMO files** are v17 (split root + group files: `name.wmo`, `name_000.wmo`, `name_001.wmo`...)
5. **Model files** are M2 format (MD20 header), with external `.skin` files for render batches
6. **MCVT/MCNR** store vertices in **interleaved** order (9-8-9-8... alternating rows)
7. **WDT MAIN** uses **row-major** indexing (tileY*64+tileX), 8 bytes per entry (flags + asyncId)
8. **MPHD flags** control alpha format: bit 0x4 = bigAlpha (8-bit vs 4-bit)

### Coordinate System

Same as Alpha: X=North, Y=West, Z=Up. File positions stored as (X, Z, Y).
- MapOrigin = 17066.66666
- ChunkSize = 533.33333

---

## Research Tasks

### Task 1: MCNK Header — Definitive 3.3.5 Layout

**Goal**: Document the complete LK MCNK header for cross-reference with our implementation.

**Method**:
1. Search for "KNCM" (0x4B4E434D) to find MCNK chunk handler
2. Trace header field accesses from the base pointer
3. Document all 128 bytes of the header

**What we believe** (verify):
```
0x00: flags (uint32)
0x04: indexX (uint32)
0x08: indexY (uint32)
0x0C: nLayers (uint32)
0x10: nDoodadRefs (uint32)
0x14: ofsHeight (uint32)     → MCVT
0x18: ofsNormal (uint32)     → MCNR
0x1C: ofsLayer (uint32)      → MCLY
0x20: ofsRefs (uint32)       → MCRF
0x24: ofsAlpha (uint32)      → MCAL
0x28: sizeAlpha (uint32)
0x2C: ofsShadow (uint32)     → MCSH
0x30: sizeShadow (uint32)
0x34: areaId (uint32)
0x38: nMapObjRefs (uint32)
0x3C: holes (uint32)         → 16-bit hole mask in low word
0x40-0x4F: lowQualityTextureMap[16]
0x50: predTex (uint32)
0x54: noEffectDoodad (uint32)
0x58: ofsSndEmitters (uint32) → MCSE
0x5C: nSndEmitters (uint32)
0x60: ofsLiquid (uint32)     → MCLQ
0x64: sizeLiquid (uint32)
0x68: position[3] (float×3)  → Z, X, Y (height first!)
0x74: ofsMCCV (uint32)
0x78: ofsMCLV (uint32)
0x7C: unused (uint32)
```
Header size = 128 bytes (0x80).

### Task 2: MCAL Alpha Map Formats

**Goal**: Verify all three alpha map formats: 4-bit, 8-bit (bigAlpha), and RLE compressed.

**Method**:
1. Find MCAL reader function
2. Trace the format selection logic (checks MCLY flags and MPHD bigAlpha)
3. For 4-bit: verify pixel ordering (row-major vs column-major)
4. For RLE: verify compression header format

**What we believe**:
- 4-bit: 2048 bytes, expand low nibble then high nibble, `value * 17` to scale to 0-255
- 8-bit (bigAlpha): 4096 bytes, direct copy
- RLE: header byte bit 7 = fill, bits 0-6 = count

### Task 3: MCLQ Liquid Structure

**Goal**: Document LK MCLQ completely for cross-reference with Alpha findings.

**Method**:
1. Find MCLQ reader (search for the chunk offset access in MCNK handler)
2. Document header and vertex grid layout
3. Check liquid type determination

### Task 4: M2 Model Complete Structure

**Goal**: Document the M2 header and all referenced data blocks for our M2 reader implementation.

**Method**:
1. Search for "MD20" (0x3032444D) or "MD21" (0x3132444D)
2. Find the M2 loading function
3. Trace all header field accesses
4. Document texture, vertex, submesh, bone, and animation structures

**Priority fields for rendering**:
- Vertex format (position, normal, texcoord, bone weights)
- Texture array (filename offsets)
- Render flags / blend modes
- Submesh definitions
- .skin file loading (render batch definitions)

### Task 5: WMO v17 Loading Path

**Goal**: Document WMO v17 root and group file loading for our viewer implementation.

**Method**:
1. Find WMO root loader (search for "MOHD" reversed = "DHOM")
2. Trace group file loading (how does it construct `name_XXX.wmo` paths?)
3. Document MOGP group chunk structure
4. Document material (MOMT) structure for rendering

**Priority for rendering**:
- MOHD header (nGroups, nPortals, etc.)
- MOMT material definitions (texture, blend, flags)
- MOGP group geometry (MOVI indices, MOVT vertices, MONR normals, MOTV texcoords)
- MOBA render batch definitions

### Task 6: MDDF/MODF Placement Entries

**Goal**: Verify LK placement formats match our implementation.

**Method**:
1. Find MDDF reader (search for "FDDM" reversed)
2. Find MODF reader (search for "FDOM" reversed)
3. Verify entry sizes and field layouts
4. Check coordinate handling in the placement transform pipeline

---

## Tips for This Binary

- **No PDB** — but this is the most studied WoW binary. Many function addresses are known from community research.
- wowdev.wiki is a good starting point but has errors — always verify against the binary
- The binary uses a lot of `CMap*`, `CMapObj*`, `CM2Model*`, `CWorldModelGroup*` class hierarchies
- Look for vtable patterns to identify class methods
- Error strings and debug logging calls are present and very helpful
- The MPQ loading path (`SFile*` functions from StormLib) is well-documented

---

## Output Format

For each finding, provide:
1. **Function address** in this binary
2. **Structure definition** in C-style notation
3. **Comparison with wowdev.wiki** — matches / differs (detail differences)
4. **Comparison with our implementation** — matches / differs
5. **Confidence level** (high/medium/low)
