# WoWClient.exe 0.6.0 (Build 3592) - Ghidra Analysis Summary

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Architecture**: x86 (32-bit)
**PDB**: NOT available (used 0.5.3 PDB as reference)
**Analysis Date**: 2026-02-09

---

## Executive Summary

This analysis examined WoW Alpha 0.6.0 (build 3592) to determine format transitions between the Alpha format (0.5.x) and later versions (1.x+).

### Key Finding: **0.6.0 is a Transitional Build**

Some format changes had already occurred, while others were still pending.

---

## Format Determination Summary

| Format Aspect | 0.6.0 Status | Style |
|---------------|--------------|-------|
| WDT/ADT Split | **Separate ADT files** | LK-Style ✓ |
| MCVT/MCNR Layout | **Non-interleaved** | Alpha-Style |
| WMO Version | **v14 (monolithic)** | Alpha-Style |
| Model Format | **MDX (MDLX magic)** | Alpha-Style |
| MCNK Header | **Similar to 0.5.3** | Alpha-Style |
| FourCC Byte Order | **Forward ('MCNK')** | LK-Style ✓ |

---

## Task Reports

| Task | File | Determination |
|------|------|---------------|
| Task 1: WDT/ADT Format | [task-01-wdt-adt-format-detection.md](task-01-wdt-adt-format-detection.md) | LK-Style (separate files) |
| Task 2: MCVT/MCNR Layout | [task-02-mcvt-mcnr-vertex-layout.md](task-02-mcvt-mcnr-vertex-layout.md) | Alpha-Style (non-interleaved) |
| Task 3: WMO Version | [task-03-wmo-format-version.md](task-03-wmo-format-version.md) | Alpha-Style (v14 monolithic) |
| Task 4: Model Format | [task-04-model-format-mdx-vs-m2.md](task-04-model-format-mdx-vs-m2.md) | Alpha-Style (MDX) |
| Task 5: MCNK Header | [task-05-mcnk-header-layout.md](task-05-mcnk-header-layout.md) | Alpha-Style (similar to 0.5.3) |
| Task 6: FourCC Order | [task-06-chunk-fourcc-byte-order.md](task-06-chunk-fourcc-byte-order.md) | LK-Style (forward) |

---

## Key Functions Identified

| Address | Purpose | 0.5.3 Reference |
|---------|---------|-----------------|
| 0x00690530 | WDT Loading | `WorldMap::LoadWDT` |
| 0x006b5010 | ADT Tile Loading | `WorldMap::LoadADT` |
| 0x006b3f70 | ADT File Reading | `CMap::LoadADTFile` |
| 0x006a6710 | MCNK Loading | `CMapChunk::Load` |
| 0x006a6d00 | MCNK Validation | `CMapChunk::ValidateChunks` |
| 0x006a7d20 | MCVT Processing | `CMapChunk::ProcessHeights` |
| 0x006a7490 | MCNR Processing | `CMapChunk::ProcessNormals` |
| 0x006b7a50 | WMO Root Loading | `CMapObj::Load` |
| 0x006b8080 | WMO Group Parsing | `CMapObjGroup::Load` |
| 0x00421250 | Model Async Complete | `CModel::AsyncComplete` |

---

## Format Transition Timeline

### Already Changed in 0.6.0
1. **WDT/ADT Split** - Separate ADT files per tile
2. **FourCC Byte Order** - Forward byte order ('MCNK' not 'KNCM')

### Still Alpha-Style in 0.6.0
1. **MCVT/MCNR Layout** - Non-interleaved vertex data
2. **WMO Format** - v14 monolithic (no group file split)
3. **Model Format** - MDX (not M2)
4. **MCNK Header** - Similar structure to 0.5.3

### Changed After 0.6.0 (Beta/Release)
1. **WMO v17** - Split root + group files
2. **M2 Format** - Replaced MDX
3. **Interleaved Vertices** - MCVT/MCNR interleaved layout

---

## Confidence Levels

| Task | Confidence |
|------|------------|
| Task 1: WDT/ADT | HIGH |
| Task 2: MCVT/MCNR | HIGH |
| Task 3: WMO | HIGH |
| Task 4: Model | HIGH |
| Task 5: MCNK | MEDIUM |
| Task 6: FourCC | HIGH |

---

## Methodology

1. Used Ghidra MCP to analyze the binary
2. Searched for chunk-related strings (e.g., "iffChunk->token=='MCNK'")
3. Decompiled functions referencing these strings
4. Analyzed comparison values for FourCC determination
5. Traced data flow for format structure analysis

---

## References

- Source string: `C:\build\buildWoW\WoW\Source\World\...` (build path)
- PDB Reference: 0.5.3 symbols used for function naming
