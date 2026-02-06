# Active Context

## Current Focus: LK/4.0.0 ADT Image Generation Fix (Jan 20, 2026)

### Critical Status

**0.5.3 Alpha ADT**: ✅ WORKING - Monolithic format, linear chunk scanning works
**LK 3.3.5 / 4.0.0 ADT**: ⚠️ JSON DATA WORKS, IMAGE GENERATION BROKEN

### Why 0.5.3 Works But LK+ Doesn't

| Aspect | 0.5.3 Alpha | LK/Cata 3.x/4.x |
|--------|-------------|-----------------|
| ADT Format | **Monolithic** - single file | **Split** - root + _obj0 + _tex0 |
| MCNK Location | **Sequential** after header chunks | **MCIN indexed** - offsets stored in MCIN chunk |
| Chunk Scanning | Linear byte scan finds MCNK | Linear scan **misses** MCNK chunks entirely |
| Chunk IDs | Little-endian, need reversal | Same - but WoWRollback handles this |

### Root Causes Found (Jan 19-20, 2026)

1. **MCIN Offset Issue**: LK/Cata MCNK chunks are NOT sequential. They're located at offsets specified in the MCIN chunk (256 × 16-byte entries). Linear scanning finds MVER→MHDR→MCIN→MTEX→etc but **skips** all MCNK chunks.

2. **posZ Addition Bug**: My fix incorrectly added `posZ` to MCVT height values. Working code stores raw MCVT floats. This caused gradient stripe pattern in heightmap images.

### Fixes Applied

- [x] Added MCIN parsing - reads 256 offset entries to locate MCNK chunks
- [x] Added `ParseMcnkAtOffset()` helper - parses MCNK at specified offset
- [x] Removed posZ from height calculation - matches working 0.5.3 pattern
- [ ] **Pending test** - waiting for file lock to clear

### JSON vs Image Status

| Output Type | LK/4.0.0 Status | Issue |
|-------------|-----------------|-------|
| height_min/max | ✅ Correct in JSON | Values like -3167 to -2666 |
| heights[] array | ✅ 256 chunks populated | All chunks have data |
| heightmap.png | ❌ Gradient stripes | Was adding posZ - FIXED |
| alpha masks | ⚠️ Need testing | May need MCAL parsing fix |
| normal maps | ⚠️ Need testing | Requires VlmChunkLayers.Normals |

---

## Architecture Differences

### 0.5.3 Alpha ADT Parsing Flow
```
1. Linear scan: MVER → MHDR → MCIN → MTEX → MMDX → MWMO → MDDF → MODF → MCNK[0..255]
2. Each MCNK found directly during scan
3. Works because MCNK chunks are sequential after header chunks
```

### LK/Cata ADT Parsing Flow (CORRECTED)
```
1. Linear scan: MVER → MHDR → MCIN → MTEX → MMDX → ... (NO MCNK found!)
2. Parse MCIN: 256 entries × 16 bytes = offset+size for each MCNK
3. For each MCIN entry: Jump to offset, parse MCNK there
4. MCNK chunks are scattered through file at MCIN-specified locations
```

---

## Key Files

| File | Purpose |
|------|---------|
| `VlmDatasetExporter.cs` | Main extraction - `ExtractFromLkAdt()` |
| `ParseMcnkAtOffset()` | NEW helper for MCIN-based parsing |
| `RenderHeightmapImage()` | 145×145 L16 heightmap generation |
| `AlphaMapService.cs` | Alpha mask decompression |

---

## Technical Notes

- **MCIN chunk**: 4096 bytes = 256 × 16 bytes (offset:4, size:4, flags:4, asyncId:4)
- **MCNK offset**: Points to MCNK signature, NOT data start
- **MCVT heights**: Store RAW floats, do NOT add posZ
- **Chunk ID reversal**: LK stores as little-endian, read as "KNCM" → reverse to "MCNK"
