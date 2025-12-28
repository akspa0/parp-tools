# MCNK Sub-Chunk Audit: Alpha v18 vs Lich King 3.3.5

## Date: 2025-10-20

## Purpose

This document provides a **definitive reference** for MCNK sub-chunk formats in both Alpha (v18) and Lich King (3.3.5) versions. The goal is to prevent format confusion that has caused texture and alpha map issues in our Alpha→LK and LK→Alpha converters.

---

## Critical Rule: NEVER MIX FORMATS

**⚠️ WARNING:** Always keep Alpha and LK parsing/writing logic completely separate. Do not attempt to "unify" code paths - the formats are fundamentally different and mixing them causes catastrophic bugs.

---

## MCNK Structure Overview

### Alpha v18 (WDT Monolithic File)
- **File Structure**: Single `.wdt` file contains all terrain data
- **MCNK Location**: 256 MCNK chunks embedded in WDT after MPHD/MAIN
- **Header Size**: 128 bytes (0x80)
- **Offsets**: Relative to **start of MCNK chunk** (includes FourCC + size)

### Lich King 3.3.5 (Split ADT Files)
- **File Structure**: Split into `_root.adt`, `_tex0.adt`, `_obj0.adt`
- **MCNK Location**: 256 MCNK chunks in root file, referenced by MCIN
- **Header Size**: 128 bytes (0x80)  
- **Offsets**: Relative to **start of MCNK chunk** (includes FourCC + size)

---

## Sub-Chunk Format Comparison

| Sub-Chunk | Alpha v18 Format | LK 3.3.5 Format | Notes |
|-----------|------------------|-----------------|-------|
| **MCVT** (Heights) | ❌ **NO HEADER** - 580 bytes raw | ❌ **NO HEADER** - 580 bytes raw | Same in both |
| **MCNR** (Normals) | ❌ **NO HEADER** - 448 bytes raw | ❌ **NO HEADER** - 448 bytes raw | Same in both |
| **MCLY** (Layers) | ✅ **HAS HEADER** - FourCC+size+data | ✅ **HAS HEADER** - FourCC+size+data | **CRITICAL**: 16 bytes per layer |
| **MCRF** (References) | ✅ **HAS HEADER** - FourCC+size+data | ✅ **HAS HEADER** - FourCC+size+data | Same in both |
| **MCSH** (Shadow) | ❌ **NO HEADER** - raw data only | ✅ **HAS HEADER** - FourCC+size+data | **DIFFERENT!** |
| **MCAL** (Alpha maps) | ❌ **NO HEADER** - raw data only | ✅ **HAS HEADER** - FourCC+size+data | **DIFFERENT!** |
| **MCSE** (Sound emitters) | ❌ **NO HEADER** - raw data only | ✅ **HAS HEADER** - FourCC+size+data | **DIFFERENT!** |
| **MCLQ** (Liquids) | ❌ **NO HEADER** - raw data only | ✅ **HAS HEADER** - FourCC+size+data (deprecated in LK, use MH2O) | **DIFFERENT!** |

---

## Key Differences: The Source of All Evil

### 1. Alpha Format Has INCONSISTENT Sub-Chunk Headers

**Alpha v18 Rules:**
- `MCVT`, `MCNR`: **NO headers** - offsets point directly to raw data
- `MCLY`, `MCRF`: **HAVE headers** - offsets point to FourCC
- `MCSH`, `MCAL`, `MCSE`, `MCLQ`: **NO headers** - offsets point directly to raw data

**Why This Matters:**
- When reading Alpha: must know which chunks have headers
- When writing Alpha: must NOT add headers to MCSH/MCAL/MCSE/MCLQ
- When calculating offsets: must account for 8-byte header where present

### 2. LK Format Is CONSISTENT - All Sub-Chunks Have Headers

**LK 3.3.5 Rules:**
- **ALL** sub-chunks have `FourCC + size + data` format
- Offsets in MCNK header point to FourCC
- Sizes stored in both chunk header AND MCNK header fields

**Why This Matters:**
- Much simpler to parse - everything is consistent
- When converting LK→Alpha: must STRIP headers from MCSH/MCAL/MCSE
- When converting Alpha→LK: must ADD headers to MCSH/MCAL/MCSE

---

## MCNK Header Field Mapping

### Alpha v18 Header (128 bytes / 0x80)

```c
struct SMChunk_Alpha {
    /*0x00*/ uint32_t flags;
    /*0x04*/ uint32_t IndexX;
    /*0x08*/ uint32_t IndexY;
    /*0x0C*/ float    radius;              // Bounding sphere radius
    /*0x10*/ uint32_t nLayers;             // Number of texture layers (0-4)
    /*0x14*/ uint32_t nDoodadRefs;         // Count for MCRF doodad refs
    /*0x18*/ uint32_t ofsHeight;           // → MCVT raw data (NO header)
    /*0x1C*/ uint32_t ofsNormal;           // → MCNR raw data (NO header)
    /*0x20*/ uint32_t ofsLayer;            // → MCLY FourCC (HAS header)
    /*0x24*/ uint32_t ofsRefs;             // → MCRF FourCC (HAS header)
    /*0x28*/ uint32_t ofsAlpha;            // → MCAL raw data (NO header)
    /*0x2C*/ uint32_t sizeAlpha;           // Size of MCAL data
    /*0x30*/ uint32_t ofsShadow;           // → MCSH raw data (NO header)
    /*0x34*/ uint32_t sizeShadow;          // Size of MCSH data
    /*0x38*/ uint32_t areaid;              // Zone/subzone packed as uint16s
    /*0x3C*/ uint32_t nMapObjRefs;         // Count for MCRF object refs
    /*0x40*/ uint16_t holes_low_res;
    /*0x42*/ uint16_t padding;
    /*0x44*/ uint16_t predTex[8];          // ReallyLowQualityTextureingMap
    /*0x54*/ uint8_t  noEffectDoodad[8];
    /*0x5C*/ uint32_t ofsSndEmitters;      // → MCSE raw data (NO header)
    /*0x60*/ uint32_t nSndEmitters;        // Count of sound emitters
    /*0x64*/ uint32_t ofsLiquid;           // → MCLQ raw data (NO header)
    /*0x68*/ uint8_t  padding[24];         // Unused in Alpha
};
```

### LK 3.3.5 Header (128 bytes / 0x80)

```c
struct SMChunk_LK {
    /*0x00*/ uint32_t flags;               // Has additional flags vs Alpha
    /*0x04*/ uint32_t IndexX;
    /*0x08*/ uint32_t IndexY;
    /*0x0C*/ uint32_t nLayers;             // NO radius field!
    /*0x10*/ uint32_t nDoodadRefs;
    /*0x14*/ uint32_t ofsHeight;           // → MCVT raw data (NO header)
    /*0x18*/ uint32_t ofsNormal;           // → MCNR raw data (NO header)
    /*0x1C*/ uint32_t ofsLayer;            // → MCLY FourCC (HAS header)
    /*0x20*/ uint32_t ofsRefs;             // → MCRF FourCC (HAS header)
    /*0x24*/ uint32_t ofsAlpha;            // → MCAL FourCC (HAS header!)
    /*0x28*/ uint32_t sizeAlpha;           // Size from MCAL chunk
    /*0x2C*/ uint32_t ofsShadow;           // → MCSH FourCC (HAS header!)
    /*0x30*/ uint32_t sizeShadow;          // Size from MCSH chunk
    /*0x34*/ uint32_t areaid;              // Just area ID, not packed
    /*0x38*/ uint32_t nMapObjRefs;
    /*0x3C*/ uint16_t holes_low_res;
    /*0x3E*/ uint16_t unknown_but_used;
    /*0x40*/ uint16_t predTex[8];
    /*0x50*/ uint8_t  noEffectDoodad[8];
    /*0x58*/ uint32_t ofsSndEmitters;      // → MCSE FourCC (HAS header!)
    /*0x5C*/ uint32_t nSndEmitters;
    /*0x60*/ uint32_t ofsLiquid;           // → MCLQ FourCC (HAS header, deprecated)
    /*0x64*/ uint32_t sizeLiquid;
    // Rest differs significantly from Alpha
};
```

### Critical Header Differences

| Field | Alpha v18 | LK 3.3.5 | Impact |
|-------|-----------|----------|--------|
| **0x0C** | `radius` (float) | `nLayers` (uint32) | **MAJOR DIFFERENCE** - field meanings shift! |
| **0x28** `ofsAlpha` | Points to raw data | Points to FourCC | Must strip/add header on convert |
| **0x2C** `ofsShadow` | Points to raw data | Points to FourCC | Must strip/add header on convert |
| **0x38** `areaid` | Packed uint16 zone+sub | Just uint32 area ID | Requires unpacking/repacking |
| **0x5C** `ofsSndEmitters` | Points to raw data | Points to FourCC | Must strip/add header on convert |

---

## MCLY Sub-Chunk: Texture Layers (CRITICAL FOR TEXTURES)

### Format (Same in Alpha and LK - Both Have Headers)

```c
struct SMLayer {
    /*0x00*/ uint32_t textureId;         // Index into MTEX chunk
    /*0x04*/ uint32_t flags;              // Animation, alpha map, compression flags
    /*0x08*/ uint32_t offsetInMCAL;       // Offset into MCAL for this layer's alpha map
    /*0x0C*/ int32_t  effectId;           // GroundEffectTexture ID (or -1)
};
// Each layer = 16 bytes
// Up to 4 layers per MCNK
```

### Common Bugs

1. **Writing empty MCLY**: Setting `nLayers=0` but including MCLY data anyway
2. **Wrong offsetInMCAL**: Not accounting for previous layers' alpha map sizes
3. **Flags mismatch**: Alpha flags differ from LK flags (compression, cube map, etc.)

---

## MCAL Sub-Chunk: Alpha Maps (CRITICAL FOR BLENDING)

### Alpha Format
- **NO chunk header** - just raw alpha map data
- Size stored in MCNK header `sizeAlpha` field
- Each layer (except first) has an alpha map
- Offset stored in `MCLY[n].offsetInMCAL` is relative to MCAL data start

### LK Format  
- **HAS chunk header** - `'MCAL' + size + data`
- Size stored in both chunk header AND MCNK `sizeAlpha` field
- Same alpha map formats as Alpha (uncompressed, compressed, 4-bit)

### Alpha Map Formats (Same in both versions)

1. **Uncompressed 4096**: 64×64 uint8 alpha values (4096 bytes)
2. **Uncompressed 2048**: 64×64 uint4 packed values (2048 bytes)
3. **Compressed**: RLE-encoded (WDT flag + MCLY flag required)

### Common Bugs

1. **Adding headers to Alpha MCAL**: Fatal - Alpha expects raw data
2. **Wrong offset calculation**: Must account for all previous layers
3. **Forgetting padding**: Chunks must be even-byte aligned

---

## MCSH Sub-Chunk: Shadow Map

### Alpha Format
- **NO chunk header** - just raw 512-byte shadow map
- Can be 63×63 or 64×64 depending on "fix alpha map" flag
- Most commonly 512 bytes (64×64×1 bit = 4096 bits = 512 bytes)

### LK Format
- **HAS chunk header** - `'MCSH' + size + data`
- Same shadow map data as Alpha

### Common Bugs

1. **Adding header to Alpha MCSH**: Fatal error
2. **Wrong size calculation**: Must match sizeShadow in header

---

## Conversion Logic

### LK → Alpha Conversion

```csharp
// Pseudocode for correct LK→Alpha conversion

// 1. Read LK MCNK
byte[] lkMcnkData = ReadLkMcnk(adtFile, offset);
McnkHeader_LK header = ParseLkHeader(lkMcnkData);

// 2. Extract sub-chunks (LK all have headers)
byte[] mcvtRaw = ExtractSubChunk(lkMcnkData, header.ofsHeight); // NO header in LK either
byte[] mcnrRaw = ExtractSubChunk(lkMcnkData, header.ofsNormal); // NO header in LK either
byte[] mclyWhole = ExtractSubChunk(lkMcnkData, header.ofsLayer); // Has header in both
byte[] mcrfWhole = ExtractSubChunk(lkMcnkData, header.ofsRefs); // Has header in both

// 3. CRITICAL: Strip headers from MCSH/MCAL/MCSE for Alpha
byte[] mcshWhole = ExtractSubChunk(lkMcnkData, header.ofsShadow);
byte[] mcshRaw = StripChunkHeader(mcshWhole); // Remove 8-byte header!

byte[] mcalWhole = ExtractSubChunk(lkMcnkData, header.ofsAlpha);
byte[] mcalRaw = StripChunkHeader(mcalWhole); // Remove 8-byte header!

byte[] mcseWhole = ExtractSubChunk(lkMcnkData, header.ofsSndEmitters);
byte[] mcseRaw = StripChunkHeader(mcseWhole); // Remove 8-byte header!

// 4. Build Alpha MCNK with correct format
byte[] alphaMcnk = BuildAlphaMcnk(
    header.IndexX, header.IndexY,
    mcvtRaw, mcnrRaw,    // No headers
    mclyWhole, mcrfWhole, // Keep headers
    mcshRaw, mcalRaw, mcseRaw // NO headers!
);
```

### Alpha → LK Conversion

```csharp
// Pseudocode for correct Alpha→LK conversion

// 1. Read Alpha MCNK
byte[] alphaMcnkData = ReadAlphaMcnk(wdtFile, offset);
McnkHeader_Alpha header = ParseAlphaHeader(alphaMcnkData);

// 2. Extract sub-chunks (Alpha format)
byte[] mcvtRaw = ExtractRawData(alphaMcnkData, header.ofsHeight, 580);
byte[] mcnrRaw = ExtractRawData(alphaMcnkData, header.ofsNormal, 448);
byte[] mclyWhole = ExtractWithHeader(alphaMcnkData, header.ofsLayer); // Has header
byte[] mcrfWhole = ExtractWithHeader(alphaMcnkData, header.ofsRefs); // Has header

// 3. CRITICAL: Add headers to MCSH/MCAL/MCSE for LK
byte[] mcshRaw = ExtractRawData(alphaMcnkData, header.ofsShadow, header.sizeShadow);
byte[] mcshWhole = WrapWithHeader("MCSH", mcshRaw); // Add 8-byte header!

byte[] mcalRaw = ExtractRawData(alphaMcnkData, header.ofsAlpha, header.sizeAlpha);
byte[] mcalWhole = WrapWithHeader("MCAL", mcalRaw); // Add 8-byte header!

byte[] mcseRaw = ExtractRawData(alphaMcnkData, header.ofsSndEmitters, header.nSndEmitters * emitterSize);
byte[] mcseWhole = WrapWithHeader("MCSE", mcseRaw); // Add 8-byte header!

// 4. Build LK MCNK with correct format
byte[] lkMcnk = BuildLkMcnk(
    header.IndexX, header.IndexY,
    mcvtRaw, mcnrRaw,     // No headers in LK either
    mclyWhole, mcrfWhole,  // Keep headers
    mcshWhole, mcalWhole, mcseWhole // NOW have headers!
);
```

---

## Current Code Issues

### Problem 1: AlphaMcnkBuilder.cs (Lines 260-280)

**BUG**: Creating Chunk wrappers for ALL sub-chunks, including MCSH/MCAL/MCSE

```csharp
// WRONG - This adds headers to everything!
var mcshChunk = new Chunk("MCSH", mcshRaw.Length, mcshRaw);
var mcalChunk = new Chunk("MCAL", mcalRaw.Length, mcalRaw);
var mcseChunk = new Chunk("MCSE", mcseRaw.Length, mcseRaw);

byte[] mcshWhole = mcshChunk.GetWholeChunk();  // Includes header!
byte[] mcalWhole = mcalChunk.GetWholeChunk();  // Includes header!
byte[] mcseWhole = mcseChunk.GetWholeChunk();  // Includes header!
```

**FIX**: Only wrap MCLY and MCRF, use raw data for others

```csharp
// CORRECT - Only MCLY and MCRF get headers in Alpha
var mclyChunk = new Chunk("MCLY", mclyRaw.Length, mclyRaw);
var mcrfChunk = new Chunk("MCRF", mcrfRaw.Length, mcrfRaw);

byte[] mclyWhole = mclyChunk.GetWholeChunk();  // Has header
byte[] mcrfWhole = mcrfChunk.GetWholeChunk();  // Has header

// mcshRaw, mcalRaw, mcseRaw used directly - NO headers!
```

### Problem 2: Offset Calculation (Lines 278-294)

**BUG**: Calculating offsets assuming all chunks have headers

```csharp
// WRONG - mcshWhole, mcalWhole, mcseWhole have extra 8 bytes!
int totalSubChunkSize = alphaMcvtRaw.Length + mcnrRaw.Length + 
                        mclyWhole.Length + mcrfWhole.Length + 
                        mcshWhole.Length + mcalWhole.Length + mcseWhole.Length;
```

**FIX**: Use raw sizes for MCSH/MCAL/MCSE

```csharp
// CORRECT - Use raw sizes where appropriate
int totalSubChunkSize = alphaMcvtRaw.Length + mcnrRaw.Length + 
                        mclyWhole.Length + mcrfWhole.Length + 
                        mcshRaw.Length + mcalRaw.Length + mcseRaw.Length;
```

### Problem 3: Write Operations (Lines 371-378)

**BUG**: Writing wrapped chunks (with headers) for MCSH/MCAL/MCSE

```csharp
// WRONG - Writes headers for MCSH, MCAL, MCSE!
ms.Write(mcshWhole, 0, mcshWhole.Length);  // Writes 'MCSH' + size + data
ms.Write(mcalWhole, 0, mcalWhole.Length);  // Writes 'MCAL' + size + data
ms.Write(mcseWhole, 0, mcseWhole.Length);  // Writes 'MCSE' + size + data
```

**FIX**: Write raw data only

```csharp
// CORRECT - Write raw data only for MCSH, MCAL, MCSE
if (mcshRaw.Length > 0) ms.Write(mcshRaw, 0, mcshRaw.Length);
if (mcalRaw.Length > 0) ms.Write(mcalRaw, 0, mcalRaw.Length);
if (mcseRaw.Length > 0) ms.Write(mcseRaw, 0, mcseRaw.Length);
```

---

## Testing Strategy

### 1. Unit Tests with Synthetic Data

Create minimal test MCNKs:
- Alpha MCNK with known sub-chunk layout
- LK MCNK with known sub-chunk layout
- Verify round-trip conversion

### 2. Integration Tests with Real Data

Use actual game files:
- `test_data/0.5.3/RazorfenDowns.wdt` (Alpha)
- `test_data/3.3.5/Kalidar_XX_YY.adt` (LK)
- Compare extracted sub-chunks byte-by-byte

### 3. Regression Tests

After fixing code:
- Re-convert known good test cases
- Verify output matches expected structure
- Test in actual game client (ultimate validation)

---

## Action Items

### Immediate Fixes Required

1. ✅ **Create this audit document** - DONE
2. ⏳ **Fix `AlphaMcnkBuilder.cs`**:
   - Remove Chunk wrappers for MCSH/MCAL/MCSE
   - Fix offset calculations
   - Fix write operations
3. ⏳ **Add unit tests**:
   - Test Alpha MCNK parsing
   - Test LK MCNK parsing  
   - Test round-trip conversion
4. ⏳ **Validate with real data**:
   - Compare outputs against authentic Alpha WDTs
   - Test in Alpha 0.5.3 client

### Future Improvements

1. Create strict type system to prevent format mixing
2. Add validation checks for chunk header presence
3. Create visualization tool to inspect MCNK structure
4. Document any remaining edge cases

---

## References

- `z_wowdev.wiki/ADT_v18.md` - Alpha/LK ADT specification
- `WoWRollback/docs/Fix-MCLY-MCAL-MCSH-Format.md` - Previous fix attempt
- `WoWRollback/docs/MCNK-Header-Comparison.md` - Header field analysis
- `src/gillijimproject-csharp/WowFiles/Alpha/McnkAlpha.cs` - Alpha parsing code
- `src/gillijimproject-csharp/WowFiles/LichKing/McnkLk.cs` - LK parsing code

---

## Conclusion

The root cause of texture/alpha map issues is **format confusion** between Alpha and LK MCNK sub-chunks. The key insight is:

> **Alpha v18 is INCONSISTENT** - some sub-chunks have headers (MCLY, MCRF), others don't (MCSH, MCAL, MCSE, MCLQ)
>
> **LK 3.3.5 is CONSISTENT** - ALL sub-chunks have headers

Any code that doesn't respect this fundamental difference will produce corrupted output. The fix is straightforward but requires careful attention to which chunks get headers in which format.

**Next Step**: Implement the fixes in `AlphaMcnkBuilder.cs` with extreme care to not break working code.
