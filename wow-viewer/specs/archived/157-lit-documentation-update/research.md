# Research: LIT File Format Documentation

## Overview

This research consolidates findings from the wow-viewer implementation for LIT file format documentation, comparing against the original wowdev.wiki LIT page.

## Version Matrix

| Version | Hex | Name | Track Count | Group Stride | Parameter Bands | Notes |
|---------|-----|------|-------------|--------------|-----------------|-------|
| 2 | 0x00000002 | Pre-alpha v2 | 9 | 0xA24 (per data set) | No | Single partial profile (rawLightCount=-1), two data sets |
| 83 | 0x80000003 | Alpha v8.3 | 14 | 0x1140 | No | Standard 4-group layout |
| 84 | 0x80000004 | Alpha v8.4 | 18 | 0x1550 | No | Standard 4-group layout |
| 85 | 0x80000005 | Alpha v8.5 | 18 | 0x15F0 | Yes (4×10) | Standard 4-group layout |

## File Structure

### Global Header (8 bytes)
```
uint32 version
int32  rawLightCount
```

**Implementation Note**: `rawLightCount` of -1 indicates the pre-alpha v2 single-partial layout. All other versions use non-negative counts.

### Light Headers (64 bytes each, count = max(rawLightCount, 1) for v2, or rawLightCount for others)

All versions share the same 64-byte header structure:

```
int32  chunkX          // -1 for default/global light
int32  chunkY          // -1 for default/global light
int32  chunkRadius     // -1 for default/global light
float  positionX       // Client fixed-point X (1/36 world units)
float  positionY       // Client fixed-point Z (1/36 world units) 
float  positionZ       // Client fixed-point Y (1/36 world units)
float  radius          // Core radius in fixed-point (1/36 world units)
float  dropoff         // Falloff distance in fixed-point (1/36 world units) [v83+ only; v2 has reserved uint32]
char[32] name          // Null-terminated, 0xFD also terminates
```

**Implementation Note**: The v2 pre-alpha header lacks a separate `dropoff` field - the final 4 bytes are reserved (read as uint32 and discarded). The position components are stored as XZY (file Y=world Z, file Z=world Y).

**Spatial Coordinate Conversion** (Implementation Finding):
- LIT spatial records use client fixed-point at 1/36 world units
- File stores position as XZY (second component = Z, third = Y)
- Conversion: `WorldPosition = (X/36, Z/36, Y/36)`
- Renderer position: `RendererPosition = (mapOrigin - WorldPosition.Y, mapOrigin - WorldPosition.X, WorldPosition.Z)`

### Light Groups

Each light has 4 groups (Clear=0, Storm=1, ClearWater=2, StormWater=3) for standard versions.
v2 partial profile has 2 data sets: Primary (Partial) and Alternate (LegacyPartialAlternate).

#### Group Header (per track)
```
int32 trackLength[trackCount]  // 0-32 keyframes per track
```

#### Color Tracks (per track, 32 slots each)
```
int32 timeOfDay[32]    // 0-2880 (inclusive), 2880 wraps to 0
uint32 packedBgrx[32]  // BGRX format: B | (G<<8) | (R<<16) | (unused<<24)
```

**Implementation Note**: Only the first `trackLength` slots are valid. Time 2880 is treated as 0 for cyclic interpolation. Colors are stored as BGRX (blue in lowest byte).

#### Float Bands (v83-v85: 6 bands × 32 samples; v2: 2 bands × 32 samples)
```
float samples[32]  // Per band
```

Band semantics (v83-v85):
- Band 0: Fog End
- Band 1: Fog Start Scalar
- Bands 2-5: Sky bands (4 bands)
- v85 only: 4 parameter bands × 10 samples each (after cloudMask)

**Implementation Note**: v2 pre-alpha only has 2 float bands (observed values: ~10000 and 0.25 for first data set). The remaining legacy prefix (60 bytes) is consumed for alignment but not assigned guessed semantics.

#### Additional Fields (v83-v85 only)
```
int32 highlightSky   // Sky highlight index
int32 cloudMask      // Cloud mask
```

## Version-Specific Details

### Version 2 (Pre-alpha) - areatest.lit

**File**: `World/Azeroth/areatest.lit` (observed in 0.5.3.3368 client)

**Structure**:
- Header: version=2, rawLightCount=-1
- 1 Light Header (64 bytes, "Global Light", default position)
- 60-byte legacy prefix (15×int32, all zeros in observed file)
- Data Set 0: Primary Partial (9 tracks, 2 float bands, 0xA24 bytes)
- Data Set 1: Legacy Partial Alternate (9 tracks, 2 float bands, 0xA24 bytes)

**Track Count**: 9 (vs 14/18 in later versions)
**Group Stride**: 0xA24 per data set (vs 0x1140/0x1550/0x15F0 for full groups)
**Parameter Bands**: None
**HighlightSky/CloudMask**: None

**Observed Track Data** (from test fixture):

**Data Set 0 (Primary/Partial):**
- Track 0 (DirectColor): 4 keyframes at times 0, 720, 1440, 2160
- Track 8 (Unknown): 3 keyframes at times 0, 720, 1440
- Float Band 0: First sample 10000.0 (likely Fog End)
- Float Band 1: First sample 0.25 (likely Fog Start Scalar)

**Data Set 1 (LegacyPartialAlternate):**
- **No color tracks** (all track lengths = 0)
- Float Band 0: First sample 12000.0
- Float Band 1: First sample 0.5

**Implementation Note**: This is the only known v2 file. The second data set (`LegacyPartialAlternate`) contains **only float bands** (no color tracks) with different fog values. Its semantic purpose is not established — it may represent an alternate fog configuration or an earlier iteration of the format. It is retained for inspection rather than discarded. Only the primary `Partial` data set drives lighting evaluation.

### Version 83 (0x80000003)
- Track Count: 14
- Group Stride: 0x1140
- No parameter bands
- Standard 4-group layout

### Version 84 (0x80000004)
- Track Count: 18
- Group Stride: 0x1550
- No parameter bands
- Standard 4-group layout

### Version 85 (0x80000005)
- Track Count: 18
- Group Stride: 0x15F0
- Has parameter bands (4 × 10 samples)
- Standard 4-group layout

## Light Group Kinds

| Kind | Value | Description |
|------|-------|-------------|
| Clear | 0 | Clear weather |
| Storm | 1 | Storm weather |
| ClearWater | 2 | Clear underwater |
| StormWater | 3 | Storm underwater |
| Partial | 4 | Single partial profile (v2, modern partial) |
| LegacyPartialAlternate | 5 | Second data set in v2 pre-alpha (semantic unknown) |

## Color Track Indices (Standard Layout)

Based on implementation and LIT.md references:
- Track 0: Direct Color
- Track 1: Ambient Color
- Track 2: Sky Top
- Track 3: Sky Upper
- Track 4: Sky Middle
- Track 5: Sky Lower
- Track 6: Sky Horizon
- Track 7: Fog Color
- Track 8: Unknown (shadow opacity? - not promoted without client proof)
- Tracks 9-17: Additional tracks in v84/v85

**Implementation Note**: Track 8 is explicitly left as "Unknown" in the implementation. Shadow opacity semantics are an inference unless separately recovered. MCSH terrain visibility is a different signal.

## Time System

- 2880 time units per day (Light/LIT clock)
- 24 real minutes per cycle (0.5.3 contract)
- Keyframe times are inclusive 0..2880
- Cyclic linear interpolation: 2880 wraps to 0

## Validation Rules (Implementation)

1. File must be exactly expected length (no trailing data)
2. Track lengths must be 0..32
3. Keyframe times must be 0..2880
4. Version must be one of: 2, 0x80000003, 0x80000004, 0x80000005
5. rawLightCount must be -1 (v2 only) or >= 0
6. All headers read before any group payloads (strict ordering)
7. Group stride must match exactly

## Differences from wowdev.wiki

**Original wiki covers**: Basic structure, versions v83-v85, general track layout

**Implementation adds**:
1. **v2 pre-alpha partial layout** - completely undocumented on wiki
2. **Exact group strides** for each version
3. **Strict header-before-payload ordering** (not interleaved)
4. **Spatial coordinate conversion** (XZY fixed-point at 1/36, Y/Z swap)
5. **BGRX color format** documentation
6. **Float band semantics** (fog end, fog start scalar, sky bands)
7. **Parameter bands** in v85
8. **Validation rules** (exact size, track limits, time bounds)
9. **LegacyPartialAlternate** group kind for v2 second data set
10. **Track 8 = Unknown** (not shadow opacity without proof)

## References

- Implementation: `wow-viewer/src/core/WowViewer.Core.IO/Lit/LitProfileReader.cs`
- Models: `wow-viewer/src/core/WowViewer.Core.IO/Lit/LitProfileModels.cs`
- Runtime: `wow-viewer/src/viewer/WoWViewer/Terrain/LitLoader.cs`
- Tests: `wow-viewer/tests/WowViewer.Core.Tests/LitProfileReaderTests.cs`
- Coordinate Transform: `wow-viewer/src/core/WowViewer.Core/Lit/LitCoordinateTransform.cs`