# LIT

**LIT** files contain lighting information for Alpha-era World of Warcraft maps (0.5.3 through ~0.8.0). They provide time-keyframed color and float tracks for different weather and water conditions.

> **[Implementation Note]**: This documentation extends the original wowdev.wiki LIT page with findings from the wow-viewer implementation (`WowViewer.Core.IO.Lit`). Implementation-specific observations are marked with `[Implementation Note: ...]` blocks. Original wiki terminology and structure are preserved where possible.

---

## File Structure

### Header (8 bytes)

| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | uint32 | `version` | Format version |
| 0x04 | int32 | `rawLightCount` | Number of lights, or -1 for v2 partial layout |

**Known Versions:**

| Version | Hex | Era | Track Count | Group Stride | Parameter Bands |
|---------|-----|-----|-------------|--------------|-----------------|
| 2 | 0x00000002 | Pre-alpha (0.5.3) | 9 | 0xA24 (per data set) | No |
| 83 | 0x80000003 | Alpha v8.3 | 14 | 0x1140 | No |
| 84 | 0x80000004 | Alpha v8.4 | 18 | 0x1550 | No |
| 85 | 0x80000005 | Alpha v8.5 | 18 | 0x15F0 | Yes (4×10) |

> **[Implementation Note]**: `rawLightCount = -1` is only valid for version 2 and indicates the pre-alpha single-partial layout. All other versions require `rawLightCount >= 0`. The reader validates exact file length against the expected layout for each version.

---

## Light Header (64 bytes each)

All versions share the same 64-byte header structure. Headers are stored contiguously **before** any group payloads.

| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | int32 | `chunkX` | ADT chunk X coordinate (-1 = default/global light) |
| 0x04 | int32 | `chunkY` | ADT chunk Y coordinate (-1 = default/global light) |
| 0x08 | int32 | `chunkRadius` | Chunk radius (-1 = default/global light) |
| 0x0C | float | `positionX` | **Client fixed-point X** (1/36 world units) |
| 0x10 | float | `positionY` | **Client fixed-point Z** (1/36 world units) |
| 0x14 | float | `positionZ` | **Client fixed-point Y** (1/36 world units) |
| 0x18 | float | `radius` | Core radius (fixed-point, 1/36 world units) |
| 0x1C | float | `dropoff` | Falloff distance (fixed-point, 1/36 world units) |
| 0x20 | char[32] | `name` | Null-terminated name (0xFD also terminates) |

> **[Implementation Note]**: The position components are stored in **XZY order** (file Y = world Z, file Z = world Y). This is a client fixed-point format at **1/36 world units per unit**.
>
> **Spatial Conversion (Implementation Finding):**
> ```csharp
> // Raw fixed-point XZY → Semantic game-world XYZ
> WorldPosition = (positionX / 36, positionZ / 36, positionY / 36)
> 
> // Game-world XYZ → Renderer coordinates (mapOrigin = 17066.666... for Azeroth)
> RendererPosition = (mapOrigin - WorldPosition.Y, mapOrigin - WorldPosition.X, WorldPosition.Z)
> ```
>
> **Version 2 Difference**: The v2 pre-alpha header lacks a separate `dropoff` field. The final 4 bytes (offset 0x1C) are reserved (read as uint32 and discarded to preserve alignment). `dropoff` is effectively 0 for v2.

### Default/Global Light Detection

A light is the default/global light when:
```
chunkX == -1 && chunkY == -1 && chunkRadius == -1
```

---

## Light Groups

Each light contains multiple groups for different weather/water conditions. **All headers are read first, then all group payloads** (strict ordering, not interleaved).

### Standard Layout (v83, v84, v85)

4 groups per light, in fixed order:

| Index | Kind | Description |
|-------|------|-------------|
| 0 | Clear | Clear weather |
| 1 | Storm | Storm weather |
| 2 | ClearWater | Clear underwater |
| 3 | StormWater | Storm underwater |

### Version 2 Partial Layout

Single light (`rawLightCount = -1`) with **2 data sets** instead of 4 groups:

| Index | Kind | Description |
|-------|------|-------------|
| 0 | Partial | Primary partial profile |
| 1 | LegacyPartialAlternate | Second data set (semantic unknown) |

> **[Implementation Note]**: The `LegacyPartialAlternate` group kind is an implementation-assigned name for the second data set in the v2 pre-alpha layout. Its semantic purpose is not established; it is retained for inspection rather than discarded. Only the primary `Partial` data set drives lighting evaluation.

---

## Group Payload Structure

### Track Length Array
```
int32 trackLength[trackCount]  // 0-32 keyframes per track
```

### Color Tracks (per track, 32 slots each)
```
int32 timeOfDay[32]    // 0-2880 inclusive, 2880 wraps to 0
uint32 packedBgrx[32]  // BGRX format
```

Only the first `trackLength` slots are valid. Remaining slots are padding (zeros).

> **[Implementation Note]**: Color tracks use **cyclic linear interpolation** over the inclusive 0..2880 domain. Time 2880 is treated as 0 for interpolation. The 2880 units/day matches the Alpha 0.5.3 day/night cycle (24 real minutes per cycle).

#### BGRX Color Format
```
packedBgrx = Blue | (Green << 8) | (Red << 16) | (Unused << 24)
// Each component 0-255, decoded to 0.0-1.0 float range
```

> **[Implementation Note]**: Colors are stored as **BGRX** (blue in lowest byte), not RGB. The unused byte (alpha) is typically 0.

### Float Bands

Fixed-count float arrays retained in disk order.

| Version | Band Count | Samples/Band | Semantics |
|---------|------------|--------------|-----------|
| v2 | 2 | 32 | Band 0: Fog End, Band 1: Fog Start Scalar (observed) |
| v83-v84 | 6 | 32 | 0:FogEnd, 1:FogStartScalar, 2-5:Sky bands (4) |
| v85 | 6 + 4 param | 32 + 10 | Same as v84 plus 4 parameter bands × 10 samples |

> **[Implementation Note]**: v2 pre-alpha only exposes 2 float bands (observed first samples: ~10000.0 and 0.25). The remaining legacy prefix (60 bytes = 15×int32 before the data sets) is consumed for alignment but not assigned guessed semantics. v85 adds 4 parameter bands (10 samples each) after `cloudMask`.

### Additional Fields (v83-v85 only)
```
int32 highlightSky   // Sky highlight index
int32 cloudMask      // Cloud mask
```

---

## Version-Specific Details

### Version 2 (0x00000002) — Pre-alpha Partial (areatest.lit)

**Observed File**: `World/Azeroth/areatest.lit` (0.5.3.3368 client)

**Complete Layout:**
```
Header (8 bytes):     version=2, rawLightCount=-1
Light Header (64B):   "Global Light", default position (all -1, zeros)
Legacy Prefix (60B):  15×int32 (all zeros in observed file)
Data Set 0 (0xA24B):  9 tracks, 2 float bands (Partial)
Data Set 1 (0xA24B):  9 tracks, 2 float bands (LegacyPartialAlternate)
Total: 8 + 64 + 60 + 0xA24 + 0xA24 = 0x1490 bytes
```

**Track Count**: 9 (vs 14/18 in later versions)

**Observed Track Data** (from implementation test fixtures):

**Data Set 0 (Primary/Partial):**
- Track 0 (DirectColor): 4 keyframes at times 0, 720, 1440, 2160
- Track 8 (Unknown): 3 keyframes at times 0, 720, 1440
- Float Band 0: First sample 10000.0 (likely Fog End)
- Float Band 1: First sample 0.25 (likely Fog Start Scalar)

**Data Set 1 (LegacyPartialAlternate):**
- **No color tracks** (all track lengths = 0)
- Float Band 0: First sample 12000.0
- Float Band 1: First sample 0.5

> **[Implementation Note]**: This is the only known v2 file. The wow-viewer implementation handles it as a special case: single light, no standard groups, two data sets with reduced track count (9). The second data set (`LegacyPartialAlternate`) contains **only float bands** (no color tracks) with different fog values. Its semantic purpose is not established — it may represent an alternate fog configuration or an earlier iteration of the format. It is retained for inspection rather than discarded. Only the primary `Partial` data set drives lighting evaluation.

### Version 83 (0x80000003)
- Track Count: 14
- Group Stride: 0x1140
- 4 groups × 14 tracks × 32 slots + 6 float bands × 32 + highlightSky + cloudMask

### Version 84 (0x80000004)
- Track Count: 18
- Group Stride: 0x1550
- 4 groups × 18 tracks × 32 slots + 6 float bands × 32 + highlightSky + cloudMask

### Version 85 (0x80000005)
- Track Count: 18
- Group Stride: 0x15F0
- Same as v84 plus 4 parameter bands × 10 samples

---

## Color Track Indices (Standard Layout)

Based on implementation and observed data:

| Index | Semantic | Notes |
|-------|----------|-------|
| 0 | Direct Color | Primary directional light color |
| 1 | Ambient Color | Ambient fill color |
| 2 | Sky Top | Zenith sky color |
| 3 | Sky Upper | Upper sky band |
| 4 | Sky Middle | Middle sky band |
| 5 | Sky Lower | Lower sky band |
| 6 | Sky Horizon | Horizon sky color |
| 7 | Fog Color | Fog color |
| 8 | Unknown | **[Implementation Note]**: Not promoted to shadow opacity without client proof. MCSH terrain visibility is a separate signal. |
| 9-17 | Additional | Present in v84/v85; semantics not fully established |

> **[Implementation Note]**: Track 8 is explicitly left as "Unknown" in the implementation. Shadow opacity semantics are an inference unless separately recovered.

---

## Time System

- **2880 time units per day** (Light/LIT clock)
- **24 real minutes per cycle** (0.5.3 contract)
- Keyframe times: inclusive 0..2880
- Cyclic interpolation: 2880 wraps to 0

> **[Implementation Note]**: The wow-viewer `WorldTimeCycle` uses this same 2880-unit clock. `ToTimeUnits(normalizedTime)` and `FromTimeUnits(timeUnits)` provide conversion to/from 0..1 normalized game time.

---

## Validation Rules (Implementation)

The wow-viewer reader enforces:

1. **Exact file length** — no trailing data allowed
2. **Track lengths** — 0..32 keyframes
3. **Keyframe times** — 0..2880 inclusive
4. **Supported versions only** — 2, 0x80000003, 0x80000004, 0x80000005
5. **rawLightCount** — -1 (v2 only) or >= 0
6. **Headers before payloads** — strict ordering
7. **Group stride exact match** — byte-perfect consumption

---

## Known Files

| Map | File | Version | Notes |
|-----|------|---------|-------|
| Azeroth | `areatest.lit` | 2 | Only known v2 file |
| Various | `lights.lit` | 83-85 | Standard layout |
| Various | `light.lit` | 83-85 | Alternate name |

> **[Implementation Note]**: The wow-viewer discovers all `.lit` files directly in the map folder (`World/<map>` or `World/Maps/<map>`), not just the conventional names.

---

## Differences from Original Wiki

| Aspect | Original Wiki | Implementation Findings |
|--------|---------------|------------------------|
| Version 2 | Not documented | Fully specified (partial layout, 9 tracks, 2 data sets) |
| Group Strides | Not specified | Exact strides per version |
| Header/Payload Order | Not specified | Strict: all headers, then all payloads |
| Spatial Coordinates | Not specified | XZY fixed-point at 1/36, Y/Z swap |
| Color Format | Not specified | BGRX packed |
| Float Band Semantics | Partial | FogEnd, FogStartScalar, 4 Sky bands, v85 parameter bands |
| Parameter Bands | Not documented | v85 only, 4×10 samples |
| Track 8 | Not specified | Unknown (not shadow opacity without proof) |
| Validation | Not specified | Exact length, bounds, ordering |

---

## References

- **Implementation**: `WowViewer.Core.IO.Lit.LitProfileReader`
- **Models**: `WowViewer.Core.IO.Lit.LitProfileModels`
- **Runtime Loader**: `WoWViewer.Terrain.LitLoader`
- **Coordinate Transform**: `WowViewer.Core.Lit.LitCoordinateTransform`
- **Tests**: `WowViewer.Core.Tests.LitProfileReaderTests`
- **Time Cycle**: `WowViewer.Core.Maps.WorldTimeCycle`

---

*Last updated: 2026-08-16 | Based on wow-viewer implementation (0.5.3.3368 client corpus)*