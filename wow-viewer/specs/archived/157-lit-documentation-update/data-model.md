# Data Model: LIT Documentation Update

## Entities

### LITFile
Represents a complete LIT file.

| Field | Type | Description |
|-------|------|-------------|
| sourcePath | string | Archive or filesystem path |
| version | uint32 | Format version (2, 0x80000003, 0x80000004, 0x80000005) |
| rawLightCount | int32 | -1 for v2 partial, >=0 for standard |
| trackCount | int | Number of color tracks per group |
| groupStride | int | Byte size of one group |
| lights | LightProfile[] | Array of light profiles |
| isSinglePartialProfile | bool | True if rawLightCount == -1 |

### LightProfile
One spatial light entry with its weather/water groups.

| Field | Type | Description |
|-------|------|-------------|
| index | int | Light index in file |
| header | LightHeader? | Spatial header (null for modern partial) |
| groups | LightGroupProfile[] | 4 groups (standard) or 2 data sets (v2) |
| isPartial | bool | True for single-partial profiles |

### LightHeader
64-byte spatial header (fixed layout across versions).

| Field | Type | Description |
|-------|------|-------------|
| index | int | Light index |
| chunkX | int | ADT chunk X (-1 = default/global) |
| chunkY | int | ADT chunk Y (-1 = default/global) |
| chunkRadius | int | Chunk radius (-1 = default/global) |
| position | Vector3 | Raw fixed-point XZY (1/36 world units) |
| radius | float | Core radius (fixed-point) |
| dropoff | float | Falloff distance (fixed-point, 0 for v2) |
| name | string | Null-terminated name (max 31 chars) |
| isDefault | bool | chunkX==-1 && chunkY==-1 && chunkRadius==-1 |
| worldPosition | Vector3 | Semantic game-world XYZ (computed) |
| worldRadius | float | Radius in world units (computed) |
| worldDropoff | float | Dropoff in world units (computed) |
| worldOuterRadius | float | Max(worldRadius, worldRadius + max(worldDropoff, 0)) |

### LightGroupProfile
One weather/water lighting group.

| Field | Type | Description |
|-------|------|-------------|
| index | int | Group index (0-3 for standard, 0-1 for v2) |
| kind | LightGroupKind | Clear, Storm, ClearWater, StormWater, Partial, LegacyPartialAlternate |
| tracks | ColorTrack[] | Color tracks (version-dependent count) |
| floatBands | FloatBand[] | Float array bands (6 for v83-v85, 2 for v2) |
| highlightSky | int? | Sky highlight index (v83-v85 only) |
| cloudMask | int? | Cloud mask (v83-v85 only) |
| parameterBands | FloatBand[] | Parameter bands (v85 only, 4×10) |
| encodedSize | int | Actual bytes consumed |

### ColorTrack
Time-keyframed color track with cyclic interpolation.

| Field | Type | Description |
|-------|------|-------------|
| index | int | Track index (0-17) |
| declaredLength | int | Number of valid keyframes (0-32) |
| keyframes | ColorKeyframe[] | Sorted keyframes |
| evaluationKeyframes | ColorKeyframe[] | Time-sorted for interpolation |

### ColorKeyframe
Single time/color sample.

| Field | Type | Description |
|-------|------|-------------|
| timeOfDay | int | 0-2880 (inclusive) |
| packedBgrx | uint32 | BGRX packed color |
| color | Vector3 | Decoded RGB (0-1 range) |

### FloatBand
Float array retained in disk order.

| Field | Type | Description |
|-------|------|-------------|
| index | int | Band index |
| samples | float[] | 32 samples (10 for parameter bands) |

## Enums

### LightGroupKind
```
Clear = 0
Storm = 1
ClearWater = 2
StormWater = 3
Partial = 4
LegacyPartialAlternate = 5
```

## Validation Rules

1. File length must exactly match expected layout
2. Track lengths: 0-32
3. Keyframe times: 0-2880
4. Version must be supported
5. rawLightCount: -1 (v2 only) or >=0
6. All headers before payloads
7. Group stride exact match

## Relationships

```
LITFile (1) → (N) LightProfile
LightProfile (1) → (1) LightHeader (optional)
LightProfile (1) → (4 or 2) LightGroupProfile
LightGroupProfile (1) → (trackCount) ColorTrack
ColorTrack (1) → (declaredLength) ColorKeyframe
LightGroupProfile (1) → (6 or 2) FloatBand
LightGroupProfile (1) → (4) FloatBand (v85 parameter bands)