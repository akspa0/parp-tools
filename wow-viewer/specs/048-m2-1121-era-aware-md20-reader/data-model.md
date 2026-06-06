# Data Model: 1.12.1 Era-Aware MD20 Reader

## Domain Model

```
                +--------------------+
                |  M2Era1121Version |
                |  enum: V100|V101  |
                +---------+---------+
                          |
                          v
+-------------+     +-----+-------+     +-------------------+
| M2Dispatch  |     |             |     |  M2Era1121EraTag  |
|  Result     | <-- |  Reader     | --> |  enum: Mdlx|      |
+-------------+     |  Dispatch   |     |  Md20_1X_V100|    |
| M2Document  |     |             |     |  Md20_1X_V101|    |
| EraTag      |     +-----+-------+     |  Md20_3X_V108|    |
| IsSuccess   |           |             |  Unknown          |
| ErrorMsg    |           v             +-------------------+
+-------------+     +-----+-------+
                   |  M2Era1121   |
                   |  ModelReader |
                   +-------------+
```

## Records / Types

### `M2Era1121Version` (enum)
- `V100 = 0x100`
- `V101 = 0x101`

### `M2Era1121EraTag` (enum)
- `Mdlx = 0`
- `Md20_1X_V100 = 1`
- `Md20_1X_V101 = 2`
- `Md20_3X_V108 = 3`
- `Unknown = 99`

### `M2Era1121EraTagExtensions`
- `ToDisplayString()`:
  - `Mdlx` → `"MDLX (chunked)"`
  - `Md20_1X_V100` → `"1.12.1 (MD20 v0x100)"`
  - `Md20_1X_V101` → `"1.12.1 (MD20 v0x101)"`
  - `Md20_3X_V108` → `"3.3.5 (MD20 v0x108)"`
  - `Unknown` → `"Unknown (MD20 v?)"`

### `M2Era1121Constants`
- `Magic = 0x3032444D` (little-endian `"MD20"`)
- Header layout offsets (1.12.1):
  - `NameCountOffset = 0x08`
  - `NameOffsetOffset = 0x0C`
  - `FlagsOffset = 0x10`
  - `GlobalLoopCountOffset = 0x14`
  - `GlobalLoopOffsetOffset = 0x18`
  - `SequenceCountOffset = 0x1C`
  - `SequenceOffsetOffset = 0x20`
  - `SequenceStride = 0x6C`
  - `ColorCountOffset = 0x44`
  - `ColorOffsetOffset = 0x48`
  - `ColorStride = 0x1C`
  - `TextureWeightCountOffset = 0x54`
  - `TextureWeightOffsetOffset = 0x58`
  - `TextureWeightStride = 0x08`
  - `ViewCountOffset = 0x3C`
  - `ViewOffsetOffset = 0x40`
  - `ViewStride = 0x2C`
  - `LightCountOffset = 0xAC`
  - `LightOffsetOffset = 0xB0`
  - `LightStride = 0x0C`
  - `CameraCountOffset = 0xB4`
  - `CameraOffsetOffset = 0xB8`
  - `CameraStride = 0x2C`
  - `RibbonCountOffset = 0xC4`
  - `RibbonOffsetOffset = 0xC8`
  - `RibbonStride = 0x7C`
  - `ParticleCountOffset = 0xCC`
  - `ParticleOffsetOffset = 0xD0`
  - `ParticleStride = 0xDC`
  - `V101ExtraCountOffset = 0xDC` (0x101 only)
  - `V101ExtraOffsetOffset = 0xE0` (0x101 only)
  - `V101ExtraStride = 0x1F8` (0x101 only)

### `M2DispatchResult` (record, new in dispatcher)
- `Document: M2ModelDocument`
- `Era: M2Era1121EraTag`
- `IsSuccess: bool` (true if document is non-null)
- `ErrorMessage: string?` (null on success)

### `M2ModelDocument` (existing, unchanged)
The reader populates:
- `Identity` (via `M2ModelIdentity.FromPath(sourcePath)`)
- `Version` (set to 0x100 or 0x101)
- `Flags` (read from `FlagsOffset = 0x10`)
- `ModelName` (read from `NameOffset`/`NameCount`, ASCII bytes)
- `GlobalLoops` (read from `GlobalLoopOffset`/`GlobalLoopCount`, 4-byte M2Track-style)
- `Sequences` (read from `SequenceOffset`/`SequenceCount`, 0x6c stride)
- `Colors` (read from `ColorOffset`/`ColorCount`, 0x1c stride)
- `TextureWeights` (read from `TextureWeightOffset`/`TextureWeightCount`, 0x08 stride)
- `TextureTransforms` (placeholder for the 0x30 records)
- `Lights` (read from `LightOffset`/`LightCount`, 0x0c stride — position + radius)
- `Cameras` (read from `CameraOffset`/`CameraCount`, 0x2c stride)
- `Ribbons` (read from `RibbonOffset`/`RibbonCount`, 0x7c stride)
- `Particles` (read from `ParticleOffset`/`ParticleCount`, 0xdc stride)
- `ViewCount` (read from `ViewCountOffset = 0x3C`; per-record view contents deferred to a future slice)

## Validation

- Header span: ≥ 0xD4 (0x100) or ≥ 0xE8 (0x101) bytes.
- Magic: `0x3032444D`.
- Version: `0x100` or `0x101`.
- Bounds checks: `count * stride + offset ≤ streamLength` for every table.
- Truncation: surfaces as a "not yet mapped" log + skip; does not throw `EndOfStreamException`.
- Non-finite floats: surfaces as a warning + zero-fill; does not throw.

## Open Schema Gaps

- `M2ModelDocument` has no field for the 1.12.1 view record's per-record sub-tables (9 nested sub-tables per view). Currently only `ViewCount` is exposed.
- `M2ModelDocument` has no field for 1.12.1 camera per-frame data (`CameraPerFrameStride = 0xD4`).
- `M2ModelDocument` has no field for the 1.12.1 vertex/normal layout (0x0C positions + 0x0C normals).
- `M2ModelDocument` has no field for 1.12.1 bone lookup pairs (0x4 per entry).

These are schema-level follow-ups for spec 049 / 050; the 048 MVP does not land them per A-006.
