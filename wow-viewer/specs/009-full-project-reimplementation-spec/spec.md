# Full Project Reimplementation Specification

**Feature:** 009 — Complete Reimplementation Spec
**Date:** 2026-05-22
**Purpose:** Design specification sufficient to fully reimplement the MdxViewer + wow-viewer project as a single new project in a new repository from scratch, preserving all functionality.

---

## 1. Executive Summary

This project is a **World of Warcraft data viewer, terrain AI pipeline, and format toolkit**. It reads game client files (MPQ archives containing ADT, WDT, WMO, M2, MDX, BLP, DBC, PM4, and related formats), renders terrain and 3D models for inspection, converts terrain between expansion eras, and trains neural networks to reconstruct terrain geometry from minimap images.

The system has three major product surfaces:
1. **Desktop Viewer** — Interactive 3D viewer for WoW terrain, models, and worlds
2. **Format Toolkit** — CLI tools for inspection, conversion, and data extraction
3. **Terrain AI Pipeline** — Harvest → Dataset → Train → Inference pipeline for terrain reconstruction

### Key Statistics
- **~200+ game file format chunk readers/writers** across 15+ WoW file formats
- **6 supported client builds** (0.5.3 through 4.0.0)
- **5 independent ML models** (V16.1 family: height, normal, holes, liquid, texture composition)
- **~35,000+ lines of C#** (core libraries, tools, viewer)
- **~8,000+ lines of Python** (dataset building, training, inference)
- **~140 test files** with broad format coverage

---

## 2. Architecture Overview

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Product Surfaces                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Desktop  │  │   CLI    │  │  Harvest │  │  Python    │  │
│  │ Viewer   │  │  Tools   │  │  Stream  │  │  Training  │  │
│  │ (App)    │  │ (Inspect,│  │  (C#→Py) │  │  Scripts   │  │
│  │          │  │ Converter│  │          │  │            │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │              │              │               │         │
│  ┌────┴──────────────┴──────────────┴───────────────┴──────┐ │
│  │              Runtime Pipeline                           │ │
│  │  M2 Animation/Skinning  │  World Composition  │  Effects│ │
│  └──────────────────────────┬──────────────────────────────┘ │
│                             │                                │
│  ┌──────────────────────────┴──────────────────────────────┐ │
│  │              Core I/O Layer                             │ │
│  │  ADT │ WDT │ WMO │ M2/MDX │ BLP │ DBC │ PM4 │ MPQ    │ │
│  │  Readers + Writers + Converters + Archive Access        │ │
│  └──────────────────────────┬──────────────────────────────┘ │
│                             │                                │
│  ┌──────────────────────────┴──────────────────────────────┐ │
│  │              Domain Models (Core)                       │ │
│  │  Maps │ WMO │ M2 │ MDX │ BLP │ PM4 │ Chunks │ Files   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Core Libraries | C# / .NET 10 | Domain models, format readers, I/O |
| Rendering | Silk.NET (OpenGL 3.3) | GPU rendering, windowing, input |
| UI | ImGui (via Silk.NET) | All viewer panels and controls |
| Compression | SharpZipLib | MPQ/PKWare decompression |
| Texture Decode | SereniaBLPLib + BCnEncoder.Net | BLP texture reading |
| 3D Export | SharpGLTF.Toolkit | glTF/GLB export |
| DBC Database | DBCD + WoWDBDefs | DBC/DB2 table reading |
| Archive Access | NativeMpqService (P/Invoke) | MPQ archive reading |
| ML Training | Python 3.11+ / PyTorch | Model training and inference |
| Dataset Storage | Zarr v3 + Blosc | Array storage with compression |
| Metadata | Parquet (PyArrow) | Index and placement data |
| Python Env | uv | Environment management |

### 2.3 Repo Independence

The project must be extractable as a standalone repository. No source file may reference paths outside the project root. All shared code lives in `src/core/`. External dependencies are vendored in `libs/`.

---

## 3. Binary Format Specifications

### 3.1 Chunk Header (Universal)

Every WoW chunked format uses an 8-byte header:

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0x00 | 4 | char[4] | FourCC tag (e.g. `MVER`, `MPHD`, `MAIN`) |
| 0x04 | 4 | uint32 LE | Payload size in bytes |

Odd-sized chunks are padded to 2-byte alignment.

### 3.2 WDT (World Definition Table)

#### Alpha WDT

**Chunks read in order:**

| Order | FourCC | Purpose |
|-------|--------|---------|
| 1 | `MVER` | Version (single uint32) |
| 2 | `MPHD` | Map header — contains offsets to MDNM/MONM |
| 3 | `MAIN` | Main tile index — 64x64 entries |

**MPHD layout (≥16 bytes):**

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0x00 | 4 | uint32 | Flags |
| 0x04 | 4 | int32 | Absolute file offset to `MDNM` chunk |
| 0x08 | 4 | (reserved) | |
| 0x0C | 4 | int32 | Absolute file offset to `MONM` chunk |

**MAIN layout:**
- Total size: 65536 bytes (64 × 64 × 16 bytes per entry)
- Each entry: **16 bytes**
- Entry at `(tileX, tileY)` is at index `tileY * 64 + tileX`
- Offset 0x00: `int32 LE` = absolute offset to embedded ADT data. If 0, tile is absent.

**MDNM / MONM:** Null-terminated UTF-8 string tables for doodad (`.m2`) and WMO model names.

#### LK WDT

Standard MAIN entry is **8 bytes** (vs Alpha's 16):

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 4 | uint32 | `flags` (bit 0 = HasAdt, bit 1 = AllWater, bit 2 = Loaded) |
| 0x04 | 4 | uint32 | `asyncId` |

### 3.3 ADT (Area Data Tile)

#### Alpha Embedded ADT (inside WDT)

Tile data starts at the absolute offset from MAIN. Begins with `MHDR` chunk:

**Alpha MHDR layout (128-byte header after chunk header):**

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0x00 | 4 | int32 | Relative offset to `MCIN` |
| 0x04 | 4 | int32 | Relative offset to `MTEX` |
| 0x08 | 4 | (reserved) | |
| 0x0C | 4 | int32 | Relative offset to `MDDF` |
| 0x10 | 4 | (reserved) | |
| 0x14 | 4 | int32 | Relative offset to `MODF` |

All offsets relative to `adtOffset + 8` (start of MHDR payload).

**MCIN:** 256 entries × 16 bytes. Each entry offset 0x00: `int32 LE` = absolute offset to MCNK chunk. Chunk index `i` → chunkX = `i % 16`, chunkY = `i / 16`.

#### LK Split ADT

LK ADTs split into three files: root `.adt`, `_tex0.adt`, `_obj0.adt`. The reader merges all three via the same `ParseAdtStream` loop. Reversed FourCCs are also recognized (`KNCM` = `MCNK` reversed).

#### MCNK Header (Both Eras) — 128 bytes

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 4 | uint32 | `flags` |
| 0x04 | 4 | int32 | `indexX` (0..15) |
| 0x08 | 4 | int32 | `indexY` (0..15) |
| 0x10 | 4 | int32 | `layerCount` (Alpha) / `nLayers` (LK) |
| 0x18 | 4 | int32 | Relative offset to `MCVT` |
| 0x1C | 4 | int32 | Relative offset to `MCNR` |
| 0x20 | 4 | int32 | Relative offset to `MCLY` |
| 0x28 | 4 | int32 | Relative offset to `MCAL` |
| 0x2C | 4 | int32 | `sizeMcal` |
| 0x30 | 4 | int32 | Relative offset to `MCSH` |
| 0x34 | 4 | int32 | `sizeMcsh` |
| 0x3C | 4 | int32 | `nMapObjRefs` / `holeMask` (LK: offset 0x3C) |
| 0x5C | 4 | int32 | `mcnkChunksSize` (total sub-chunk data size) |
| 0x64 | 4 | int32 | Relative offset to `MCLQ` |
| 0x70 | 4 | float32 | `baseHeight` (LK only — added to MCVT heights) |
| 0x74 | 4 | int32 | Relative offset to `MCCV` |

All sub-chunk offsets relative to `mcnkOffset + 8 + 128`.

**Key difference:** LK MCVT heights are **relative** to `baseHeight` at offset 0x70. Alpha MCVT values are absolute.

#### MCVT — 580 bytes = 145 × float32

145 height values in a **staggered grid** of 17 rows:
- Even rows (0,2,4,...,16): 9 vertices each
- Odd rows (1,3,5,...,15): 8 vertices each
- Total: 9×9 + 8×8 = **145** vertices per chunk

Each vertex: `float32 LE` — world-space Z (absolute for Alpha, relative+baseHeight for LK).

Index layout: even rows at offset `outerRow * 9 + col`, odd rows at offset `81 + innerRow * 8 + col`.

#### MCNR — 435 bytes = 145 × 3 bytes

145 packed normals, same staggered-grid vertex order. Each normal: 3 `sbyte` values (nx, nz, ny) — **note swizzled XZY order**.

Decoded as: `float = sbyte / 127.0`, clamped to [-1, 1].

#### MCLY — 16 bytes per layer

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0x00 | 4 | uint32 | Texture ID (index into MTEX name table) |
| 0x04 | 4 | uint32 | Flags (bit 9 = `0x200` = big alpha) |
| 0x08 | 4 | uint32 | Alpha offset (relative to MCAL data start) |
| 0x0C | 4 | uint32 | (reserved/effect ID) |

#### MCAL Decoding (4 Encoding Types)

**Encoding 1: Packed4Bit** (2048 bytes source → 4096 bytes output)

Source: 2048 bytes. Each source byte encodes 2 alpha pixels using 4-bit nibbles.

```
for each byte in source:
    low_nibble  = (byte & 0x0F) * 17   // 4-bit -> 8-bit expansion
    high_nibble = ((byte >> 4) & 0x0F) * 17
    output[pixel++] = low_nibble
    output[pixel++] = high_nibble      // column 31: high = low
```

Output: 64 rows × 64 pixels. **Edge fix applied:** last column = column 62; last row = row 62; bottom-right = (62,62).

**Encoding 2: Compressed** (RLE, variable-length → 4096 bytes)

```
while writePos < 4096 and readPos < sourceEnd:
    control = source[readPos++]
    fill = (control & 0x80) != 0
    count = control & 0x7F
    if count == 0: continue
    if fill:
        value = source[readPos++]
        output[writePos .. writePos+count] = value
    else:
        output[writePos .. writePos+count] = source[readPos .. readPos+count]
```

**Encoding 3: BigAlpha** (4096 bytes → 4096 bytes) — Direct copy.

**Encoding 4: BigAlphaFixed** (truncated BigAlpha with expansion) — Used when big-alpha data is truncated to < 4096 bytes but ≥ 63×63. Replicates last available byte per row, copies row 62 to row 63.

**MCAL Span Resolution:**
- span ≥ 4096 → force BigAlpha
- span ≤ 2048 → force Packed4Bit
- LKStrict and span > 0 && span < 2048 and flag 0x200 not set → force Compressed

#### MDDF (Doodad Placements) — 36 bytes per entry

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0x00 | 4 | int32 | `nameId` (index into MDNM name table) |
| 0x04 | 4 | int32 | `uniqueId` |
| 0x08 | 4 | float32 | `filePosX` (raw ADT X) |
| 0x0C | 4 | float32 | `filePosY` (raw ADT Y) |
| 0x10 | 4 | float32 | `filePosZ` (raw ADT Z / up) |
| 0x14 | 4 | float32 | `fileRotX` (degrees) |
| 0x18 | 4 | float32 | `fileRotY` (degrees) |
| 0x1C | 4 | float32 | `fileRotZ` (degrees) |
| 0x20 | 2 | uint16 | `scale` (divided by 1024 for float) |

**Coordinate transform:** `rendererX = 17066.666 - filePosZ`, `rendererY = 17066.666 - filePosX`, `rendererZ = filePosY`

#### MODF (WMO Placements) — 64 bytes per entry

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0x00 | 4 | int32 | `nameId` (index into MONM name table) |
| 0x04 | 4 | int32 | `uniqueId` |
| 0x08-0x1C | 48 | float[12] | Position (3) + Rotation (3) + Extents (6) |
| 0x38 | 2 | uint16 | `flags` |
| 0x3A | 2 | (padding) | |

#### MCCV (Vertex Colors) — 580 bytes

145 RGBA vertex colors (4 bytes each). R, G, B extracted as `byte / 255.0`, alpha discarded.

#### MCSH (Shadow) — up to 512 bytes

Shadow map: 64×64 bitmask, packed as 8 bytes per row. Each byte encodes 8 horizontal pixels; bit 0 = leftmost.

#### MCLQ (Legacy Liquid)

- Offset 0x00: `float32` min height
- Offset 0x04: `float32` max height
- Offset 0x08: 81 vertex heights at 8 bytes each (float32 + 4 reserved bytes)
- Offset 0x290: 64-byte tile flags

### 3.4 BLP (Texture Format)

#### BLP2 Header — 148 bytes

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 4 | char[4] | Signature `"BLP2"` |
| 0x04 | 4 | uint32 | `Version` |
| 0x08 | 1 | byte | `Compression` (0=Palettized, 1=JPEG, 2=DXTC) |
| 0x09 | 1 | byte | `AlphaDepthBits` |
| 0x0A | 1 | byte | `PixelFormat` |
| 0x0B | 1 | byte | `MipMapType` |
| 0x0C | 4 | uint32 | `Width` |
| 0x10 | 4 | uint32 | `Height` |
| 0x14 | 64 | uint32[16] | `MipMapOffsets` |
| 0x54 | 64 | uint32[16] | `MipMapSizes` |

Post-header: Palettized → 1024-byte palette (256 × RGBA). JPEG → uint32 `jpegHeaderSize` + header bytes.

#### BLP0/BLP1 Header — 156 bytes

Same as BLP2 but with `Compression` as uint32 at offset 0x04 and different field offsets.

### 3.5 M2 (Modern Model)

**Signature:** `MD20` at offset 0x00. Minimum header size: 0x110 bytes.

#### M2 Header (Fixed Offsets)

| Offset | Size | C# Type | Field |
|--------|------|---------|-------|
| 0x00 | 4 | char[4] | Signature ("MD20") |
| 0x04 | 4 | uint32 | Version |
| 0x08 | 4 | uint32 | NameCount |
| 0x0C | 4 | uint32 | NameOffset |
| 0x10 | 4 | uint32 | Flags |
| 0x1C | 4 | uint32 | SequenceCount |
| 0x20 | 4 | uint32 | SequenceOffset |
| 0x2C | 4 | uint32 | BoneCount |
| 0x30 | 4 | uint32 | BoneOffset |
| 0x3C | 4 | uint32 | VertexCount |
| 0x40 | 4 | uint32 | VertexOffset |
| 0x44 | 4 | uint32 | ViewCount |
| 0x50 | 4 | uint32 | TextureCount |
| 0x54 | 4 | uint32 | TextureOffset |
| 0x70 | 4 | uint32 | RenderFlagCount |
| 0x74 | 4 | uint32 | RenderFlagOffset |
| 0xA0 | 12 | float[3] | BoundsMin |
| 0xAC | 12 | float[3] | BoundsMax |
| 0xB8 | 4 | float | BoundsRadius |

#### M2 Track (Animation Block) — Stride: 0x14 (20 bytes)

| Offset | Size | C# Type | Field |
|--------|------|---------|-------|
| 0x00 | 2 | uint16 | InterpolationType (0=None, 1=Linear, 2=Hermite, 3=Bezier) |
| 0x02 | 2 | uint16 | GlobalSequenceIndex (0xFFFF = none) |
| 0x04 | 4 | uint32 | TimestampArray.Count |
| 0x08 | 4 | uint32 | TimestampArray.Offset |
| 0x0C | 4 | uint32 | ValueArray.Count |
| 0x10 | 4 | uint32 | ValueArray.Offset |

#### M2 Vertex Format — Stride: 0x30 (48 bytes)

| Offset | Size | C# Type | Field |
|--------|------|---------|-------|
| 0x00 | 12 | float[3] | Position |
| 0x0C | 4 | byte[4] | BoneWeight0-3 (read as Vector4(w0/255, ...)) |
| 0x10 | 4 | byte[4] | BoneIndex0-3 |
| 0x14 | 12 | float[3] | Normal |
| 0x20 | 8 | float[2] | TextureCoords0 |
| 0x28 | 8 | float[2] | TextureCoords1 |

#### M2 Sequence Definition — Stride: 0x40 (64 bytes)

| Offset | Size | C# Type | Field |
|--------|------|---------|-------|
| 0x00 | 2 | uint16 | AnimationId |
| 0x04 | 4 | uint32 | Duration (ms) |
| 0x08 | 4 | float | MoveSpeed |
| 0x0C | 4 | uint32 | Flags |
| 0x10 | 2 | int16 | Frequency |
| 0x14 | 4 | uint32 | ReplayMinimum |
| 0x18 | 4 | uint32 | ReplayMaximum |
| 0x1C | 2 | uint16 | BlendTimeIn |
| 0x1E | 2 | uint16 | BlendTimeOut |
| 0x20 | 12 | float[3] | BoundsMin |
| 0x2C | 12 | float[3] | BoundsMax |
| 0x3C | 2 | int16 | VariationNext |
| 0x3E | 2 | uint16 | AliasNext |

#### M2 Bone Definition — Stride: 0x58 (88 bytes)

| Offset | Size | C# Type | Field |
|--------|------|---------|-------|
| 0x00 | 4 | int32 | KeyBoneId |
| 0x04 | 4 | uint32 | Flags |
| 0x08 | 2 | int16 | ParentBone |
| 0x0A | 2 | uint16 | SubmeshId |
| 0x0C | 4 | uint32 | BoneNameCrc |
| 0x10 | 20 | Track<Vector3> | TranslationTrack |
| 0x24 | 20 | Track<M2CompQuaternion> | RotationTrack |
| 0x38 | 20 | Track<Vector3> | ScalingTrack |
| 0x4C | 12 | float[3] | Pivot |

**M2CompQuaternion** (4 × int16, 8 bytes): value < 0 → `(value + 32768) / 32767.0`; value ≥ 0 → `(value - 32767) / 32767.0`. Identity: `(32767, 32767, 32767, -1)`.

#### M2 Skin Profile (.skin)

**Signature:** `SKIN`. Header:

| Offset | Size | C# Type | Field |
|--------|------|---------|-------|
| 0x00 | 4 | char[4] | Signature ("SKIN") |
| 0x04 | 4 | uint32 | VertexLookupCount |
| 0x08 | 4 | uint32 | VertexLookupOffset |
| 0x0C | 4 | uint32 | TriangleIndexCount |
| 0x10 | 4 | uint32 | TriangleIndexOffset |
| 0x1C | 4 | uint32 | SubmeshCount |
| 0x20 | 4 | uint32 | SubmeshOffset |
| 0x24 | 4 | uint32 | BatchCount |
| 0x28 | 4 | uint32 | BatchOffset |

**Skin Submesh — Stride: 0x30 (48 bytes):**

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 2 | uint16 | SkinSectionId |
| 0x02 | 2 | uint16 | Level |
| 0x04 | 2 | uint16 | VertexStart |
| 0x06 | 2 | uint16 | VertexCount |
| 0x08 | 2 | uint16 | IndexStart |
| 0x0A | 2 | uint16 | IndexCount |

**Skin Batch — Stride: 0x18 (24 bytes):**

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 1 | byte | Flags |
| 0x02 | 2 | uint16 | ShaderId |
| 0x04 | 2 | uint16 | SkinSectionIndex |
| 0x08 | 2 | int16 | ColorIndex |
| 0x0A | 2 | uint16 | RenderFlagsIndex |
| 0x0E | 2 | uint16 | TextureCount |
| 0x10 | 2 | uint16 | TextureComboIndex |
| 0x12 | 2 | uint16 | TextureCoordComboIndex |
| 0x14 | 2 | uint16 | TransparencyComboIndex |

#### M2 Render Flag — Stride: 4 bytes

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 2 | uint16 | Flags |
| 0x02 | 2 | uint16 | RawBlendMode (0=Opaque, 1=AlphaKey, 2=AlphaBlend, 3=NoAlphaAdd, 4=Add, 5=Mod, 6=Mod2X, 7=BlendAdd) |

### 3.6 MDX (Legacy Model)

**Signature:** `MDLX`. Sequential tagged chunks with FourCC + uint32 size.

**Key chunks:** `VERS` (version, only 1300/1400), `MODL` (name, 0x50 bytes), `GEOS` (geometry), `BONE` (skeleton), `MTLS` (materials), `TEXS` (textures), `PIVT` (pivot points).

**MDX GEOS sub-chunks:** `VRTX` (positions), `NRMS` (normals), `UVBS` (UVs), `PVTX` (indices), `GNDX` (vertex groups), `MTGC`/`MATS` (matrix groups).

**MDX BONE:** Length-prefixed nodes with `KGTR` (translation), `KGRT` (rotation), `KGSC` (scaling) track sub-chunks. Pivots from separate `PIVT` chunk (count = chunkSize / 12, each Vector3 = 12 bytes).

**MDX MTLS:** Materials with layers. Each layer has `BlendMode`, `TextureId`, `TransformId`, optional `KMTA` (alpha track), `KMTE` (emissive track).

### 3.7 WMO (World Map Object)

#### WMO Root Chunks (in order)

`MVER`, `MOMO` (optional wrapper), `MOHD`, `MOTX`, `MOMT`, `MOGN`, `MOGI`, `MOSB`, `MOPV`, `MOPT`, `MOPR`, `MOVV`, `MOVB`, `MOLT`, `MODS`, `MODN`, `MODD`, `MFOG`, `MCVP`

#### MOHD — 64 bytes

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 4 | uint32 | MaterialCount |
| 0x04 | 4 | uint32 | GroupCount |
| 0x08 | 4 | uint32 | PortalCount |
| 0x0C | 4 | uint32 | LightCount |
| 0x10 | 4 | uint32 | DoodadNameCount |
| 0x14 | 4 | uint32 | DoodadPlacementCount |
| 0x18 | 4 | uint32 | DoodadSetCount |
| 0x24 | 12 | float[3] | BoundsMin |
| 0x30 | 12 | float[3] | BoundsMax |
| 0x3C | 4 | uint32 | Flags |

#### MOMT Material — 64/48/44 bytes (version-dependent)

Standard 64-byte layout:

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 4 | uint32 | Flags |
| 0x04 | 4 | uint32 | Shader |
| 0x08 | 4 | uint32 | BlendMode |
| 0x0C | 4 | uint32 | Texture1Offset (into MOTX) |
| 0x18 | 4 | uint32 | Texture2Offset |
| 0x24 | 4 | uint32 | Texture3Offset |

#### WMO Group Chunks (inside MOGP)

`MOPY` (material info), `MOVI` (vertex indices), `MOVT` (vertices Vector3[]), `MONR` (normals), `MOTV` (UVs), `MOBA` (batch info), `MOLR` (light refs), `MOBN` (BSP nodes), `MOCV` (vertex colors), `MLIQ` (liquid), `MODR` (doodad refs).

#### Portal System

**MOPV Portal Vertex — 12 bytes:** float X, Y, Z.

**MOPT Portal Info — 20 bytes:** uint16 StartVertex, uint16 VertexCount, float Normal.X/Y/Z, float PlaneDistance.

**MOPR Portal Reference — 8 bytes:** uint16 PortalIndex, uint16 GroupIndex, int16 Side (+1/-1).

#### Doodad System

**MODS Doodad Set — 32 bytes:** char[20] Name, uint32 StartIndex, uint32 Count, uint32 Flags.

**MODD Doodad Placement — 40 bytes:** uint32 NameIndex (masked `& 0x00FFFFFF`), float[3] Position, float[4] Rotation (quaternion), float Scale, uint32 ColorBGRA.

### 3.8 WDL (World Data Low-res)

**Chunks:** `MVER`, `MAOF` (64×64 = 4096 uint32 offsets), per-tile `MARE`.

**MARE payload:** 1090 bytes — int16[289] outer heights (17×17 grid) + int16[256] inner heights (16×16 grid).

### 3.9 LIT (Lighting)

**Header — 8 bytes:** uint32 VersionNumber, int32 LightCount.

**Light Entry — 64 bytes:** int32 ChunkX, ChunkY, ChunkRadius; float Position.X/Y/Z; float LightRadius, LightDropoff; char[32] Name.

### 3.10 MPQ Archive

**Magic:** `0x1A51504D` = `"MPQ\x1A"`.

**Header v0 — 32 bytes:**

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0x00 | 4 | uint32 | Magic |
| 0x04 | 4 | uint32 | HeaderSize |
| 0x08 | 4 | uint32 | ArchiveSize |
| 0x0C | 2 | uint16 | FormatVersion (0 or 1) |
| 0x0E | 2 | uint16 | SectorSizeShift (sector size = `512 << shift`) |
| 0x10 | 4 | uint32 | HashTableOffset |
| 0x14 | 4 | uint32 | BlockTableOffset |
| 0x18 | 4 | uint32 | HashTableEntries |
| 0x1C | 4 | uint32 | BlockTableEntries |

**HashEntry — 16 bytes:** Name1 (uint32), Name2 (uint32), Locale (uint16), Platform (uint16), BlockIndex (uint32). Special: `0xFFFFFFFF` = empty, `0xFFFFFFFE` = deleted.

**BlockEntry — 16 bytes:** BlockOffset, BlockSize (compressed), FileSize (uncompressed), Flags.

**Flag bits:** 31=FlagExists, 24=FlagSingleUnit, 17=FlagFixKey, 16=FlagEncrypted, 9=FlagCompressed.

**Encryption:** Hash string algorithm with CryptTable (0x500 entries, seed = 0x00100001, `seed = (seed * 125 + 3) % 0x2AAAAB`). Block decryption: `seed += CryptTable[0x400 + (key & 0xFF)]`, temp = data[i] ^ (key + seed), key rotation.

**Compression types (first byte mask):** 0x01=Huffman, 0x02=Zlib, 0x08=Pkware, 0x10=BZip2, 0x80=LZMA.

---

## 4. Rendering Pipeline Specifications

### 4.1 OpenGL Version

All shaders use `#version 330 core`.

### 4.2 Terrain Rendering

#### Vertex Format (Interleaved, Stride = 44 bytes)

| Location | Name | Type | Components | Offset |
|----------|------|------|------------|--------|
| 0 | `aPosition` | float | 3 (vec3) | 0 |
| 1 | `aNormal` | float | 3 (vec3) | 12 |
| 2 | `aTexCoord` | float | 2 (vec2) | 24 |
| 3 | `aChunkSlice` | uint | 1 (UNSIGNED_BYTE) | separate VBO, 1 byte |
| 4 | `aTexIndices` | uvec4 | 4 (UNSIGNED_SHORT×4) | separate VBO, 8 bytes |
| 5 | `aFallbackColor` | float | 3 (vec3) | 32 |

Index buffer: uint32.

#### Shader Uniforms

**Vertex:** `uView` (mat4), `uProjection` (mat4).

**Fragment:** `uDiffuseArray` (sampler2DArray, unit 0), `uAlphaShadowArray` (sampler2DArray, unit 1), `uDiffuseLayerCount` (int), `uLightDirection` (vec3, hardcoded `(-0.45, -0.55, 0.70)`), `uLightColor` (vec3, `(0.80, 0.82, 0.78)`), `uAmbientColor` (vec3, `(0.28, 0.30, 0.34)`).

#### Texture Units

| Unit | Target | Content |
|------|--------|---------|
| 0 | Texture2DArray | Diffuse tileset array (all chunk textures, resampled to 64/128/256, RGBA8, LinearMipmapLinear, Repeat) |
| 1 | Texture2DArray | Alpha-shadow array (64×64×256 slices, RGBA8, Linear, ClampToEdge) |

#### Geometry

8×8 sub-cells per chunk, 17×17 outer vertices per chunk with 8×8 inner center vertices. Hole-mask bits disable 2×2-cell triangles. Each cell produces 4 fan triangles from center vertex. World-space UV: `(-worldY, -worldX) * (8.0 / 33.333)`.

#### Render Pass Order

1. Clear (fog color, color+depth)
2. Sky (fullscreen triangle, no depth test)
3. Terrain tiles (depth test Lequal, one DrawElements per tile)
4. Hole overlay (alpha blend)
5. Instance markers (alpha blend, point sprites)

### 4.3 M2 Model Rendering

#### Vertex Format (Stride = 32 bytes)

| Location | Name | Type | Components | Offset |
|----------|------|------|------------|--------|
| 0 | `aPos` | float | 3 | 0 |
| 1 | `aNormal` | float | 3 | 12 |
| 2 | `aTexCoord` | float | 2 | 24 |

**No bone weights in GPU format** — skinning applied on CPU before render. Index buffer: uint32.

#### Shader Uniforms

**Vertex:** `uView`, `uProj` (mat4), `uHasUvTransform` (bool), `uUvTranslation` (vec2), `uUvScale` (vec2), `uUvRotation` (vec2 — 2D rotation matrix columns).

**Fragment:** `uLightDir` (vec3, `(-0.5, 0.8, 0.35)` normalized), `uLightColor`, `uAmbientColor`, `uBaseColor`, `uEmissiveColor`, `uAlpha`, `uHasTexture`, `uTexture0` (sampler2D, unit 0), `uAlphaCutout` (bool, discard if alpha < 0.5), `uReceivesLighting` (bool).

#### Two-Pass Rendering

1. **Opaque pass:** blend disabled, depth write enabled
2. **Transparent pass:** blend enabled, back-to-front sorted

Per-command blend modes: Additive/NoAlphaAdd/BlendAdd → `SrcAlpha, One`. Mod/Mod2X → `DstColor, Zero`. Default → `SrcAlpha, OneMinusSrcAlpha`.

### 4.4 WMO Rendering

#### Vertex Format (Stride = 32 bytes)

Same as M2: vec3 position, vec3 normal, vec2 texcoord. Index buffer: uint16.

#### Shader Uniforms

`uLightDir` (vec3, `(0.35, 0.45, 1.0)` normalized), `uAmbientColor` (`(0.30, 0.30, 0.34)`), `uBaseColor`, `uHasTexture`, `uTexture0`, `uAlphaTestThreshold` (0.5 for AlphaKey, 0.0 otherwise), `uUseTextureAlpha` (bool).

Fragment: `light = max(dot(normalize(vNormal), normalize(uLightDir)), 0.18)`. Final: `shaded * clamp(ambient + light, 0, 1.75)`.

#### Two-Pass Rendering

1. **Opaque pass:** insertion order
2. **Transparent pass:** sorted by `DistanceSquared(cameraPosition, sortCenter)` descending (far-first)

### 4.5 World Composition (8 Layers)

| # | Kind | Name | Description |
|---|------|------|-------------|
| 0 | Sky | "Spherical Sky" | Procedural camera-centered backdrop |
| 1 | SkyboxBackdrop | "Skybox Backdrop" | Skybox model placements |
| 2 | Wdl | "Far Terrain (WDL)" | Low-detail terrain |
| 3 | Terrain | "ADT Terrain Quilt" | Full terrain tiles |
| 4 | Liquid | "Water/lava" | Liquid surfaces |
| 5 | Wmo | "World Models" | WMO geometry |
| 6 | Doodad | "Doodads" | MDX/M2 geometry |
| 7 | Overlay | "Overlays" | Debug/editor overlay |

### 4.6 World Frame Pass Execution Order

```
1. RenderLighting()
2. IF SkyVisible: RenderSky(), RenderSkyboxBackdrop()
3. IF WdlVisible: RenderWdl()
4. IF TerrainVisible: RenderTerrain()
5. IF NOT ObjectsVisible: RETURN
6. PrepareObjectPhase()  -- animation, opaque/translucent routing
7. IF WmosVisible: RenderWmoOpaque()
8. IF DoodadsVisible: RenderMdxOpaque()
9. IF LiquidVisible: RenderLiquid()
10. IF DoodadsVisible: RenderMdxTransparent()
11. IF OverlayVisible: RenderOverlay()
```

### 4.7 M2 Frame Pipeline Stages

```
1. M2AnimatedRenderStateEvaluator.Evaluate()  → M2AnimatedRenderState
2. M2BonePoseEvaluator.Evaluate()             → M2BonePoseState
3. M2SkinnedRenderModelBuilder.ApplyPose()     → M2SkinnedRenderModel
4. M2RenderConsumerFrameStateBuilder.Build()   → M2RenderConsumerFrameState
5. M2ParticleRibbonRuntimeEvaluator.Evaluate() → M2EffectRuntimeState
6. M2SceneSubmissionEntryBuilder.Build*Entries() → M2SceneSubmissionEntry[]
7. M2SceneSubmissionCoordinator.BuildPlan()    → M2SceneSubmissionPlan
8. M2RenderFrameBuilder.Build()                → M2RenderFrame
9. M2SoftwareVisualSnapshotBuilder.Build()     → M2SoftwareVisualSnapshot
10. M2RuntimeGoldenFrameBuilder.Build()        → M2RuntimeGoldenFrame
```

### 4.8 Draw Call Batching

**Batch limits:** MaxVertices = 65535, MaxIndices = 98304.

**Sort key order:** Family → ModelKey → TextureSortKey → EffectKey → StateBucket → DepthSortValue (descending for transparent) → EntryKey.

**Batch sharing:** Two entries share a batch if Family, ModelKey, EffectKey, TextureSortKey, StateBucket, IsTransparent, IsAdditive all match.

**Family policies:** Core → batched. Projected → batched, dedicated state scope. Doodad → batched if BatchDoodads flag set. Ribbon → always direct, dedicated state. Particle → batched if BatchParticles flag, dedicated state. Callback/HitTest → always direct.

### 4.9 Framebuffer Format (All Renderers)

- Color: `Rgba8` (Linear, ClampToEdge)
- Depth: `DepthComponent24` (renderbuffer)
- Common OpenGL state: depth test enabled (`Lequal`), face culling disabled

---

## 5. Terrain AI Pipeline Specifications

### 5.1 Zarr v3 Dataset Format

#### Per-Tile Arrays

| Array | Shape | Dtype | Fill | Description |
|-------|-------|-------|------|-------------|
| `height_257` | (N, 257, 257) | float32 | 0.0 | World-space height map |
| `normal_xyz` | (N, 257, 257, 3) | float32 | 0.0 | Unit-length normal vectors |
| `normal_mask` | (N, 257, 257) | bool | False | Normal validity mask |
| `alpha_256` | (N, 256, 256, 4) | float32 | 0.0 | MCAL alpha blend weights [0,1] |
| `holes_16` | (N, 16, 16) | bool | False | Hole mask |
| `liquid_mask` | (N, 256, 256) | float32 | 0.0 | Liquid coverage [0,1] |
| `liquid_height` | (N, 256, 256) | float32 | 0.0 | Liquid surface height |
| `object_mask` | (N, 257, 257) | bool | False | Object occlusion mask |
| `object_precise_mask` | (N, 257, 257) | float32 | 0.0 | Precise object coverage |
| `object_instance_mask` | (N, 257, 257) | int32 | 0 | Per-instance object IDs |
| `object_filtered_mask` | (N, 257, 257) | float32 | 0.0 | Filtered object mask |
| `mddf_mask` | (N, 257, 257) | float32 | 0.0 | WMO placement mask |
| `modf_mask` | (N, 257, 257) | float32 | 0.0 | M2 placement mask |
| `mcnk_flags_16` | (N, 16, 16) | int32 | 0 | MCNK chunk flags |
| `minimap_rgb` | (N, 256, 256, 3) | uint8 | 0 | Baked minimap RGB |
| `shadow_mask` | (N, 256, 256) | float32 | 0.0 | MCSH shadow mask |
| `mcly_texture_ids` | (N, 16, 16, 4) | int32 | -1 | Texture IDs per MCLY layer |
| `mcly_layer_mask` | (N, 16, 16, 4) | float32 | 0.0 | MCLY layer visibility |

#### Companion Files

- `index.parquet` — tile_id, build, map, tile_x, tile_y, height_mean, height_std, has_* flags
- `placements.parquet` — per-tile object placements (mddf/modf with nameId, uniqueId, pos, rot, scale, bounding box, asset_path)
- `_resume_state.json` — build progress tracking
- `harvest_metrics.json` — tile counts, signal coverage
- `signal_validation.json` — per-signal coverage report

#### Default Compression

Codec: `BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")`.

#### Chunk Shapes (Multi-Tile Stores)

| Array | Chunk Shape |
|-------|-------------|
| `height_257` | (64, 257, 257) |
| `normal_xyz` | (64, 257, 257, 3) |
| `alpha_256` | (64, 256, 256, 4) |
| `minimap_rgb` | (64, 256, 256, 3) |
| `holes_16` | (1024, 16, 16) |
| `liquid_mask` | (64, 256, 256) |
| `mcnk_flags_16` | (256, 16, 16) |
| `mcly_texture_ids` | (1024, 16, 16, 4) |

### 5.2 Streaming Protocol (C# → Python)

#### Frame Format

Each blob on stdout:
```
[4 bytes] magic: "NPZB" (legacy) or "ARRY" (new)
[4 bytes] blob byte length (LE uint32, max 50,000,000)
[N bytes] blob payload
```

Terminated by `[4 bytes] "ENDS"` sentinel.

#### ARRY Binary Format

```
[4 bytes] "ARRY" magic (ASCII)
[4 bytes] metadata JSON length (LE uint32)
[N bytes] metadata JSON (UTF-8)

For each array:
  [4 bytes]     name length (LE uint32)
  [N bytes]     name (UTF-8)
  [4 bytes]     ndim (LE uint32)
  [4*ndim bytes] shape (LE uint32 each dimension)
  [8 bytes]     dtype ASCII string, null-padded (e.g. "<f4\0\0\0\0\0")
  [8 bytes]     data byte length (LE uint64)
  [N bytes]     raw array data (numpy-compatible)

[4 bytes] "ENDS" magic (ASCII)
[4 bytes] 0x00000000 (padding)
```

**Supported dtypes:** `<f4` (float32), `<f8` (float64), `<i4` (int32), `<u4` (uint32), `<i2` (int16), `<u2` (uint16), `|u1` (uint8), `|i1` (int8), `|b1` (bool).

### 5.3 Compositing Algorithm

#### MCAL Alpha Blend (4-Layer Hierarchical)

Given raw MCAL alpha values `alpha_pack (H, W, 4)` with `a1, a2, a3, a4`:

```
w0 = 1.0 - a1
w1 = a1 * (1.0 - a2)
w2 = a1 * a2 * (1.0 - a3)
w3 = a1 * a2 * a3 * (1.0 - a4)

weights = stack([w0, w1, w2, w3], axis=-1)
total = weights.sum(axis=-1, keepdims=True)
where(total > 1e-6, weights / total, 0.0)
```

#### Synthetic Minimap

```
synthetic_rgb = tensordot(weights, PLACEHOLDER_COLORS, axes=([2], [0]))
return synthetic_rgb.clip(0.0, 1.0)
```

**Placeholder Colors (RGB):**

| Layer | RGB (/255) |
|-------|------------|
| 0 | (0.549, 0.706, 0.784) |
| 1 | (0.392, 0.549, 0.627) |
| 2 | (0.431, 0.510, 0.471) |
| 3 | (0.510, 0.471, 0.392) |

**Residual:** `residual = real_minimap - synthetic_minimap`

### 5.4 ML Model Architectures

#### V16 Monolithic (~15.6M params)

**Encoder:** ConvNeXt V2 Nano (pretrained from timm, `features_only=True`)

| Stage | Stride | Channels | Spatial (256 input) |
|-------|--------|----------|---------------------|
| e0 | 4 | 80 | 64×64 |
| e1 | 8 | 160 | 32×32 |
| e2 | 16 | 320 | 16×16 |
| e3 | 32 | 640 | 8×8 |

**Decoder:**

| Module | Operation | In/Out | Spatial |
|--------|-----------|--------|---------|
| bottleneck | ConvBlock(640, 640) | 640→640 | 8×8 |
| dec3 | UpFuse(640, 320, 320) | upsample+skip→320 | 16×16 |
| dec2 | UpFuse(320, 160, 160) | upsample+skip→160 | 32×32 |
| dec1 | UpFuse(160, 80, 80) | upsample+skip→80 | 64×64 |
| dec0 | ConvBlock(80, 64) | 80→64 | 64×64 |

**ConvBlock(in_ch, out_ch):** Conv2d(in, out, 3, padding=1, bias=False) → BatchNorm → ReLU → Conv2d(out, out, 3, padding=1, bias=False) → BatchNorm → ReLU.

**UpFuse(in_ch, skip_ch, out_ch):** Upsample(2, bilinear, align_corners=True) → cat([skip, x]) → ConvBlock(in+skip, out).

**Heads (all from 64-ch d0):**

| Head | Architecture | Output |
|------|-------------|--------|
| height | Conv(64,32,3)→ReLU→Upsample(257)→Conv(32,1,1) | (B,1,257,257) raw |
| normals | Conv(64,32,3)→ReLU→Upsample(257)→Conv(32,3,1)→Tanh | (B,3,257,257) |
| alpha | Conv(64,32,3)→ReLU→Upsample(256)→Conv(32,4,1)→Sigmoid | (B,4,256,256) |
| holes | AdaptiveAvgPool(16)→Conv(64,1,1)→Sigmoid | (B,1,16,16) |
| liquid | Conv(64,32,3)→ReLU→Upsample(256)→Conv(32,1,1)→Sigmoid | (B,1,256,256) |
| mcly | AdaptiveAvgPool(16)→Conv(64,64,3)→ReLU→Conv(64,64,1) | (B,64,16,16)→reshape(4,16,16,16) logits |

#### V16.1 Independent Models (5 models, shared backbone architecture)

All share `_UNetBackbone`:

| Module | Layers | Channels |
|--------|--------|----------|
| enc0 | ConvBlock(3, 64) | 3→64 |
| enc1 | MaxPool(2) + ConvBlock(64, 96) | 64→96 |
| enc2 | MaxPool(2) + ConvBlock(96, 160) | 96→160 |
| enc3 | MaxPool(2) + ConvBlock(160, 224) | 160→224 |
| bottleneck | ConvBlock(224, 224) | 224→224 |
| dec3 | UpBlock(224, 224, 160) | →160 |
| dec2 | UpBlock(160, 160, 96) | →96 |
| dec1 | UpBlock(96, 96, 64) | →64 |
| dec0 | UpBlock(64, 64, 32) | →32 |

Output: d0 (B,32,256,256), pooled16 = AdaptiveAvgPool(d0, (16,16)) → (B,32,16,16).

**V16.1 ConvBlock(in_ch, out_ch, mid_ch=None):** Conv2d(in, mid, 3, padding=1, bias=False) → BatchNorm → ReLU → Conv2d(mid, out, 3, padding=1, bias=False) → BatchNorm → ReLU.

**V16.1 UpBlock(in_ch, skip_ch, out_ch):** Upsample(2, bilinear, align_corners=True) → Conv2d(in, out, 1, bias=False) → cat([skip, x]) → ConvBlock(out+skip, out).

| Model | Head | Output |
|-------|------|--------|
| V161HeightModel | Conv(32,32,3)→ReLU→Upsample(257)→Conv(32,1,1) | (B,1,257,257) |
| V161NormalModel | Conv(32,32,3)→ReLU→Upsample(257)→Conv(32,3,1)→Tanh | (B,3,257,257) |
| V161HolesModel | Conv(32,16,3)→ReLU→AdaptiveAvgPool(16)→Conv(16,1,1)→Sigmoid | (B,1,16,16) |
| V161LiquidModel | mask_head: Conv(32,32,3)→ReLU→Conv(32,1,1)→Sigmoid; type_head: Conv(32,32,3)→ReLU→Conv(32,5,1) | mask (B,1,256,256) + type (B,5,16,16) |
| V161TexcompModel | alpha_head: Conv(32,32,3)→ReLU→Conv(32,4,1)→Sigmoid; mask_head: Conv(32,32,3)→ReLU→Conv(32,4,1)→Sigmoid; ids_head: Conv(32,64,3)→ReLU→Conv(64,64,1)→view(4,16,16,16) | alpha (B,4,256,256) + mask (B,4,16,16) + ids (B,4,16,16,16) |

### 5.5 Loss Functions (Exact Formulas)

#### Height Loss
```
loss = masked_mean(|pred - target|, weight_257)
```

#### Normal Loss
```
pred_n = normalize(pred, dim=1, eps=1e-6)
target_n = normalize(target, dim=1, eps=1e-6)
cosine = 1.0 - (pred_n * target_n).sum(dim=1, keepdim=True)
vec_l1 = |pred_n - target_n|.mean(dim=1, keepdim=True)
nz_l2 = (pred_n[:,2:3] - target_n[:,2:3]) ** 2
hard_region_weight = 1.0 + (normal_detail_boost * hard_region_signal)
loss = masked_mean(cosine, train_mask)
      + 0.35 * masked_mean(vec_l1, train_mask)
      + 0.15 * masked_mean(nz_l2, train_mask)
```

Where `train_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight * (1 - what_plate_flag) * hard_region_weight`.

`hard_region_signal = clamp(0.50 * height_grad + 0.25 * normal_grad + 0.25 * max(alpha_grad, mcly_grad), 0, 4) * terrain_valid_mask`.

#### Holes Loss
```
bce = binary_cross_entropy(pred, target, reduction="none")
loss = masked_mean(bce, weight_16)
```

#### Liquid Loss
```
mask_loss = weighted_l1(pred_mask, target_mask, weight_256)
type_ce = cross_entropy(pred_type, target_type, reduction="none")
loss = mask_loss + 0.5 * masked_mean(type_ce, type_valid * weight_16)
```

#### Texcomp Loss
```
alpha_loss = weighted_l1(pred_alpha, alpha_target, weight_256)
mask_bce = BCE(pred_mask, mcly_mask, reduction="none")
mask_loss = masked_mean(mask_bce, weight_16)
id_ce = cross_entropy(pred_ids, mcly_ids, reduction="none")
id_loss = masked_mean(id_ce, mcly_mask * weight_16)
recomposed = recompose_from_mcly_alpha(pred_alpha, pred_ids, pred_mask)
recompose_loss = weighted_l1(recomposed, input_minimap, weight_256)
loss = alpha_loss + 0.35 * mask_loss + 0.25 * id_loss + 0.5 * recompose_loss
```

### 5.6 Dataset Loading

#### Normalization
- `minimap_rgb`: `/ 255.0` to float32 [0, 1]
- `height_raw`: `(height_raw - h_mean) / (h_std + 1e-8)` (per-tile z-score)
- `alpha_256`: clip [0, 1]
- `liquid_mask`: clip [0, 1]

#### Augmentation (Train Only)

Random 3-bit transform (8 possibilities):
- Bit 0: horizontal flip — flip all spatial dims on axis=1; flip normal_x sign
- Bit 1: vertical flip — flip all spatial dims on axis=0; flip normal_y sign
- Bit 2: 90° rotation — `np.rot90(k=1)` all spatial; swap normal_x = old_ny, normal_y = -old_nx

#### Terrain-Valid Mask
```
terrain_valid_mask = normal_mask * (1 - clip(object_presence, 0, 1)) * (1 - clip(liquid_mask * 0.85, 0, 1))
zeroed entirely if what_plate_flag > 0.5
```

#### What-Plate Flag
```
is_blank = (|height|_max ≤ 1e-6 AND height_std ≤ 1e-6 AND alpha_cov ≤ 1e-4 AND mcly_cov ≤ 1e-4 AND liquid_cov ≤ 1e-4 AND object_cov ≤ 1e-4)
```

### 5.7 Training Configuration

#### Optimizer
```python
AdamW(lr=2e-4, weight_decay=0.05)
```

#### LR Schedule
```python
CosineAnnealingLR(T_max=epochs, eta_min=0.0)
```

On resume: `lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * completed_epoch / T_max)) / 2`

#### AMP / Mixed Precision
```python
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not no_amp))
with torch.amp.autocast("cuda", enabled=...):
    loss, metrics, outputs = task.loss_fn(model, batch, device, args)
```

#### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Grad Accumulation
```python
scaler.scale(loss / grad_accum_steps).backward()
# Every grad_accum_steps batches:
scaler.unscale_(optimizer)
clip_grad_norm_(max_norm=1.0)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

#### `torch.compile`
Applied when available and not `--no-compile` and CUDA. Wrapped in try/except; falls back gracefully.

#### Difficulty Bucket Sampling
```python
PROFILES = {
    "uniform": {"easy": 1.0, "medium": 1.0, "hard": 1.0, "pathological": 1.0},
    "v16_1_1_normal": {"easy": 1.0, "medium": 1.75, "hard": 3.5, "pathological": 1.25},
}
```

#### Checkpoint Format
```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,
    "best_val": float,
    "best_epoch": int | None,
    "task": str,
}
```

### 5.8 Inference Protocol

#### Input
- Load source Zarr store + index.parquet
- Per batch: `minimap_rgb[tile_id].astype(float32) / 255.0` → (B, 3, 256, 256)

#### Output Zarr Arrays

| Array | Shape | Dtype |
|-------|-------|-------|
| `height_pred_257` | (N, 257, 257) | float32 |
| `normal_pred_xyz` | (N, 257, 257, 3) | float32 |
| `holes_pred_16` | (N, 16, 16) | float32 |
| `liquid_pred_mask_256` | (N, 256, 256) | float32 |
| `liquid_type_pred_16` | (N, 16, 16) | int16 |
| `alpha_pred_256` | (N, 256, 256, 4) | float32 |
| `mcly_mask_pred_16x16x4` | (N, 16, 16, 4) | float32 |
| `mcly_id_pred_16x16x4` | (N, 16, 16, 4) | int16 |
| `recomposed_pred_rgb_256` | (N, 256, 256, 3) | float32 |

### 5.9 Curation Metrics

**Grayscale:** `0.299 * R + 0.587 * G + 0.114 * B`

**Edge strength (L∞ gradient):** `gx[:, 1:] = |x[:, 1:] - x[:, :-1]|; gy[1:, :] = |x[1:, :] - x[:-1, :]; edge = max(gx, gy)`

**Alpha painted coverage:** `alpha[:, :, 1:].max(axis=2)` (excludes layer 0 base)

**Normal relief:** `sqrt(nx² + ny²) * normal_mask`

**Difficulty buckets:** easy, medium, hard, pathological

---

## 6. CLI Tool Specifications

### 6.1 Inspect Tool

```
WowViewer.Tool.Inspect <command> [options]
```

| Command | Description |
|---------|-------------|
| `archive build-listfile-cache` | Build MPQ listfile cache |
| `audio alpha-area` | Inspect Alpha area audio catalogs |
| `blp inspect` | BLP texture inspection |
| `m2 inspect` | Full M2 model inspection (skin, animation, bone, render, golden frame, visual snapshot) |
| `mdx inspect` | MDX model inspection |
| `mdx export-json` | MDX JSON export |
| `mdx chunk-carriers` | MDX chunk analysis |
| `mdx skin-diagnostics` | MDX skinning diagnostics |
| `map inspect` | Map/WDT/ADT inspection |
| `lit inspect` | LIT lighting file inspection |
| `pm4 inspect` | PM4 pathmap inspection |
| `pm4 research` | PM4 research analysis |
| `wmo inspect` | WMO root+group inspection |
| `wmo render-doc` | WMO render document assembly |

### 6.2 Converter Tool

```
WowViewer.Tool.Converter <command> [options]
```

| Command | Description |
|---------|-------------|
| `detect` | File format detection |
| `dataset-list-maps` | List terrain-trainable maps |
| `dataset-scan` | Scan map tiles, build manifests |
| `dataset-merge` | Merge manifests |
| `dataset-split-pm4` | Split PM4/non-PM4 subsets |
| `dataset-audit` | Audit training samples |
| `dataset-curate` | Curate by quality thresholds |
| `dataset-build-cache` | Build tensor cache shards |
| `extract-map` | Extract map tile data |
| `export-tex-json` | Export ADT texture as JSON |
| `extract-v10-tensors` | V10 tensor pack extraction |
| `dataset-build-v10-stage1` | V10 stage1 batch extraction |
| `ml-corpus` | ML corpus generation |
| `ml-audit-signals` | ML signal auditing |
| `ml-harvest-brushes` | Brush imprint harvesting |
| `ml-generate-controls` | Synthetic control generation |
| `ml-repair-normalmaps` | Normal map repair |
| `ml-synth-no-liquid` | Synthetic terrain without liquid |
| `terrain-patch-adt` | ADT terrain patching |
| `mine-v10-brushes` | V10 brush mining |
| `mine-v10-mcly` | V10 MCLY dictionary mining |
| `label-v10-mcly` | V10 MCLY label manifest |
| `mine-v10-mcal-compositions` | V10 MCAL composition mining |
| `mine-v10-mcal-brushes` | V10 MCAL brush dictionary |
| `mine-v10-height-profiles` | V10 height profile mining |
| `mine-v10-prefab-cells` | V10 prefab cell detection |
| `convert-alpha-to-lk` | Alpha ADT → LK ADT |
| `convert-split-adt-to-lk` | Split ADT → LK ADT |
| `convert-lk-to-alpha` | LK ADT → Alpha ADT |
| `convert-wmo-v17-to-v17` | WMO v14 → v17 |
| `convert-wmo-v14-to-v17` | WMO v17 → v14 |
| `convert-m2-to-mdx` | M2 → MDX |
| `convert-mdx-to-m2` | MDX → M2 |
| `validate-roundtrip` | Round-trip validation |

### 6.3 Harvest Tool

```
WowViewer.Tool.Harvest <command> [options]
```

| Command | Description |
|---------|-------------|
| `harvest-tile` | Extract shard from single ADT tile |
| `harvest-map` | Batch-extract all tiles from map directory |
| `harvest-map-mpq` | Batch-extract from MPQ archives |
| `harvest-stream` | Stream V16-ready raw tile blobs to stdout |
| `extract-unified` | Extract shard from MPQ-archived tile |
| `synthetic-minimap` | Composite tilesets + alpha → synthetic minimap |
| `discover-maps` | List terrain-trainable maps |

**harvest-stream options:** `--tile-workers` (default=max(1,min(16,cpu_count))), `--stream-profile` ("v16" or "full"), `--limit`, `--build`, `--client-root`, `--map`.

### 6.4 Dataset Build CLI

```
build_v16_dataset.py build --build <ver> --allow-zarr-write [--limit N] [--maps name] [--resume] [--tile-workers N] [--codec lz4] [--clevel 1] [--shuffle shuffle]
```

Subcommands: `build`, `stats`, `validate-signals`, `repair-index`, `patch-liquids`, `patch-objects`, `merge-builds`.

### 6.5 Training CLI

```
train_v16_1_<task>.py --builds <list> --curation-manifest <path> [options]
```

Key args: `--batch-size` (8), `--epochs` (50), `--lr` (2e-4), `--weight-decay` (0.05), `--device` (auto), `--seed` (42), `--val-fraction` (0.1), `--train-max-tiles`, `--train-epoch-tiles`, `--bucket-sampling-profile` (v16_1_1_normal), `--target-vram-gb`, `--autotune-batch-size`, `--normal-detail-boost` (1.0), `--resume-checkpoint`, `--no-amp`, `--no-compile`.

### 6.6 Inference CLI

```
infer_v16_1.py --build <ver> --height-checkpoint <path> --normal-checkpoint <path> --holes-checkpoint <path> --liquid-checkpoint <path> --texcomp-checkpoint <path> [--batch-size 8] [--device auto]
```

---

## 7. Client Build Support

| Build | Version | Expansion | Notes |
|-------|---------|-----------|-------|
| 0.5.3 | 3368 | Alpha | Embedded ADT, unique formats |
| 0.5.5 | 3494 | Alpha | Similar to 0.5.3 |
| 0.7.0 | 3694 | Alpha | Later Alpha build |
| 3.0.1 | 8303 | WotLK | Split ADT, standard formats |
| 3.3.5 | 12340 | WotLK | Primary reference build |
| 4.0.0 | 11927 | Cataclysm | Cataclysm changes |

---

## 8. Functional Requirements

### FR-001: Multi-Era Terrain Reading
Read terrain from all 6 supported builds including Alpha-embedded ADT, LK split ADT, and Cataclysm variants.

### FR-002: Format Conversion
Convert terrain Alpha↔LK, M2↔MDX, WMO v14↔v17 with round-trip validation.

### FR-003: 3D Rendering
Render terrain, M2, WMO, and complete world scenes with texturing, object placement, liquid, and sky.

### FR-004: Interactive Viewer
Desktop viewer with camera navigation, object selection, and diagnostic overlays.

### FR-005: Dataset Harvesting
Harvest terrain into Zarr via streaming C#→Python protocol with no intermediate files.

### FR-006: ML Training
Train independent terrain models with curation, augmentation, and resumable checkpoints.

### FR-007: ML Inference
Run models on minimap inputs producing Zarr prediction outputs.

### FR-008: Format Inspection
Inspect any supported game file via CLI.

### FR-009: Archive Access
Read from MPQ archives and loose filesystem paths.

### FR-010: glTF Export
Export terrain, M2, WMO to glTF/GLB.

### FR-011: PM4 Analysis
Read, analyze, visualize PM4 pathmap data.

### FR-012: Data Validation
Validate signal coverage, training readiness, and inference quality.

---

## 9. Non-Functional Requirements

### NFR-001: Repo Independence
Extractable standalone repository with no external path references.

### NFR-002: Real-Data Validation
All format claims validated against real staged game client data.

### NFR-003: Streaming-First
Dataset pipelines stream C#→Python via stdout, no intermediate files.

### NFR-004: Residual Model Chain
Each model predicts exactly one residual signal. No monolithic models.

### NFR-005: Buildability
Zero errors on .NET 10 + Python 3.11+.

### NFR-006: Test Coverage
All readers, writers, converters tested against real game files.

### NFR-007: Performance
Terrain AOI streaming with bounded GPU upload budgets.

### NFR-008: Extensibility
Support future Vulkan backend, WebGL delivery, ML content seams.

---

## 10. User Stories

### P1 — Critical

**US-001:** As a developer, I want to read any WoW game file format so that I can inspect its contents.
- Given a game client root, when I run the inspect tool, then all supported formats are detected and summarized.

**US-002:** As a developer, I want to render WoW terrain in 3D so that I can visually inspect terrain data.
- Given a WDT file, when I open it in the viewer, then terrain tiles render with correct textures and heightmaps.

**US-003:** As a developer, I want to harvest terrain data into Zarr datasets so that I can train ML models.
- Given a staged client, when I run the harvest pipeline, then a Zarr store is produced with all required signals.

**US-004:** As a developer, I want to train terrain reconstruction models so that I can predict terrain from minimaps.
- Given a Zarr dataset, when I run training, then models converge and produce valid predictions.

**US-005:** As a developer, I want to convert terrain between expansion eras so that I can work with cross-era data.
- Given Alpha or LK terrain, when I run conversion, then output is validated against ground truth.

### P2 — Important

**US-006:** As a developer, I want to render M2 models in the viewer.
- Given an M2 file, when I open it, then the model renders with geometry, textures, and animation.

**US-007:** As a developer, I want to render WMO models in the viewer.
- Given a WMO file, when I open it, then the model renders with portals, doodads, and materials.

**US-008:** As a developer, I want to view complete world scenes.
- Given a map, when I open a world session, then terrain, M2s, WMOs, and liquids render together.

**US-009:** As a developer, I want to export terrain and models to glTF.
- Given loaded terrain/models, when I export, then a valid GLB file is produced.

**US-010:** As a developer, I want to analyze PM4 pathmap data.
- Given a PM4 file, when I run analysis, then research outputs and visualizations are produced.

### P3 — Nice to Have

**US-011:** Vulkan rendering backend.
**US-012:** WebGL browser output.
**US-013:** ML-driven content generation.

---

## 11. Implementation Phases

### Phase 1: Core I/O (Foundation)
Domain models, chunked file reader, archive access, all format readers/writers/converters, unit tests.

### Phase 2: Runtime Pipeline
M2 runtime (animation, skinning, bone evaluation), world runtime (terrain, liquid, visibility, composition), MDX runtime.

### Phase 3: GPU Rendering
OpenGL 3.3 renderer, terrain/M2/WMO/liquid rendering, world composition, sky dome.

### Phase 4: Desktop Viewer
Application shell, workspace modes, navigator/inspector panels, camera controls, object selection.

### Phase 5: CLI Tools
Inspect tool, converter tool, harvest tool.

### Phase 6: Terrain AI Pipeline
Streaming protocol, Zarr builder, curation, D1/R1/V16/V16.1 models, training, inference, validation.

### Phase 7: Export and Analysis
glTF/GLB export, terrain image import/export, PM4 workbench, terrain analysis.

### Phase 8: Advanced Features
Vulkan backend, WebGL output, audio engine, ML content generation.

---

## 12. Success Criteria

1. All 6 supported client builds load and render correctly
2. All format conversions produce round-trip validated output
3. Terrain AI models converge and predict within quality thresholds
4. Desktop viewer renders terrain, M2, WMO, and world scenes interactively
5. CLI tools cover all inspection, conversion, and harvesting workflows
6. Test suite passes with zero failures against real game data
7. Project builds cleanly with zero errors on .NET 10 + Python 3.11+
8. Project is extractable as standalone with no external references
