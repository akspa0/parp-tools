# Full Project Reimplementation Specification

**Feature:** 009 — Complete Reimplementation Spec
**Date:** 2026-05-22
**Purpose:** Design specification sufficient to fully reimplement the MdxViewer + wow-viewer project as a single new project in a new repository from scratch, preserving all functionality.
**Scope:** 4+ years of personal WoW format reverse engineering, terrain AI research, and viewer development — condensed into a single design notebook.

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

### Design Notebook Index

This specification is organized as a layered reference. Sections 1–12 define the **what** (requirements, architecture, formats at a glance). Sections 13–28 are the **how** (deep-dives into every subsystem with exact algorithms, byte layouts, shader source, and hard-won edge cases that only exist in code).

---

## Design Notebook — Table of Contents with Synopses

### Part I: Foundations (Sections 1–7)

**§1 Executive Summary** — Project scope, statistics, and this index.

**§2 Architecture Overview** — The four-layer stack: Domain Models → Core I/O → Runtime Pipeline → Product Surfaces. Technology stack (C#/.NET 10, Silk.NET/OpenGL, Python/PyTorch, Zarr v3). Repo independence constraint: the project must be extractable as a standalone repository with zero external path references.

**§3 Binary Format Specifications** — The byte-level truth for every WoW file format the project reads or writes. Covers WDT (Alpha and LK variants), ADT (embedded and split), MCNK chunks, MCVT heightmap (145-vertex staggered grid), MCNR normals (XZY-swizzled sbytes), MCLY texture layers, MCAL alpha decoding (all 4 encoding types: Packed4Bit, Compressed/RLE, BigAlpha, BigAlphaFixed), MDDF/MODF placement structs, MCCV vertex colors (BGR swizzle), MCSH shadow maps, BLP textures (BLP0/1/2 headers), M2 models (MD20 header, 48-byte vertex format, 20-byte animation tracks, skin profiles), MDX legacy models (MDLX, geoset sub-chunks), WMO root and groups (MOHD, MOMT, portal system, doodad system), WDL low-res heightmaps, LIT lighting, and MPQ archive structure (hash/block tables, encryption, compression). Every FourCC, every field offset, every struct stride.

**§4 Rendering Pipeline Specifications** — GPU-side contracts. OpenGL 3.3 core shaders. Vertex attribute layouts for all renderers (terrain: 44-byte interleaved + separate chunk-slice/tex-index VBOs; M2: 32-byte position+normal+UV; WMO: 32-byte). Shader uniform bindings and texture unit assignments. Draw call batching sort keys and batch-sharing rules. World composition 8-layer stack (Sky → SkyboxBackdrop → Wdl → Terrain → Liquid → Wmo → Doodad → Overlay). Pass execution order. M2 10-stage frame pipeline (animation → bone pose → skinning → effects → scene submission → render frame). Framebuffer formats (Rgba8 + DepthComponent24).

**§5 Terrain AI Pipeline Specifications** — The complete ML dataset contract. Zarr v3 store layout with 18 signal arrays (height, normals, alpha, holes, liquid, objects, shadows, MCLY) and their exact shapes, dtypes, and chunk sizes. Parquet index format. Streaming protocol: C# harvester writes ARRY binary blobs to stdout, Python reads and writes directly to Zarr — no intermediate files. ARRY format byte layout. Compositing algorithm: 4-layer hierarchical MCAL alpha blend (`w0=1-a1, w1=a1*(1-a2), ...`). Placeholder colors. All 5 V16.1 model architectures with exact layer specs (ConvNeXt V2 Nano encoder, U-Net decoder, task-specific heads). All 5 loss function formulas (height L1 masked, normal cosine+vec_l1+nz_l2, holes BCE, liquid mask+type CE, texcomp alpha+mask+id+recompose). Training config (AdamW, cosine LR, AMP, grad clipping, difficulty bucket sampling). Inference output contract. Curation metrics (grayscale, edge strength, alpha painted, normal relief, difficulty buckets).

**§6 CLI Tool Specifications** — Every command surface. Inspect tool: 14 commands for format diagnostics (archive, audio, blp, m2, mdx, map, lit, pm4, wmo). Converter tool: 30+ commands for format conversion and dataset management (detect, dataset-*, convert-*, ml-*, mine-*, validate-roundtrip). Harvest tool: 7 commands for dataset extraction (harvest-tile, harvest-map, harvest-map-mpq, harvest-stream, extract-unified, synthetic-minimap, discover-maps). Dataset build CLI subcommands. Training CLI arguments. Inference CLI arguments.

**§7 Client Build Support** — The 6 supported WoW client builds (0.5.3 through 4.0.0), their expansion era, and format characteristics (Alpha embedded ADT, LK split ADT, Cataclysm changes).

### Part II: Requirements and Planning (Sections 8–12)

**§8 Functional Requirements** — 12 FRs covering multi-era terrain reading, format conversion, 3D rendering, interactive viewer, dataset harvesting, ML training, ML inference, format inspection, archive access, glTF export, PM4 analysis, and data validation.

**§9 Non-Functional Requirements** — 8 NFRs: repo independence, real-data validation, streaming-first pipelines, residual model chain, buildability, test coverage, performance (AOI streaming), extensibility (Vulkan, WebGL, ML content).

**§10 User Stories** — 13 stories across P1/P2/P3 priorities with Given/When/Then acceptance scenarios.

**§11 Implementation Phases** — 8 phases from Core I/O foundation through advanced features (Vulkan, WebGL, audio, ML content generation).

**§12 Success Criteria** — 8 measurable outcomes.

### Part III: Deep-Dives — The Code Is the Truth (Sections 13–28)

These sections capture what only exists in working code. Documentation is often wrong or outdated; the codebase is the authoritative source. Each deep-dive extracts exact algorithms, constants, edge cases, and open questions from the implementation.

**§13 Deep-Dive: PM4 Format (Practically Unknown)** — The pathmap format that took years of research to partially decode. 16 recognized chunks with binary record layouts and field-level confidence ratings (Known/Partial/Open). The 4-level hierarchy model: Region (MSHD.Field04) → Object (CK24) → Sub-object (MSLK.GroupObjectId) → Surfaces (MSUR) + Positions (MPRL). CK24 type classification (nav mesh, M2 interior/exterior, WMO). `MSLK.TypeFlags` now has first partial family buckets from real-data inspection (`0x03` M2 top surfaces, `0x10` interior WMO floors, `0x12` exterior WMO solid surfaces), but grouping ownership is still not fully closed. Cross-tile merge via MSCN connector keys with Union-Find. Coordinate transforms (3 axis conventions, 2 coordinate modes, planar transform scoring). Verified vs partial linkages. 10 open research questions ranked by impact. This section is a living research document — not a solved spec.

**§14 Deep-Dive: WMO Portal Visibility (BFS Flood-Fill)** — The algorithm that determines which WMO interior groups are visible. BFS from camera through portal adjacency graph. Two early-out paths (camera inside root → all visible; no exterior groups → frustum only). Portal reveal distances (exterior: 1024, interior: 3072). Traversal depth limits (exterior: 1, interior: 4). Known simplifications: portal plane-side test not used, BSP tree not traversed.

**§15 Deep-Dive: Terrain Edge Cases and Special Handling** — The accumulated terrain knowledge from 4 years of format work. 3-phase seam stitching (edge → corner → predicted edge anchoring). MCNK subchunk size inflation (declared sizes unreliable for MCNR/MCAL/MCSH). Coordinate system transform (world↔file, axes swapped and negated through MapOrigin=17066.666). Normal XZY swap. MCCV BGR swizzle. MCAL packed 4-bit column 31 edge case. Legacy edge fix (corrupted last column/row). BigAlphaFixed Cataclysm truncation fix. MCAL force-compressed logic. Residual alpha synthesis. Heightmap sentinel 0.0f. Normal Z-flip guarantee. Format profile matrix per build.

**§16 Deep-Dive: M2 Animation System** — The animation evaluation pipeline. Hermite and Bezier cubic interpolation (exact formulas). Quaternion cubic as linear blend of components + normalize (NOT slerp — this is the WoW engine's approach, not standard). Compressed quaternion encoding (4 × int16 with asymmetric decode around 32767/-32768). 4-influence bone skinning with 3-level bone index resolution. Skin vertex resolution two-pass (direct lookup + GlobalVertexOffset fallback).

**§17 Deep-Dive: GLSL Shaders (Complete Source)** — All 5 shader programs embedded in the C# viewer: terrain (textured quad with alpha layer blending), sky backdrop (procedural gradient + starfield via FNV-1a hash), M2 model (directional lighting + UV transform), MDX model (half-Lambert diffuse + bone skinning + sphere env map), WMO model (half-Lambert with minimum 0.18 ambient floor). Full GLSL 330 core source for both vertex and fragment shaders.

**§18 Deep-Dive: Converter Algorithms** — The exact conversion pipelines. WMO v14→v17: split monolithic file, upconvert MOMT 44/48→64 bytes, upconvert MOGI 40→32 bytes, rename MOIN→MOVI, upconvert MOPY 4→2 bytes. WMO v17→v14: reverse with group splitting (max 384 groups, max 49151 vertices), portal layout remapping, overflow merging. MDX→M2: build M2 v0x108 with 48-byte vertices, single skin per geoset. M2→MDX: compress quaternion encoding, accumulate sequence start times.

**§19 Deep-Dive: WMO Liquid System** — MLIQ parsing (30-byte header, 8-byte vertex records). Orientation auto-fit: test 4 rotations, score by overflow penalty × 1000 + center distance, legacy default orientation 2. Liquid type dispatch from Ghidra (nibble→water/magma/slime, ocean flag override). Color assignment per type. Vertex mapping for 4 orientations. Tile size: 4.16666f (1/8th of map chunk).

**§20 Deep-Dive: Vertex Lighting Fallback Chain** — Three-tier fallback for WMO vertex colors: (1) direct BGRA vertex colors with luminosity ≥ 10/255 rejection, (2) v14 lightmap sampling (MOLV UVs + MOLD pixels + MOLM infos) with luminosity ≥ 0.08 rejection, (3) all-white fallback. Why these thresholds exist: prevents black lighting from corrupting the scene.

**§21 Deep-Dive: M2 Draw Call Batching** — Batch limits (65535 vertices, 98304 indices). Sort key hierarchy (Family → ModelKey → TextureSortKey → EffectKey → StateBucket → DepthSortValue → EntryKey). Batch sharing rules (7 fields must match). Family policies (Core=batched, Projected=dedicated state, Doodad=batched if flag, Ribbon/Particle=dedicated). M2RuntimeOptions flags.

**§22 Deep-Dive: Triangle Winding Conversion** — WoW uses CW front faces, OpenGL uses CCW. Swap `indices[t+1] <-> indices[t+2]` at buffer upload time. Applied to WMO, M2, and terrain index buffers.

**§23 Deep-Dive: M2 Effect Recipe Classification** — Diffuse family selection (None/T1/T1T2/T1T2T3/T1T2T3T4/Projected). Combiner family from blend mode (Opaque/AlphaKey/Decal/Add/Mod/Mod2X/Fade). State bucket bitfield for render state sorting (10 bits encoding blend mode, depth write, alpha test, two-sided, unshaded, additive, projected).

**§24 Deep-Dive: M2 Particle/Ribbon Effects** — Particle classification (7 blend types → effect keys). State bucket encoding (blend type | emitter type | particle type | head/tail). Max 65535 particles per emitter. Ribbon edge count estimation. Geometry estimation (particles: 4 verts + 6 indices per quad; ribbons: 2 verts per edge point).

**§25 Deep-Dive: Object Masking System (Training Breakthrough)** — The breakthrough that made ML training viable. Six 257×257 output masks: binary, precise (soft-edged), instance ID, doodad-only, WMO-only, filtered (excludes nature/clutter). Four candidate projection modes for world→tile coordinate conversion (handles coordinate ambiguity across Alpha/LK/Cata/WoD). MODF bounding box flip through origin. PaintCircle/PaintSoftCircle algorithms. Three-tier WMO fallback: exact mesh footprint (Sutherland-Hodgman clipping + edge-function rasterization) → chunk-coverage from MCRF/MCRW → AABB bounds → circle. Object filtered mask: regex exclusions for 30+ nature keywords, size thresholds (3m extent, 6m² area, 8m height with 1.35 aspect). Shadow residual: isolates terrain self-shadowing by subtracting object mask from shadow map.

**§26 Deep-Dive: Liquid System Across All Versions** — Complete version-handling matrix (Alpha through Cata+). Unified liquid priority chain: MH2O > MCLQ > WL* (strict fallback, NO blending). MH2O per-layer mapping with offset/width/height. MCLQ 129×129 → 257×257 bilinear upsample. WL loose file 4×4 block rasterization with distance-weighted blending. Alpha flat per-chunk fill. 8 edge cases including the viewer using MCLQ>MH2O (opposite priority from unified builder).

**§27 Deep-Dive: World Rendering Pipeline (Open Map → Pixels)** — Complete 7-phase pipeline. Session bootstrap (WDT resolution, archive catalog, fuzzy map directory lookup). 3×3 tile window construction. Object instance construction with the exact MDX legacy rotation matrix. Visibility culling: vision cone factor from dot product, cone-adjusted cull distance, 3 quality profiles (Quality/Balanced/Performance). Asset inventory streaming with budget (2 loads/frame, 4ms max). Pass coordination: 11-pass render order. GPU terrain rendering with Texture2DArray and alpha shadow array. Sky rendering with procedural gradient and starfield. Marker rendering (GL_POINTS, gold for WMO, blue for MDX).

**§28 Deep-Dive: Legacy WMO Renderer (Reference Implementation)** — The complete 2,564-line reference renderer. 5-pass pipeline (opaque shell → doodad opaque → liquid → doodad transparent → transparent shell). Portal BFS with all constants (32f padding, 192f near-root distance, 1024/3072 portal reveal, depth 1/4). WMO 48-byte vertex format with vertex lighting. Half-Lambert + baked lighting shader. Deferred loading budgets (1 load/frame, 2ms max). M2 skin resolution 3-level fallback. Vertex lighting 3-tier fallback. Blend state from Ghidra (EGxBlend). Render queue (opaque front-to-back, transparent back-to-front). 20 specific features NOT yet ported to wow-viewer — the remaining work.

---

## How to Use This Spec

1. **For format work:** Start at §3 (byte layouts), then check the relevant §15+ deep-dive for edge cases.
2. **For rendering work:** Start at §4 (GPU contracts), then §17 (shader source), then §27 (world pipeline).
3. **For ML/dataset work:** Start at §5 (dataset contract), then §25 (object masking), then §26 (liquid).
4. **For conversion work:** Start at §18 (converter algorithms), then §15 (terrain edge cases).
5. **For PM4 research:** §13 is the entire research state — knowns, unknowns, hypotheses, open questions.
6. **For porting legacy features:** §28 lists everything not yet ported, with exact source locations.

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

#### V16.1 Independent Models (5 models, shared backbone architecture) — LANDED

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

#### V16.1 Helper Functions (New)

**`compute_compositor_weights_torch(alpha_pack)`:** Converts raw 4-layer MCAL alpha pack into normalized compositor weights using the hierarchical formula: `w0=1-a1, w1=a1*(1-a2), w2=a1*a2*(1-a3), w3=a1*a2*a3*(1-a4)`. Normalized by total weight.

**`recompose_from_mcly_alpha(pred_alpha, pred_ids, pred_mask)`:** Reconstructs terrain RGB from predicted texture IDs + alpha pack + layer mask using a 16-entry texture palette (`_TEXTURE_PALETTE_16`). Used by V16.1 texcomp model for recomposition loss.

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

Where `train_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight * (1 - what_plate_flag)` (whiteplate flag) `* hard_region_weight`.

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

#### V16 Dataset (`V16Dataset`)
- Source: Zarr v3 stores with Parquet index
- Split: deterministic train/val via `np.random.RandomState(seed).permutation(n)`
- Normalization: minimap `/255.0`, height z-scored per-tile, alpha/liquid clipped [0,1]
- Augmentation: random 3-bit transform (hflip, vflip, rot90) with correct normal vector transforms

#### V16.1 Dataset (`V161Dataset`) — LANDED

Enriched dataset for V16.1 trainers with curation manifest integration:

**Curation integration:** Loads curation index, filters tiles by `keep` flag, attaches per-tile quality/usefulness/difficulty scores and bucket labels.

**MCNK flag processing:** `_flags_to_liquid_type()` converts MCNK flags to coarse liquid types: 0=none, 1=water (0x04), 2=ocean (0x08), 3=magma (0x10), 4=slime (0x20).

**Object mask integration:** Reads `object_filtered_mask`, `mddf_mask`, `modf_mask`, computes `object_presence_257 = max(mddf, modf)`.

**Terrain valid mask:**
```
terrain_valid_mask = normal_mask * (1 - clip(object_presence, 0, 1)) * (1 - clip(liquid_mask * 0.85, 0, 1))
zeroed entirely if `what_plate_flag` (whiteplate) > 0.5
```

**Weight maps at 3 scales:**
- `weight_257 = 1.0 - clip(object_filtered_mask, 0, 1)` (terrain-valid weight)
- `weight_256 = crop_257_to_256(weight_257)`
- `weight_16 = downsample_256_to_16(weight_256)`

**Output keys (35+ tensors):** `input`, `height_raw`, `height_norm`, `height_mean`, `height_std`, `normals`, `normal_mask`, `alpha`, `holes`, `liquid_mask`, `liquid_height`, `liquid_type_16`, `liquid_type_valid_16`, `mcly_ids`, `mcly_mask`, `mcnk_flags_16`, `weight_257/256/16`, `mddf_mask`, `modf_mask`, `object_presence_257`, `alpha_painted_256`, `terrain_valid_mask_257`, `mcly_any_16`, `what_plate_flag` (whiteplate flag), `curation_*` scores, `has_normals`, `has_alpha`, and more.

### 5.7 Training Configuration

#### V16.1 Training System — LANDED

The V16.1 training system (`train_v16_1_common.py`, 1534 lines) is a shared infrastructure for all 5 task-specific trainers.

**Task Registry (`TaskSpec` dataclass):**
```python
TASKS = {
    "height":  TaskSpec(V161HeightModel,  _height_loss,  save_height_preview),
    "normal":  TaskSpec(V161NormalModel,  _normal_loss,  save_normal_preview),
    "holes":   TaskSpec(V161HolesModel,   _holes_loss,   save_holes_preview),
    "liquid":  TaskSpec(V161LiquidModel,  _liquid_loss,  save_liquid_preview),
    "texcomp": TaskSpec(V161TexcompModel, _texcomp_loss, save_texcomp_preview),
}
```

**Loss functions (exact formulas):**

Height: `masked_mean(|pred - target|, weight_257)`

Normal: `cosine + 0.35 * vec_l1 + 0.15 * nz_l2` with terrain-valid masking, liquid weighting, object instance weighting, and hard-region boosting via `--normal-detail-boost`.

Holes: `binary_cross_entropy(pred, target, reduction="none")` masked by `weight_16`.

Liquid: `weighted_l1(mask) + 0.5 * cross_entropy(type)` masked by `type_valid * weight_16`.

Texcomp: `alpha_L1 + 0.35 * mask_BCE + 0.25 * id_CE + 0.5 * recompose_L1`.

**Batch autotuning:** Probes a ladder of batch sizes against `--target-vram-gb`, runs forward+backward per candidate, picks largest that fits. Safety factor: 0.85 with `torch.compile`, 0.92 without.

**Deterministic epoch sampler:** Custom PyTorch `Sampler` with per-epoch subset rotation, build-balanced sampling, difficulty-bucket weighted sampling, and JSONL audit logging.

**Difficulty bucket profiles:**
```python
PROFILES = {
    "uniform": {"easy": 1.0, "medium": 1.0, "hard": 1.0, "pathological": 1.0},
    "v16_1_1_normal": {"easy": 1.0, "medium": 1.75, "hard": 3.5, "pathological": 1.25},
}
```

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

---

## 13. Deep-Dive: PM4 Format (Practically Unknown)

### 13.1 Chunk Inventory (16 Recognized Chunks)

| Chunk | Full Name | Status |
|-------|-----------|--------|
| `MVER` | Version | Known |
| `MSHD` | Map Scene Header | Partially understood |
| `MSLK` | Map Scene Link | Partially understood |
| `MSPV` | Map Scene Path Vertices | Known |
| `MSPI` | Map Scene Path Indices | Known |
| `MSVT` | Map Scene Vertex | Known |
| `MSVI` | Map Scene Vertex Indices | Known |
| `MSUR` | Map Scene Surface | Partially understood |
| `MSCN` | Map Scene Nodes | Hypothesized role |
| `MPRL` | Map Position Ref List | Partially understood |
| `MPRR` | Map Position Ref Range | Partially understood |
| `MDBH` | Map Destructible Building Header | Partially understood |
| `MDBI` | Map Destructible Building Index | Partially understood |
| `MDBF` | Map Destructible Building Filename | Partially understood |
| `MDOS` | Map Destructible Object State | Partially understood |
| `MDSF` | Map Destructible Surface | Partially understood |

### 13.2 Binary Record Layouts

**MSHD Header (32 bytes):**
```
Offset  Type  Field      Status
0x00    uint  Field00    Unknown. Candidate: tile-level count.
0x04    uint  Field04    Region ID. Groups tiles into level-designer areas. (PROMOTED)
0x08    uint  Field08    Unknown. Appears to mirror Field00 in many files.
0x0C-0x1C  uint[5]      All zero across entire development corpus. Placeholders.
```
- Field04 == 1 = empty stub region (140 of 502 tiles). Active tiles never have Field04=1.
- Field0C-Field1C are ALL zero across the full development corpus.

**MSUR Entry (32 bytes):**
```
Offset  Type   Field        Confidence
0x00    byte   GroupKey     LOW — grouping/diagnostic semantics open
0x01    byte   IndexCount   MEDIUM
0x02    byte   AttributeMask LOW — bit meanings remain open
0x03    byte   Padding
0x04    Vector3 Normal     Surface normal (partially used)
0x10    float  Height      MEDIUM — behaves like signed plane-distance
0x14    uint   MsviFirstIndex First index into MSVI
0x18    uint   MscnRefIndex  MEDIUM — index into MSCN (NOT MDOS — naming corrected)
0x1C    uint   PackedParams  MEDIUM — CK24 derived from this
```
PackedParams decoding: `Ck24 = (PackedParams >> 8) & 0x00FF_FFFF`, `Ck24Type = (byte)((PackedParams >> 24) & 0xFF)`.

**MSLK Entry (24 bytes):**
```
Offset  Type   Field              Status
0x00    byte   TypeFlags          PARTIAL - observed surface-family buckets: 0x03=M2 tops, 0x10=interior WMO floors, 0x12=exterior WMO solids
0x01    byte   Subtype            OPEN — often floor/layer-like
0x02    ushort Padding
0x04    uint   GroupObjectId      LOW — NOT confirmed as full-object identity; keep distinct from TypeFlags surface-family buckets
0x08    int    MspiFirstIndex     First index into MSPI path stream
0x0C    byte   MspiIndexCount     OPEN — indices vs triangles ambiguity
0x10    uint   LinkId             PARTIAL — sentinel tiles decoded
0x14    ushort RefIndex           PRIMARY target is MSUR, but mismatches exist
0x16    ushort SystemFlag         OPEN — 0x8000 dominates
```
LinkId decoded: if high 16 bits == 0xFFFF, low 16 bits encode tile coordinates: `tileY = (low >> 8) & 0xFF`, `tileX = low & 0xFF`.

**MPRL Entry (24 bytes):**
```
Offset  Type   Field     Status
0x00    ushort Unk00     OPEN
0x02    short  Unk02     OPEN — often -1
0x04    ushort Unk04     MEDIUM — heading angle (degrees = value * 360/65536)
0x06    ushort Unk06     OPEN — often 0x8000
0x08    Vector3 Position World-space position (MEDIUM)
0x14    short  Unk14     OPEN — floor/level indicator
0x16    ushort Unk16     MEDIUM — 0=normal, nonzero=terminator
```

### 13.3 The 4-Level Hierarchy Model

```
Level 0: MSHD.Field04 (Region) — spans multiple ADT tiles
  Level 1: CK24 (Object) — WMO or M2 collision mesh
    Level 2: MSLK.GroupObjectId (Sub-object) — linked surface sets
             note: do not conflate with MSLK.TypeFlags, which now looks more like per-surface family classification
      Level 3: Individual MSUR surfaces + MPRL positions
```

**CK24 Type Classification:**
- 0x00 = Nav mesh
- 0x40-0x41 = M2 Interior
- 0x42-0x43 = WMO
- 0xC0-0xC3 = M2 Exterior
- CK24 = 0 = nav mesh spanning entire map

**Cross-tile merge:** 21.6% of distinct CK24 values appear in 2+ tiles. Top cross-tile CK24 spans 13+ tiles. Merge uses MSCN connector keys with Union-Find. Keys quantized to 2.0-unit grid. Merge criteria: adjacent tiles, ≥2 shared keys, bounds overlap (32-unit padding) OR center distance ≤256.

### 13.4 Coordinate Transforms

**Constants:**
```
TileSize = 533.33333f
HalfMapExtent = 32 * TileSize = 17066.66656f
```

**Two Coordinate Modes:** `TileLocal` (0..533.33 range) and `WorldSpace`.

**Three Axis Conventions:** `XZPlaneYUp`, `XYPlaneZUp` (default WoW), `YZPlaneXUp`.

**Planar Transform:** `Pm4PlanarTransform(bool SwapPlanarAxes, bool InvertU, bool InvertV)` — 2-4 candidates tested per mode. Footprint scoring: 85% overlap + 15% centroid distance. Decisive margin: 512.0.

**PM4-to-World (XYPlaneZUp):**
```
localU = pm4Vertex.Y; localV = pm4Vertex.X; localUp = pm4Vertex.Z
if SwapPlanarAxes: swap(localU, localV)
if TileLocal:
  mappedU = InvertU ? TileSize - localU : localU
  mappedV = InvertV ? TileSize - localV : localV
  worldX = tileY * TileSize + mappedU
  worldY = tileX * TileSize + mappedV
else (WorldSpace):
  worldX = InvertU ? -localU : localU
  worldY = InvertV ? -localV : localV
world = (worldX, worldY, localUp)
```

**MPRL Position Conversion:** `ConvertMprlPositionToWorld(refPos) = (refPos.X, refPos.Z, refPos.Y)` — Y and Z swapped.

**MSCN Hypothesis:** Axis-swapped companion geometry stream. Swapped XY overlap consistently beats raw overlap for MSCN-to-mesh bounds comparison. Not proven with ADT/object ground truth yet.

### 13.5 Verified Linkages

| Relationship | Status |
|---|---|
| MSVI → MSVT | Verified |
| MSPI → MSPV | Verified |
| MSUR.Msvi window → MSVI | Verified |
| MDSF.MsurIndex → MSUR | Verified |
| MDSF.MdosIndex → MDOS | Verified |
| MSUR._0x18 → MSCN | Partial (many fits, significant misses) |
| MSLK.RefIndex → MSUR | Partial (primary target, but mismatches) |
| MPRR.Value1 → MPRL or MSVT | Partial (mixed-mode hypothesis) |

### 13.6 Region-Aware Object Grouping (New)

A 3-phase grouping system (`Pm4RegionObjectGrouper`) decomposes PM4 data into a hierarchical object model:

**Phase 1: Region grouping** — Group tiles by `MSHD.Field04` value. Each distinct Field04 value (excluding 1 = empty stub) becomes a `Pm4Region` containing all tiles sharing that region ID.

**Phase 2: Object grouping** — Within each region, group MSUR surfaces by CK24 value (derived from `MSUR.PackedParams >> 8 & 0x00FF_FFFF`). Each CK24 group becomes a `Pm4RegionObject` with multi-tile tracking.

**Phase 3: Sub-object partitioning** — Within each object, partition surfaces by `MSLK.GroupObjectId` using Union-Find. Each partition becomes a `Pm4SubObject` containing surfaces, position refs, bounds, and average height. Keep `MSLK.TypeFlags` as a parallel classification signal, not as the partition key, until field ownership is fully proven.

**Output types:**
- `Pm4SubObject`: surfaces + position refs + bounds + average height
- `Pm4RegionObject`: CK24 object containing multiple sub-objects
- `Pm4Region`: all objects sharing an MSHD.Field04 value
- `Pm4RegionGroupingReport`: full map-directory analysis report

**Position decoding** (`Pm4ObjectPositionDecoder`):
- Resolves MPRL positions to world coordinates
- Two-pass approach: first via GroupObjectId linking, then fallback via direct surface-to-MSLK matching
- Heading computation: mean heading from MPRL entries using circular mean (sin/cos averaging)
- Uses existing `Pm4PlacementMath.ResolveCoordinateMode` and `ResolvePlacementSolution` infrastructure

### 13.7 Open Research Questions (Ranked by Impact)

1. MSCN coordinate transform: Is swapped-XY correct?
2. CK24ObjectId identity mapping: Real UniqueID or sub-identifier?
3. MSLK.RefIndex final semantics: What are non-MSUR target domains?
4. MSLK.TypeFlags/Subtype final semantics: extend or falsify the current partial buckets (`0x03` M2 tops, `0x10` interior WMO floors, `0x12` exterior WMO solids), then close how Subtype refines them
4. MSHD.Field00/Field08 relationship to Field04
5. MSLK.MspiIndexCount: Indices or triangles?
6. MPRR.Value1/Value2 full semantics
7. MPRL.Unk02/Unk06/Unk14 full semantics
8. Nav mesh (CK24=0) interaction with regions
9. Cross-region merge rules
10. MDOS.buildingIndex link type

### 13.8 Cross-Tile Statistics (Development Corpus)

- 616 total PM4 files, 502 non-empty tiles
- 227 distinct Field04 values (regions)
- 21.6% of CK24 values span 2+ tiles
- Field04=1 on 140/502 tiles (empty stubs)

---

## 14. Deep-Dive: WMO Portal Visibility (BFS Flood-Fill)

### 14.1 Algorithm Overview

The renderer uses a **BFS flood-fill through portals** to determine which WMO groups are visible from the camera position.

### 14.2 Initialization

1. Build `portalGroups` dictionary: portal index → set of groups it belongs to
2. For each group, collect `_groupPortalRefs[groupIndex]` — indices into PortalRefs
3. Compute `_portalCenters[portalIndex]` — average position of all portal polygon vertices
4. Build `_groupPortalNeighbors[groupIndex]` — list of `PortalNeighbor(neighborGroup, portalIndex)` tuples. Two groups are neighbors if they share a portal. Dedup key: `((long)portalIndex << 32) | (uint)neighborGroup`.

### 14.3 Runtime BFS

**Step 1: Transform camera to WMO-local space:** `localCameraPos = Vector3.Transform(cameraPos, inverseModelMatrix)`.

**Step 2: Near-root fast path.** If camera is inside the WMO root AABB (expanded by 32 units) OR within `ComputeNearRootFullVisibilityDistance()`:
```
largestDimension = max(extents.X, extents.Y, extents.Z)
scaledDistance = largestDimension * 0.75
fogLimitedDistance = max(192, fogEnd * 0.75)
nearRootDist = max(192, min(scaledDistance, fogLimitedDistance))
```
ALL groups become visible immediately. No BFS needed.

**Step 3: Seed selection:**
- Camera inside root: find groups containing camera, enqueue at depth 0
- Camera outside root: find exterior groups (`flags & 0x8 != 0`) that are frustum-visible, enqueue at depth 0
- If queue empty: fall back to nearest group by AABB distance
- If nearest group within near-root distance: show all groups

**Step 4: BFS traversal:**
```
portalRevealDistance = cameraInsideRoot
    ? max(InteriorPortalRevealDistance=3072, min(fogEnd, 5000))
    : max(ExteriorPortalRevealDistance=1024, min(fogEnd * 0.4, 1800))
maxTraversalDepth = cameraInsideRoot ? 4 : 1

while queue not empty:
    (groupIndex, depth) = dequeue
    runtimeVisible[groupIndex] = true
    if depth >= maxTraversalDepth: continue
    for each neighbor in groupPortalNeighbors[groupIndex]:
        if neighbor already visited: continue
        if !ShouldTraversePortal(neighbor, ...): continue
        enqueue neighbor at depth + 1
```

**Step 5: `ShouldTraversePortal`:**
- Compute portal center in world space
- Check `distanceSq <= portalRevealDistanceSq`
- If camera inside root: always traverse (distance check only)
- If camera outside root: traverse only if neighbor is frustum-visible OR is exterior group

### 14.4 Known Simplifications

1. **Portal plane-side test NOT used:** The renderer uses portal center distance rather than testing the camera's relationship to the portal plane normal. The `Side` field from WmoPortalRef and the `Normal`/`PlaneDistance` from WmoPortalDetail are parsed but NOT used in the visibility BFS.

2. **BSP tree NOT traversed:** The renderer does group-level culling via BFS + frustum + distance, but does NOT use the per-group BSP tree (MOBN/MOBF) for face-level occlusion culling within a group.

---

## 15. Deep-Dive: Terrain Edge Cases and Special Handling

### 15.1 Seam Stitching (Three-Phase Algorithm)

**Phase 1 — Edge stitching:** For each tile, average the shared edge row/column with its neighbor. Simple `(a + b) * 0.5f`. Both sides mutated in-place.

**Phase 2 — Corner stitching:** For each corner where up to 4 tiles meet, gather all contributions, compute uniform average, write back. Uses `HashSet<(CornerX, CornerY)>` to ensure each corner processed exactly once.

**Phase 3 — Predicted edge anchoring:** For ML-generated heightmaps that don't cover all tiles: only copy from neighbors NOT in the predicted set. Prevents overwriting predicted data.

### 15.2 MCNK Subchunk Size Inflation (CRITICAL)

The subchunk header's size field is **unreliable** for MCNR, MCAL, and MCSH. The authoritative sizes come from:
- MCNR: fixed known size = 0x1C0 (448 bytes)
- MCAL: MCNK header offset 0x28 (`sizeMcal`)
- MCSH: MCNK header offset 0x30 (`sizeMcsh`)

```csharp
// When encountering MCNR:
consumedSize = Math.Max(declaredSize, McnrConsumedSize); // 448

// When encountering MCAL:
consumedSize = Math.Max(declaredSize, headerMcalSize - ChunkHeader.SizeInBytes);

// When encountering MCSH:
consumedSize = Math.Max(declaredSize, headerMcshSize - ChunkHeader.SizeInBytes);
```

### 15.3 Coordinate System Transform (CRITICAL)

WoW ADT file format uses a **different coordinate system** than the 3D world:

```csharp
// World -> ADT file:
rawX = MapOrigin - worldPosition.Y;  // 17066.666 - Y
rawY = MapOrigin - worldPosition.X;  // 17066.666 - X
rawZ = worldPosition.Z;              // Z unchanged

// ADT file -> World:
worldX = MapOrigin - rawY;
worldY = MapOrigin - rawX;
worldZ = rawZ;
```

Axes are **swapped and negated** through `MapOrigin = 17066.666f`. Z (height) is preserved.

### 15.4 Normal XZY Swap

MCNR stores normals in **XZY order** (not XYZ):
```
Byte 0: X component
Byte 1: Z component (vertical!)
Byte 2: Y component
```

Encoding: `clamp(value, -1, 1) * 127`, cast to signed byte, reinterpreted as unsigned.

### 15.5 MCCV BGR Swizzle

Vertex colors are stored **BGR+Alpha**, not RGB:
```csharp
colors[idx]     = (byte)(b * 255f); // Blue first
colors[idx + 1] = (byte)(g * 255f); // Green second
colors[idx + 2] = (byte)(r * 255f); // Red third
colors[idx + 3] = 0xFF;              // Alpha = fully opaque
```

### 15.6 MCAL Packed 4-Bit Column 31 Edge Case

At the end of each row (column 31), the high nibble is **discarded** and replaced with the low nibble value. This handles the fact that 32 bytes × 2 pixels = 64 pixels per row, but the last pixel pair only uses the low nibble.

### 15.7 Legacy Edge Fix (Hard-Won Bug Fix)

The rightmost column and bottom row of packed 4-bit alpha are **corrupted by the packing process**:
```csharp
// Fix rightmost column (column 63) of every row
for (int row = 0; row < 64; row++)
    alpha[(row * 64) + 63] = alpha[(row * 64) + 62];

// Copy row 62 to row 63 (bottom row)
Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);

// Bottom-right corner = row 62, column 62
alpha[(64 * 64) - 1] = alpha[(62 * 64) + 62];
```

### 15.8 BigAlphaFixed (Cataclysm Truncation Fix)

Cataclysm sometimes truncates big alpha to 63×63 (3969 bytes). Expansion:
```csharp
for row = 0 to 62:
    bytes_available = min(63, source_end - readPos)
    copy source to output[row*64 .. row*64+bytes_available]
    output[row*64 + 63] = output[row*64 + max(0, bytes_available - 1)]  // replicate
// row 63 = copy of row 62
```

### 15.9 MCAL Force-Compressed (LK Strict Only)

In LK Strict mode, if `maxLength > 0 && maxLength < 2048` and the compressed flag is NOT set, the code **forces** the compressed flag:
```csharp
if (maxLength > 0 && maxLength < 2048)
    effectiveFlags |= CompressedAlphaFlag;
```
This handles cases where the MCLY flag is missing but data is clearly RLE-compressed.

### 15.10 Residual Alpha Synthesis

For overlay layers with no direct alpha data, the last active overlay gets residual coverage:
```csharp
alpha_last = 1.0 - sum(alpha_prev_overlays)
```
Without this, the base layer would show through everywhere.

### 15.11 Heightmap Sentinel 0.0f

Height 0.0 means "no data" in Alpha format. Gap-filling propagates from nearest non-zero neighbor. Known limitation: could theoretically overwrite legitimate sea-level heights. In practice, Alpha terrain heights are rarely exactly 0.0f at grid vertices.

### 15.12 Normal Z-Flip Guarantee

Computed normals are always flipped to have positive Z:
```csharp
if (normal.Z < 0) normal = -normal;
```
Ensures terrain surfaces never have downward-facing normals.

### 15.13 Format Profile Matrix

| Build | Alpha Decode | Big Alpha Mask | Liquid Profile |
|-------|-------------|----------------|----------------|
| 0.6.0-0.7.0 | LegacySequential | 0 | MCLQ fallback |
| 3.0.1 (8303) | LichKingStrict | 0x4\|0x80 | MH2O+MCLQ |
| 3.3.5 (12340) | LichKingStrict | 0x4\|0x80 | MH2O+MCLQ |
| 4.0.x | Cataclysm400 | 0x4\|0x80 | MH2O only |

### 15.14 Liquid Chunk Edge Stitching (New)

After liquid chunks are built in `LkToAlphaConverter`, adjacent chunk edges are stitched for seamless water surfaces:

**Horizontal stitching:** For each pair `(cx, cy)` and `(cx+1, cy)`, average the rightmost column of the left chunk with the leftmost column of the right chunk (9×9 grid stride).

**Vertical stitching:** For each pair `(cx, cy)` and `(cx, cy+1)`, average the bottom row of the top chunk with the top row of the bottom chunk.

After stitching, `MinHeight` and `MaxHeight` are recomputed on both chunks. Only stitches chunks where `Heights.Length >= 81` (full 9×9 grid present).

### 15.15 WMO External Group File Loading (New)

When no embedded MOGP groups exist in a WMO root file but `ReportedGroupCount > 0`, the reader loads external group files:
- Path pattern: `{baseName}_{groupIndex:D3}.wmo` (zero-padded 3-digit index)
- Loading: tries asset reader callback first, falls back to filesystem
- Enables footprint projection for split-group WMOs common in some WoW builds

---

## 16. Deep-Dive: M2 Animation System

### 16.1 Track Evaluation Pipeline

For each animated property (color, transparency, translation, rotation, scaling):
1. If no keyframes, return type-specific fallback
2. Resolve duration: global sequence or sequence-specific
3. Sample time: `timeMs % period` (no wrapping if duration=0)
4. Interpolation dispatch: None/Step, Linear, Hermite, Bezier

### 16.2 Hermite Cubic Interpolation

```
t2 = factor * factor; t3 = t2 * factor
h00 = 2*t3 - 3*t2 + 1
h10 = t3 - 2*t2 + factor
h01 = -2*t3 + 3*t2
h11 = t3 - t2
result = start.Value*h00 + start.OutTangent*h10 + end.Value*h01 + end.InTangent*h11
```

### 16.3 Bezier Cubic Interpolation

```
inv = 1 - factor
b0 = inv^3; b1 = 3*inv^2*factor; b2 = 3*inv*factor^2; b3 = factor^3
result = start.Value*b0 + start.OutTangent*b1 + end.InTangent*b2 + end.Value*b3
```

### 16.4 Quaternion Cubic (NOT Slerp-Based)

For Hermite/Bezier quaternion tracks, the `CombineCubic` method converts all four Quaternion operands to Vector4, does a weighted sum of all four components independently, then re-normalizes. This is a **linear blend of quaternion components**, not slerp-based spherical interpolation. This is the WoW engine's approach.

### 16.5 Compressed Quaternion Encoding

Four int16 values (8 bytes total):
```
Offset 0x00: short (Y)
Offset 0x02: short (negated X)
Offset 0x04: short (Z)
Offset 0x06: short (W)
```

Decoding: `if value < 0: (value + 32768) / 32767.0; else: (value - 32767) / 32767.0`. Clamped to [-1, 1]. Identity: `(32767, 32767, 32767, -1)` ≈ `(0, 0, 0, -1)`.

### 16.6 Bone Skinning (4 Influences Per Vertex)

```
For each influence (0..3):
    weight = vertex.BoneWeights[influence]
    if weight <= 0: skip
    boneIndex = ResolveBoneIndex(renderModel, section, vertex.BoneIndices[influence])
    skinnedPosition += Transform(vertex.Position, pose.Matrices[boneIndex]) * weight
    skinnedNormal  += TransformNormal(vertex.Normal, pose.Matrices[boneIndex]) * weight
    totalWeight += weight

if totalWeight <= 0: use original unskinned position/normal
if |totalWeight - 1.0| > 0.0001: normalize by dividing by totalWeight
```

**Bone index resolution (three-level fallback):**
1. Scoped: `section.BoneComboIndex + sectionBoneIndex` → BoneLookup
2. Direct: `sectionBoneIndex` → BoneLookup
3. Raw passthrough

### 16.7 Skin Vertex Resolution (Two-Pass)

1. First: `globalIndex = skin.VertexLookup[localSkinVertexIndex]`
2. Fallback: `globalIndex = skin.VertexLookup[localSkinVertexIndex] + skin.GlobalVertexOffset`

---

## 17. Deep-Dive: GLSL Shaders (Complete Source)

### 17.1 Terrain Fragment Shader

```glsl
#version 330 core
in vec3 vWorldPosition;
in vec3 vNormal;
in vec2 vTexCoord;
flat in uint vChunkSlice;
flat in uvec4 vTexIndices;
in vec3 vFallbackColor;

uniform sampler2DArray uDiffuseArray;
uniform sampler2DArray uAlphaShadowArray;
uniform int uDiffuseLayerCount;
uniform vec3 uLightDirection;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;

out vec4 FragColor;

bool HasLayer(uint textureIndex)
{
    return textureIndex != 65535u && int(textureIndex) < uDiffuseLayerCount;
}

void main()
{
    vec3 normal = normalize(vNormal);
    float ndotl = max(dot(normal, normalize(uLightDirection)), 0.0);
    vec3 lighting = uAmbientColor + (uLightColor * ndotl);
    float texScale = 8.0 / 33.333;
    vec2 diffuseUv = vec2(-vWorldPosition.y, -vWorldPosition.x) * texScale;
    vec4 alphaShadow = texture(uAlphaShadowArray, vec3(vTexCoord, float(vChunkSlice)));

    bool has0 = HasLayer(vTexIndices.x);
    bool has1 = HasLayer(vTexIndices.y);
    bool has2 = HasLayer(vTexIndices.z);
    bool has3 = HasLayer(vTexIndices.w);

    vec3 result = vFallbackColor * lighting;
    if (has0)
        result = texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.x))).rgb * lighting;
    if (has1)
        result = mix(result, texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.y))).rgb * lighting, alphaShadow.r);
    if (has2)
        result = mix(result, texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.z))).rgb * lighting, alphaShadow.g);
    if (has3)
        result = mix(result, texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.w))).rgb * lighting, alphaShadow.b);

    FragColor = vec4(result, 1.0);
}
```

### 17.2 Sky Backdrop Fragment Shader

```glsl
#version 330 core
in vec2 vClip;
uniform mat4 uInverseViewProjection;
uniform vec3 uCameraPosition;
uniform vec3 uZenithColor;
uniform vec3 uHorizonColor;
uniform vec3 uFogColor;
uniform float uBackdropStrength;
uniform vec3 uBackdropTint;
uniform float uBackdropSeed;
out vec4 FragColor;

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main()
{
    vec4 farPoint = uInverseViewProjection * vec4(vClip, 1.0, 1.0);
    vec3 worldPoint = farPoint.xyz / farPoint.w;
    vec3 ray = normalize(worldPoint - uCameraPosition);
    float up = clamp(ray.z * 0.5 + 0.5, 0.0, 1.0);
    float dome = smoothstep(0.18, 0.96, up);
    float horizonBand = exp(-abs(ray.z) * 5.5);
    vec3 color = mix(uHorizonColor, uZenithColor, dome);
    color = mix(color, uFogColor, horizonBand * 0.34);
    if (uBackdropStrength > 0.0)
    {
        float azimuth = atan(ray.y, ray.x) / 6.2831853 + 0.5 + (uBackdropSeed * 0.37);
        float latitude = acos(clamp(ray.z, -1.0, 1.0)) / 3.1415926;
        vec2 shellCell = floor(vec2(azimuth * 96.0, latitude * 42.0));
        float star = step(0.988, hash21(shellCell + uBackdropSeed));
        float zenithMask = smoothstep(0.30, 0.88, up);
        float shellBand = smoothstep(0.04, 0.42, abs(ray.z)) * (1.0 - smoothstep(0.78, 1.0, abs(ray.z)));
        vec3 shell = mix(uBackdropTint, vec3(0.86, 0.82, 0.66), star * zenithMask);
        color = mix(color, shell, uBackdropStrength * (0.22 + shellBand * 0.38 + star * 0.65));
    }
    FragColor = vec4(color, 1.0);
}
```

### 17.3 M2 Fragment Shader

```glsl
#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uBaseColor;
uniform vec3 uEmissiveColor;
uniform float uAlpha;
uniform bool uHasTexture;
uniform sampler2D uTexture0;
uniform bool uAlphaCutout;
uniform bool uReceivesLighting;
out vec4 FragColor;

void main()
{
    vec4 texel = uHasTexture ? texture(uTexture0, vTexCoord) : vec4(1.0);
    float finalAlpha = clamp(texel.a * uAlpha, 0.0, 1.0);
    if (uAlphaCutout && finalAlpha < 0.5)
        discard;

    vec3 shaded = texel.rgb * uBaseColor;
    if (uReceivesLighting)
    {
        vec3 normal = normalize(vNormal);
        float diffuse = max(dot(normal, normalize(-uLightDir)), 0.0);
        shaded *= clamp(uAmbientColor + (uLightColor * diffuse), vec3(0.0), vec3(1.5));
    }

    shaded += uEmissiveColor;
    FragColor = vec4(shaded, finalAlpha);
}
```

### 17.4 MDX Fragment Shader (with Skinning and Half-Lambert)

```glsl
#version 330 core
in vec3 vNormal;
in vec3 vViewNormal;
in vec2 vTexCoord;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uBaseColor;
uniform vec3 uEmissiveColor;
uniform float uAlpha;
uniform bool uHasTexture;
uniform sampler2D uTexture0;
uniform bool uAlphaCutout;
uniform float uAlphaThreshold;
uniform bool uReceivesLighting;
uniform bool uUseTextureAlpha;
uniform bool uPremultiplyAlpha;
uniform bool uSphereEnvMap;
out vec4 FragColor;

void main()
{
    vec2 texCoord = vTexCoord;
    if (uSphereEnvMap)
    {
        vec3 viewNormal = normalize(vViewNormal);
        if (!gl_FrontFacing) viewNormal = -viewNormal;
        texCoord = viewNormal.xy * 0.5 + 0.5;
    }

    vec4 texel = uHasTexture ? texture(uTexture0, texCoord) : vec4(1.0);
    vec3 texRgb = texel.rgb;
    if (uPremultiplyAlpha) texRgb *= texel.a;

    float sampledAlpha = uUseTextureAlpha ? texel.a : 1.0;
    float finalAlpha = clamp(sampledAlpha * uAlpha, 0.0, 1.0);
    if ((uAlphaCutout || uAlphaThreshold > 0.0) && finalAlpha < uAlphaThreshold)
        discard;

    vec3 shaded = texRgb * uBaseColor;
    if (uReceivesLighting)
    {
        vec3 normal = normalize(vNormal);
        float NdotL = dot(normal, normalize(uLightDir));
        float diffuse = NdotL * 0.5 + 0.5;   // Half-Lambert
        diffuse = diffuse * diffuse;            // Squared for softer falloff
        shaded *= clamp(uAmbientColor + (uLightColor * diffuse), vec3(0.0), vec3(1.75));
    }

    shaded += uEmissiveColor;
    FragColor = vec4(shaded, finalAlpha);
}
```

### 17.5 WMO Fragment Shader (with Minimum Ambient Floor)

```glsl
#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;
uniform vec3 uLightDir;
uniform vec3 uAmbientColor;
uniform vec3 uBaseColor;
uniform bool uHasTexture;
uniform sampler2D uTexture0;
uniform float uAlphaTestThreshold;
uniform bool uUseTextureAlpha;
out vec4 fragColor;

void main()
{
    vec4 texel = uHasTexture ? texture(uTexture0, vTexCoord) : vec4(1.0);
    float alpha = uUseTextureAlpha ? texel.a : 1.0;
    if (uAlphaTestThreshold > 0.0 && alpha < uAlphaTestThreshold)
        discard;

    float light = max(dot(normalize(vNormal), normalize(uLightDir)), 0.18); // min ambient floor
    vec3 shaded = texel.rgb * uBaseColor;
    shaded *= clamp(uAmbientColor + vec3(light), vec3(0.0), vec3(1.75));
    fragColor = vec4(shaded, alpha);
}
```

---

## 18. Deep-Dive: Converter Algorithms

### 18.1 WMO v14→v17 Conversion

1. Parse v17 root, validate version==14
2. Extract embedded MOGP groups
3. Upconvert MOMT: pad 44/48-byte entries to 64 bytes (zero-fill remainder)
4. Upconvert MOGI: 40-byte entries → 32 bytes (copy bytes [8..40) → [0..32))
5. Convert each group: rename MOIN→MOVI, upconvert MOPY 4-byte→2-byte (drop extra u16)
6. Build output: MVER(17) + root chunks in canonical order + MVER(17) + MOGP per group

### 18.2 WMO v17→v14 Conversion (Much More Complex)

1. Downconvert MOMT: 64→48 bytes (truncate last 16)
2. Read all group files
3. Convert each group: downconvert MOPY 2→4 bytes, downconvert MOBA firstIndex u32→u16, rename MOVI→MOIN
4. **Split oversized groups** if any batch has firstIndex > ushort.MaxValue or vertex count > 49151
5. **Handle portal layout** remapping when groups are split
6. **Merge overflow** if total groups > 384: spatial bucket splitting + group merging
7. Build v14 root: MVER(14) + MOMO wrapper + all root chunks + embedded group payloads

### 18.3 MDX→M2 Conversion

1. Read MDX geometry + bones + materials
2. Build M2 header (version 0x108, header size 0x130)
3. Build sequences from MDX sequence summaries
4. Build bones from MDX bone summaries (parent, pivot, empty animation tracks)
5. Build geometry per geoset: vertex format 48 bytes (pos + weights + boneIndices + normal + uv0 + uv1)
6. Build skin: one submesh per geoset, one batch per geoset
7. Single material layer from first resolved material

### 18.4 M2→MDX Conversion

1. Read M2 geometry + skin
2. Build triangle indices by remapping skin.VertexLookup → geometry vertex index
3. Build single material layer from first skin batch
4. Write MDX chunks: VERS→MODL→SEQS→GLBS→TEXS→MTLS→GEOS→BONE→PIVT
5. Compressed quaternion encoding: `xq = round(x * 2^21)`, `yq = round(y * 2^20)`, `zq = round(z * 2^20)`, packed into uint32

---

## 19. Deep-Dive: WMO Liquid System

### 19.1 MLIQ Parsing

Header: 30 bytes — xverts(4), yverts(4), xtiles(4), ytiles(4), cornerX/Y/Z(4 each), matId(2).

Vertex heights: 8 bytes each (4 bytes flow/filler + 4 bytes float height).

Tile flags: 1 byte per tile. `(tileFlags[t] & 0x0F) == 0x0F` means no liquid.

### 19.2 Orientation Auto-Fit

Tests 4 rotations (0/90/180/270), scores by overflow outside group bounds (weighted 1000×) + center distance. Default/tie-break = orientation 2 (90° CCW).

Effective rotation = `(autoOrientation + baselineRotation + userRotation) & 3`.

### 19.3 Liquid Type Dispatch (Per 0.8.0 Ghidra Spec)

- Nibble 0/4/8 → water
- Nibble 2/6 → magma
- Nibble 3/7 → slime
- Ocean flag: if `group.Flags & 0x80000` and type is water → ocean

### 19.4 Color Assignment

| Type | RGBA |
|------|------|
| Water | (0.15, 0.35, 0.65, 0.55) |
| Ocean | (0.10, 0.25, 0.55, 0.60) |
| Magma | (0.85, 0.25, 0.05, 0.70) |
| Slime | (0.20, 0.65, 0.10, 0.65) |

### 19.5 Vertex Mapping (4 Orientations)

```
orientation 0: (cornerX + i*tileSize, cornerY + j*tileSize)       -- no rotation
orientation 1: (cornerX + j*tileSize, cornerY - i*tileSize)       -- 90° CW
orientation 2: (cornerX - j*tileSize, cornerY + i*tileSize)       -- 90° CCW (legacy default)
orientation 3: (cornerX - i*tileSize, cornerY - j*tileSize)       -- 180°
```

Tile size: `4.16666f` (1/8th of map chunk).

---

## 20. Deep-Dive: Vertex Lighting Fallback Chain

Three-tier fallback for WMO vertex colors:

1. **Direct vertex colors:** If group.VertexColors count matches vertex count AND average luminosity > 10/255, decode BGRA packed colors.
2. **Lightmap sampling:** If v14 lightmap data exists (MOLV UVs + MOLD pixels + MOLM infos), sample nearest pixel for each vertex across all faces, average per-vertex. Requires average luminosity > 0.08/1.0.
3. **Fallback:** All vertices get `Vector4.One` (white).

---

## 21. Deep-Dive: M2 Draw Call Batching

### 21.1 Batch Limits

MaxVertices = 65535, MaxIndices = 98304.

### 21.2 Sort Key Order

Family → ModelKey → TextureSortKey → EffectKey → StateBucket → DepthSortValue (descending for transparent) → EntryKey.

### 21.3 Batch Sharing Rules

Two entries share a batch if ALL match: Family, ModelKey, EffectKey, TextureSortKey, StateBucket, IsTransparent, IsAdditive.

A batch is flushed when: key changes, direct/batch mode switches, vertex count exceeds limit, or index count exceeds limit.

### 21.4 Family Policies

| Family | Batching | Always Direct | Dedicated State |
|--------|----------|---------------|-----------------|
| Core | Yes | No | No |
| Projected | Yes | No | Yes |
| Doodad | If BatchDoodads | No | No |
| Ribbon | No | Yes | Yes |
| Particle | If BatchParticles | No | Yes |
| Callback | No | Yes | Yes |
| HitTest | No | Yes | Yes |

### 21.5 M2RuntimeOptions Flags

```
None = 0
UseZFill = 0x1
UseClipPlanes = 0x2
UseThreads = 0x4
Faster = 0x8
BatchDoodads = 0x20
BatchParticles = 0x80
ForceAdditiveParticleSort = 0x100
```

---

## 22. Deep-Dive: Triangle Winding Conversion

At buffer upload time, WoW uses CW front faces but OpenGL uses CCW. The converter swaps `indices[t+1] <-> indices[t+2]` for every triangle during GPU upload.

---

## 23. Deep-Dive: M2 Effect Recipe Classification

### Diffuse Family Selection
- Projected (batch flags 0x4 or geoset index 0x2) → Projected
- 0 textures → None, 1 → T1, 2 → T1T2, 3 → T1T2T3, 4+ → T1T2T3T4

### Combiner Family Selection
| BlendMode | Combiner |
|-----------|----------|
| Opaque | Opaque |
| AlphaKey | AlphaKey |
| AlphaBlend | Decal |
| NoAlphaAdd | Add |
| Add | Add |
| Mod | Mod |
| Mod2X | Mod2X |
| BlendAdd | Fade |

### State Bucket Bitfield
```
bits [0:3]   = blendMode
bit  [4]     = depthWrite
bit  [5]     = alphaTest
bit  [6]     = isTwoSided
bit  [7]     = !receivesLighting (unshaded)
bit  [8]     = isAdditive
bit  [9]     = isProjected
```

---

## 24. Deep-Dive: M2 Particle/Ribbon Effects

### Particle Classification
- Blending types 0-6 map to effect keys (Particle_Opaque through Particle_BlendAdditive)
- Additive types: 1, 4, 5, 6
- State bucket: bits [0:7]=blendingType, [8:15]=emitterType, [16:19]=particleType, [20:23]=headOrTail
- Max 65535 particles per emitter
- Estimated vertex count: count × 4 (quad), index count: count × 6

### Ribbon Classification
- Effect key: "Ribbon_Material_{sortKey}" or "Ribbon_Default"
- State bucket: bits [0:15]=materialSortKey, [16:23]=textureRows, [24:31]=textureColumns
- Estimated vertex count: edgeCount × 2, index count: max(0, edgeCount-1) × 6

---

## 25. Deep-Dive: Object Masking System (Training Breakthrough)

### 25.1 Six Output Masks (All 257×257)

| Mask | Type | Purpose |
|------|------|---------|
| `objectMask257` | float | Binary presence of ANY object (doodad or WMO) |
| `objectPreciseMask257` | float | Soft-edged presence with radius from scale |
| `objectInstanceMask257` | int | Unique instance ID per object |
| `mddfMask257` | float | Binary presence of doodads (MDDF) only |
| `modfMask257` | float | Binary presence of WMOs (MODF) only |
| `objectFilteredMask257` | float | Like objectMask but excluding vegetation/clutter |

### 25.2 Placement Coordinate Transform

The WoW ADT file format uses a **different coordinate system** than the 3D world:

```csharp
// World -> ADT file:
rawX = MapOrigin - worldPosition.Y;  // 17066.666 - Y
rawY = MapOrigin - worldPosition.X;  // 17066.666 - X
rawZ = worldPosition.Z;              // Z unchanged

// ADT file -> World:
worldX = MapOrigin - rawY;
worldY = MapOrigin - rawX;
worldZ = rawZ;
```

Axes are **swapped and negated** through `MapOrigin = 17066.666f`. Z (height) is preserved.

### 25.3 MODF Bounding Box Flip (Critical Edge Case)

The MODF BB corners are in WoW client space but the origin flip inverts which is min vs max:
```csharp
BoundsMin = (MapOrigin - bbMaxY, MapOrigin - bbMaxX, bbMinZ)
BoundsMax = (MapOrigin - bbMinY, MapOrigin - bbMinX, bbMaxZ)
```

### 25.4 World-Space → Tile-Local 257×257 Grid Conversion

`TryProjectPlacementToTilePixel` tries **four candidate projection modes**:

```
Candidates:
  (U, V) = (position.X / 533.333 - tileX,  position.Z / 533.333 - tileY)           [WorldXZ]
  (U, V) = ((17066.666 - position.Z) / 533.333 - tileX, (17066.666 - position.X) / 533.333 - tileY)  [OriginZX]
  (U, V) = (position.X / 533.333 - tileX,  position.Y / 533.333 - tileY)           [WorldXY]
  (U, V) = ((17066.666 - position.Y) / 533.333 - tileX, (17066.666 - position.X) / 533.333 - tileY)  [OriginYX]
```

**Selection:** Candidate valid if `U ∈ [-0.25, 1.25]` AND `V ∈ [-0.25, 1.25]`. Scoring: `score = -(|U - 0.5| + |V - 0.5|)` — closest to tile center wins.

**Final pixel:** `pixelX = Clamp(round(best.U * 256), 0, 256)`, `pixelY = Clamp(round(best.V * 256), 0, 256)`.

### 25.5 Doodad (MDDF) Mask Computation

For each placement:
1. Project center to tile pixel (px, py)
2. `radiusBinary = 2.0` (fixed 2-pixel radius)
3. `radiusPrecise = Max(1.5, placement.Scale * 2.0)` (scale-dependent)
4. `PaintCircle(mask, px, py, radiusBinary, value=1.0)` — hard binary
5. `PaintSoftCircle(preciseMask, px, py, radiusPrecise)` — linear falloff
6. `PaintCircle(instanceMask, px, py, radiusBinary, value=instanceId)` — unique ID
7. `PaintCircle(mddfMask, px, py, radiusBinary, value=1.0)` — doodad-only
8. If `ShouldIncludeDoodadInFilteredMask()`: paint filtered mask

**PaintCircle:** Iterate square of radius, skip if `dx² + dy² > r²`, write value.

**PaintSoftCircle:** Iterate square of `radius * 1.5`, `alpha = 1.0 - min(1.0, dist / radius)`, `buffer[y,x] = max(buffer[y,x], alpha)`.

#### Model-Bound Doodad Filtering (New)

When `assetReader` is available, the system loads actual `.m2`/`.mdx` model files at build time to read their bounding boxes:

```
DoodadModelMetadata { BoundsMin, BoundsMax }  // cached per model path

For each doodad:
  1. Load model metadata via assetReader (results cached in doodadModelCache)
  2. Compute localSize = |BoundsMax - BoundsMin|
  3. scale = Max(placement.Scale, 0.01)
  4. planarExtentX = localSize.X * scale
  5. planarExtentY = localSize.Z * scale
  6. planarMaxExtent = max(planarExtentX, planarExtentY)
  7. planarArea = planarExtentX * planarExtentY
  8. height = localSize.Y * scale

  isSmallClutter = planarMaxExtent ≤ 3.0  OR  planarArea ≤ 6.0
  isTallClutter  = height ≥ 8.0  AND  height ≥ planarMaxExtent × 1.35

  include if NOT isSmallClutter AND NOT isTallClutter
```

When `assetReader` is null, falls back to simple heuristic: `include if placement.Scale > 0.35`.

### 25.6 WMO (MODF) Mask Computation — Three-Tier Fallback

**Tier 1: Exact WMO mesh footprint** (when assetReader available):
1. Load WmoRenderDocument (with wmoCache for dedup)
2. Resolve transform via `ResolveWmoPlacementTransform()`:
   - Tests 3 candidate transforms: translation-only, legacy with Z-flip, legacy without Z-flip
   - Score: transform MODF-declared BoundsMin/BoundsMax through each candidate, sum absolute differences vs ADT-reported bounds. Lowest error wins.
3. Resolve projection mode via `TryResolveProjectionMode()`:
   - Tests 4 `TileProjectionMode` values (WorldXZ, OriginZX, WorldXY, OriginYX)
   - Score: count of in-range sample points + penalty for out-of-range. Highest in-range count wins.
4. For each WMO group mesh (loaded via `WmoRenderDocumentReader` with external group file support):
   - For each triangle (i0, i1, i2):
     - Transform vertices through placement transform
     - Project to tile pixels via `TryProjectToTilePixel` (auto-resolves projection mode)
     - Clip triangle to tile bounds [0, 256] × [0, 256] via Sutherland-Hodgman polygon clipping
     - Rasterize clipped polygon via edge-function triangle rasterization

**Sutherland-Hodgman clipping:** Generic polygon clipper against 4 edges (X≥0, X≤256, Y≥0, Y≤256). Each edge test: compute intersection point, add to output polygon if inside.

**Edge-function rasterization:** For each pixel (x, y) in triangle bounding box:
```
sampleX = x + 0.5, sampleY = y + 0.5   (pixel center)
w0 = EdgeFunction(v1, v2, sample)
w1 = EdgeFunction(v2, v0, sample)
w2 = EdgeFunction(v0, v1, sample)
if (w0 ≥ 0 ∧ w1 ≥ 0 ∧ w2 ≥ 0) ∨ (w0 ≤ 0 ∧ w1 ≤ 0 ∧ w2 ≤ 0):
    buffer[y, x] = max(buffer[y, x], value)
```
Edge function: `EdgeFunction(ax, ay, bx, by, px, py) = (px - ax) * (by - ay) - (py - ay) * (bx - ax)`

**WMO render document assembly:** When no embedded MOGP groups exist, loads external group files `{basename}_{groupIndex:D3}.wmo` via asset reader or filesystem. This enables footprint projection for split-group WMOs common in some WoW builds.

**Tier 2: Chunk-coverage fallback** (MCRF/MCRW per-chunk references):
1. Build `WmoPlacementChunkCoverage16[placementCount, 16, 16]` from MCRW (Cata+) or MCRF subchunks
2. Auto-detect 0-based vs 1-based reference indexing
3. For each covered chunk: paint 16×16-pixel rect

**Tier 3: Bounding-box fallback** (MODF bounds):
1. Validate bounds: min.X < max.X AND min.Y < max.Y AND not NaN
2. Project all 8 BB corners to tile pixels (auto-resolves projection mode)
3. Compute pixel AABB, paint rect + soft rect (2-pixel pad)
4. If invalid bounds: fallback to circle at placement center, radius=3.0

### 25.7 Object Filtered Mask — What Gets Excluded

**Regex exclusion** (case-insensitive, word-boundary anchored):
```
/(^|[\/_\-])(tree|trees|bush|bushes|shrub|shrubs|flower|flowers|
  plant|plants|vine|vines|fern|ferns|mushroom|mushrooms|herb|herbs|
  ivy|reed|reeds|cattail|cattails|lilypad|lilypads|kelp|
  seaweed|coral|grass|grasses|weed|weeds|rock|rocks|stone|stones|
  pebble|pebbles|gravel|twig|twigs|log|logs|stump|stumps)
  ([\/_\-.]|$)/
```

**Size-based exclusion** (when model metadata available from assetReader):
- Small clutter: `planarMaxExtent ≤ 3.0` OR `planarArea ≤ 6.0`
- Tall clutter: `height ≥ 8.0` AND `height ≥ planarMaxExtent × 1.35`
- Small doodads: `scale ≤ 0.35` (when no model metadata available)

**WMOs are ALWAYS included** — they are buildings and man-made structures.

**Doodad model metadata loading:** Both `.mdx` (legacy) and `.m2` (modern) formats are supported. MDX uses `MdxSummaryReader` for bounds; M2 uses `M2ModelReader`. Results are cached per model path in `doodadModelCache`.

### 25.8 Shadow Residual Mask

```
residual[y, x] = max(0, shadowMask256[y, x] - clamp(objectMask256[y, x], 0, 1))
```

Isolates shadows NOT caused by objects (terrain self-shadowing, baked shadowmap detail on open ground).

---

## 26. Deep-Dive: Liquid System Across All Versions

### 26.1 Version Availability Matrix

| Signal | Alpha | WotLK (3.3.5) | Cata+ | With WL files |
|--------|-------|---------------|-------|---------------|
| `mclq_*` | Yes | Yes | Rare | Yes |
| `mh2o_*` | No | Yes | Yes | Yes |
| `mcnk_flags_16` | Yes | Yes | Yes | Yes |
| `wl_*` | No | Possible | Possible | Yes |
| `unified_*` | via mclq | via mh2o>mclq | via mh2o | via mh2o>mclq>wl |

### 26.2 Unified Liquid Priority Chain

**Strict fallback — only highest-priority available source used, NO blending:**

1. **MH2O (WotLK+)** — Highest. If `mh2oHeight` + `mh2oPresence` non-null and any pixel present → use exclusively.
2. **MCLQ (pre-WotLK)** — Second. If `mclqHeight` + `mclqPresence` non-null → bilinear upsample 129×129 → 257×257.
3. **WL* loose files** — Last resort. 4×4 vertex blocks rasterized with 2-pixel radius distance-weighted blending, then globally normalized to [0,1].

### 26.3 MH2O Per-Layer Mapping

Each MH2O chunk covers a 16×16 half-step region. Per layer:
```
globalX = chunkX * 16 + layer.XOffset + localX
globalY = chunkY * 16 + layer.YOffset + localY
heights[globalY, globalX] = layer.Heights[vertexIndex]
typeMask[globalY, globalX] = (int)layer.BasicType
presenceMask[globalY, globalX] = true
```

### 26.4 MCLQ Legacy Liquid

9 vertices per chunk side (8 cells + 1 boundary), 16 chunks = 129×129 total. Per chunk: 81 heights + 64 tile flags. Tile visibility: `(tileFlags[t] & 0x0F) != 0x0F`.

### 26.5 WL Loose File Integration

Discovery: scan map directory for `*.wlw`, `*.wlm`, `*.wlq`, `*.wll`. Each 4×4 vertex block mapped to 257×257 with distance-weighted blending. WLM always forces magma type regardless of header.

### 26.6 Alpha Liquid

Flat per-chunk fill: `avgHeight = (MinHeight + MaxHeight) * 0.5f`. All 289 vertices per chunk filled with same average. Liquid type from MCNK flags: `0x04`/`0x08`=water, `0x10`=magma, `0x20`=slime.

### 26.7 Edge Cases

1. MH2O not found at expected position → fallback via MHDR header field at byte 40
2. Layer width/height clamped to [0,8]
3. MCLQ NaN/overflow guard: fills with maxHeight if height is NaN or |height| > 50000
4. WLM always forces magma type
5. WL vertex order reversed in `GetHeights4x4()`
6. MCLQ upsample skips pixels where all 4 source neighbors have no liquid
7. MH2O multiple layers: last layer's height overwrites earlier at same positions
8. Viewer uses MCLQ>MH2O priority (opposite of unified liquid builder!)

---

## 27. Deep-Dive: World Rendering Pipeline (Open Map → Pixels)

### 27.1 Phase 1: Session Bootstrap

1. Resolve client root directory
2. Open `ArchiveCatalogSession` (cached MPQ/loose file catalog)
3. Resolve map directory via `MapDirectoryLookup` (Map.dbc) or fuzzy matching
4. Read WDT file (loose overlay → on-disk → per-asset MPQ → global archive)
5. Parse WDT: `MapFileSummaryReader` → `WdtSummaryReader` → `WdtTileIndexReader`
6. Fallback: brute-force probe all 4096 tile coordinates if WDT missing

### 27.2 Phase 2: 3×3 Tile Window

For each tile in 3×3 around selected tile:
- Read root ADT → `WorldTerrainTileData` (257×257 heightmap, 256 MCNK chunks)
- Read liquid → `WorldLiquidTileData` (MCLQ/MH2O layers)
- Read placements → `AdtPlacementCatalog` (WMO + MDX placements)

**Tile coordinate convention:** `tileX` = row (Y on disk), `tileY` = column (X on disk). ADT naming: `Map_{tileY}_{tileX}.adt`.

### 27.3 Phase 3: Object Instance Construction

**WMO instances:** Bounds from ADT placement entry. Transform: simple translation. No WMO file parsing needed for preview.

**MDX instances:** Fallback bounds scaled by placement.Scale. Transform uses legacy rotation matrix:
```
Matrix4x4.CreateRotationZ(PI)
* Matrix4x4.CreateScale(scale)
* Matrix4x4.CreateRotationX(-DegreesToRadians(rotation.Y))
* Matrix4x4.CreateRotationY(-DegreesToRadians(rotation.X))
* Matrix4x4.CreateRotationZ(DegreesToRadians(rotation.Z))
* Matrix4x4.CreateTranslation(position)
```

**Skybox backdrop classification:** Must be .m2/.mdx/.mdl, must NOT contain "skylight", must contain one of: `environments/stars/`, `/skybox/`, `skybox`, `skybowl`.

### 27.4 Phase 4: Visibility Culling

**Per-object pipeline:**
1. Hide check → near-hold (384 units) → no-cull radius (512 units)
2. Frustum test → vision cone factor (`dot(toTarget, cameraForward)` mapped from [-0.35, 0.15] to [0, 1])
3. Cone-adjusted cull distance: `baseDistance * (0.45 + 0.55 * coneFactor)`
4. Hard limit: `MaxWorldObjectViewDistance = 8192`
5. Projected size threshold (profile-dependent)
6. Asset readiness check

**Three visibility profiles:**

| Profile | WMO Threshold | MDX Threshold |
|---------|--------------|--------------|
| Quality | 0 (never cull) | 0 |
| Balanced | 0.0009 | 0.0020 |
| Performance | 0.0014 | 0.0035 |

### 27.5 Phase 5: Asset Inventory Streaming

- Visible-but-not-loaded → priority queue
- Budget: 2 loads per frame, 4ms max
- State tracking: referenced/ready/pending/visible counts

### 27.6 Phase 6: Pass Coordination

**Render order:**
```
1. RenderLighting()
2. RenderSky() / RenderSkyboxBackdrop()
3. RenderWdl()
4. RenderTerrain()
--- Objects Visible Gate ---
5. PrepareObjectPhase()  -- animation + route planning
6. RenderWmoOpaque()
7. RenderMdxOpaque()
8. RenderLiquid()
9. RenderMdxTransparent()  -- back-to-front sorted
10. RenderOverlay()
```

### 27.7 Phase 7: GPU Rendering

**Terrain:** Per-tile VAO/VBO with positions, normals, UVs, chunk data. Diffuse Texture2DArray (BLP→RGBA). Alpha Shadow Texture2DArray (64×64 per chunk). Fragment: directional lighting + alpha layer blending via `mix()`.

**Sky:** Fullscreen triangle. Procedural gradient: `smoothstep(0.18, 0.96, ray.z*0.5+0.5)`. Horizon fog band: `exp(-abs(ray.z) * 5.5) * 0.34`. Optional starfield via FNV-1a hash.

**Markers:** GL_POINTS for WMO (gold) and MDX (blue) placement positions.

---

## 28. Deep-Dive: Legacy WMO Renderer (Reference Implementation)

### 28.1 Five-Pass Pipeline

```
Pass 1: Opaque shell geometry (depth write ON, no blend)
Pass 2: Doodad opaque layers (distance-culled, sorted nearest-first, capped at 1024)
Pass 3: Liquid surfaces (semi-transparent MLIQ)
Pass 4: Doodad transparent layers (back-to-front)
Pass 5: Transparent shell geometry (back-to-front by group center distance)
```

### 28.2 Portal BFS Constants

| Constant | Value |
|----------|-------|
| GroupVisibilityBoundsPadding | 32f |
| NearRootFullVisibilityDistance | 192f |
| ExteriorPortalRevealDistance | 1024f |
| InteriorPortalRevealDistance | 3072f |
| ExteriorPortalTraversalDepth | 1 |
| InteriorPortalTraversalDepth | 4 |
| DoodadCullDistance | 4000f |
| DoodadMaxRenderCount | 1024 |

### 28.3 Vertex Formats

**WMO (legacy):** 12 floats = 48 bytes: position(3) + normal(3) + UV(2) + vertexLight(4)

**M2 (legacy):** 10 floats = 40 bytes: position(3) + normal(3) + texCoord0(2) + texCoord1(2)

**Index winding:** WoW uses CW, OpenGL uses CCW. Swap `indices[t+1] <-> indices[t+2]` at upload.

### 28.4 WMO Shader (Half-Lambert + Baked Lighting)

```glsl
// Fragment
float NdotL = dot(normalize(vNormal), normalize(uLightDir));
float diffuse = NdotL * 0.5 + 0.5;   // Half-Lambert
diffuse = diffuse * diffuse;           // Squared for sharper falloff
vec3 bakedLighting = mix(vec3(1.0), clamp(vVertexLight.rgb, 0, 1), 0.6); // 60% vertex light
vec3 final = texColor.rgb * uBaseColor * (uAmbientColor + uLightColor * diffuse) * bakedLighting;
```

### 28.5 Deferred Loading Budgets

- Material textures: 1 load/frame, 2ms max
- Doodad models: 1 load/frame, 2ms max
- Doodad path resolution: data source → MPQ case-insensitive → alternate extensions (.mdx/.m2/.mdl)

### 28.6 M2 Doodad Skin Resolution (3-Level Fallback)

1. Try all `.skin` candidates from `BuildSkinCandidates()`
2. Try embedded root-profile geometry (3.3.5 build 3018303 only)
3. Try M2→MDX conversion via `M2ToMdxConverter`

### 28.7 Vertex Lighting (Three-Tier Fallback)

1. **Direct vertex colors:** BGRA packed, requires average luminosity ≥ 10/255
2. **Lightmap sampling:** v14 MOLV UVs + MOLD pixels + MOLM infos, requires luminosity ≥ 0.08
3. **Fallback:** All vertices white (Vector4.One)

### 28.8 Blend State (EGxBlend from Ghidra)

| Value | Name | Blend | Depth Write |
|-------|------|-------|-------------|
| 0 | Opaque | disabled | ON |
| 1 | Blend | SrcAlpha/OneMinusSrcAlpha | OFF |
| 2 | Add | SrcAlpha/One | OFF |
| 3 | AlphaKey | SrcAlpha/OneMinusSrcAlpha | ON (test < 0.5) |

### 28.9 Render Queue

Two-list architecture: `_opaqueItems` sorted front-to-back (early-Z), `_transparentItems` sorted back-to-front (correct alpha). Material application: texture binding, alpha test, two-sided toggle.

### 28.10 Frustum Culler (from Ghidra)

Plane extraction from view-projection matrix (6 planes). AABB test: for each plane, count corners inside; reject if zero for any plane.

### 28.11 Default Lighting

| Uniform | Default |
|---------|---------|
| fogColor | (0.6, 0.7, 0.85) |
| fogStart | 200f |
| fogEnd | 1500f |
| lightDir | normalize(0.5, 0.3, 1.0) |
| lightColor | (1.0, 0.95, 0.85) |
| ambientColor | (0.35, 0.35, 0.4) |

### 28.12 Things NOT Yet Ported to wow-viewer

1. WMO portal BFS visibility (group-level culling through portals)
2. WMO 5-pass rendering (opaque shell → doodad opaque → liquid → doodad transparent → transparent shell)
3. WMO liquid mesh building (MLIQ parsing, orientation auto-fit, type dispatch)
4. WMO vertex lighting (three-tier fallback with lightmap sampling)
5. M2 world-context rendering (skin selection, animation, bone skinning on GPU)
6. Particle rendering (billboard quads, atlas UV, blend modes)
7. TerrainManager AOI streaming (area-of-interest tile loading/unloading)
8. Full terrain renderer feature set (per-layer visibility, alpha debug, shadow display, contour lines)
9. GLB export
10. Terrain image import/export
11. PM4 workbench UI
12. WMO group controls (bounding boxes, per-group show/hide)
13. Camera click selection and terrain raycasting
14. Sky dome with day/night cycle
15. Terrain lighting + LitLoader (Alpha lights.lit)
16. Format profile registry
17. Build version catalog
18. Map discovery service (full DBC integration)
19. AreaTable service
20. TaxiPath loader
