# wow-viewer Library Completeness Plan

**Status**: Draft
**Based on**: `gillijimproject_refactor/src/MdxViewer` vs `wow-viewer/src/`
**Purpose**: Identify all gaps between the legacy viewer and the new library, and define a phased porting strategy.

---

## 1. Scope

The goal is to make `wow-viewer` a **complete, self-contained, repo-independent game file library** that:

1. Reads every file format that MdxViewer reads
2. Exports every data transformation that MdxViewer supports (converters)
3. Provides a clean foundation for GPU-accelerated rendering (viewer is a consumer, not the canonical home of logic)

**Out of scope**: Actually building the GPU-accelerated renderer. That's a separate large effort that requires its own architecture doc.

---

## 2. Gap Summary

### 2.1 File Format Readers

| Format | MdxViewer Location | wow-viewer Status |
|--------|-------------------|-------------------|
| **Alpha WDT (monolithic)** | `AlphaWdtReader` + `AlphaTerrainAdapter` | Partial — `AlphaWdtReader` exists, MCVT/MCNR/MCLY/MCAL/MCLQ parsing done, needs validation |
| **Alpha 0.6.0 split ADT** | `StandardTerrainAdapter` + `AdtAlpha` | Not tested — `AdtTensorPackBuilder` with `AdtProfile060070Baseline` should handle |
| **LK/WotLK ADT (split)** | `StandardTerrainAdapter` + `AdtLk` | Mostly done — `AdtTensorPackBuilder` + `AdtMcalDecoder` + `AdtLiquidReader` |
| **Cata 4.x split ADT** | `AdtV18` + `SplitAdt` | Partial — `AdtV23SummaryReader` exists, full decode not tested |
| **WDT (Retail)** | `Wdt` | Partial — `WdtSummaryReader` (summary only), no deep parse |
| **WDL (terrain LOD)** | `WdlParser` | Summary only — `WdlSummaryReader` exists, deep parse not ported |
| **WMO (all versions)** | `WmoRenderer` + `WmoV14Reader` | Partial — `WowViewer.Core.IO.Wmo.*` readers exist, no renderer |
| **M2 (new format)** | `M2Renderer` | Partial — `WowViewer.Core.Runtime.M2.*` exists, no viewer-side renderer |
| **MDX (old format)** | `MdxRenderer` | **Not ported** — 2866-line renderer, no library equivalent |
| **BLP (texture)** | `BlpService` + `SereniaBLPLib` | Partial — `BlpSummaryReader` exists, pixel decoding not complete |
| **MPQ archive** | `MpqDBCProvider` + `ArchiveService` | Partial — `MpqArchiveCatalog` + `AlphaArchiveReader` exist |
| **DBC/DB2** | `MpqDBCProvider` | Partial — `DbClientFileReader` exists, provider pattern not ported |
| **WL liquid files** | `WlFile` | Summary only — `WlFileReader` detection exists, deep parse not ported |
| **PM4** | `Pm4Research` | Done — `WowViewer.Core.PM4` |
| **MH2O liquid** | `Mh2oChunk` | Partial — `AdtLiquidReader` exists |
| **MCLQ liquid** | `MclqChunk` | Partial — `AdtMclqReader` exists |

### 2.2 Terrain System

| Component | MdxViewer | wow-viewer | Gap Severity |
|-----------|-----------|------------|-------------|
| `ITerrainAdapter` | ✓ | **Missing** | CRITICAL |
| `AlphaTerrainAdapter` | ✓ | **Missing** (AlphaWdtReader is partial) | CRITICAL |
| `StandardTerrainAdapter` | ✓ | **Missing** | CRITICAL |
| `TerrainChunkData` | ✓ | **Missing** | CRITICAL |
| `TerrainManager` (AOI streaming) | ✓ | **Missing** | HIGH |
| `TerrainTileMeshBuilder` | ✓ | **Missing** | HIGH |

### 2.3 Rendering System

| Component | MdxViewer Lines | wow-viewer Status | Gap Severity |
|-----------|-----------------|-------------------|-------------|
| `MdxRenderer` | ~2866 | **Missing** | CRITICAL |
| `TerrainRenderer` | ~1808 | **Missing** | CRITICAL |
| `WmoRenderer` | ~1500+ | **Missing** | CRITICAL |
| `LiquidRenderer` | ~500 | **Missing** | MEDIUM |
| `M2Renderer` | (M2 handled in Runtime) | Partial | MEDIUM |
| `ParticleRenderer` | ~500 | **Missing** | MEDIUM |
| `SkyDomeRenderer` | ~300 | **Missing** | MEDIUM |
| `MinimapRenderer` | ~400 | **Missing** (only compositor) | MEDIUM |
| `BoundingBoxRenderer` | ~150 | **Missing** | LOW |
| `ShaderProgram` | ~112 | **Missing** | HIGH |
| `RenderQueue` | ~200 | **Missing** | HIGH |
| `Material` | ~100 | **Missing** | MEDIUM |
| `FrustumCuller` | ~150 | **Missing** | MEDIUM |

### 2.4 Converters (All Missing from wow-viewer)

| Converter | File | Description |
|-----------|------|-------------|
| `LkToAlphaConverter` | `WoWMapConverter.Core` | LK split ADT → Alpha 0.5.3 monolithic WDT |
| `AlphaToLkConverter` | `WoWMapConverter.Core` | Alpha 0.5.3 WDT → LK 3.3.5 split ADT |
| `MdxToM2Converter` | `WoWMapConverter.Core` | MDX → M2 format |
| `M2ToMdxConverter` | `WoWMapConverter.Core` | M2 → MDX format |
| `WmoV14ToV17Converter` | `WoWMapConverter.Core` | WMO v14 → v17 (Cata) |
| `WmoV17ToV14Converter` | `WoWMapConverter.Core` | WMO v17 → v14 (reverse) |

---

## 3. Phase Plan

### Phase A: Foundation — Terrain Type System
**Goal**: Establish shared terrain types in `wow-viewer.Core` that both the harvest pipeline and future renderer can use.

- [ ] Define `ITerrainAdapter` interface in `wow-viewer.Core.Maps`
- [ ] Define `TerrainChunkData` in `wow-viewer.Core.Maps`
- [ ] Define `MddfPlacement`, `ModfPlacement` structs in `wow-viewer.Core.Maps`
- [ ] Define `TileLoadResult` in `wow-viewer.Core.Maps`
- [ ] Wire `AlphaWdtReader` behind `IAlphaTerrainAdapter` (internal interface)
- [ ] Wire `AdtTensorPackBuilder` to produce the same `TerrainChunkData` shape

**Dependency**: None (pure domain types)

### Phase B: Complete Harvest Pipeline
**Goal**: Full Alpha WDT → NPZ export, validated against real tiles.

- [ ] Validate `AlphaWdtReader` against known-good Alpha tiles (compare output to gillijimproject_refactor)
- [ ] Add `McnrNormalXyz` extraction to `AlphaWdtReader` (currently returns null)
- [ ] Add `McsfShadowMap` extraction to `AlphaWdtReader` (currently missing)
- [ ] Port `AlphaWdtReader` MCLQ → `MclqSurfaceHeight257` upscaling (flat plane → 257×257)
- [ ] Wire `AlphaTileData.ToPlacementCatalog()` into harvest output
- [ ] Test Alpha 0.6.0 split ADT through `AdtTensorPackBuilder` with `AdtProfile060070Baseline`

**Dependency**: Phase A

### Phase C: Port Converters
**Goal**: Bidirectional Alpha↔LK ADT conversion in `wow-viewer/src/tools/convert/`.

- [ ] Port `LkToAlphaConverter` from `WoWMapConverter.Core.Converters`
  - Parse LK split ADT (heights, normals, MCLY, MCAL, MCLQ, MDDF, MODF, MTEX)
  - Convert coordinates (XZY→XZY with MapOrigin adjustment)
  - Build monolithic WDT structure (MHDR, embedded ADTs, MAIN grid)
  - Convert liquid (LK MCLQ 804-byte → Alpha flat plane)
  - Write MCNK with non-interleaved MCVT/MCNR
  - Write MDDF/MODF with coordinate swap
- [ ] Port `AlphaToLkConverter` from `WoWMapConverter.Core.Converters`
  - Parse Alpha monolithic WDT
  - Convert to split ADT format
  - Interleave MCVT/MCNR (Alpha non-interleaved → LK interleaved)
  - Convert coordinates
  - Convert liquid (Alpha flat plane → LK MCLQ 804-byte)
- [ ] Add `WowViewer.Tool.Convert` CLI with `alpha-to-lk` and `lk-to-alpha` commands
- [ ] Cross-validate: round-trip Alpha→LK→Alpha and LK→Alpha→LK should preserve data

**Dependency**: Phase A (uses `ITerrainAdapter` types)

### Phase D: Port Mdx/M2/WMO Converters
**Goal**: Complete format conversion library.

- [ ] Port `MdxToM2Converter`
- [ ] Port `M2ToMdxConverter`
- [ ] Port `WmoV14ToV17Converter`
- [ ] Port `WmoV17ToV14Converter`

**Dependency**: Phase A, understanding of M2/MDX/WMO format libraries

### Phase E: Port Format Deep Readers
**Goal**: Fill remaining library gaps for complete format coverage.

- [ ] Deep WDT reader (not just summary)
- [ ] Deep WDL reader
- [ ] MTEX string extraction
- [ ] MMDX/MMID/MWMO/MWID name table resolution
- [ ] MHDR offset resolution service
- [ ] Format auto-detection (`FormatDetector`)

**Dependency**: Phase A

### Phase F: Renderer Architecture (Separate Effort)
**Goal**: Design and implement GPU-accelerated rendering on top of the library.

This phase is out of scope for this document and requires its own architecture spec covering:
- GPU API choice (Vulkan, DirectX, WebGPU, or cross-platform abstraction)
- Render pass architecture
- Data submission pipeline from library to renderer
- Shader management
- Resource caching

---

## 4. Architectural Principles

### 4.1 Library vs Viewer
- `wow-viewer/src/core/` = library (format readers, domain models, no rendering)
- `wow-viewer/src/viewer/` = viewer application (rendering, UI, session management)
- `wow-viewer/src/tools/` = command-line tools (harvest, convert, inspect)
- **No library code may depend on viewer code**

### 4.2 Reader Architecture
```
File bytes
  → [Format Detector] (optional, can be caller-provided)
  → [Deep Reader] (produces domain model)
  → [Domain Model] (pure C# data, no I/O)
```

All readers return domain models. Callers decide whether to call a reader based on format knowledge or detector output.

### 4.3 Converter Architecture
```
Input format → [Deep Reader] → [Domain Model] → [Converter] → [Domain Model] → [Writer] → Output format
```

Converters are read→write pipelines. They use the same deep readers and domain models as the library. No duplicate parsing logic.

### 4.4 Domain Model Ownership
- `wow-viewer.Core.Maps` owns: terrain domain types, ADT types, WDL types, WDT types
- `wow-viewer.Core.Wmo` owns: WMO domain types
- `wow-viewer.Core.M2` owns: M2/MDX domain types
- `wow-viewer.Core.Blp` owns: BLP domain types
- `wow-viewer.Core.PM4` owns: PM4 domain types

### 4.5 Repo Independence
- No source file in `wow-viewer/` may reference paths outside `wow-viewer/`
- No `.csproj` in `wow-viewer/` may reference a `.csproj` outside `wow-viewer/`
- All Python code lives under `wow-viewer/data-harvester/`
- No `.venv` outside `wow-viewer/data-harvester/`

---

## 5. Key Files Reference

### MdxViewer Source (READ-ONLY)
- Terrain adapters: `src/MdxViewer/Terrain/{AlphaTerrainAdapter,StandardTerrainAdapter,ITerrainAdapter,TerrainManager,TerrainChunkData}.cs`
- Renderers: `src/MdxViewer/Rendering/{MdxRenderer,M2Renderer,WmoRenderer,TerrainRenderer,LiquidRenderer}.cs`
- Terrain constants: `src/MdxViewer/Rendering/WoWConstants.cs`

### WoWMapConverter Source (READ-ONLY)
- Converters: `src/WoWMapConverter/WoWMapConverter.Core/Converters/{AlphaToLkConverter,LkToAlphaConverter,MdxToM2Converter,M2ToMdxConverter,WmoV14ToV17Converter}.cs`
- Terrain parsing: `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/{Mcnk,Mcal}.cs`
- Alpha parsing: `src/WoWMapConverter/WoWMapConverter.Core/Formats/Alpha/{WdtAlpha,AdtAlpha,McnkAlpha,McvtAlpha,MainAlpha}.cs`

### wow-viewer Core
- Domain types: `src/core/WowViewer.Core/Maps/`
- IO readers: `src/core/WowViewer.Core.IO/Maps/`
- Runtime M2: `src/core/WowViewer.Core.Runtime/M2/`

---

## 6. Alpha WDT Deep Reader Status

The `AlphaWdtReader` in `wow-viewer.Core.IO` is partially implemented. Below is the implementation status for each chunk:

| Chunk | Reader Status | Notes |
|-------|--------------|-------|
| MHDR | ✓ Done | Reads mcin, mtex, mddf, modf offsets |
| MPHD | ✓ Done | WMO-based detection, name table offsets |
| MAIN | ✓ Done | 64×64 grid, column-major, 16-byte entries |
| MCIN | ✓ Done | 256 × 4-byte MCNK offsets |
| MTEX | ✓ Done | Null-separated string list |
| MDDF | ✓ Done | Raw bytes → `AlphaModelPlacement[]` |
| MODF | ✓ Done | Raw bytes → `AlphaWorldModelPlacement[]` |
| MCNK.Header | ✓ Done | flags, indexX/Y, layerCount, holeMask, subchunk offsets |
| MCVT | ✓ Done | Non-interleaved 145 floats → reinterleaved → heightmap |
| MCNR | ✗ Missing | 145 normals × 3 bytes, non-interleaved → interleaved |
| MCLY | ✓ Done | 16 bytes/layer → texIds, layerMask |
| MCAL | ✓ Done | 4-bit/8-bit decode + edge fix |
| MCSH | ✗ Missing | 64×64 bits → 64×64 bytes shadow map |
| MCLQ | ✓ Done | Flat plane (minH, maxH) + base height |
| MCCV | N/A | Not present in Alpha |

Missing: `MCNR` normals extraction, `MCSH` shadow map extraction.

---

*Last updated: 2026-05-06*
