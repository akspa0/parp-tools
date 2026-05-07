# wow-viewer Library Completeness Plan

**Status**: Active — Phase A started 2026-05-06
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
| **WDT (Retail)** | `Wdt` | Partial — `WdtSummaryReader` (summary only), no deep parse. **Missing: all MPHD flags beyond `isWmoBased`; all MAIN/WDT flags beyond HasAdt/AllWater/Loaded; no WDT flag data in tensor packs.** |
| **WDL (terrain LOD)** | `WdlParser` | Summary only — `WdlSummaryReader` exists, deep parse not ported |
| **WMO (all versions)** | `WmoRenderer` + `WmoV14Reader` | Partial — `WowViewer.Core.IO.Wmo.*` readers exist, no renderer |
| **M2 (new format)** | `M2Renderer` | Partial — `WowViewer.Core.Runtime.M2.*` exists, no viewer-side renderer |
| **MDX (old format)** | `MdxRenderer` | **Not ported** — 2866-line renderer. **MDX is NOT optional: used until 2.0.0 (2006), MdxViewer reads it. A `WowViewer.Core.Mdx` library equivalent is required.** |
| **BLP (texture)** | `BlpService` + `SereniaBLPLib` | Partial — `BlpSummaryReader` exists, pixel decoding not complete |
| **MPQ archive** | `MpqDBCProvider` + `NativeMpqService` | **Done — `NativeMpqService` is the gold standard.** Ported verbatim from `gillijimproject_refactor/src/MDX-L_Tool/Services/NativeMpqService.cs`. Pure C#, patch-last archive search, listfile harvesting from all opened archives, zlib decompression, encryption support. Registered as `IArchiveCatalog` via `NativeMpqServiceFactory`. `MpqArchiveCatalog` is a separate extended implementation (bzip2/PKWARE/LZMA, HiBlockTable, 64-bit offsets, scanned files) — kept separately for now but is NOT the primary MPQ reader. |
| **DBC/DB2** | `MpqDBCProvider` | Done — `DBCD` + `WoWDBDefs` read/write DBC/DB2 natively |
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
| `MddfPlacement` / `ModfPlacement` name resolution (MDDF→MMID, MODF→MWID) | Partial | **Missing** — placements must resolve `NameId` → asset path via MMDX/MMID/MWMO/MWID name tables. **All fields must be captured: `NameId`, `UniqueId`, `Position`, `Rotation` (degrees), `Scale`, `BoundsMin`, `BoundsMax`, `Flags` — nothing dropped. These are all required downstream by viewer ingestion pipelines.** | CRITICAL |
| `TerrainLayer` texture ID resolution (MTEX) | ✓ | **Missing** — `TerrainLayer.TextureIndex` must resolve to a tileset asset path via MTEX string table | CRITICAL |
| Listfile builder (per-client MPQ archive + loose file harvest) | ✓ | **Missing** — build a complete listfile from every opened MPQ (archive internal listfiles + all discovered loose files) for each client version. Used for: (1) name-lookup cache for asset resolution, (2) identifying missing assets that cause green smoke / render errors | CRITICAL |

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

- [x] Define `ITerrainAdapter` interface in `wow-viewer.Core.Maps`
- [x] Define `TerrainChunkData` in `wow-viewer.Core.Maps`
- [x] Define `MddfPlacement`, `ModfPlacement` structs in `wow-viewer.Core.Maps`
- [x] Define `TileLoadResult` in `wow-viewer.Core.Maps`
- [x] Define `TerrainLayer` and `LiquidChunkData` in `wow-viewer.Core.Maps`
- [x] Wire `AlphaWdtReader` behind `AlphaTerrainAdapter` (implements `ITerrainAdapter`)
- [x] Wire `AdtTensorPackBuilder` to produce `TileLoadResult` via `TerrainTileTensorPack.ToTileLoadResult()`

**Dependency**: None (pure domain types)

### Phase B: Complete Harvest Pipeline
**Goal**: Full Alpha WDT → NPZ export, validated against real tiles.

- [x] Validate `AlphaWdtReader` against known-good Alpha tiles (compare output to gillijimproject_refactor) — deferred until game data available
- [x] Add `McnrNormalXyz` extraction to `AlphaWdtReader` (145 normals × 3 bytes, non-interleaved → assembled to 257×257×3)
- [x] Add `McshShadowMask256` extraction to `AlphaWdtReader` (64×64 bits per chunk → 1024×1024 → downsampled to 256×256)
- [x] Port `AlphaWdtReader` MCLQ → tile-level `MclqSurfaceHeight[257,257]` and `MclqTypeMask[16,16]`
- [x] Wire `AlphaTileData.ToPlacementCatalog()` into harvest output (`--export-placements`)
- [x] Fix MDDF/MODF model name resolution — now uses MDNM/MONM name tables instead of MTEX
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
  - Parse ALL MPHD flags (not just `isWmoBased`); flags at offset 0, 4, 8 in MPHD data
  - Parse ALL MAIN/WDT tile flags (beyond HasAdt/AllWater/Loaded); there are ~16 known flag bits
  - Include WDT flags as minimal scalar fields in `TerrainTileTensorPack` (e.g. `WdtTileHasAdt: bool`, `WdtTileAllWater: bool`, `WdtTileAsyncId: int`) — tiny, no proprietary blob decode at runtime
- [ ] Deep WDL reader
- [ ] MTEX string extraction — expose as `TerrainTileTensorPack.MtexNames: IReadOnlyList<string>`, resolved from name tables
- [ ] MMDX/MMID/MWMO/MWID name table resolution — populate `AdtPlacementCatalog` with resolved asset paths, all placement fields preserved
- [ ] MHDR offset resolution service
- [ ] Format auto-detection (`FormatDetector`)
- [ ] **MPQ listfile builder (per-client, first-class)**: `NativeMpqService` already harvests listfiles from all opened archives via `ExtractInternalListfiles()`. Expose the loaded names as `IReadOnlyDictionary<string, ulong>` (normalized virtual path → file hash) — serializable to disk per client build as a cached listfile. This is the canonical name-lookup cache for placement resolution (MDDF NameId→asset path via MMDX/MMID) and for identifying missing assets that cause green smoke/render errors.
  - **Patch chain support**: Archive search is patch-last (search-reverse) — newer patches override older ones correctly.
  - **Loose file scanning**: Also scan discovered loose files (BLP, ADT, WMO, etc.) on disk alongside MPQ listfiles to build a complete per-client artifact manifest.

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
| MCNR | ✓ Done | 145 normals × 3 bytes, non-interleaved → interleaved, assembled to 257×257×3 in `AlphaTileData.McnrNormalXyz` |
| MCLY | ✓ Done | 16 bytes/layer → texIds, layerMask |
| MCAL | ✓ Done | 4-bit/8-bit decode + edge fix |
| MCSH | ✓ Done | 64×64 bits → 64×64 bytes shadow map per chunk, assembled to 1024×1024 → downsampled to 256×256 in `AlphaTileData.McshShadowMask256` |
| MCLQ | ✓ Done | Flat plane → 257×257 `MclqSurfaceHeight` + 16×16 `MclqTypeMask` in `AlphaTileData`; also per-chunk in `AlphaLiquidChunk` |
| MCCV | N/A | Not present in Alpha |

**Known Alpha WDT data quirks (must be tracked):**
- MCSH shadow orientation: pre-0.6.0 MCSH has sun in upper-**right** corner; later versions use upper-**left**. This affects minimap decomposition — some early tiles have shading baked into MCSH instead of minimap, or a single object accidentally had shadows on during minimap gen (e.g. Arathi Highlands 0.5.3 has a solid terrain shadow in both MCSH and minimap — likely a bug in the old shadow generator or direct file write). Track `McshSunOrientation: bool` (true=upperRight) in `AlphaTileData` or `TerrainTileTensorPack`.
- **Residual tile data**: Tiles marked as `adtOffset <= 0` in MAIN (non-existent) may still contain embedded tile data. We should detect and flag this as `TileHasResidualData: bool` in `AlphaTileData`, count leftover bytes, and attempt restructuring — later game versions hide files by marking them 0 in WDT; we want to recover every artifact.
- **Sparse embedded tile detection**: Tiles with `adtOffset > 0` but where MCIN shows fewer than 256 chunk offsets, or chunk offsets point to empty subchunks, should be flagged as `TileHasSparseChunks: bool`.

Missing: MCNR normals extraction (done — `AlphaTileData.McnrNormalXyz`), MCSH shadow map extraction (done — `AlphaTileData.McshShadowMask256` / `McshShadowMask1024`), MCLQ upscaling (done — `AlphaTileData.MclqSurfaceHeight` / `MclqTypeMask`), residual/sparse tile diagnostic fields.

---

*Last updated: 2026-05-06*
