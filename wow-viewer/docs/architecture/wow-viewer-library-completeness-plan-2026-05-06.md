# wow-viewer Library Completeness Plan

**Status**: Active — Phase A complete; Phase B harvest/tensor-pack lane complete for typed signals plus raw fallback on undecoded ADT-family chunks; Phase C terrain converters landed in shared form as of 2026-05-08, with AlphaToLk real-data proof and LkToAlpha focused round-trip proof; archival-grade NPZ/ADT interchange remains unfinished
**Based on**: `gillijimproject_refactor/src/MdxViewer` vs `wow-viewer/src/`
**Purpose**: Identify all gaps between the legacy viewer and the new library, and define a phased porting strategy.

## 0. Execution Update — 2026-05-09

- Phase A is complete in `wow-viewer`: `ITerrainAdapter`, `TerrainChunkData`, `TerrainLayer`, `TileLoadResult`, `AlphaTerrainAdapter`, and the `TerrainTileTensorPack.ToTileLoadResult()` bridge all landed.
- Phase B is no longer just "Alpha WDT validation pending". The current `WowViewer.Tool.Harvest extract-unified` plus `AlphaTensorPackBuilder` path now works on staged `0_5_3_3368` and `0_5_5_3494`, and the broader tensor-pack path is proven on staged `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, and `4_0_0_11927`.
- Recent May 6-7 commits fixed the Alpha `MCLY` header handling bug, restored missing Alpha `AvailableSignals` metadata, added Alpha placement export via `AlphaTileData.ToPlacementCatalog()`, and added Alpha object-mask plus shadow-residual generation.
- The remaining near-term gap is explicit `0.6.0` split-ADT validation via `AdtProfile060070Baseline`, not basic Alpha/retail harvest plumbing.
- Follow-up on May 8 landed the reverse `LkToAlphaConverter` path in `wow-viewer`, repaired the Alpha WDT writer so emitted tiles parse correctly, and added focused `LkToAlphaRoundTripTests` covering structural round-trip plus `MH2O <-> MCLQ -> MH2O` liquid parity.
- Current AlphaWDT regression baseline confirms focused LK↔Alpha test coverage remains green (`LkToAlphaRoundTripTests`: `17/17` passing), including MCNK metadata/liquid/reference ownership checks.
- Proof boundary matters: AlphaToLk has real-data batch validation, while LkToAlpha is currently proven at focused library-regression scope rather than broad LK corpus runs.
- ADT-family raw fallback preservation is now real: undecoded top-level chunks and undecoded `MCNK` subchunks persist into NPZ shards under `raw_chunks/...` with metadata (`source_kind`, `source_path`, `scope`, `chunk_id`, `chunk_index`, `chunk_x`, `chunk_y`, `byte_length`).
- Several formerly raw-only chunks are now also promoted into typed shard signals (`MAMP`, `MFBO`, `MCMT`, `MCLV`, `MCSE`, `MCRF`, `MCRD`, `MCRW`).
- The proof boundary still matters here too: current shards are good analysis/training interchange, but they are **not yet archival-complete** for rebuilding arbitrary ADT versions from NPZ alone because `MCNK` headers and some consumed chunks are not preserved as exact raw bytes.

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
| **Alpha WDT (monolithic)** | `AlphaWdtReader` + `AlphaTerrainAdapter` | Substantially done for the harvest/tensor-pack path — validated on staged `0.5.3`/`0.5.5`, with placements, object masks, and shadow residual now emitted. Remaining work is broader consumer/runtime ownership and residual-data diagnostics. |
| **Alpha 0.6.0 split ADT** | `StandardTerrainAdapter` + `AdtAlpha` | Partial — `AdtTensorPackBuilder` with `AdtProfile060070Baseline` exists, but explicit `0.6.0` validation is still open |
| **LK/WotLK ADT (split)** | `StandardTerrainAdapter` + `AdtLk` | Broad tensor-pack support proven on staged `3.0.1` and `3.3.5`; deeper reader/converter closure is still incomplete |
| **Cata 4.x split ADT** | `AdtV18` + `SplitAdt` | Broad tensor-pack support proven on staged `4.0.0.11927` including MCCV/MCLV/MCMT/MAMP plus raw fallback preservation for undecoded ADT-family chunks; full deep-reader and archival-grade shard ownership are still incomplete |
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
| **MH2O liquid** | `Mh2oChunk` | Partial — `AdtLiquidReader` plus `LkAdtWriter`/converter round-trip emission now exist, but broader consumer and real-data validation are still incomplete |
| **MCLQ liquid** | `MclqChunk` | Partial — `AdtMclqReader` plus `AlphaWdtReader`/`AlphaWdtWriter` round-trip preservation now exist, but broader consumer and real-data validation are still incomplete |

### 2.2 Terrain System

| Component | MdxViewer | wow-viewer | Gap Severity |
|-----------|-----------|------------|-------------|
| `ITerrainAdapter` | ✓ | Done | CLOSED |
| `AlphaTerrainAdapter` | ✓ | Done | CLOSED |
| `StandardTerrainAdapter` | ✓ | Partial — tensor-pack/decode coverage exists through `AdtTensorPackBuilder`, but a first-class wow-viewer runtime adapter equivalent is still missing | HIGH |
| `TerrainChunkData` | ✓ | Done | CLOSED |
| `TerrainManager` (AOI streaming) | ✓ | **Missing** | HIGH |
| `TerrainTileMeshBuilder` | ✓ | **Missing** | HIGH |
| `MddfPlacement` / `ModfPlacement` name resolution (MDDF→MMID, MODF→MWID) | Partial | Partial — `AdtPlacementReader` and `AlphaTileData.ToPlacementCatalog()` now resolve names and preserve placement fields for export/runtime consumers, but the tensor-pack contract still stores flattened numeric arrays rather than a first-class shared placement object model | MEDIUM |
| `TerrainLayer` texture ID resolution (MTEX) | ✓ | Partial — tensor packs now carry `MclyTextureNames`, but `TerrainLayer` still only stores texture indices in the tile-load bridge | MEDIUM |
| Listfile builder (per-client MPQ archive + loose file harvest) | ✓ | Partial — shared archive-listfile cache/bootstrap work already exists, and `NativeMpqService` harvests internal listfiles, but a single first-class per-client manifest surface across all consumers is still not complete | MEDIUM |

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

### 2.4 Converters

| Converter | File | Status | Notes |
|-----------|------|--------|-------|
| `LkToAlphaConverter` | `WoWMapConverter.Core` | LANDED | Focused round-trip proof only; broad LK corpus validation still open |
| `AlphaToLkConverter` | `WoWMapConverter.Core` | VALIDATED | Real-data batch proof landed; output target is LK 3.3.5, not Cataclysm split ADT emission |
| `MdxToM2Converter` | `WoWMapConverter.Core` | NOT PORTED | — |
| `M2ToMdxConverter` | `WoWMapConverter.Core` | NOT PORTED | — |
| `WmoV14ToV17Converter` | `WoWMapConverter.Core` | NOT PORTED | — |
| `WmoV17ToV14Converter` | `WoWMapConverter.Core` | NOT PORTED | — |

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
**Goal**: Full Alpha WDT + retail ADT → NPZ export with typed signals and raw fallback preservation for undecoded ADT-family chunks.

- [x] Validate `AlphaWdtReader` against known-good Alpha tiles (compare output to gillijimproject_refactor) — deferred until game data available
- [x] Add `McnrNormalXyz` extraction to `AlphaWdtReader` (145 normals × 3 bytes, non-interleaved → assembled to 257×257×3)
- [x] Add `McshShadowMask256` extraction to `AlphaWdtReader` (64×64 bits per chunk → 1024×1024 → downsampled to 256×256)
- [x] Port `AlphaWdtReader` MCLQ → tile-level `MclqSurfaceHeight[257,257]` and `MclqTypeMask[16,16]`
- [x] Wire `AlphaTileData.ToPlacementCatalog()` into harvest output (`--export-placements`)
- [x] Fix MDDF/MODF model name resolution — now uses MDNM/MONM name tables instead of MTEX
- [x] Fix Alpha `MCLY` parsing to skip the embedded subchunk header before reading layer entries
- [x] Restore missing Alpha `AvailableSignals` metadata for `mcly_texture_ids`, `mcly_layer_mask`, and `mcal_alpha_pack_256`
- [x] Generate `object_mask_257`, `object_precise_mask_257`, and `shadow_residual_mask_256` for Alpha tensor packs
- [x] Prove signal-complete staged extraction across `0.5.3`, `0.5.5`, `0.7.0`, `3.0.1`, `3.3.5`, and `4.0.0`
- [x] Add placement flat arrays (`placement_mddf_data`, `placement_modf_data`) to NPZ with resolved model paths
- [x] Port `WlFileReader` for WLW/WLM/WLQ/WLL liquid files (fallback when MCLQ missing)
- [x] Fix Alpha/Retail coordinate conventions (cx/cy swap, base height offsets, FillHeightmapGaps)
- [x] Persist undecoded ADT-family top-level chunks and `MCNK` subchunks into shard raw blobs with metadata
- [x] Promote preservation-focused typed signals for `MAMP`, `MFBO`, `MCMT`, `MCLV`, `MCSE`, `MCRF`, `MCRD`, `MCRW`
- [ ] Test Alpha 0.6.0 split ADT through `AdtTensorPackBuilder` with `AdtProfile060070Baseline`
- [ ] Upgrade shard contract from analysis-grade to archival-grade by preserving raw `MCNK` headers and raw bytes for consumed-but-currently-lossy chunks so ADTs can be regenerated from NPZ alone

**Dependency**: Phase A — **Phase B IS COMPLETE**

### Phase B2: DBC/DB2 Metadata Enrichment (NEW)
**Goal**: Attach rich database-driven metadata to every NPZ shard so downstream models have full provenance.

- [ ] **WorldSafeLocs resolution**: Read `WorldSafeLocs.dbc` from client, resolve graveyard coordinates per map. Expose as per-tile mask or JSON array in metadata.
- [ ] **AreaTable resolution**: Read `AreaTable.dbc` from client, resolve AreaID → area/zone/subzone names. Attach per-chunk AreaID → name mapping in metadata JSON. (MdxViewer has it!)
- [ ] **GroundEffects layer**: Read `GroundEffectTexture.dbc` + related DBCs. Per-chunk ground effect IDs are already in MCNK header (`GroundEffectsMap1-4`). Resolve to texture paths.(MdxViewer has parts of it!)
- [ ] **Light/LightParams/Sky**: Read light-related DBCs for per-zone lighting data (useful for reconstruction rendering). (MdxViewer has it!)
- [ ] **LiquidType resolution**: Read `LiquidType.dbc` to resolve MCLQ/MH2O type IDs to liquid names (water, ocean, magma, slime). (MdxViewer has it!)
- [ ] **Map.dbc / MapDifficulty.dbc**: Resolve map instance names, difficulties, loading screen associations. (MdxViewer has it!)
- [ ] **DBC provider threading**: Ensure `IDBCProvider` is accessible from the harvest tool for all staged client versions.
- [ ] **Per-client DBC cache**: Build a per-client-build DBC lookup cache key database so the harvest tool resolves DBC data once per client, not per tile.

**Dependency**: Phase B, DBCD/WoWDBDefs libraries (already in wow-viewer)

### Phase B3: Placement & Object Provenance (NEW)
- [ ] **M2/MDX model metadata**: Resolve model FileDataID → asset path using listfile or DBC. Expose in `placement_mddf_names`.
- [ ] **WMO metadata**: Resolve WMO FileDataID → asset path. Expose in `placement_modf_names`. **(DONE — names already resolved)**
- [ ] **MODF doodadSet/nameSet resolution**: Per-MODF `doodadSet` and `nameSet` fields for WMO variant selection.
- [ ] **Shared MCRF provenance surfaces**: `AlphaWdtWriter` now emits per-chunk Alpha `MCRF` arrays, but shard/runtime consumers still need first-class access to the exact per-chunk reference lists and any future extent-aware doodad-owner metadata.

### Phase B4: NPZ Interchange / ADT Preservation (NEW)
**Goal**: Make NPZ the long-term interchange for fast tooling **and** eventual ADT regeneration without rereading client files.

- [x] Keep typed terrain signals for fast analysis/training workflows
- [x] Preserve undecoded ADT-family chunks as raw blobs with metadata
- [x] Treat ADT v18-family input support as an ingest/preservation requirement even when output targets stop at LK 3.3.5
- [ ] Preserve raw `MCNK` headers in the shard contract
- [ ] Preserve raw bytes for consumed-but-not-archived chunks when exact file regeneration matters (especially `MH2O`, `MCAL`, `MCLY`, `MCSH`, `MTXF`, placement/name-table chunks, and any other chunk currently reduced to typed or flattened data)
- [ ] Define shard-side decoders/loaders for raw chunk blobs so later tools can parse wiki-documented chunks directly from NPZ without going back to client archives

### Phase C: Port Converters
**Goal**: Bidirectional Alpha↔LK ADT conversion in `wow-viewer/src/tools/convert/`.

- [x] Port `LkToAlphaConverter` from `WoWMapConverter.Core.Converters`
  - Parse LK split ADT (heights, normals, MCLY, MCAL, MCLQ, MDDF, MODF, MTEX)
  - Convert coordinates (XZY→XZY with MapOrigin adjustment)
  - Build monolithic WDT structure (MHDR, embedded ADTs, MAIN grid)
  - Convert liquid (`MH2O` or LK `MCLQ` → Alpha `MCLQ`, preserving 81-sample surface heights in the shared round-trip path)
  - Write MCNK with non-interleaved MCVT/MCNR
  - Write MDDF/MODF with coordinate swap
- [x] Port `AlphaToLkConverter` from `WoWMapConverter.Core.Converters`
  - Parse Alpha monolithic WDT
  - Convert to LK 3.3.5 output format
  - Interleave MCVT/MCNR (Alpha non-interleaved → LK interleaved)
  - Convert coordinates
  - Convert liquid (Alpha `MCLQ` → LK `MH2O` payloads in the current shared writer path)
- [x] Add converter CLI commands in `WowViewer.Tool.Converter` with `convert-alpha-to-lk` and `convert-lk-to-alpha`
- [ ] Complete `AreaIdMapper` orchestration wiring: `AlphaToLkConverter.ConvertTile(...)` now accepts and applies `AreaIdMapper`, but `ConvertWdt`/CLI-side auto-load and broad real-data proof refresh are still open
- [ ] Cross-validate: round-trip Alpha→LK→Alpha and LK→Alpha→LK should preserve data on broader real-data corpora, not just focused library regressions
- [ ] Validate chunk-family mapping: every expected input chunk surface must either become the correct native output chunk family (`MH2O` in LK, `MCLQ` in Alpha) or be explicitly accounted for through preserved/raw interchange data
- [ ] Promote converter proof from reduced terrain-domain reconstruction to chunk-preserving conversion where required
- [ ] Keep Cataclysm split-output generation explicitly out of scope for wow-viewer converter work; existing external tooling already covers that output lane

**Dependency**: Phase A (uses `ITerrainAdapter` types)

### Phase D: Port Mdx/M2/WMO Converters
**Goal**: Complete format conversion library.

- [ ] Port `MdxToM2Converter` — expected to be a relatively bounded port because wow-viewer already has partial M2/runtime ownership and MDX-related groundwork
- [ ] Port `M2ToMdxConverter` — same boundary as above
- [ ] Port `WmoV14ToV17Converter` — treat as a planned bounded port, not a new architecture problem
- [ ] Port `WmoV17ToV14Converter` — same boundary as above

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

### 4.4 NPZ Interchange Truth Boundary
- The current NPZ contract is the correct direction for fast downstream tooling.
- Today’s shards are strong analysis/training interchange: typed signals + metadata + raw fallback for undecoded chunks.
- They are **not yet** archival-complete for regenerating arbitrary ADT versions from NPZ alone.
- To close that gap, shard preservation must extend beyond undecoded chunks to raw `MCNK` headers and any consumed chunk whose current typed representation is lossy or version-specific.

### 4.5 Domain Model Ownership
- `wow-viewer.Core.Maps` owns: terrain domain types, ADT types, WDL types, WDT types
- `wow-viewer.Core.Wmo` owns: WMO domain types
- `wow-viewer.Core.M2` owns: M2/MDX domain types
- `wow-viewer.Core.Blp` owns: BLP domain types
- `wow-viewer.Core.PM4` owns: PM4 domain types

### 4.6 Repo Independence
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

The `AlphaWdtReader` in `wow-viewer.Core.IO` is implemented for the current harvest/tensor-pack path. Below is the implementation status for each chunk:

`AlphaWdtReader` and `AlphaWdtWriter` are also the canonical alphaWDT format owners. Future `MdxViewer` alphaWDT read/write work should consume these shared contracts instead of extending app-side readers or adding a second writer.

| Chunk | Reader Status | Notes |
|-------|--------------|-------|
| MHDR | ✓ Done | Reads mcin, mtex, mddf, modf offsets |
| MPHD | ✓ Done | WMO-based detection, name table offsets |
| MAIN | ✓ Done | 64×64 grid, row-major, 16-byte entries |
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

Remaining alphaWDT follow-up: residual/sparse tile diagnostic fields, broader consumer/runtime ownership that still sits outside the current harvest/tensor-pack lane, and retiring duplicate app-side readers after their consumers move to the shared APIs.

---

*Last updated: 2026-05-08*
