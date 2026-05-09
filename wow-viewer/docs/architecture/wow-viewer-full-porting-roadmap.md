# wow-viewer Complete Porting Roadmap

**Version**: 1.4 — 2026-05-09
**Status**: Active — Phase C (Converters) IN PROGRESS — AlphaToLk validated, LkToAlpha landed with focused regression proof; NPZ interchange promoted, archival-grade ADT regeneration from shards not yet complete
**Parent**: `gillijimproject_refactor` → `wow-viewer` full refactor

---

## Project Architecture

```
                    ┌──────────────────────┐
                    │   CLI Toolkit (5th)   │  extract, harvest, convert, inspect, bake
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Model Trainer (4th)  │  v11 ConvNeXt + per-signal residual models
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Dataset Builder (3rd) │  NPZ tensor shards from game clients
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   WOW Library (2nd)   │  ADT/WDT/WMO/M2/PM4/BLP/DBC readers + writers
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Viewer/Editor (1st)  │  OpenGL renderer, map editor
                    └──────────────────────┘
```

Built bottom-up: **library** → **dataset** → **trainer** → **CLI** → **viewer**. Each layer depends only on layers below it.

---

## Phase Status

### Phase A: Terrain Type System ← DONE
Shared domain types (`TerrainChunkData`, `TerrainLayer`, `LiquidChunkData`, `MddfPlacement`, `ModfPlacement`, `TileLoadResult`, `ITerrainAdapter`, `AlphaTerrainAdapter`, `AlphaTileData`). Both Alpha and retail terrains surface through the same interface.

### Phase B: Complete Harvest Pipeline ← DONE
Full Alpha WDT + Retail ADT → NPZ export, validated against real tiles across 6 game clients. The current shard contract is strong for analysis/training and already preserves undecoded ADT-family chunks as raw blobs, but it is not yet archival-complete for arbitrary ADT regeneration.

| Component | Status |
|-----------|--------|
| Alpha WDT (0.5.3/0.5.5): MCVT/MCNR/MCLY/MCAL/MCSH/MCLQ | DONE |
| Retail ADT (0.7.0/3.x/4.x): MCVT/MCNR/MCLY/MCAL/MCSH/MH2O/MCLQ/MCCV/MCLV/MCMT/MAMP | DONE |
| MDDF/MODF placement extraction with resolved model names | DONE |
| Object footprint masks (binary + anti-aliased) | DONE |
| Shadow residual masks | DONE |
| Minimap BLP resolution (md5translate) | DONE |
| WL* loose-file liquid fallback | DONE |
| NPZ shard serialization with metadata | DONE |
| Raw fallback blobs for undecoded ADT-family chunks (`raw_chunks/...`) | DONE |
| Promoted typed preservation signals (`MAMP`, `MFBO`, `MCMT`, `MCLV`, `MCSE`, `MCRF`, `MCRD`, `MCRW`) | DONE |
| Batch harvest command (`harvest-map-mpq`) | DONE |
| IsAlphaWdt detection (MAIN size check) | DONE |
| Coordinate convention fix (IndexX→col, IndexY→row) | DONE |
| Base height offset (0x68 Alpha, 0x70 LK) | DONE |
| FillHeightmapGaps | DONE |
| 0.6.0 AdtProfile060070Baseline test | NOT YET |
| Archival-grade shard contract for full ADT regeneration without reharvest | NOT YET |

**Current NPZ truth boundary:**
- Today’s shards preserve typed signals plus raw fallback blobs for chunks we do not yet decode.
- That is good enough for fast downstream analysis, training, and later decoder work on unknown chunks without immediately rereading client files.
- It is **not** yet enough to guarantee regeneration of arbitrary ADT versions from shard data alone, because some consumed surfaces are stored only in typed/derived form and `MCNK` headers are not preserved raw.
- To make NPZ the full archival interchange, we still need raw preservation for every non-structural top-level ADT-family chunk, every `MCNK` header, and every `MCNK` subchunk, even when a typed decoder also exists.

### Phase C: Converters ← IN PROGRESS

Port conversion engine from `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/`:

| Converter | Source | Purpose | Status |
|-----------|--------|---------|--------|
| `AlphaToLkConverter` | `WoWMapConverter.Core.Converters` | Alpha 0.5.3 WDT → LK 3.3.5 ADT/WDT/WDL | VALIDATED |
| `LkToAlphaConverter` | `WoWMapConverter.Core.Converters` | LK split ADT → Alpha 0.5.3 monolithic WDT | LANDED — 100% terrain/alpha round-trip validated |
| `MdxToM2Converter` | `WoWMapConverter.Core.Converters` | MDX → M2 model format | NOT PORTED |
| `M2ToMdxConverter` | `WoWMapConverter.Core.Converters` | M2 → MDX model format | NOT PORTED |
| `WmoV14ToV17Converter` | `WoWMapConverter.Core.Converters` | WMO v14 → v17 (Cata) | NOT PORTED |
| `WmoV17ToV14Converter` | `WoWMapConverter.Core.Converters` | WMO v17 → v14 | NOT PORTED |

**New wow-viewer files for AlphaToLk (May 8, 2026)**:
- `WowViewer.Core/Maps/LkAdtData.cs` — domain types for LK ADT output (`LkAdtData`, `LkMcnkData`, `LkMclyEntry`, `LkMddfEntry`, `LkModfEntry`)
- `WowViewer.Core.IO/Maps/WdlWriter.cs` — WDL binary writer with `ExtractTileHeightsFromAlpha()` helper
- `WowViewer.Core.IO/Maps/LkWdtWriter.cs` — LK 3.3.5 WDT binary writer (MVER+MPHD+MAIN+MWMO+MODF)
- `WowViewer.Core.IO/Maps/LkAdtWriter.cs` — LK 3.3.5 monolithic ADT binary writer (MVER, MHDR with offset patching, MCIN, MTEX, MMDX/MMID, MWMO/MWID, MDDF, MODF, 256 MCNK chunks)
- `WowViewer.Core.IO/Maps/AlphaToLkConverter.cs` — Conversion orchestration: reads Alpha WDT via `AlphaWdtReader`, converts `AlphaTileData` → `LkAdtData`, writes .wdt + .wdl + .adt files
- `WowViewer.Tool.Converter/AlphaToLkCommand.cs` — CLI command `convert-alpha-to-lk --input <wdt> --output <dir> [--verbose|-v]`

**AlphaToLk architecture notes for a fresh chat**:
- Writer consumes existing `AlphaWdtReader` and `AlphaTileData` — no duplication of read logic
- Uses `FourCC` from `WowViewer.Core.Chunks` for correct on-disk byte order (reversed for WoW files)
- MHDR offsets computed via two-pass build: first pass writes chunks and tracks offsets, then patches MHDR and MCIN
- MCNK position: `posX = -((ChunkSubSize * cx) + ChunkSize * tileX - ChunkSize * 32))`, `posY = -((ChunkSubSize * cy) + ChunkSize * tileY - ChunkSize * 32))`
- `AlphaWorldModelPlacement` has no `Scale` field; MODF scale defaults to 1.0 (encoded as 1024 in binary)
- AreaID currently defaults to 0 for all chunks (crosswalk not yet implemented)
- Output target is LK 3.3.5 semantics. Cataclysm-era split `_tex0.adt` / `_obj0.adt` output is out of scope for wow-viewer because existing tooling already covers that lane.
- WDL writer produces MVER+MWMO+MWID+MODF+MAOF+MARE/MAHO chunks per tile with 17×17 outer + 16×16 inner heights

**AlphaToLk validation results (May 8, 2026)**:
- PVPZone01 (0.5.5, 25 tiles): 25/25 converted, 0 failed
- Azeroth (0.5.5, 755 tiles): 755/755 converted, 0 failed (42s)
- Kalimdor (0.5.5, 972 tiles): 972/972 converted, 0 failed (61s)
- EmeraldDream (0.5.5, 256 tiles): 256/256 converted, 0 failed
- Shadowfang (0.5.5, 25 tiles): 25/25 converted, 0 failed
- WDT output validates with `map inspect` (correct MVER+MPHD+MAIN+MWMO+MODF structure)
- ADT output validates with `map inspect` (correct MVER+MHDR+MCIN+MTEX+MMDX/MMID+MWMO/MWID+MDDF+MODF+256 MCNK structure)
- Alpha layer encoding: big-alpha (LK profile), all overlay layers decoded successfully
- MDDF/MODF placement data preserved with coordinate transform (Azeroth tile 28,28: 410 doodads, 5 WMOs)
- Bugs fixed during validation: ChunkedFileReader misuse on monolithic WDTs, MHDR/MCIN empty payload, MPHD size mismatch, MAIN index formula (tileX×64+tileY → tileY×64+tileX)
- Note: 0.5.5.3494 Azeroth has 755 tiles vs 0.5.3.3368's 685; the extra ~70 tiles were added in the 0.5.5 patch

**Remaining Phase C work**:
1. **AreaID crosswalk** — `AreaIdMapper` already exists in `WowViewer.Core.IO/Dbc/AreaIdMapper.cs`; still needs wiring into `AlphaToLkConverter`
2. **Round-trip chunk-family validation** — prove that every expected input surface either maps to the correct output chunk family (`MH2O` for LK output, `MCLQ` for Alpha output) or is explicitly accounted for as out-of-band preserved/raw data
3. **Broad LK corpus validation for `LkToAlpha`** — current proof covers 100% terrain geometry and alpha mask parity on Alpha 0.5.5 roundtrips, plus `MH2O <-> MCLQ -> MH2O` parity, but broad batch conversion of native LK maps is still open
4. **Full chunk-preserving conversion** — current converter lane is still a reduced terrain-domain reconstruction, not chunk-for-chunk spec closure
5. **ADTv18 ingest boundary** — input support should tolerate full ADT v18-family chunk inventories, with undecoded or future-version chunks preserved through the NPZ/raw-blob path for later reinterpretation

**Validation truth boundary**:
- `AlphaToLk` has real-data batch proof.
- `LkToAlpha` is landed and fully validated for 100% terrain and alpha roundtrip parity against real Alpha 0.5.5 data, but broad batch conversion of native LK maps is still open.
- Neither direction should currently be described as full ADT-family chunk preservation.
- Native output chunk types remain the target: use `MH2O` for LK output and `MCLQ` for Alpha output. A temporary `MCLQ`-in-LK diagnostic path is acceptable only as a stopgap proof tool, not as the desired end state.

### Phase D: Deep Format Readers

Port remaining format parsers from `gillijimproject_refactor` to `wow-viewer`:

| Reader | Source | Gap |
|--------|--------|-----|
| WDT retail (MPHD flags, MAIN flags) | `WoWMapConverter.Core.Formats.LichKing/Wdt.cs` | Current is summary-only |
| WDL low-res terrain | `MdxViewer/WdlParser.cs` | Summary-only, no deep parse |
| WMO all versions | `MdxViewer/WmoRenderer.cs` + `WoWMapConverter.Core.Formats.Wmo/` | Partial |
| M2 runtime | `MdxViewer/M2Renderer.cs` | Partial (Runtime.M2 exists, no renderer) |
| MDX runtime | `MdxViewer/MdxRenderer.cs` (2,866 lines) | Not ported |
| BLP pixel decode | `MdxViewer/BlpService.cs` | Summary-only |
| DBC/DB2 read | `WoWMapConverter.Core.Formats.DBC/` | Done via DBCD/WoWDBDefs |
| MPQ listfile per-client | `NativeMpqService` | Done |
| Format auto-detection | `WoWMapConverter.Core.Formats/FormatDetector.cs` | Not ported |
| WMO v14 → v22 (full version range) | `WoWMapConverter.Core.Wmo/` | Partial |
| MH2O liquid (deep decode) | `WoWMapConverter.Core.Formats.LichKing/Mh2oChunk.cs` | Partial |

### Phase E: DBC/DB2 Metadata Enrichment

Attach database-driven metadata to every NPZ shard:

| Task | Source DBC | Value |
|------|-----------|-------|
| AreaTable resolution | `AreaTable.dbc` | Chunk AreaID → zone/subzone name |
| WorldSafeLocs | `WorldSafeLocs.dbc` | Graveyard coordinates per map |
| LiquidType lookup | `LiquidType.dbc` | MCLQ/MH2O type → water/ocean/magma/slime |
| Map/MapDifficulty | `Map.dbc` + `MapDifficulty.dbc` | Instance names, reset timers |
| GroundEffects | `GroundEffectTexture.dbc` | Per-chunk ground effect → texture path |
| Light/Sky | `Light.dbc` + `LightParams.dbc` + `LightSkybox.dbc` | Per-zone lighting |
| WMO AreaTable | `WMOAreaTable.dbc` | WMO group → area mapping |
| Sound entries | `SoundEntries.dbc` + `ZoneMusic.dbc` | Per-zone ambient audio metadata |

**Implementation**: Use `DBCD` + `WoWDBDefs` libraries (already in `wow-viewer`). Build per-client DBC lookup cache. Expose as string arrays in NPZ metadata JSON.

### Phase F: Placement & Object Provenance

| Task | Source |
|------|--------|
| M2 model metadata (FileDataID → asset path) | Listfile + DBC |
| WMO metadata (FileDataID → asset path) | Listfile + DBC — DONE |
| MODF doodadSet/nameSet resolution | `WMOAreaTable.dbc` |
| MCRF per-chunk reference arrays | MCNK subchunk raw data |
| PM4 placement SQLite builder | Port `build_pm4_sqlite.py` |
| Prefab object placement detection | Port `build_prefab_library.py` |
| Minimap-object association | Port `build_minimap_object_lookup.py` |
| Object visibility masks per tile | World position → pixel projection |

### Phase G: ML Model Architecture (V14+)

Current V11 trainer (`train_v11.py`): single ConvNeXt V2 backbone predicting all outputs.

V14 target architecture: **per-signal residual models**. Each model is tiny, independent, predicts one residual signal:

```
H1: minimap → coarse height (17×17)
H2: minimap + H1 → medium height (65×65)  
H3: minimap + H1 + H2 → full height (257×257)
H4: height signals → normals
H5: minimap + heights → alpha pack (texture composition)
H6: alpha + heights → minimap reconstruction (for verification)
H7: minimap + heights → liquid detection
H8: height → hole prediction
```

Each model trains independently. If H3 improves, only H3's checkpoint changes. Full architecture: `wow-viewer/docs/architecture/v14-model-and-refactor-plan-2026-05-06.md`.

### Phase H: Synthetic Data Pipeline

| Component | Purpose |
|-----------|---------|
| `MinimapBakeService` port | Full tileset compositing with shadow de-baking |
| `SynthesizedTrainingService` port | Matched deformed-minimap/heightmap training pairs |
| `HeightmapBakeService` port | 256×256 + 4096×4096 grayscale heightmap images |
| `TileStitchingService` port | Chunk-level assembly (shadows, alpha atlases, liquid maps) |
| `TerrainTileBakeService` port | Coherent 257×257 tile heightmap from 256 MCNK chunks |
| `ShadowMapService` port | MCSH 64×64 bitmap ↔ PNG codec |
| `AlphaMapService` port | RLE-compressed, 4-bit, 8-bit alpha ↔ PNG codec |

### Phase I: Viewer/Editor Shell

| Component | Source | Purpose |
|-----------|--------|---------|
| World session management | `MdxViewer/WorldScene.cs` | AOI tile streaming, map loading |
| Navigator/inspector panels | `MdxViewer/ViewerApp*.cs` | Tile browser, placement inspector |
| Minimap overlay | `MdxViewer/MinimapRenderer.cs` | 2D minimap panel |
| WDL terrain preview | `MdxViewer/WdlTerrainRenderer.cs` | Low-res 3D terrain mesh |
| OpenGL renderer port | `MdxViewer/Rendering/*.cs` | GPU-accelerated terrain/WMO/M2 rendering |
| Map editor | `MdxViewer export tools` | Terrain brush painting, object placement |
| GLB export | `MdxViewer/Export/MapGlbExporter.cs` | Terrain tile → glTF mesh |

### Phase J: CLI Toolkit

| Command | Source | Status |
|---------|--------|--------|
| `extract-unified` | New | DONE |
| `harvest-map-mpq` | New | DONE |
| `harvest-tile / harvest-map` | New | DONE (disk paths) |
| `convert-alpha-to-lk` | New (May 8) | VALIDATED |
| `convert-lk-to-alpha` | New (May 8) | LANDED — 100% terrain/alpha round-trip validated |
| `ml-list-maps` | `WoWMapConverter.Cli.RunMlListMapsAsync` | NOT PORTED |
| `ml-export` | `VlmDatasetExporter.ExportMapAsync` | NOT PORTED (replaced by direct NPZ) |
| `ml-harvest` | `MkDatasetHarvester.HarvestAsync` | NOT PORTED |
| `ml-bake / ml-bake-heightmap` | `MinimapBakeService` / `HeightmapBakeService` | NOT PORTED |
| `convert` (lk→alpha) | `WoWMapConverter.Cli` converters | REPLACED by `convert-lk-to-alpha` |
| `inspect` (PM4, ADT, WDT) | Existing `WowViewer.Tool.Inspect` | PARTIAL |
| `synthetic-minimap` | `MinimapBakeService` | STUB (not wired) |
| `ml-corpus / ml-batch` | `export_ml_corpus.ps1` | NOT PORTED |

---

## Component Ownership Map

| wow-viewer project | Owns |
|--------------------|------|
| `WowViewer.Core` | Domain types, `ITerrainAdapter`, `TileLoadResult`, `TerrainTileTensorPack`, `LkAdtData`, `LkMcnkData` |
| `WowViewer.Core.IO` | All file readers + writers: ADT, WDT, WMO, M2, BLP, DBC, MPQ, PM4, WDL, WL |
| `WowViewer.Core.IO.Maps` | `AlphaWdtReader`, `AdtTensorPackBuilder`, `AlphaTensorPackBuilder`, `AlphaTerrainAdapter`, `AdtTerrainWriter`, `AdtPlacementWriter`, `NpzTileSerializer`, `WlFileReader`, `Md5TranslateResolver`, `AlphaToLkConverter`, `LkToAlphaConverter`, `AlphaWdtWriter`, `LkAdtWriter`, `LkWdtWriter`, `WdlWriter` |
| `WowViewer.Core.PM4` | PM4 format reader, linkage, MPRL, MSCN |
| `WowViewer.Core.Runtime` | M2 runtime, skin profiles, render passes |
| `WowViewer.Tool.Harvest` | `extract-unified`, `harvest-map-mpq`, `harvest-tile`, `harvest-map`, `synthetic-minimap` |
| `WowViewer.Tool.Convert` | `convert-alpha-to-lk`, dataset/ML utilities |
| `WowViewer.Tool.Inspect` | PM4/ADT/WDT format inspection |
| `WowViewer.App` | Viewer shell, terrain rendering, minimap overlay |
| `data-harvester/` | Python: training, inference, quilt visualization, NPZ analysis |

---

## Known Working Data

### Verified Tiles (Azeroth 32,32)

| Client | Tiles | Height Range | MDDF | MODF | Signals |
|--------|-------|-------------|------|------|---------|
| 0.5.3.3368 | 685 | 38-186 | 758 | 6 | 11 (no object/residual — fixed) |
| 0.5.5.3494 | 755 | — | — | — | 11 |
| 0.7.0.3694 | 755 | — | — | — | 11 + texture_names |
| 3.0.1.8303 | 687 | — | — | — | 11 + texture_names |
| 3.3.5.12340 | 687 | 0-186 | 764 | 7 | 11 + texture_names, placement names |
| 4.0.0.11927 | 839 | — | — | — | 12 + MCCV |

### Weak Signal Recovery (Proven)
User has demonstrated: terrain data recovery from MCAL/MCLY correlations in sparse tiles, enabling training on "lost" data within the existing corpus. This feeds into the V14 per-signal residual model architecture.

---

## Reference Implementation Sources

**READ-ONLY** — do not modify. Port logic, don't rewrite:

| Source | What it contains |
|--------|-----------------|
| `gillijimproject_refactor/src/MdxViewer/Terrain/AlphaTerrainAdapter.cs` | Alpha WDT per-chunk extraction |
| `gillijimproject_refactor/src/MdxViewer/Terrain/StandardTerrainAdapter.cs` | Retail ADT per-chunk extraction (2,289 lines) |
| `gillijimproject_refactor/src/gillijimproject-csharp/WowFiles/Alpha/` | Alpha WDT/ADT/MCNK low-level parsers |
| `gillijimproject_refactor/src/gillijimproject-csharp/WowFiles/LichKing/` | LK ADT/MCNK binary format (AdtLk.cs, McnkLk.cs, Mhdr.cs) |
| `gillijimproject_refactor/src/gillijimproject-csharp/WowFiles/ChunkHeaders.cs` | McnkHeader struct (0x80 bytes), McnkAlphaHeader struct |
| `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/` | Converters, VLM pipeline, minimap services |
| `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs` | Reference AlphaToLk implementation (930 lines, WDL generation, AreaID crosswalk) |
| `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/LkToAlphaConverter.cs` | Reference LkToAlpha implementation (~965 lines, Alpha MCNK builder) |
| `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Builders/` | AlphaMainBuilder, AlphaMcnkBuilder, AlphaMhdrBuilder |
| `gillijimproject_refactor/src/WoWRollback/Core/Services/PM4/Wdt335Writer.cs` | Reference LK WDT writer |
| `gillijimproject_refactor/src/MdxViewer/Rendering/` | OpenGL renderer (M2, WMO, terrain) |
| `gillijimproject_refactor/scripts/` | Python: export orchestrator, curation, training |

---

## Implementation Order

Priorities flow from foundation upward. Library + dataset tooling first, rendering last.

### Priority 1: Library + Dataset ← CURRENT FOCUS
- **Phase B** (Harvest Pipeline) — DONE for typed signals plus raw fallback on undecoded chunks; archival-grade shard completeness still open
- **Phase C** (Converters) — `AlphaToLk` validated, `LkToAlpha` landed with focused proof; remaining work is AreaID crosswalk, round-trip chunk-family validation, broad validation, and model/WMO converters
- **Phase D** (Deep Format Readers) — complete every file reader port
- **Phase E** (DBC Metadata) — enrich shards with database-driven metadata
- **Phase F** (Placement Provenance) — MCRF arrays, PM4, prefab library

### Priority 2: Model Training
- **Phase G** (V14 Models) — per-signal residual architecture
- **Phase H** (Synthetic Data) — training pair generation, shadow de-baking

### Priority 3: Dataset Curation
- On-the-fly curation to minimize raw data processing
- Shard validation at every level (data sanity, decode verification, round-trip testing)
- Manifest-driven subset selection for focused training

### Priority 4: CLI Completion
- **Phase J** — every command from reference ported

### Priority 5: Viewer/Editor
- **Phase I** — OpenGL renderer, UI panels, map editor

### NPZ as Interchange Format
The NPZ tensor shard format is the **strategic interchange target** for downstream tooling:
- Open format (NumPy .npz = ZIP + .npy), readable by any language
- Self-describing (metadata JSON with signal inventory, build provenance, coordinate conventions)
- Fast to load for analysis, visualization, training, and transformation tooling
- Already carries typed terrain signals plus raw fallback blobs for undecoded ADT-family chunks

**Current truth boundary:**
- Today’s shard contract is excellent for ML and analysis workflows.
- It is **not yet archival-complete** for regenerating arbitrary ADT versions from shards alone.
- To make NPZ the full interchange for reconstruction/conversion, the shard contract still needs raw preservation for all non-structural top-level ADT-family chunks, every `MCNK` header, and every `MCNK` subchunk, even when typed decoders exist.

The decode-once → NPZ path is still the right direction. The remaining work is to make that NPZ contract archival-grade rather than only analysis-grade.

--- 
*Built from: `gillijimproject_refactor/plans/` (30+ architectural plans), `AGENTS.md` guardrails, and 2 months of active refactoring across 6 game clients.*
