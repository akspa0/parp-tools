# wow-viewer Complete Porting Roadmap

**Version**: 1.1 — 2026-05-07
**Status**: Active — Phase C (Converters) IN PROGRESS — AlphaToLk first pass complete
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
                    │   WOW Library (2nd)   │  ADT/WDT/WMO/M2/PM4/BLP/DBC readers
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
Full Alpha WDT + Retail ADT → NPZ export, validated against real tiles across 6 game clients:

| Component | Status |
|-----------|--------|
| Alpha WDT (0.5.3/0.5.5): MCVT/MCNR/MCLY/MCAL/MCSH/MCLQ | DONE |
| Retail ADT (0.7.0/3.x/4.x): MCVT/MCNR/MCLY/MCAL/MCSH/MH2O/MCLQ/MCCV | DONE |
| MDDF/MODF placement extraction with resolved model names | DONE |
| Object footprint masks (binary + anti-aliased) | DONE |
| Shadow residual masks | DONE |
| Minimap BLP resolution (md5translate) | DONE |
| WL* loose-file liquid fallback | DONE |
| NPZ shard serialization with metadata | DONE |
| Batch harvest command (`harvest-map-mpq`) | DONE |
| IsAlphaWdt detection (MAIN size check) | DONE |
| Coordinate convention fix (IndexX→col, IndexY→row) | DONE |
| Base height offset (0x68 Alpha, 0x70 LK) | DONE |
| Subchunk reader overflow guards | DONE |
| FillHeightmapGaps | DONE |
| 0.6.0 AdtProfile060070Baseline test | NOT YET |

### Phase C: Converters

Port conversion engine from `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/`:

| Converter | Source | Purpose | Status |
|-----------|--------|---------|--------|
| `AlphaToLkConverter` | `WoWMapConverter.Core.Converters` | Alpha 0.5.3 WDT → LK 3.3.5 split ADT | DONE (first pass) |
| `LkToAlphaConverter` | `WoWMapConverter.Core.Converters` | LK split ADT → Alpha 0.5.3 monolithic WDT | NOT PORTED |
| `MdxToM2Converter` | `WoWMapConverter.Core.Converters` | MDX → M2 model format | NOT PORTED |
| `M2ToMdxConverter` | `WoWMapConverter.Core.Converters` | M2 → MDX model format | NOT PORTED |
| `WmoV14ToV17Converter` | `WoWMapConverter.Core.Converters` | WMO v14 → v17 (Cata) | NOT PORTED |
| `WmoV17ToV14Converter` | `WoWMapConverter.Core.Converters` | WMO v17 → v14 | NOT PORTED |

**New wow-viewer files for AlphaToLk**:
- `WowViewer.Core/Maps/LkAdtData.cs` — domain types for LK ADT output
- `WowViewer.Core.IO/Maps/WdlWriter.cs` — WDL binary writer
- `WowViewer.Core.IO/Maps/LkWdtWriter.cs` — LK WDT binary writer
- `WowViewer.Core.IO/Maps/LkAdtWriter.cs` — LK ADT binary writer
- `WowViewer.Core.IO/Maps/AlphaToLkConverter.cs` — conversion orchestration

**Remaining AlphaToLk work**: CLI command, real-data validation, AreaID crosswalk, split ADT support

**Validation**: Bidirectional round-trip preserves data. User has already proven `2.0.0→0.5.3` and `3.3.5→Alpha WDT` conversions work from prior screenshots.

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
| `ml-list-maps` | `WoWMapConverter.Cli.RunMlListMapsAsync` | NOT PORTED |
| `ml-export` | `VlmDatasetExporter.ExportMapAsync` | NOT PORTED (replaced by direct NPZ) |
| `ml-harvest` | `MkDatasetHarvester.HarvestAsync` | NOT PORTED |
| `ml-bake / ml-bake-heightmap` | `MinimapBakeService` / `HeightmapBakeService` | NOT PORTED |
| `convert` (alpha→lk, lk→alpha) | `WoWMapConverter.Cli` converters | NOT PORTED |
| `inspect` (PM4, ADT, WDT) | Existing `WowViewer.Tool.Inspect` | PARTIAL |
| `synthetic-minimap` | `MinimapBakeService` | STUB (not wired) |
| `ml-corpus / ml-batch` | `export_ml_corpus.ps1` | NOT PORTED |

---

## Component Ownership Map

| wow-viewer project | Owns |
|--------------------|------|
| `WowViewer.Core` | Domain types, `ITerrainAdapter`, `TileLoadResult`, `TerrainTileTensorPack` |
| `WowViewer.Core.IO` | All file readers: ADT, WDT, WMO, M2, BLP, DBC, MPQ, PM4, WDL, WL |
| `WowViewer.Core.IO.Maps` | `AlphaWdtReader`, `AdtTensorPackBuilder`, `AlphaTensorPackBuilder`, `AlphaTerrainAdapter`, `NpzTileSerializer`, `WlFileReader`, `Md5TranslateResolver` |
| `WowViewer.Core.PM4` | PM4 format reader, linkage, MPRL, MSCN |
| `WowViewer.Core.Runtime` | M2 runtime, skin profiles, render passes |
| `WowViewer.Tool.Harvest` | `extract-unified`, `harvest-map-mpq`, `harvest-tile`, `harvest-map`, `synthetic-minimap` |
| `WowViewer.Tool.Convert` | Bidirectional format converters |
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
| `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/` | Converters, VLM pipeline, minimap services |
| `gillijimproject_refactor/src/MdxViewer/Rendering/` | OpenGL renderer (M2, WMO, terrain) |
| `gillijimproject_refactor/scripts/` | Python: export orchestrator, curation, training |

---

## Implementation Order

Priorities flow from foundation upward. Library + dataset tooling first, rendering last.

### Priority 1: Library + Dataset ← CURRENT FOCUS
- **Phase B** (Harvest Pipeline) — DONE
- **Phase C** (Converters) — bidirectional format conversion
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
The NPZ tensor shard format serves as a **long-term, future-proof interchange** for ADT terrain data:
- Open format (NumPy .npz = ZIP + .npy), readable by any language
- Self-describing (metadata JSON with signal inventory, build provenance, coordinate conventions)
- Lossless (stores raw decoded values, not rendered images)
- Decoupled from proprietary WoW binary formats
- Single shard = all extractable signals from one tile → perfect for ML pipelines

This eliminates dependency on proprietary `.adt`/`.wdt` formats for downstream consumers. The library handles the decode-once → NPZ path; models and tools consume only NPZ.

--- 
*Built from: `gillijimproject_refactor/plans/` (30+ architectural plans), `AGENTS.md` guardrails, and 2 months of active refactoring across 6 game clients.*
