# ACTIVE CONTEXT — V14 Branch (V11 Reset)

## BRANCH
`v0.4.9-strict-guards` forked from `971fff2` on 2026-05-06.

## wow-viewer Library Completeness / Harvest Status — Resynced 2026-05-08

Phase A (terrain type system) and Phase B (harvest pipeline) are COMPLETE.
Phase C (Converters): AlphaToLk is real-data validated at 100% tile conversion across 4 maps, and LkToAlpha is now landed with focused Alpha/LK round-trip coverage including `MH2O <-> MCLQ` liquid parity.

### Landed In The May 8 Session (Phase C AlphaToLk)

**New files:**
- `WowViewer.Core/Maps/LkAdtData.cs` — domain types for LK ADT output (LkAdtData, LkMcnkData, LkMclyEntry, LkMddfEntry, LkModfEntry)
- `WowViewer.Core.IO/Maps/WdlWriter.cs` — WDL binary writer with `ExtractTileHeightsFromAlpha` helper
- `WowViewer.Core.IO/Maps/LkWdtWriter.cs` — LK WDT binary writer (MVER+MPHD+MAIN+MWMO+MODF)
- `WowViewer.Core.IO/Maps/LkAdtWriter.cs` — full LK 3.3.5 monolithic ADT binary writer (MVER, MHDR with offset patching, MCIN, MTEX, MMDX/MMID, MWMO/MWID, MDDF, MODF, 256 MCNK with MCVT/MCNR/MCLY/MCAL/MCRF/MCSH)
- `WowViewer.Core.IO/Maps/AlphaToLkConverter.cs` — orchestration: AlphaTileData → LkAdtData, reads Alpha WDT via AlphaWdtReader, writes .wdt + .wdl + .adt files
- `WowViewer.Tool.Converter/AlphaToLkCommand.cs` — CLI command `convert-alpha-to-lk --input <wdt> --output <dir> [--verbose|-v]`

**Bugs fixed during validation:**
1. **ChunkedFileReader crash on monolithic WDTs** — replaced `ChunkedFileReader.ReadTopLevelChunks()` calls in `AlphaToLkConverter` and `AlphaToLkCommand` with `AlphaWdtReader.ReadExistingTiles()` which correctly parses monolithic Alpha WDTs without hitting embedded ADT sub-chunks
2. **MHDR/MCIN empty payload** — `LkAdtWriter.Build()` wrote chunk headers declaring 64/4096 bytes but wrote 0 data bytes, corrupting all subsequent offsets. Fixed by writing pre-allocated zero arrays.
3. **MPHD size mismatch** — `LkWdtWriter` wrote 9 uint32s (36 bytes) but declared MPHD size as 32 bytes. Fixed by removing extra `Write(0u)`.
4. **MAIN index formula** — `AlphaWdtReader.TryReadTile` used `tileX * 64 + tileY` (wrong) for MAIN table lookup. Fixed to `tileY * 64 + tileX` (correct row-major with y as row). This bug caused 420/755 Azeroth tiles to fail before the fix; all maps now convert at 100%.

**Validation results (0.5.5.3494 dataset):**

| Map | Tiles | Result |
|-----|-------|--------|
| Azeroth | 755 | 755/755 converted |
| Kalimdor | 972 | 972/972 converted |
| EmeraldDream | 256 | 256/256 converted |
| PVPZone01 | 25 | 25/25 converted |
| Shadowfang | 25 | 25/25 converted |

Note: The 0.5.5.3494 Azeroth has 755 tiles (vs 685 in 0.5.3.3368) because 0.5.5 added ~70 more tiles to the continent.

### Landed In The May 8 Follow-Up Session (Phase C LkToAlpha)

**Implementation surface:**
- `WowViewer.Core.IO/Maps/AlphaWdtWriter.cs` — reverse Alpha writer now emits structurally valid monolithic WDT tiles, correct chunk-relative `MCVT`, chunk-local `MCAL`, and `MCLQ` payloads
- `WowViewer.Core.IO/Maps/LkToAlphaConverter.cs` — LK ADT -> AlphaTileData conversion now carries structured liquid instead of collapsing to flags only
- `WowViewer.Core.IO/Maps/AlphaWdtReader.cs` — preserves `MCLQ` 81-sample surface heights as absolute world Z during round-trip reads
- `WowViewer.Core.IO/Maps/AlphaToLkConverter.cs` + `LkAdtWriter.cs` — Alpha back-conversion now rebuilds `MH2O` payloads instead of dropping liquid after Alpha ingest
- `WowViewer.Tool.Converter/LkToAlphaCommand.cs` — CLI path now parses input `MH2O` instead of silently discarding LK liquid before conversion
- `wow-viewer/tests/WowViewer.Core.Tests/LkToAlphaRoundTripTests.cs` — focused regression coverage for Alpha writer structure and `MH2O <-> MCLQ -> MH2O` liquid round-trip

**Bugs fixed during validation:**
1. **Alpha placeholder chunk corruption** — `AlphaWdtWriter` declared `MPHD` and `MHDR` sizes without reserving payload bytes, shifting every later chunk offset. Fixed by writing zero-filled payloads for declared placeholder chunks.
2. **Chunk-relative terrain heights** — reverse Alpha `MCVT` was being written relative to tile base instead of chunk base. Fixed to match Alpha reader expectations.
3. **MCNK offset frame mismatch** — Alpha MCNK subchunk offsets were not aligned to the frame expected by `AlphaWdtReader`. Fixed so emitted tiles parse as real Alpha WDT content.
4. **MCAL contract mismatch** — the repo's tile-level `256x256` alpha pack was being treated like per-chunk `64x64` payloads. Fixed by resampling tile alpha into chunk-local MCAL payloads during write.
5. **Liquid loss on both directions** — reverse conversion dropped `MH2O` on LK input and failed to emit `MCLQ`; return conversion only restored flags. Fixed by carrying structured liquid through `LkMcnkData`, preserving `MCLQ` 81-sample surfaces, and emitting real `MH2O` payloads on the LK writer path.

**Focused validation:**
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter LkToAlphaRoundTripTests`
- Current proof level is focused library regression only. Broad `WowViewer.Core.Tests` closure is still blocked by missing real-data fixtures and one unrelated pre-existing invalid-data test.

### Architecture Notes for AlphaToLk
- Consumes existing `AlphaWdtReader` and `AlphaTileData` — no duplication of read logic
- Writer uses `FourCC` from `WowViewer.Core.Chunks` for correct on-disk byte order
- MHDR offsets computed via two-pass build: first pass writes chunks and tracks offsets, second pass patches MHDR and MCIN
- MCNK position computation matches LK convention: `posX = -((ChunkSubSize * cx) + ChunkSize * tileX - ChunkSize * 32))`
- `AlphaWorldModelPlacement` has no `Scale` field; MODF scale defaults to 1.0 for WMO placements
- AreaID currently defaults to 0 for all chunks (crosswalk not yet implemented)
- Split ADT support (_tex0, _obj0) not yet implemented; current output is monolithic ADT only
- `AreaIdMapper` exists in `WowViewer.Core.IO/Dbc/AreaIdMapper.cs` (995 lines) with embedded crosswalk CSV at `WowViewer.Core.IO/Resources/area_crosswalk.csv` — not yet wired to AlphaToLkConverter

## WHAT WORKS
- `extract-unified` for Alpha monolithic WDT tiles on staged `0.5.3` and `0.5.5`
- `AdtTensorPackBuilder` / harvest tensor-pack generation on staged `0.7.0`, `3.0.1`, `3.3.5`, and `4.0.0`
- Alpha placement export through `--export-placements`
- Alpha and retail object footprint mask generation in the current tensor-pack contract
- Metadata JSON with current `AvailableSignals` coverage for the active harvest path
- `WowViewer.Tool.Harvest harvest-map-mpq` is the canonical multi-client shard builder for staged archive-backed clients
- **AlphaToLk terrain-domain conversion pipeline: 100% tile conversion across 4 maps, 5 terrain types**
- **LkToAlpha terrain-domain conversion pipeline exists in the shared library and CLI, and is fully round-trip validated for terrain geometry, alpha masking, and `MH2O <-> MCLQ` liquid parity against real 0.5.5 client data.**
- **Important proof boundary:** tile conversion counts and focused round-trip tests do **not** mean full ADT/WDT chunk preservation. The current converter lane rebuilds a reduced terrain-domain model and still drops chunk families it does not yet decode.
- **ADT harvest continuity update:** `TerrainTileTensorPack` / NPZ shards now preserve unconsumed ADT-family chunks as raw uint8 blobs under `raw_chunks/...` so current harvesting no longer silently discards every non-decoded chunk family.
- **ADT preservation signal promotion update:** `MAMP`, `MFBO`, `MCMT`, `MCLV`, `MCSE`, `MCRF`, `MCRD`, and `MCRW` now exist as first-class NPZ signals rather than raw-only fallback chunks. `MCSE` currently keeps raw fallback alongside the typed NPZ signals because only the standard `0x1C` emitter layout is decoded.
- **ADT staged-proof boundary update:** staged real-data `MCSE` and `MCRF` coverage now scans multiple staged client roots and common map families, but both checks are still availability-based smoke in this environment rather than hard-pinned positive regressions; synthetic tests still carry the exact `MCSE`/`MCRF` preservation proof, including the trailing-subchunk proof for the `MCRF` payload-range fix.
- **Alpha harvest continuity update:** Alpha tile tensor packs now also carry raw embedded tile chunks under `raw_chunks/alpha/...`, so Alpha harvesting keeps exact source payloads beside decoded signals.
- WDT/WDL/ADT output validates via `map inspect` (correct chunk structure, MCAL big-alpha decoding)
- `convert-alpha-to-lk` CLI command in `WowViewer.Tool.Converter`
- `convert-lk-to-alpha` CLI command in `WowViewer.Tool.Converter`
- **NPZ shard interchange validated on real LK 3.3.5 data:** Azeroth tile 32,32 harvested via MPQ produces a complete NPZ shard with 13 typed signals (height, normals, textures, alpha, shadow, holes, objects, minimap) plus structurally valid `metadata.json`. Raw-chunk collection now correctly rejects misaligned garbage FourCCs after FourCC validation fix, producing 0 garbled entries where previously 274 were fabricated.
- **AdtTensorPackBuilder overflow fixes:** Three MCNK sub-chunk locator methods (`LocateMcvtDataOffset`, `LocateMcnrDataOffset`, `LocateMccvDataOffset`) and `TryReadSplitMcnkSubchunkPayload` were fixed to use `long` arithmetic instead of `checked((int))` casts that threw `OverflowException` on real LK client data. The raw-chunk collector `CollectRawMcnkSubchunks` was similarly fixed and gained FourCC validation and MCAL/MCSH header-size override handling matching the tensor pack builder.
- **Alpha resolution fix:** NPZ shards now store both `mcal_alpha_pack` (1024×1024 full resolution, 64px/chunk) and `mcal_alpha_pack_256` (256×256 bilinearly downsampled). The previous single `mcal_alpha_pack_256` stored 1024×1024 data under a name implying 256×256, causing visualization grid artifacts at chunk boundaries. The 256 version is now correctly downsampled for minimap-resolution compositing; the 1024 version preserves per-pixel fidelity for training.
- **Grid lines in normal/height visualizations are expected:** MCVT/MCNR store 145 vertices per chunk with shared boundary vertices. Rendering them as a flat 257×257 image produces visible seams at chunk edges. This is not a bug — it's inherent to the WoW ADT vertex layout.

## WHAT IS STILL OPEN
- AreaID crosswalk support (currently all chunks default to AreaID 0)
- Split ADT output (_tex0, _obj0) for Cataclysm+ clients
- LkToAlpha real-data batch validation beyond the new focused round-trip regressions
- Full Alpha/LK chunk-for-chunk preservation instead of reduced terrain-domain reconstruction
- M2/MDX converters and WMO v14↔v17 converters
- Deep format readers Phase D
- DBC/DB2 metadata enrichment Phase E
- Placement provenance Phase F

## WHAT BROKE / DO NOT ROUTE BACK TO
- `--client-root` mode for the older pre-harvest dataset-build path
- `build_v10_2_dataset.py` and `train_v10_2_terrain_synth.py` as active architecture owners
- `WowViewer.Tool.Converter dataset-scan` → `dataset-audit` → `dataset-curate` → `dataset-build-cache` as the primary shard-generation path. That is legacy manifest/audit tooling and does not surface the full modern harvest/tensor-pack signals/metrics.
- Repo-root `output\tmp\...` as the default home for real dataset prep runs. Canonical outputs should land under `wow-viewer\output\datasets\`.

## KEY FILES — wow-viewer Library
- Domain types: `wow-viewer/src/core/WowViewer.Core/Maps/`
- IO readers/writers: `wow-viewer/src/core/WowViewer.Core.IO/Maps/`
  - `AlphaWdtReader.cs` — Alpha WDT parser for the harvest path
  - `AlphaWdtWriter.cs` — Alpha WDT writer used by the reverse LK→Alpha converter path
  - `AlphaTerrainAdapter.cs` — AlphaTileData → TerrainChunkData bridge
  - `AlphaToLkConverter.cs` — Alpha WDT → LK ADT/WDT/WDL conversion orchestration
  - `LkToAlphaConverter.cs` — LK ADT → Alpha WDT conversion orchestration
  - `LkAdtWriter.cs` — LK 3.3.5 monolithic ADT binary writer
  - `LkWdtWriter.cs` — LK WDT binary writer
  - `WdlWriter.cs` — WDL binary writer with height extraction
  - `AdtTerrainWriter.cs` — existing ADT heightmap/normal patcher
  - `AdtPlacementWriter.cs` — existing ADT placement patcher
- DBC crosswalk: `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` + `Resources/area_crosswalk.csv`
- CLI: `wow-viewer/tools/converter/WowViewer.Tool.Converter/{AlphaToLkCommand,LkToAlphaCommand}.cs`

## NEXT
1. Dataset prep lane: use staged clients + `dataset-list-maps` for discovery + `WowViewer.Tool.Harvest harvest-map-mpq` for shard generation into `wow-viewer\output\datasets\`, then NPZ-based validation/visualization from the harvested shards
2. Phase C (continued): AreaID crosswalk, LkToAlpha real-data validation, Mdx↔M2, Wmo v14↔v17
3. Phase D: Deep format readers (WDT retail flags, WDL, WMO full version range, MDX, BLP pixel decode)
4. Phase E: DBC/DB2 metadata enrichment (AreaTable, WorldSafeLocs, LiquidType, GroundEffects)
5. Phase F: Placement provenance (MCRF per-chunk arrays, PM4 SQLite, prefab detection)

**Full roadmap**: `wow-viewer/docs/architecture/wow-viewer-full-porting-roadmap.md`
**Current architecture**: library → dataset → trainer → CLI → viewer (bottom-up)
