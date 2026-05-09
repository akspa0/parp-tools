# ACTIVE CONTEXT — V14 Branch (V11 Reset)

## BRANCH
`v0.4.9-strict-guards` forked from `971fff2` on 2026-05-06.

## wow-viewer Library Completeness / Harvest Status — Resynced 2026-05-09

Phase A (terrain type system) and Phase B (harvest pipeline) are COMPLETE.
Phase C (Converters): AlphaToLk is real-data validated at 100% tile conversion across 4 maps. LkToAlpha now has **real-data MdxViewer validation** of 4.0.0 Azeroth (839/839 tiles) against staged 0.5.3 client, with asset filtering and tileset bundling.

### Landed In The May 9 Session (Phase C Alpha WDT Writer Fixup + MdxViewer Validation)

**AlphaWdtWriter structural fixes (critical):**
- MAIN grid order corrected to legacy column-major (`tileX * 64 + tileY`). Previous row-major (`tileY * 64 + tileX`) caused wrong tiles to be mapped to positions.
- Empty MCNK removal eliminated: removed `BuildEmptyMcnk`. Legacy `McnkAlpha` always reads 256 MCNK per tile; all 256 MCNKs are now emitted with full subchunk payloads.
- MCRF always emitted: legacy `McnkAlpha` unconditionally reads MCRF data. Previously omitted when empty.
- MCLY/MCAL offsets always populated (not conditionally zeroed).
- `chunkBaseHeight` mirrored at both MCNK header offsets `0x68` and `0x6C` for compatibility across reader code paths.

**AlphaWdtReader fixes:**
- `ReadExistingTiles`: MAIN index `x = i / 64, y = i % 64` (was `x = i % 64, y = i / 64`).
- `TryReadTile`: `mainEntryIndex = tileX * 64 + tileY` (was `tileY * 64 + tileX`).

**LkAdtWriter FourCC fix:**
- All chunk IDs (`MCNK`, `MCVT`, `MCNR`, `MCLY`, `MCAL`, `MCRF`, `MCSH`, `MCCV`, `MCLV`, top-level `MVER`/`MHDR`/`MCIN`) switched from `Encoding.ASCII.GetBytes()` to `FourCC.FromString().ToFileBytes()` to match repository-wide I/O boundary convention. This fixed the return-path test where `MapFileSummaryReader` could not detect ADT files.

**Added `WriteAlphaWdt_UsesLegacyMainOrderAndMcnkSubchunkContract` test:**
- Validates MAIN ordering uses legacy column-major (`tileX * 64 + tileY`).
- Validates all 256 MCNKs have valid MCLY, MCRF subchunk headers.
- Validates MCNK subchunk data does not overrun bounds.
- 3/3 `LkToAlphaRoundTripTests` pass.

**Asset name fixup (`--target-client-root`):**
- Accepts target client root, loads its MPQ catalog (or scans Alpha per-asset `.wmo.MPQ`/`.mdx.MPQ` wrappers).
- Builds `HashSet<string>` of available asset paths.
- Filters each tile's MDDF/MODF placements against the set.
- Result: 243,585 MDX + 1,287 WMO placements filtered from 839-tile Azeroth conversion (4.0.0 → 0.5.3).

**Tileset bundling (`--bundle-tilesets`):**
- Collects unique texture paths from all converted tiles.
- Reads each BLP from source client (4.0.0) via `NativeMpqService`.
- Writes to `tilesets/{map_name}/` preserving directory structure.
- Fixes up WDT MTEX references to point to local tilesets path (`tilesets/{map_name}/...`).
- Result: 327/327 textures extracted from Azeroth 4.0.0.

**MdxViewer validation results:**
- **Terrain-only mode**: loads and renders successfully. No TerrainAdapter errors. 11 tiles (2816 chunks) rendering. `ExitCode=0`. Capture saved.
- **Filtered-placements mode**: loads with v14 WMOs through Alpha `.wmo.MPQ` resolution. No crash (filtering prevents missing-asset fallback). `ExitCode=0`. Capture saved.
- See `wow-viewer/README.md` for complete validation workflow.

**New CLI flags for `convert-lk-to-alpha`:**
- `--target-client-root <dir>` / `-tcr` — filter placements against target client
- `--terrain-only` / `-to` — strip all placements (crash-proof validation)
- `--bundle-tilesets` / `-bt` — extract textures alongside output
- `--limit <N>` / `-n` — limit tile count for testing

### Architecture Notes for LkToAlpha
- Legacy reference readers: `gillijimproject-csharp/WowFiles/Alpha/{WdtAlpha,AdtAlpha,McnkAlpha}.cs`
- `McnkAlphaHeader` struct has specific subchunk offset fields that must point into the MCNK data area (after 8-byte chunk header + 128-byte MCNK header)
- MCVT/MCNR are raw (no chunk headers); MCLY/MCRF/MCSH/MCAL/MCLQ are chunked (FourCC + size + data)
- When `McnkAlpha` reads a subchunk with `new Chunk(file, offset)`, the offset must point to the chunk's FourCC, not its data
- Column-major MAIN indexing: mainEntryIndex = tileX * 64 + tileY
- Always emit MCRF (even with 0 refs), MCLY (even with 0 layers), MCAL (even with 0 nonzero alphas), MCSH (if data exists)

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
