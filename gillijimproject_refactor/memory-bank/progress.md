# PROGRESS — V14 Branch (V11 Reset)

## POSITION
- V10 pipeline: dead. Two-stage nonsense with broken archive path.
- V11: v9 proven pipeline + ConvNeXt backbone + MCAL/MCLY multi-task + proper training infra.
- V14: wow-viewer library completeness effort, modular terrain model system, and a now-proven harvest/tensor-pack lane from Alpha `0.5.x` through `4.0.0`.

## VALIDATED
| What | Status |
|------|--------|
| MpqArchiveCatalog probe fix | DONE — break→continue, 256 limit |
| MCAL/MCLY in v9 BuildDirectCacheEntry | DONE |
| V11 model forward pass | DONE — 35.5M, 6 output heads |
| V11 training loop (GPU) | DONE — AMP, EMA, cosine, uncertainty loss |
| V11 inference + OBJ export | DONE |
| Channel layout audit | DONE — 26ch, shadow removed, MCCV at 3x dropout |
| Cache memory (LRU 2GB cap) | DONE |
| Zero-samples / empty vocab guards | DONE |
| wow-viewer Phase A: Terrain type system | DONE |
| wow-viewer: NativeMpqService port | DONE — pure C#, no StormLib |
| wow-viewer: AlphaTileData.ToTileLoadResult | DONE |
| wow-viewer: TerrainTileTensorPack.ToTileLoadResult | DONE |
| wow-viewer: Harvest tool extract-unified | DONE |
| wow-viewer: MDDF/MODF model name resolution | DONE |
| wow-viewer: Harvest --export-placements | DONE |
| wow-viewer: Alpha object/precise masks | DONE |
| wow-viewer: Alpha shadow residual mask | DONE |
| **Alpha 0.5.3/0.5.5: all 11 signals** | **DONE** |
| **Retail 3.x: all 11 signals** | **DONE** |
| **Cata 4.0.0: 12 signals** | **DONE** — + MCCV vertex colors |
| **0.7.0 extraction** | **DONE** — AdtProfile0703694 |
| **Alpha object mask projection** | **DONE** |
| **Minimap via Md5TranslateResolver** | **DONE** |
| **All staged clients pass** | **DONE** — 0.5.3, 0.5.5, 0.7.0, 3.0.1, 3.3.5, 4.0.0 |
| **Placement flat arrays in NPZ** | **DONE** |
| **Placement model names resolved** | **DONE** |
| **BuildKey provenance** | **DONE** |
| **WL* loose-file liquid fallback** | **DONE** |
| **Coordinate fixes** | **DONE** |
| **Phase C: AlphaToLk writer infrastructure** | **DONE** — WdlWriter, LkWdtWriter, LkAdtWriter, AlphaToLkConverter |
| **Phase C: AlphaToLk CLI command** | **DONE** — convert-alpha-to-lk in WowViewer.Tool.Converter |
| **Phase C: AlphaToLk real-data tile conversion validation** | **DONE** — 755/755 Azeroth (0.5.5), 972/972 Kalimdor, 256/256 EmeraldDream, 25/25 PVPZone01, 25/25 Shadowfang |
| **Phase C: LkToAlpha reverse converter + writer path** | **DONE** — `LkToAlphaConverter`, `AlphaWdtWriter`, `convert-lk-to-alpha` |
| **Phase C: LkToAlpha focused round-trip validation** | **DONE** — `LkToAlphaRoundTripTests` prove Alpha writer structure and `MH2O <-> MCLQ -> MH2O` parity |
| **Phase C: LkToAlpha real-data MdxViewer render** | **DONE** — 839/839 tiles, terrain + WMOs + MCLQ render, ExitCode=0 |
| **Phase C: LkToAlpha asset name fixup** | **DONE** — `--target-client-root` filters 244k+ missing placements |
| **Phase C: LkToAlpha tileset bundling** | **DONE** — `--bundle-tilesets` extracts 327 textures, fixes up paths |
| **Phase C: LkToAlpha AlphaWdtWriter structural parity** | **DONE** — MAIN row-major per 0.5.3 client, always emit MCNK+MCRF, FourCC I/O convention |
| **Phase C: Placement orientation proof** | **DONE** — Ghidra-confirmed Alpha MDDF/MODF position and rotation transforms; LK writer MODF bounds now round-trip through the shared reader |
| **Phase C: ADT shard raw-chunk preservation** | **DONE** — unconsumed ADT-family chunks now persist into NPZ shards as raw uint8 blobs with metadata instead of being dropped outright |
| **Phase C: ADT preservation signal promotion** | **DONE** — `MAMP`, `MFBO`, `MCMT`, `MCLV`, `MCSE`, `MCRF`, `MCRD`, and `MCRW` now persist as first-class NPZ entries rather than raw-only fallback blobs; `MCSE` currently retains raw fallback beside the typed signals, and staged real-data `MCSE`/`MCRF` coverage is broadened smoke rather than a pinned positive regression |
| **Phase C: Alpha shard raw-chunk preservation** | **DONE** — Alpha tile tensor packs now preserve raw embedded tile chunks under `raw_chunks/alpha/...` alongside decoded signals |

## IN PROGRESS
| What | Status |
|------|--------|
| Multi-client full shard dataset prep | SWITCHED TO HARVEST PATH — use `WowViewer.Tool.Harvest harvest-map-mpq` on staged clients, not converter `dataset-scan` manifests |
| Phase C: AlphaToLk AreaID crosswalk | NOT YET — `AreaIdMapper` exists in `WowViewer.Core.IO/Dbc/`, not yet wired to converter |
| Phase C: LkToAlpha real-data MdxViewer validation | DONE — 839/839 tiles, terrain renders, WMOs load, ExitCode=0 |
| Phase C: Alpha/LK full chunk preservation | OPEN — current converter lane is still a reduced terrain-domain reconstruction, not chunk-for-chunk spec closure. Gap inventory documented below. |
| Phase C: Mdx↔M2 converters | NOT PORTED |
| Phase C: Wmo v14↔v17 converters | NOT PORTED |
| Phase C: Legacy MdxViewer → wow-viewer port | LONG-RANGE — `WowViewer.App` shell exists but needs world-session, terrain, WMO, M2 rendering

## CHUNK PRESERVATION GAP INVENTORY (2026-05-08)

### Both converters (AlphaToLk and LkToAlpha) drop these chunks:

| Source Chunk | Where | Data Survives? | Notes |
|---|---|---|---|
| **MFBO** | ADT root top-level | NO | Flight bounding planes are read by inspect/harvest but the converters never carry MFBO to output |
| **MCCV** | MCNK sub-chunk | NO | Vertex colors are read by inspect/harvest (`mccv_rgb` signal) but converters drop them |
| **MCLV** | MCNK sub-chunk | NO | Baked vertex lighting is read by harvest (`mclv_lighting_bytes`) but converters drop it |
| **AreaId** | MCNK header field | NO | Always defaults to 0 in output. AreaIdMapper exists but is not wired |
| **MCSH** | MCNK sub-chunk (Alpha) | PARTIAL | Alpha MCSH is 1-bit-per-pixel (512 bytes); LK MCSH is 1-byte-per-pixel (4096 bytes). AlphaToLk upsamples; LkToAlpha downsamples. Format is correct but re-encoded, not byte-identical |
| **MCAL** | MCNK sub-chunk | PARTIAL | Alpha uses 4-bit packed (2048 bytes); LK uses big-alpha or compressed. Conversion re-encodes correctly but not byte-identical |
| **MCVT** | MCNK sub-chunk | PARTIAL | Alpha interleaves outer/inner; LK uses sequential 145 floats. Layout is reorganized, not byte-identical |
| **MCNR** | MCNK sub-chunk | PARTIAL | Alpha uses Z-up component order; LK uses Y-up. Components are swapped, not byte-identical |

### LkToAlpha additionally drops these chunks:

| Source Chunk | Where | Data Survives? | Notes |
|---|---|---|---|
| **MCRF** | MCNK sub-chunk | NO | Per-chunk doodad/WMO reference indices are read from LK but never written to Alpha MCNK |
| **MH2O fidelity** | ADT root top-level | PARTIAL | Only min/max height, 9x9 vertex heights, and 8x8 tile flags survive. Lost: fishableMask, deepMask, liquidTypeId beyond basic type, vertexFormat, depth data, UV coords |
| **MTXF** | tex0 top-level | NO | Texture animation/transformation flags are never carried |
| **MAMP** | tex0 top-level | NO | Terrain texture sizing parameter (never needed for Alpha conversion which doesn't produce split ADTs) |
| **MMID/MWID** | obj0 top-level | NO | Name offset indexes. The string name data (MMDX/MWMO) IS carried, but the offset index tables are rebuilt from scratch |

### AlphaToLk additionally drops these chunks:

| Source Chunk | Where | Data Survives? | Notes |
|---|---|---|---|
| **MCLQ (Alpha source)** | MCNK sub-chunk | PARTIAL | Alpha liquid heights/flags are carried into MH2O format in LK output, but Alpha MCLQ's per-vertex liquid data is simplified |
| **MDNM/MONM (Alpha source)** | WDT top-level | PARTIAL | Model and WMO names ARE resolved and carried into LK MMDX/MMID/MWMO/MWID, but the original Alpha string table offsets are lost |

### Priority order for fixes:

1. **AreaId** — already have AreaIdMapper, just needs wiring (medium scope, high value)
2. **MCRF** — per-chunk doodad/WMO refs matter for placement fidelity (medium scope)
3. **MFBO** — read by harvest already, just needs pass-through in converters (small scope)
4. **MCCV/MCLV** — read by harvest already, need writer support in both converters (medium scope)
5. **MH2O full fidelity** — significant rewrite of liquid path, carry more fields (large scope)
6. **MTXF** — small chunk, straightforward carry (small scope)
7. **MAMP** — irrelevant for Alpha conversion but matters for split-ADT LK output (small scope)

## ALPHA WDT FORMAT — REVERTED COMMITS AND OPEN ISSUES (2026-05-10)

### Reverted commits (maps crashing 0.5.3 client)
Both commits after `47cbb435` were reverted because they broke 0.5.3 client map rendering:
- **`8bcb7045`** "fix: write MCRF as raw uint32 data (not FourCC-wrapped)" — REVERTED. Ghidra confirms MCRF is raw uint32 in the 0.5.3 client, but writing it as raw broke the client. The MCRF FourCC-wrapper path currently works; the raw-uint32 Ghidra finding may apply to a different version or need more investigation.
- **`d52bda9b`** "fix: alpha MCNR normal encoding and liquid flag regeneration" — REVERTED. Two changes in one commit:
  1. **MCNR encoding changed from `(X, Z, Y)` to `(-Y, Z, -X)`**: Ghidra confirms the client decodes MCNR as `(-Y, Z, -X)`, but changing the writer to this format broke rendering. The `(X, Z, Y)` format (which matches the LK convention) currently produces working output. This may indicate the alpha client also reads MCNR in `(X, Z, Y)` order from some code path, or there is a coordinate system mismatch elsewhere that the old format happened to mask.
  2. **Liquid flags changed from `McnkFlags & 0x3C` to `ClassifyAlphaLiquidType` switch**: The new logic produced flags like `0x04` (water only) instead of `0x3C` (all liquid bits), which may have changed liquid rendering behavior.

### Current working state (commit `47cbb435`)
- MCNR: written as `(X, Z, Y)` per byte — `byte[0]=X, byte[1]=Z, byte[2]=Y`
- MCRF: wrapped in FourCC chunk (`MCRF` + size + data)
- Liquid flags: `McnkFlags & 0x3C` (preserves original 4.x flags with only liquid bits)
- MDDF/MODF position: `file = (MapOrigin - world.Y, world.Z, MapOrigin - world.X)` — Ghidra-verified
- MDDF/MODF rotation: `file = (world.pitch_deg, world.yaw_deg - 180, world.roll_deg)` — Ghidra-verified, yaw offset = π
- MODF bounds: `file.t = (MapOrigin - world.min.Y, world.max.Z, MapOrigin - world.min.X)`, `file.b = (MapOrigin - world.max.Y, world.min.Z, MapOrigin - world.max.X)` — Ghidra-verified
- Maps load and render in 0.5.3 client without crashes

### Open issues — OBJECTS AND PLACEMENTS STILL BROKEN
- **Object placement coordinates may be in the wrong coordinate space.** The Ghidra-verified transforms are applied to `(world.X, world.Y, world.Z)` where X=north, Y=west, Z=up, and the conversion to file placement space uses `MapOrigin` subtraction. But the source 4.x MODF/MDDF data might use a different axis convention than what our code assumes for `Position.X/Y/Z`.
- **MCRF population logic** (from `47cbb435`) tests containment and bounds overlap but may not correctly map 4.x placement indices to alpha tile indices.
- **MCNR format needs definitive resolution.** Ghidra says `(-Y, Z, -X)` but `(X, Z, Y)` works. Need to determine why — possibly the alpha client has a different code path, or the wow-viewer's coordinate conventions make `(X, Z, Y)` produce identical results for the common case of mostly-upward-facing normals.

### Ghidra-verified MDDF/MODF details (still correct, not dependent on reverted changes)
- SMDoodadDef size = 0x24 (36 bytes): nameId(0x00), uniqueId(0x04), pos(0x08), rot(0x14), scale(0x20), flags(0x22)
- SMMapObjDef size = 0x40 (64 bytes): nameId(0x00), uniqueId(0x04), pos(0x08), rot(0x14), extents.t(0x20), extents.b(0x2C), flags(0x38), doodadSet(0x3A), nameSet(0x3C), scale(0x3E)
- Client position transform: `world = (MapOrigin - file.Z, MapOrigin - file.X, file.Y)`
- Client rotation transform: `world_rot = (file_rot.Z × π/180, file_rot.X × π/180, file_rot.Y × π/180 + π)`
- Client bounds transform: `world_min = (-extents.t.Z + MapOrigin, -extents.t.X + MapOrigin, extents.b.Y)`, `world_max = (-extents.b.Z + MapOrigin, -extents.b.X + MapOrigin, extents.t.Y)`
- Scale: MDDF scale = uint16 × (1/1024), MODF scale = uint16 × (1/1024) at offset 0x3E (alpha padding, not scaling)
- MCRF in `CreateRefs` passed `(17066.666, 17066.666, 0.0)` as position offset to `CreateDoodadDef`/`CreateMapObjDef`

## NOT YET
- Explicit Alpha 0.6.0 split ADT validation via `AdtProfile060070Baseline`
- Full extraction run on 6 staged game clients (800-1500+ shards) via `harvest-map-mpq` into `wow-viewer/output/datasets/`
- Production training run (300 epochs)
- Model evaluation on held-out tiles
- DBC/DB2 metadata enrichment (WorldSafeLocs, AreaTable, GroundEffects, LiquidType)
- MCRF per-chunk reference arrays
- MODF doodadSet/nameSet resolution
- PM4 masks for development map build (4.0.0.12304 loose files)
- Development map extraction pipeline (wow-viewer/test_data/original_development)

## BUGS FIXED IN ALPHAtoLK VALIDATION (2026-05-08)
1. **ChunkedFileReader crash on monolithic WDTs** — replaced with `AlphaWdtReader.ReadExistingTiles()`
2. **MHDR/MCIN empty payload** — wrote declared-size chunk headers with 0 data bytes; fixed by writing pre-allocated zero arrays
3. **MPHD size mismatch** — wrote 9 uint32s (36 bytes) but declared 32 bytes; fixed by removing extra `Write(0u)`
4. **MAIN index formula** — `tileX * 64 + tileY` was wrong; fixed to `tileY * 64 + tileX` (row-major with y as row). This caused 420/755 Azeroth tiles to fail before the fix.

## VALIDATED: LKTOALPHA REAL-DATA MDXVIEWER RENDER (2026-05-09)
- 4.0.0 Azeroth (839 tiles) → Alpha WDT → MdxViewer with 0.5.3 staged client
- Terrain renders: up to 11 tiles (2816 chunks) simultaneously
- WMOs resolved via Alpha `.wmo.MPQ` wrappers (v14 format)
- MCLQ liquid data parsed and rendered
- WDL parsed correctly: 839/4096 tiles with MARE data
- ExitCode=0, capture saved

## BUGS FIXED IN LKTOALPHA ALPHAWDTWRITER STRUCTURAL REPAIR (2026-05-09)
1. **MAIN grid order confirmed** — Ghidra shows 0.5.3 `CMap::PrepareArea(x,y)` indexes `areaInfo` as `tileY * 64 + tileX` after `CMap::LoadWdt` reads raw `MAIN`; `AlphaWdtWriter`, `AlphaWdtReader`, and focused tests use that row-major contract.
2. **Empty MCNK omission** — Writer skipped MCNKs with all-zero heights via `BuildEmptyMcnk`, but legacy `McnkAlpha` always reads 256 MCNK entries. Removed `BuildEmptyMcnk` — all 256 MCNKs are now emitted with full subchunk structure.
3. **MCRF conditionally emitted** — Writer only emitted MCRF when `mcrfRaw.Length > 0`. Legacy `McnkAlpha` unconditionally reads MCRF at offset 0x24. Fixed by always wrapping MCRF (even with 0-byte payload).
4. **MDDF/MODF conditionally emitted** — 0.5.3 `CMapArea::Create` unconditionally asserts both embedded tile placement chunks. Fixed `AlphaWdtWriter` to emit empty `MDDF` and `MODF` chunk headers when a tile has no placements of that type.
5. **Placement orientation and MODF bounds** — Ghidra confirms Alpha file positions are `(origin - worldY, worldZ, origin - worldX)` and file rotations are consumed as `(fileZ, fileX, fileY + 180deg)` in client axes. `AlphaWdtWriter` matched this contract; `LkAdtWriter` bounds axes were corrected and covered by `LkAdtWriter_RoundTripsModfBoundsWithReaderOrientation`.
6. **MCLY/MCAL offsets conditionally zeroed** — When `mclyWhole.Length > 0` was false, offset was set to 0, but legacy reader expects non-zero offsets. Fixed by always populating cursor-based offsets.
7. **LkAdtWriter chunk ID byte order** — Used `Encoding.ASCII.GetBytes()` which writes forward-order FourCC. Changed to `FourCC.FromString().ToFileBytes()` which writes the reversed FourCC expected by `wow-viewer`'s I/O boundary convention. This caused `MapFileSummaryReader` to fail detecting ADT family.

## BUGS FIXED IN LKTOALPHA ROUND-TRIP VALIDATION (2026-05-08)
1. **Alpha placeholder chunk payloads** — `AlphaWdtWriter` declared `MPHD`/`MHDR` payload sizes but wrote zero bytes, corrupting all later offsets. Fixed by writing explicit zero-filled payload buffers.
2. **Reverse Alpha terrain heights** — `MCVT` heights were written relative to tile base instead of chunk base. Fixed to match Alpha chunk semantics.
3. **Alpha MCNK offset frame mismatch** — emitted subchunk offsets did not line up with `AlphaWdtReader`. Fixed so written tiles parse as structurally valid Alpha WDTs.
4. **Tile alpha vs chunk alpha mismatch** — the shared `256x256` alpha contract was being reused as if it were already chunk-local `MCAL`. Fixed by resampling during write.
5. **Liquid parity gap** — LK input `MH2O` was dropped before conversion and Alpha return conversion rebuilt only flags. Fixed by carrying structured liquid through the shared models, preserving `MCLQ` 81-sample heights, and emitting real `MH2O` again on the LK writer path.
6. **Heightmap validation gap** — `LkToAlphaConverter` lacked `AlphaWdtReader`'s `FillHeightmapGaps` logic, causing missing heights at chunk borders to fail roundtrip validation. Fixed by porting the logic to the converter pipeline.
7. **Complete Alpha Mask Loss** — `LkAdtWriter` was dropping alpha chunks entirely because `AlphaToLkConverter` extracted data from transposed X/Y indices `[cx, cy, l]` instead of `AlphaWdtReader`'s populated `[cy, cx, l]`. Fixed by correcting array accesses in `AlphaToLkConverter` to perfectly align with `AlphaTileData` layout.
8. **Binary Chunk Tag Corruption** — `LkAdtWriter` used endian-swapping `FourCC.ToFileBytes()` for `MCNK` subchunks (`MCVT`, etc.), breaking binary validation. Fixed to use `ASCII.GetBytes`.

## FOCUSED VALIDATION NOTE (2026-05-08)
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter LkToAlphaRoundTripTests` passes with `2/2` tests.
- Do not describe this as full suite closure: the broader `WowViewer.Core.Tests` run still hits missing `wow-viewer/test_data/development` fixtures and one unrelated pre-existing invalid-data test.

## BUGS FIXED IN NPZ SHARD VALIDATION (2026-05-08)
1. **OverflowException in MCNK sub-chunk locators** — `LocateMcvtDataOffset`, `LocateMcnrDataOffset`, `LocateMccvDataOffset`, and `TryReadSplitMcnkSubchunkPayload` all used `checked((int)header.Size)` casts that threw `OverflowException` when encountering MCNK sub-chunk headers with `uint` sizes > `int.MaxValue`. Fixed by converting all three locators and `TryReadSplitMcnkSubchunkPayload` to `long` arithmetic.
2. **Garbled raw-chunk FourCCs in AdtRawChunkBlobCollector** — `CollectRawMcnkSubchunks` lacked FourCC validation, MCAL/MCSH header-size overrides, and `long` arithmetic. When scanning past valid MCNK sub-chunks, it fabricated 274 garbage "raw chunks" with binary FourCC IDs. Fixed by adding `IsValidAdtFourCC` validation, MCAL/MCSH consumed-size overrides from MCNK header fields at offsets 0x28/0x30, and `long` arithmetic throughout.
3. **Stale AdtTextureFile constructor call** — `WowViewer.Tool.Converter/Program.cs` called `AdtTextureFile(...)` without the `mampValue` parameter added in a prior refactor. Fixed by passing `null`.

## ALPHA RESOLUTION FIX (2026-05-08)
- **NPZ alpha naming mismatch**: `mcal_alpha_pack_256` key stored 1024×1024×4 data (full per-pixel resolution at 64 px/chunk × 16 chunks) but the "256" suffix implied 256×256 resolution. Visualization scripts interpreted it as 256×256, showing a stretched 1024×1024 image with visible 64-pixel chunk-boundary seams.
- **Fix**: Added `McalAlphaPack` (1024×1024 full resolution) as a new property alongside `McalAlphaPack256` (256×256 bilinearly downsampled). `NpzTileSerializer` now writes both `mcal_alpha_pack` and `mcal_alpha_pack_256`. The 256 version is correctly downsampled for minimap-resolution compositing; the full 1024 version preserves per-pixel alpha for training. Updated `AdtTensorPackBuilder`, `AlphaTensorPackBuilder`, `TerrainTileTensorPack.ToTileLoadResult` (now uses `McalAlphaPack` for 64px-per-chunk slice indexing), and harvest tool's synthetic minimap compositor (uses `McalAlphaPack256` for 256×256 compositing, which is correct).
- **Grid artifacts in normals are expected**: MCVT/MCNR store 145 vertices per chunk with shared boundary vertices between neighbors. When rendered as a plain 257×257 image, chunk-boundary edges create visible grid patterns. This is not a bug — it's how the WoW ADT format works. The raw data is correct.

## VALIDATED SHARD STRUCTURE (2026-05-08)
- Azeroth 32,32 from 3.3.5 client via `extract-unified`: 13 typed signals (height_257, mcnr, mcly, mcal, mcsh, hole_mask, object masks, shadow residual, minimap, etc.) plus structurally valid `metadata.json`.
- Raw-chunk collection correctly produces 0 garbled entries (was 274 before FourCC validation fix) for standard LK root MCNKs where all sub-chunks are in the consumed set.
- Build: `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — 0 errors, 12 warnings (nullable, CA2014).

## WORKFLOW CORRECTION (2026-05-08)
- Do not route full multi-client dataset generation through `WowViewer.Tool.Converter dataset-scan` / `dataset-audit` / `dataset-curate` / `dataset-build-cache`.
- Those commands remain useful as legacy manifest/audit helpers, but they are not the canonical full-signal shard builder and can miss newer harvest/tensor-pack coverage and metrics.
- Use `WowViewer.Tool.Harvest harvest-map-mpq` for staged archive-backed clients and `harvest-map` for loose on-disk maps.
- Default real dataset outputs belong under `wow-viewer/output/datasets/`, not repo-root `output/tmp/`.
