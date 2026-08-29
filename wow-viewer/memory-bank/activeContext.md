# Active Context — wow-viewer

Last updated: 2026-08-29

**Spec 191 lane (2026-08-29, checkpoint 40).** Procedural Garden Museum Map Generator & Dense Calibration Corpus — COMPLETE:
1. **Procedural 3D Mesh Terrain & Pedestal Sculpting Hooked Up**:
   - `ProceduralTerrainSculptor`, `ProceduralTexturePainter`, `AdaptiveLayoutPacker`, `SemanticAssetClassifier`, `IGenerativeMapSurface` integrated cleanly in `WowViewer.Core.IO.Procedural`.
   - `RosettaTilesetGenerator.BuildTileAdt` generates authentic continuous MCVT heightmaps and raised podium pedestals (`CreateChunkHeights`), with smooth bevel transitions and gradient slope constraints.
   - `RosettaTilesetGenerator.BuildMap` samples terrain height (`groundZ = options.PedestalHeightMeters`) so 3D exhibit models sit flush on the pedestal platforms.
2. **4-Layer Alpha Texture Splatting & Clean Center Plaza Floor**:
   - `BuildTileCheckersCanvas` renders an outer decorative checkerboard border frame (Layer 2) and explicitly clears the center exhibit plaza ($R \le 0.30 \times \text{CellSize}$) with value `0`, ensuring the 3D model silhouette stands out against clean neutral ground.
   - `BuildTileAlphaCanvas` generates anti-aliased grid cell boundary lines (Layer 1).
   - Minimap rendering in `RosettaMinimapPainter` upgraded with shaded pedestals, checker perimeter rings, and clean floor plaques.
3. **Minimap Directory & File Output Cleanup**:
   - Streamlined output paths so Alpha 0.5.3 minimap BLPs and TRS files write exclusively to canonical `Textures\Minimap\{map}\` and post-Alpha to `World\Minimaps\{map}\`, eliminating redundant directory dumping.
4. **Validation & Verification**:
   - All 72 Rosetta unit tests pass green (100%). Solution builds cleanly with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 36).** Minimap TRS / BLP Multi-Directory Emission & Map ID / Port Command Reporting — COMPLETE:
1. Enhanced `RosettaMinimapPainter.GenerateMinimapTrs` and `WriteMinimapTrs` to generate authentic Alpha TRS format with relative entries under `dir: {map}` and write `minimap.trs` / `md5translate.trs` globally and into per-map directories (`World\Maps\{map.MapName}\`).
2. Expanded `rosetta-generate` minimap BLP writer to emit all 4 coordinate / padding variants (`map{Y:D2}_{X:D2}.blp`, `map{X:D2}_{Y:D2}.blp`, `map{Y}_{X}.blp`, `map{X}_{Y}.blp`) across all standard directories (`Textures\Minimap\`, `World\Textures\Minimap\`, `World\Minimaps\`, `World\Maps\{map}\`, and `World\Maps\{map}\Minimap`).
3. Added `InitializeMinimapSupport()` to overlay attachment in `ViewerApp.cs` and expanded search paths to include `OverlayRoots` and `LooseRoots`, resolving minimaps dynamically in viewer.
4. Upgraded `rosetta-generate` console summary to display prominent Map IDs, map names, tile bounds, first exhibit spawn world coordinates (`X`, `Y`, `Z`), and ready-to-copy in-game teleport commands (`.worldport <mapId> <x> <y> <z>` and `.go xyz <x> <y> <z> <mapId>`).
5. All 74 Rosetta and Core IO unit tests pass green (100%). Solution builds cleanly with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 35).** Loose File DBC Resolution & Valid Map ID Display in Viewer — COMPLETE:
1. Fixed `MpqDBCProvider` to accept `IDataSource` and check loose files (`_dataSource.ReadFile`) across all table candidate paths (`DBFilesClient\Map.dbc`, `DBC\Map.dbc`, etc.) *before* falling back to raw MPQs.
2. Added `"DBFilesClient"`, `"DBC"`, `"dbfilesclient"`, `"dbc"` to `IndexedLooseExtensions` / `dataDirs` in `MpqDataSource.cs` so loose DBC files in game folders and attached loose overlays are scanned, indexed, and accessible.
3. Updated `ViewerApp.cs` to pass `_dataSource` into `MpqDBCProvider(mpqDs.ArchiveReader, _dataSource)` on data source loading and overlay attachment, ensuring `MapDiscoveryService` loads the patched `Map.dbc` and resolves custom maps with their authentic 500+ Map IDs (`[500] Rosetta Exhibit (Development_M200)`) instead of falling back to synthetic `[custom]` entries.
4. Updated `ArchiveReaderDbcProvider` in `WowViewer.Core.IO.Dbc` with optional `Func<string, byte[]?>? looseFileReader` delegate for unified offline tool and test support.
5. All 74 Rosetta and Core IO unit tests pass green (100%). Solution builds cleanly with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 34).** DBC Real Client Patching via DBCD & WoWDBDefs — COMPLETE:
1. Updated `RosettaDbcGenerator` in `WowViewer.Core.IO.Dbc` with `PatchAndSaveClientDbcs` to load existing client `Map.dbc` and `AreaTable.dbc` from client MPQ archives / directories using `DBCD` and `WoWDBDefs` definition schemas (`Map.dbd`, `AreaTable.dbd`), preserving all real client records and appending new Rosetta exhibit continents and designkit exhibit zones.
2. Added `InferClientBuild` to automatically detect client builds from paths and format flags against known `Map.dbd` BUILD ranges (e.g. `3.3.5.12340`, `0.5.3.3368`, `1.12.1.5875`), and `TryFindDefinitionsDirectory` to locate `WoWDBDefs/definitions/`.
3. Implemented safe column mapping in `PatchAndSaveClientDbcs` supporting both scalar fields and multi-locale `string[]` arrays with `_mask` handling across Alpha, Vanilla, TBC, and LK client schemas.
4. Integrated `PatchAndSaveClientDbcs` with `ArchiveReaderDbcProvider` in `rosetta-generate` CLI command in `WowViewer.Tool.Inspect` (`Program.cs`) with automatic fallback to standalone DBC creation if client DBCs are absent.
5. Authored comprehensive unit tests in [`RosettaTilesetGeneratorTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaTilesetGeneratorTests.cs) verifying build inference, definition discovery, and real 3.3.5 DBC patching + DBCD round-trip loading. All 71 Rosetta unit tests pass green (100%). Solution builds cleanly with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 33).** Phase 4 Companion ADT Synthesizer (US4) — COMPLETE:
1. Created `RosettaCompanionAdtSynthesizer` in `WowViewer.Core.IO.Maps` to scan PM4 file directories and identify orphan PM4 tiles lacking companion `.adt` or `_obj0.adt` files.
2. Implemented minimal compliant companion ADT synthesis (`BlankAdtFactory` / `LkAdtWriter`) generating standard valid ADTs with flat terrain and authentic headers (`MVER`, `MHDR`, `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`, `MCNK`).
3. Added SHA256 cryptographic provenance reporting (`RosettaCompanionProvenanceReport`) recording source PM4 files, output paths, timestamps, options, and content hashes to distinguish synthetic data from authentic game data (FR-009, FR-010, SC-003).
4. Implemented safe overwrite protection skipping existing companions by default and `--overwrite` flag.
5. Added CLI command `rosetta-synthesize-companions` in `WowViewer.Tool.Inspect` (`Program.cs`).
6. Authored 6 comprehensive unit tests in [`RosettaCompanionAdtSynthesizerTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaCompanionAdtSynthesizerTests.cs). All 69 Rosetta tests pass green (100%).

**Spec 190 lane (2026-08-28, checkpoint 32).** Phase 3 Deterministic PM4 Lookup Engine (US3):
1. Created `RosettaPm4LookupEngine` in `WowViewer.Core.PM4.Matching` for pure deterministic PM4 object identification against `RosettaReferenceLibrary` without ML/LLMs.
2. Implemented tri-state classification (`Identified`, `Ambiguous`, `NoReference`, `Ineligible`) with score floor ($0.45$), ambiguity window ($0.03$), and bounding tolerance pruning (`FindCandidatesByBounds`).
3. Added granular signal evidence evaluation (`AspectRatio`, `MajorSpan`, `Volume`, `Footprint`, `TypeFlags`) surfacing exact reasons for match agreement and discrepancy.
4. Added `CompareWithLegacyScorer` and `Pm4ReconciliationInputAdapter.BuildRosettaCorpusReferences` connecting the global Rosetta reference library directly into the Spec 176 reconciliation pipeline.
5. Added `rosetta-pm4-match` CLI command in `WowViewer.Tool.Inspect` (`Program.cs`) producing detailed console diagnostics and structured JSON match reports.
6. Authored 7 comprehensive unit tests in [`RosettaPm4LookupEngineTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaPm4LookupEngineTests.cs). All 63 Rosetta unit tests and all 82 Editor tests pass green (100%).

**Spec 190 lane (2026-08-28, checkpoint 31).** Alpha 0.5.3 Native Client Ergonomics, First-Class DBCs, WDL Mesh, Map Splitting & +20Z Exhibits:
1. Created `RosettaDbcGenerator` in `WowViewer.Core.IO.Dbc` to generate authentic binary `Map.dbc` (5 fields) and `AreaTable.dbc` (14 fields) for Alpha 0.5.3, registering Rosetta maps as first-class outdoor continents (Map ID 500+) and named designkit exhibit zones (Area ID 5000+).
2. Added automatic `.wdl` distant low-resolution terrain mesh generation (`WdlWriter.Build`) alongside `.wdt` files.
3. Added `GenerateMinimapTrs` and `WriteMinimapTrs` to `RosettaMinimapPainter` for `minimap.trs` / `md5translate.trs` mapping with `Azeroth` aliases.
4. Added kind-based map splitting (`SplitAssetKinds`), partitioning models (`.mdx`/`.m2`) into `{map}_MDX` and world models (`.wmo`) into `{map}_WMO`, with an 800-tile map budget ceiling to prevent client engine memory exhaustion.
5. Implemented bounding-box centering offset (`(X, Y) -= boundsCenter`) and $+20\text{Z}$ elevation ($Z = \text{groundZ} + \max(0, -\text{bounds.Min.Z}) + 20\text{m}$), placing models comfortably floating in the air above flat walkable terrain ($Z = 0$) with zero ground clipping or saw-tooth ridge trapping.
6. All 56 Rosetta unit tests pass green (100%).

**Spec 190 lane (2026-08-28, checkpoint 30).** Phase 2 Reference Library Builder (US2):
1. Created `RosettaReferenceLibrary` and `RosettaReferenceAsset` data model with full bounding, span, volume, footprint, aspect ratio, subpart bounds, and signal dictionary properties, with 100% interoperability with `Pm4AssetMatchScorer` via `ToAssetReferenceSignalRecord()`.
2. Created `RosettaCorpusReader` to decode synthetic Rosetta placements and geometry from `rosetta-manifest.json`, in-memory `RosettaGenerationResult`, and Zarr datastores (`RosettaObjectLibrary`).
3. Created `RosettaReferenceLibrarySelfTest` enforcing the $\ge 99.0\%$ Top-1 identification accuracy requirement with exact and perturbed (jittered) bounding box tests.
4. Added CLI commands `rosetta-build-library` and `rosetta-library-selftest`, plus `--emit-library` and `--library-output` flags in `rosetta-generate`.
5. Created custom `Vector3JsonConverter` and `Vector2JsonConverter` for clean JSON serialization and round-tripping.
6. Added 7 unit tests in `RosettaReferenceLibraryTests.cs`. All 50 Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 29).** Alpha WDT Row-Major Indexing & Visual Calibration:
1. Fixed transposed tile indexing bug in [`AlphaTerrainAdapter.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs): `TileExists` and `LoadTileWithPlacements` now index `_adtOffsets` as row-major `tileY * 64 + tileX` instead of column-major `tileX * 64 + tileY`. This was the root cause of objects being placed over mismatched terrain tiles/labels in Alpha WDT maps when $tileX \ne tileY$.
2. Added visual calibration tools in [`RosettaAlphaPainter.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/RosettaAlphaPainter.cs) with `DrawCircleOutline` and `DrawBullseyePattern` (concentric rings from 30m to 240m, full crosshair axes, and cardinal direction indicators: `NORTH (-Y)`, `SOUTH (+Y)`, `WEST (-X)`, `EAST (+X)`).
3. Connected calibration bullseye generation to [`RosettaTilesetGenerator.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/RosettaTilesetGenerator.cs) for empty tiles and test patterns.
4. All 43 focused Rosetta unit tests pass green. Full solution builds with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 28).** Cross-Era Extension Resolution, Datastore Diff Engine & UI Integration:
1. Implemented seamless 3-way cross-era model extension shifting (`.mdx` $\leftrightarrow$ `.mdl` $\leftrightarrow$ `.m2`) in `WorldAssetManager`, `WmoRenderer`, and `ViewerApp`, allowing maps from any era (such as Rosetta 0.5.3 or Alpha WMOs) to resolve models when loaded over newer clients (1.12.1 / 3.3.5) and vice-versa.
2. Built `RosettaBuildMetadata` and `RosettaBuildDiff` with `ComputeBuildDiff` in `RosettaObjectLibrary` to compare any two client builds directly from the Zarr datastore without re-processing.
3. Added `rosetta-datastore-diff` CLI command in `WowViewer.Tool.Inspect` (`Program.cs`).
4. Added "Load from Rosetta Datastore..." menu item and interactive modal in `ViewerApp` with Data Version, Map Name, Base Game Version dropdowns, and live cross-build diff statistics.
5. Documented Phased Maps and Rosetta Datastore in `USERGUIDE.md` and `README.md`.
6. All 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 27).** Minimap & Placed Object Coordinate Alignment:
1. Resolved minimap tile loading mismatch by generating dual-convention minimaps (`map{tileY:D2}_{tileX:D2}.blp` for standard viewer queries and `map{tileX:D2}_{tileY:D2}.blp`) across all four standard directories (`Textures/Minimap/{mapName}`, `Textures/Minimap/{mapName.ToLowerInvariant()}`, `World/Minimaps/{mapName}`, `World/Minimaps/{mapName.ToLowerInvariant()}`).
2. Fixed Alpha WDT streaming converter invocation in `Program.cs` which previously inverted `(tileX, tileY)` coordinate arguments.
3. Enforced strict transitive weak ordering in `kitAssets.Sort` comparator using floor bucketing.
4. Added path separator normalization to `SanitizeLabel` and `RosettaMinimapPainter` label formatting, and made `IndexOfOrAdd` case-insensitive.
5. All 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 26).** Deduplication Normalization & Minimap Path/Font Optimization:
1. Fixed duplicate asset enumeration by normalizing all paths to backslashes before `Distinct(StringComparer.OrdinalIgnoreCase)` and adding deduplication to `RosettaTilesetGenerator.Generate`.
2. Corrected minimap tile naming from `map{tileY}_{tileX}.blp` to canonical `map{tileX:D2}_{tileY:D2}.blp`.
3. Emitted minimap BLPs to both `Textures/Minimap/{mapName}/` and `World/Minimaps/{mapName}/` for cross-era client and viewer compatibility.
4. Built a 3×5 bitmap font for minimap plaques with dynamic capacity calculation to fit 30 characters per cell without clipping.
5. 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 25).** High-Legibility MCAL & Minimap Text Rendering:
1. Switched font pixel metrics in `AppendClass` from coarse 4.16m MCCV lattice sizing to 2 MCAL texels (~1.04m), boosting capacity from 5 to 21+ characters per line across 2-3 line label bands.
2. Updated `SanitizeLabel` to preserve natural lowercase and uppercase casing so handwriting glyphs render with proper ascenders and descenders.
3. Updated `WrapLabel` with smart path separator wrapping and middle-elision with `...`.
4. Replaced blurry minimap downsampling with direct 5×7 glyph rendering on high-contrast parchment plaques (`ColorPlaque = (248, 245, 238)`, `ColorInk = (15, 15, 20)`) over warm sand terrain.
5. 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 24).** Unified Multi-Version Zarr Datastore & Interchange Engine:
1. Created `RosettaDatastoreWriter` implementing a versioned Zarr v3 datastore with global content-addressed asset deduplication (`global_assets/catalog.parquet`).
2. Deduplicates assets across client builds by deterministic SHA1 hash (`objlib_<hash>`), storing shared assets once with incremented reference counts.
3. Slices heights, MCAL alpha, checkers alpha, and raw 24-bit RGB minimaps into chunked Zarr v3 arrays.
4. Built `RosettaObjectLibrary` for high-performance C# O(1) asset lookups, spatial bounding queries, and on-demand chunk loading.
5. Integrated `--datastore` / `--emit-zarr` CLI flags in `rosetta-generate`, plus `rosetta-datastore-info` and `rosetta-datastore-query` commands.
6. 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 23).** Painted terrain grid lines and cell perimeter demarcation:
1. Added `DrawRectOutline` and `DrawLine` to `RosettaAlphaPainter` and updated `BuildTileAlphaCanvas` to paint 1.2m perimeter borders and 1.0m object/label divider lines onto the 1024×1024 MCAL canvas.
2. Updated `FillRect` with `Math.Max` for seamless non-destructive alpha blending across overlapping lines and text.
3. Connected `options.PaintCellBorders` through `BuildTileAdt` in both LK and Alpha generation pipelines.
4. 39 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 22).** Streaming tile generation and writing to eliminate out-of-memory crashes:
1. Added streaming `AlphaWdtWriter.Write` directly to `FileStream`, writing tiles one-by-one to disk and patching the 64 KB `MAIN` table and `MPHD` offsets at the end.
2. Made 1024×1024 text and checkers canvas generation on-demand via `BuildTileAlphaCanvas` and `BuildTileCheckersCanvas`, avoiding gigabytes of canvas allocations across all planned tiles in memory.
3. Updated `RosettaMinimapPainter` to render downsampled text on-demand.
4. Cleanly handled `--overwrite` across all output files (Alpha WDT, LK ADT/WDT/WDL, and minimap BLPs).
5. Verified on 0.5.3 client: 1,720 tiles, 11,663 placements, 920 MB monolithic WDT written smoothly without OOM.
6. 38 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 21).** Diagnostic `checkers.blp` texturing on indented terrain floors:
1. Painted `tileset\generic\checkers.blp` under objects on the indented pedestal floor via `CheckersCanvas`, replicating authentic Blizzard model testing terrain.
2. Supported 3-layer MCNK structures in `BuildTileAdt` (Layer 0: Sand, Layer 1: Checkers pad under object, Layer 2: Handwriting ink label) with concatenated MCAL chunks and correct offsets.
3. Added `ResolveCheckersTexture` in `Program.cs` and `--checkers-texture` CLI option.
4. Rendered matching checkered pattern on pedestal plateaus in `RosettaMinimapPainter`.
5. 38 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 20).** Museum exhibit layout, handwriting font, compact cells & Westfall sand:
1. `rosetta-generate` defaults to `CellChunks = 4, LabelBandChunks = 1`, packing 16 cells per tile (133.33m cells with 100m object area) so small objects don't waste empty space.
2. Museum curation ordering: Models sorted by size (smallest -> largest) first, followed by WorldModels sorted by size.
3. Added 5x7 cursive handwriting font in `RosettaTextPainter` supporting uppercase ('A'-'Z'), lowercase ('a'-'z'), digits, and punctuation, baked into the 1024x1024 MCAL Layer 1 texture map.
4. Set default ground texture to `tileset\westfall\westfallsand.blp` and prioritized Westfall/Westwood sand auto-selection.
5. 37 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-27, checkpoint 19).** Negative pedestal height support (sunken object viewing dips):
1. Enabled negative pedestal heights (`PedestalHeightMeters = -10f` default; e.g. `-10.0`, `-20.0`) in `RosettaGeneratorOptions` and `Program.cs` to give sunken viewing dips with beveled ramps for WMOs whose origins sit below the mesh.
2. Updated `CreateChunkHeights` to compute sunken dips when `p.Height < 0` with bevel ramp scaling.
3. 34 focused Rosetta tests pass green, including `Generate_PedestalHeights_NegativeSunkenDipWithBevel`.

**Spec 190 lane (2026-08-27, checkpoint 18).** Alpha WDT MCAL/MCLY tile transform alignment:
1. `rosetta-generate --format alpha` in `Program.cs` now passes `(tile.TileY, tile.TileX)` to `AlphaWdtWriter.Build`, using the exact same coordinate transform as `LkWdtWriter.Write` and `RosettaMinimapPainter` (`map{tileY}_{tileX}.blp`).
2. No protected format readers or writers were modified; `AlphaWdtWriter.cs` and `AlphaWdtReader.cs` remain completely untouched.
3. MCAL/MCLY texture layers now align with the correct map tiles and minimap tiles.
4. 33 focused Rosetta tests pass green, all 17 LkToAlpha tests pass green. Real 0.5.3 client visual proof remains operator-owned.

**Spec 190 lane (2026-08-27, checkpoint 17).** Correction: Alpha Rosetta map bytes were not the proven defect; the retained fix is minimap-only.
1. `rosetta-generate --format alpha` writes the monolithic map bytes directly from `AlphaWdtWriter.Build(...)`; no Rosetta post-write MDDF/MODF coordinate mutation remains.
2. `RosettaMinimapPainter` now places model/WMO pins at the object-band center used by `RosettaTilesetGenerator`, not the whole label cell center.
3. No map renderer, terrain adapter, scene-loading, culling, protected writer, or generated map placement byte contract was changed.
4. Noggit3 evidence: its minimap is WDL horizon-derived and skips WDL object chunks; no accessible Noggit/Noggit-Red per-object minimap denylist or classifier was found in this pass.
5. 33 focused Rosetta tests pass, including a minimap marker pixel regression. Real 0.5.3 client visual proof remains operator-owned.

**Spec 190 lane (2026-08-27, checkpoint 16).** Alpha 0.5.3 candidate asset discovery in `rosetta-generate`:
1. Updated `RunRosettaGenerate` in `WowViewer.Tool.Inspect/Program.cs` to scan Alpha-era `.mdx.mpq`, `.mdl.mpq`, `.m2.mpq`, `.wmo.mpq`, and `.blp` single-file wrappers on disk with robust enumeration options.
2. Verified on Alpha client roots: era detection correctly classifies `.mdx`/`.mdl` models and discovers candidate assets.
3. 32 focused Rosetta tests pass green; solution builds clean.

**Spec 190 lane (2026-08-27, checkpoint 15).** Minimap generation and runtime tile coordinate readout:
1. `RosettaMinimapPainter` & `Blp2Writer`: 256×256 DXT1-compressed BLP2 minimap tiles written under `Textures/Minimap/{mapName}/map{tileY}_{tileX}.blp` with cell borders, pedestals, downsampled text labels, and model/WMO markers.
2. Bottom status bar now displays `Tile: {tileY:D2}_{tileX:D2}` matching ADT and minimap filenames whenever runtime scene is loaded.
3. 32 focused unit tests pass green; solution builds clean.

**Spec 190 lane (2026-08-27, checkpoint 14).** Museum-grade presentation and full-scale continent container support:
1. `RosettaAlphaPainter`: 1024×1024 MCAL text rasterization (0.52 m/texel) with antialiased quincunx sampling and 4-bit nibble slicing into 2-layer MCLY/MCAL chunks.
2. Museum pedestals (`MCVT`): raised 4m plateau with 12.5m beveled ramp in the terrain mesh under every object cell to eliminate base clipping.
3. Removed artificial 512-tile Alpha WDT limitation: unified 4096-tile capacity across both Alpha and LK generation.
4. CLI options: `--ink-texture`, `--pedestal-height`, `--pedestal-bevel`. 30 focused unit tests pass.

**Spec 190 lane (2026-08-26, checkpoint 5).** Stride-scatter reverted to CONTIGUOUS row-major tile
fill from the start tile (operator rejected scattered tiles; the full ~14k-asset corpus covers the
map naturally). New CI alignment proof: every placement decoded with the viewer's rule
`(M − rawY, M − rawX)` must land inside its tile's ChunkCorner bounds. Smoke: 400 assets → 74
contiguous tiles, 345 placements, 2,578 exclusions (WMO group files). 8 focused tests pass.

**Spec 190 lane (2026-08-26, checkpoint 4).** Output is a self-consistent standalone map under
`{output}/World/Maps/{mapName}/` — folder, tile prefix (`{map}_{tileY}_{tileX}.adt`), WDT, and WDL
all derive from one map name; mismatches are refused. Tiles spread across the whole 64×64 grid on a
stride grid (tile count measured first). Smoke-verified on `C:\WoW4-data\WoW-12025`: 14,029
candidates, 60 placements, 7 tiles at stride-25, WDT parses via `map inspect`. 7 focused tests pass.
The "objects horizontal vs ADTs vertical" report came from pre-fix outputs (old transposing naming).
Next: user runs the current binary into a fresh output and confirms objects sit on their tiles.

**Spec 190 lane (2026-08-26, checkpoint 3).** Second real run root cause: the viewer loads
`{map}_{tileY}_{tileX}.adt` (column first) while the generator wrote `{tileX}_{tileY}` — every tile
loaded transposed, so terrain showed as a strip and objects landed on nonexistent tiles. Fixed file
naming, occupied-tile parsing, WDT/WDL tuple convention (MAIN read at `tileX*64+tileY`), whole-map
row-major layout spread, and label shrink-to-fit (overflow was the "nonsense" text). 7 focused tests
pass. Next: user re-run into a FRESH output dir, confirm objects render + labels legible.

**Spec 190 lane (2026-08-26, checkpoint 2).** First real run exposed three defects, all fixed and
tested: (1) placements invisible — raw MDDF/MODF coords must be `rawX = tileY*T + u`,
`rawY = tileX*T + v` because the viewer decodes `(M − rawY, M − rawX)`; (2) MCCV text scrambled —
vertex layout is interleaved 9-8 rows, not row-major 17×17; painter now mirrors the mesh builder;
(3) `LkAdtWriter` MCRF declared size was `4+…` instead of `8+…`, desyncing chunks carrying refs.
Output is now a standalone map (`{map}.wdt` with MCCV flag, flat `{map}.wdl`, optional `--pm4-dir`
PM4 copies); existing files never overwritten. 5 focused tests pass; solution builds clean; 9
pre-existing unrelated Core.Tests failures remain. Spec:
[190-rosetta-calibration-corpus](../specs/190-rosetta-calibration-corpus/spec.md). Next: user re-runs
`rosetta-generate` into a fresh output dir and confirms objects render + labels legible (user-owned),
then US2 reference-library builder.

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

**Editor Platform lane (2026-08-25).** The editor dependency chain for Spec 176 is implemented
library-first and tested: `src/core/WowViewer.Core.Editor/` (166 plugin host, 167 bridge contracts +
operations-as-data, 168 session/undo/save policy, 173 asset-integrity gate), `AdtPlacementEditor` in
`WowViewer.Core.IO/Maps` (175 placement authoring + 176 Phase 2 name-table/ID mutation; ID allocation
now keeps an original-catalog high-water mark so substitute delete+add continues the chronology),
and `Pm4ReconciliationEngine` + `Pm4ReconciliationInputAdapter` in `WowViewer.Core.PM4/Reconciliation`
(176 Phases 1–2: real guide observations from `Pm4ObjectSegmentBuilder`, canonical world→placement
composition, MSUR `_0x1C` height signal validated against the segment Z span, bounds-containment
association with explicit Conflict for competing placements, scorer candidates over a labelled
Museum self-corpus). The **viewer shell runs the real pipeline**: the Editor workbench destination's
reconciliation panel parses the actual PM4 guide and Museum ADT, and Apply goes through
`ReconciliationApplyService` — source-hash staleness refusal, session write guards, JSON provenance
sidecar (FR-016), and undoable `ReconciliationApplyOperation` restoring prior output bytes. 79 focused
`WowViewer.Core.Editor.Tests` pass; the full solution builds 0 errors. Known gaps: proposals are not
yet drawn as in-scene overlays (Phase 3 step 3), the P1 cross-tile/cross-era transfer story is
untouched, and one pre-existing real-corpus test (`Pm4RegionObjectGrouperTests`, local development
corpus) fails independently of this lane. Real Museum/PM4 visual + independent-reader proof remains
user-owned; no runtime/visual claim is made from compilation. See Specs 166–168, 173, 175, 176 and
`progress.md`.

## Current handoff

**START HERE: read the new WMO admission counters on a real Stormwind flight, then pick the rule to
fix. The instrumentation landed; the measurement has not been taken.**

- **Next bounded action, user-owned:** fly Stormwind, open Utilities > Perf > **"WMO admission (this
  frame)"**, and record the numbers into
  [Spec 151 research.md](../specs/151-portal-game-mode-surface/research.md). The panel names the
  dominant rule. **Do not change an admission rule before that reading exists.**
- **Confirmed by the Stormwind capture (2048 frames), taken after the Spec 153 fixes:**
  `PrepareObjectPhase` max **283.4 → 2.5 ms** and gone from the hitch list; `SceneMaintenance` max
  **454.5 → 3.9 ms**; unaccounted median 0.02 / p99 0.11 ms; median frame 17.40 → 6.98 ms.
- **Owner: `WmoSubmission`** — p99 154.10 / max 161.3 ms with a median of 0.71 ms, and **all 592
  recent hitches** read `<- WmoSubmission` at 153–157 ms. Stormwind submits **all districts at once**:
  7512 visible groups, 80484 draw calls, 15852 doodad submissions. **This is an admission problem,
  not batching** — 80200 of 80484 calls are correctly batched.
- **Instrumentation shipped (source proof + 11 core tests, nothing measured).** `WmoAdmissionTally` /
  `WmoAdmissionStats` in `Core.Runtime/World/Visibility` count placements and groups by admitting
  rule; `CollectVisibleWmos` has a `ref` overload proven identical to the old one; `WmoRenderer`
  records the per-group rule; the Perf panel displays both layers. Four **source** findings drove the
  counter shape, all with magnitudes still unmeasured: portal culling **cannot reject** a group
  (the decision is unioned with raw frustum visibility); WMO placements are **never rejected by the
  frustum** (`IgnoreVisionConeCulling: true` also disables the only frustum branch), which is the
  shape of the old-Ironforge-past-fog symptom; group admission is evaluated **twice per placement per
  frame** (opaque + transparent); and the recorded 7512 is a **submission** count spanning both
  passes, not distinct groups. Detail in Spec 151 `research.md`.
- Still suspect group-to-group visibility data the client uses and this renderer does not read.
- **Spec 153 Phase 4 may be moot** — it was written against `SceneMaintenance` max 454.8 ms, which no
  longer reproduces (3.9 ms). Re-measure on a route that forces `_instancesDirty` before implementing.
- **Spec 153 Phase 5 step 2 still owed:** `DeferredAssetLoads` max 442.9 ms in Stormwind against a
  3.5 ms budget. The admission policy bounded the additive overshoot; the single-load residual needs
  decode off the render thread.
- **Released:** v0.5.2.1 (commit `975d0c79`, tag `v0.5.2.1`, branch `v0.5.3-dev`). Version bumped in
  both csproj files and `ViewerApp.ViewerProductName`; `wow-viewer/CHANGELOG.md` added;
  `docs/releases/v0.5.2.1.md` is what the release workflow publishes; both READMEs updated and the
  Stormwind WMO issue is documented as a known issue rather than left for users to discover.

---

**Spec 153 detail (shipped in v0.5.2.1, confirmed by capture).** Full numbers and both capture
tables are in [Spec 153 research.md](../specs/153-renderer-hitch-and-batching/research.md).

- **Defect A was `AudioRuntime.Update`, and it was never audio.** `RefreshEmitterDiagnosticsIfDue`
  rebuilt an `AudioTriggerDiagnostic` per resident emitter (5565) on a **wall-clock 250 ms** timer on
  the render thread — which is why the "every 47–50 frames" interval drifted with framerate — and it
  ran **whether or not anything displayed the result**. A second, movement-triggered copy: `RemoveTile`
  rebuilt the list synchronously on streaming eviction. Fixed by gating the rebuild on
  `NoteEmitterDiagnosticsObserved()` (only the audio panel calls it) and making eviction invalidate
  only. **Measured 283.4 → 2.5 ms max.** General lesson: *a diagnostics surface nothing is reading
  still pays full cost unless something gates it.*
- **Defect B was a hardcoded `return true`.** `PlanVisibleMdxPasses` gave the route planner a
  `requiresUnbatchedRender` predicate whose whole body was `return true`, so 100% of opaque MDX took
  the per-instance fallback while the batching machinery sat inert. Now consumes
  `IModelRenderer.RequiresUnbatchedWorldRender` — the contract the WMO doodad path already used.
  **0/312 → 526 batched / 3 unbatched**; `MdxOpaqueSubmission` p99 30.75 → 14.12 ms. GPU instancing
  stays off (`SupportsGpuInstancedOpaque` is still `false`); the win is begin-once/submit-many state.
  `_wireframe` folded into `RequiresUnbatchedWorldRender` — the one real visual divergence between
  the paths. Live-revertible via `WorldScene.MdxOpaqueBatchingEnabled`.
- **Do not credit Defect B's fix with fixing the gallop.** Frame p99 barely moved on that capture
  (259.70 → 246.62) because the periodic stall was never the MDX cost.
- **Instrumentation is now self-defending.** `PrepareObjectPhase` has a stage timer (`StageCount`
  18 → 19) and `WorldFramePassInstrumentation` + a reflection test fail the build if a pass has no
  timer, a stage is recorded by nothing, or a stage is double-counted. Unaccounted time is now
  median 0.02 / p99 0.11 ms, so hitch attribution names a stage instead of a void.
- **Audio is scoped to the camera tile.** `Update` consulted **no tile information at all** — it
  scanned every resident tile. Now takes `TerrainManager.CameraTileX/Y` (passed in, never re-derived)
  within `WorldAudioRuntime.AudibleTileRadius` (1); the diagnostics panel uses the same window.
  **Tile keying was checked and cleared** — `AddTile`/`RemoveTile`/`EmitterKey`/`OnTileLoaded` all
  agree; scanning every tile just made it look like a keying fault.
- **OPEN: MCSE emitters read as permanently out of range; only water works.**
  `AlphaTerrainAdapter.ConvertSoundPosition` does `chunkCorner - local` on the strength of an
  unevidenced comment ("Alpha MCSE stores a chunk-local C3Vector"); the Ghidra work proved the 0x34
  **field layout**, not the frame. If it is not chunk-local, every MCSE emitter lands tens of
  thousands of units off-map — which also explains why MCNK liquid rows work, since they derive from
  the renderer's own `chunk.WorldPosition` and never touch the transform. **`McseFrameEvidence`
  measures it** (raw min/max per axis, chunk/tile/beyond counts, explicit verdict) at the top of
  Utilities > Audio. **Read that line before touching the transform** — it deliberately reports
  "inconclusive" rather than picking a winner on a mixed sample.
- **Refuted, do not revive without new evidence:** the allocation-churn hypothesis. Median
  world-render CPU is 0.33–8.58 ms and traversal maxes at 0.22 ms. Spec 152 Phases 3–5 (flatten the
  scene graph into retained draw lists, view modes) are **suspended** because they rested on that
  premise. Also ruled out with evidence: decoded-asset caching / LRU thrash (`MaxMdxCached = 0`,
  unlimited; 554 models serve 18663 instances — nothing is re-decoded, the cost is submission).
- **The measurement tool exists and works.** Utilities > Perf > Frame history: rolling per-frame
  history, hitch detection with dominant-cause attribution, unaccounted time, region peaks,
  submission batching counts, and an injected-stall self-check. Recording is allocation-free.
  Use it for every before/after. **Benchmarks: Stranglethorn Vale** (dense doodads),
  **Stormwind** (dense WMO groups).
- **No viewer test assembly exists** (`tests/` has Core, Core.Anim, Core.Curation, Core.PM4 only), so
  viewer-side changes — `McseFrameEvidence`, the audio tile window, the batching predicate — carry
  source proof plus capture, not unit tests. Moving those types into core is an open follow-up.
- **Detector lessons that made this possible:** p99 hides rare hitches (they land at p100 — use max
  and over-threshold count); ranking stages by p99 buries a rare-but-huge stage (sort by max);
  always report unaccounted time, or attribution names a 0.2 ms stage for a 350 ms frame.
- **Lower-priority target:** User-run visual/compact-window proof for the Spec 080 Phase 2E IA.
  Check Scene for only Placements/LOD, Experimental > Terrain Lab for tiles plus chunk clipboard,
  Inspect's dropdown for Archeology, MCNK/ADT, scene investigation, world context, animations, and
  actions, Utilities > Minimap for the restored route, and Navigator > World Maps for the Phase Map
  selector.
- **Related WMO/fog observation — fold this into the group-admission work above.** A screenshot showed
  distant WMO content, including old Ironforge, still visible beyond the effective fog end while
  terrain had already been culled. Same shape as the Stormwind finding: WMO geometry is being admitted
  that should not be. Treat as a concrete symptom, not a proven owner; it wants the same
  visibility/submission counters plus a trace of camera-to-bounds distance against the fog plane
  before any admission logic changes.
- **Proof owner:** Focused PM4/audio contract tests and cross-platform viewer build pass; the user owns
  real-client region-camera, streaming, archive-provenance, and audible proof. The current camera
  slice updates active tiles on mouse-look without reopening the residency lease.
  **For renderer performance specifically, the proof is now the in-viewer frame history** — the user
  flies the route and reads Utilities > Perf. Two captures (Stranglethorn, Stormwind) are recorded
  with full numbers in Spec 153 `research.md`; every renderer claim must cite one.
- **Time-of-day checkpoint:** The interactive lighting path has a pure 2,880-unit/24-minute Alpha
  clock enabled by default, with manual slider freeze/resume; Light DBC and LIT consume the same frame
  time, while synthetic minimap manifests record a frozen time-of-day mode.
- **Completed slice (latest):** `975d0c79` — Spec 153 Phases 1/2/3/5, audio tile scoping,
  `McseFrameEvidence`, and the v0.5.2.1 release (version bump, CHANGELOG, release notes, READMEs).
  `bda47bdb` — handoff repointed at WMO group admission. Both pushed to `v0.5.3-dev`; tag `v0.5.2.1`
  published with all four platform builds.
- **Completed slice (earlier):** Checkpoint commits `3bfbbba4` (accumulated audio, AreaNumber, Ghidra, and
  Zone/SubZone overlay work), `de41b183` (Spec Kit design pack), and `c70e1945` (portal phase)
  contain the work completed on this lane. Spec 151 Phase 1 now has a pure, fail-open WMO portal
  decision using transformed portal polygons/clip volumes, source-side admission, bounded
  depth/visit limits, renderer integration, and portal counters in `WmoRenderStats`; the old
  center-distance/queue traversal scaffolding is removed. Focused portal/graph tests pass 16/16 and
  the full solution Debug build passes with 0 errors. The graph evaluator is explicitly diagnostic;
  the shared runtime decision owns final renderer admission. Spec 149 now has an opt-in resident
  Zone/SubZone overlay slice: Ghidra-backed
  MCNK AreaNumber evidence, revisioned resident chunk enumeration, AreaTable-grouped footprint regions,
  distinct Zone/Subzone styling, projected labels, and unresolved-count diagnostics. Spec 148 now has a
  provenance-first world-simulator spec/plan/tasks pack;
  MCSE emitters preserve raw/transformed positions and the proven Alpha 0.5.3 0x34-byte scheduler
  fields; shared Alpha AreaNumber resolution splits high/low `ushort` zone/subzone words and follows
  `ParentAreaNum` without half-word aliases; the area contract now branches explicitly so 3.3.5+
  direct AreaTable IDs cannot be captured by Alpha AreaNumber aliases; status-bar and terrain audio now
  consume the same resolved Zone/SubZone result; the runtime exposes non-playing diagnostic rows; the audio panel
  shows IDs, coordinates, path/source, decode/backend state, terminal reason, and coordinate provenance.
  The audio runtime also exposes a residency-change-only normalized emitter snapshot and the audio
  panel can opt into source-colored 3D speaker pins without starting playback.
  Automatic ZoneMusic playback is now hard-muted behind a tested policy; its area assignment remains
  diagnostic-only so the working MCNK/MCSE water path is not affected.
  The current Spec 080 Phase 2A/2C sidebar slice replaces the visible Model/World/Tools top row with
  Quick/Inspect/Scene/Utilities/Experimental, gives selection/model/ADT/MCNK/PM4 facts one inline
  inspector route, restores MDX/M2 animation controls in Inspect, and combines terrain targeting
  with MCNK/chunk clipboard actions in Experimental Terrain Lab. Audio is owned only by Utilities.
  The viewer build and source checks pass; the full test command timed out and the
  focused core suite reports nine unrelated baseline failures.
  Main Panels utility entries now select their exact Utilities page; compact-window
  manual proof remains open. Source/file/map loading is explicitly left-sidebar-only; the right
  Scene selector now contains only Placements and LOD, while Utilities keeps an isolated page index
  so Inspect/Scene page selection cannot hide or misroute the Minimap page.
- **Main unproven gap:** **WMO group admission is measured as the renderer's dominant remaining cost
  but its cause is still not diagnosed** — the counters that name the admitting rule now exist and
  have never been read on a real flight. Reading them is the next bounded action.
  The MCSE coordinate frame is also open: measured, verdict not yet read on real data.
  The sidebar slice still needs user-owned visual proof at normal and compact
  window sizes, including selected-context transitions and legacy caller reachability. The time-of-day
  slice still needs live early-client visual proof and a
  comparison of authored minimap tint behavior; the theory that shipped minimaps captured a moving
  clock remains unproven. Spec 104's restored MDX material shader inputs still need real model/shader
  compilation and visual proof. Full BLS bytecode parity remains out of scope. Spec 151's game-mode head anchor/physics, simple-surface policy, and
  diagnostic budget remain unimplemented. Portal admission is source-tested but still needs the
  user-owned real-client visual/submission/FPS comparison. Spec 149's PM4 region
  bounds/focus, correlation UI retirement, focused area aggregation tests and default-off per-trigger
  audio controls remain open. MCSE tile/chunk
  normalization and MCNK liquid-center placement now have focused source/test proof, pending live
  runtime/audible proof. Speaker-marker placement is source-tested through the normalized snapshot
  path, pending live visual proof. The area overlay is resident chunk coverage, not a
  proven complete polygon. Automatic ZoneMusic playback is intentionally muted until its handoff is
  proven; area resolution remains diagnostic-only.
  ZoneMusic table indirection, exact `sounds.mpq` provenance, MIDI/DLS
  playback, and native MCSE callback installation remain separate proof gates. Spec 150 still lacks
  native renderer anchors, repeatable 0.5.3 baseline capture, and CPU/GPU attribution.
- **Explicitly out of scope for the next slice:** Simple-surface UI, logging-policy retirement,
  whole renderer rewrite, `.bls` bytecode loading/porting, fake audio conversion, and claims of
  visual/FPS/audible gains. Game-mode input/UI follows the pure Phase 2 runtime-core checkpoint.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| **151 Portal-aware rendering / WMO group admission** | **PRIORITY 1 — instrumentation shipped, measurement owed** | **Fly Stormwind, read Utilities > Perf > "WMO admission (this frame)", record it in Spec 151 `research.md`. Only then pick the rule to change.** |
| 153 Renderer hitch and MDX batching | Phases 1/2/3/5 shipped as v0.5.2.1 and confirmed by capture; Phase 4 likely moot | Re-measure `SceneMaintenance` before implementing Phase 4; Phase 5 step 2 (decode off the render thread) still owed |
| 152 Renderer frame-time stability / per-era lighting | Detector landed and used; its Phase 1 gate refuted the allocation hypothesis so Phases 3–5 are suspended | Owns the measurement infrastructure (done) and Phase 6 per-era terrain lighting (independent, not started, fixes 1.0.0+ darkness). |
| 151 Portal-aware rendering/game mode/simple surface | Phase 1 portal checkpoint implemented; Phase 2 open | Add pure game-mode state/physics and character-head anchor; preserve editor camera state and stop at the focused physics checkpoint. |
| **155 Asset reference inventory (expected vs catalogued vs present)** | **NEW — spec drafted, not planned** | Extend `inspect` with per-asset reference logging and corpus-wide sweeps over **WMO doodads, WMO textures, and MDX/M2 textures**. Deliverable is the disagreement between what data references, what listfiles name, and what the build contains, plus orphans as the repair donor pool. **Positive control: the Mt. Hyjal green-smoke effect objects must be flagged by an untargeted sweep** — the engine draws untextured geometry neon green, so it is verified in-world. Corpus comes from the data-access layer, never from archive internal listfiles. |
| **154 M2 reader era parity (1.x–3.0.1)** | **NEW — spec + plan drafted, not started** | Run US1 first: survey every staged build before touching a reader. Three measured defects; the "4.0.0 works" premise is contradicted by measurement and must be resolved, not assumed. |
| 104 Legacy M2/MDX rendering | 1.0.0 route complete; MDX material/effect shader checkpoint implemented with visual proof open | Validate shader compilation and translucent/reflective models against the configured client/build; keep full BLS parity separate. **Blocked in part by Spec 154** — M2 bone reading is broken outside the Alpha and late-3.x routes. |
| 149 PM4 region navigation/audio trigger controls | Draft pack; resident area overlay, MCNK liquid producer, coordinate normalization, and opt-in speaker-marker slices implemented | Add area aggregation/audio-control tests, then complete per-trigger toggles and ZoneMusic indirection; retire correlation UI only after the region checkpoint; keep world triggers default-off. |
| 150 Alpha 0.5.3 renderer performance | Draft evidence/planning pack complete; no source optimization started | Recover native world/terrain/object/resource/LOD anchors and run two repeated production `profile-render` baselines before choosing one owner. |
| 148 Artifact world simulator runtime | Phase 1 diagnostics in progress; client contract correction landed | Add ZoneMusic indirection, then finish read/decode/source-stage coverage and user real-client inspection. |
| 147 Minimap/fog/doodad instancing | Phase 2 implemented; Phase 3/4 open | User-run minimap proof, then implement fog coverage and structured batching diagnostics. |
| 146 Audio/camera playback | AreaNumber-aware area selection and master mute control implemented; client audio contracts recovered | Add ZoneMusic row resolution; MIDI/DLS and native MCSE callback proof remain gated. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks. |
| 080 WoW UI consolidation | Phase 2A tabbed sidebar IA and unified inspector source slice implemented; manual proof open | Run the five-tab visual/compact-window check, then resume the legacy route inventory before deleting old methods. |
| 143 World context and lighting | LIT source/fallback, pre-alpha v2 parser, and default-on 0.5.3 time cycle implemented with user gate | Validate live clock/manual freeze and authored-minimap tint boundaries, then continue WMO area and lighting evidence. |
| 142 World scene graph | In progress | User-run dense-WMO capture to compare internal-doodad batching against the prior placement-local path. |
| 139–141 Terrain/minimap reconstruction | Active/parked ML lanes | Reopen only for the named spec and user-run training/validation. |
| 138 Cross-era renderer research | Evidence/planning | Do not generalize one client build to every era. |
| 128–131 PM4 | Established research lane | Use the PM4 spec pack and `workstream-pm4-decode.md`. |

## Stable boundaries

- New code, tests, tools, and viewer docs go in `wow-viewer/`; the legacy tree is read-only
  reference unless a bounded compatibility fix is explicitly requested.
- Keep format readers library-first and tools thin. Do not duplicate or rewrite working client-file
  readers. Keep the Alpha/standard terrain split. `AlphaWdtWriter.cs` is frozen unless explicitly
  reopened with focused proof.
- Client roots are runtime configuration. `H:\CLIENTS` is approved; never hardcode a local client
  path. Record root, build identity, and fingerprint for client-backed proof.
- Training, GPU work, broad harvests, long captures, and real-client/runtime testing are user-run.
- The default source proof is:
  `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  and
  `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.

## Continuity rules

- Update this dashboard and `progress.md` only when the implementation handoff changes.
- Put durable technical findings in the owning workstream or architecture note, not here.
- Preserve negative results and open gates, but remove superseded narrative from the default path.
- End every handoff with: current target, proof owner, completed slice, unproven gap, next bounded
  action, and explicit out-of-scope items.
