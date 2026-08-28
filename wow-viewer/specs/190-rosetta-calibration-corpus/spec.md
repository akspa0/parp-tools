# Feature Specification: Rosetta Calibration Corpus for PM4 Object Identification

**Feature Branch**: `190-rosetta-calibration-corpus`

**Created**: 2026-08-26

**Status**: Draft

**Input**: User description: "We have access to every single object in the game. Instead of heuristic
matching that does not fully work on all PM4 tiles, generate map tiles that evenly place every game
object at a known position, decode that synthetic data back through our own pipeline to build a
complete labelled reference library, and automatically match real PM4 data against it. Also: we are
not generating ADTs for tiles that have no ADT — synthesize them so no tile is skipped."

> **Implementation checkpoint 22 (2026-08-28).** Streaming tile generation and writing to eliminate out-of-memory crashes.
>
> 1. **Streaming Alpha WDT writer (`AlphaWdtWriter.Write`).** Added streaming overload to `AlphaWdtWriter` that writes
>    tiles one-by-one directly into a `FileStream`, keeping only one tile's data in memory at a time and patching the 64 KB
>    `MAIN` index and `MPHD` offsets upon completion.
> 2. **On-demand canvas generation.** Optimized `BuildMap` and `BuildTileAdt` so 1024×1024 MCAL text and checkers canvases
>    are generated and sliced transiently per-tile via `BuildTileAlphaCanvas` and `BuildTileCheckersCanvas`, rather than
>    allocating gigabytes of canvas buffers across all planned tiles upfront.
> 3. **Streaming minimap renderer.** Updated `RosettaMinimapPainter` to render downsampled text on-demand without holding
>    pre-rendered canvases for the entire corpus.
> 4. **Safe file overwriting.** Handled `--overwrite` cleanly across all output files (Alpha WDT, LK ADT/WDT/WDL, and minimap BLPs).
> 5. **End-to-end verified on 0.5.3 client.** Successfully generated full `RosettaAlpha` map (1,720 tiles, 11,663 placements,
>    920 MB monolithic WDT, 1,720 minimap BLPs) with minimal, flat memory footprint.
>
> Verified: 38 focused `RosettaTilesetGeneratorTests` pass green, full client generation completes cleanly.
>
> **Implementation checkpoint 21 (2026-08-28).** Diagnostic `checkers.blp` texturing on indented terrain floors.
>
> 1. **Diagnostic `checkers.blp` testing pads.** Painted `tileset\generic\checkers.blp` under objects on the indented
>    pedestal floor via `CheckersCanvas`, mirroring authentic Blizzard development/testing terrain techniques.
> 2. **Multi-layer MCAL chunk blending.** Supported 3-layer MCNK structures in `BuildTileAdt` (Layer 0: Sand,
>    Layer 1: Checkers pad under object, Layer 2: Handwriting ink label) with concatenated MCAL chunks and correct offsets.
> 3. **Automatic checkers texture resolution.** Added `ResolveCheckersTexture` in `Program.cs` and CLI option `--checkers-texture`.
> 4. **Minimap checkers pattern.** Rendered matching checkered pattern on pedestal plateaus in `RosettaMinimapPainter`.
>
> Verified: 38 focused `RosettaTilesetGeneratorTests` pass green.
>
> **Implementation checkpoint 20 (2026-08-28).** Museum exhibit layout, handwriting font, compact cells & Westfall sand.
>
> 1. **Compact 4-chunk museum cells.** Set CLI default to `CellChunks = 4, LabelBandChunks = 1` in `rosetta-generate`,
>    increasing cell density from 4 to 16 cells per tile so small MDX/M2 objects do not waste large 266m areas.
> 2. **Museum exhibition ordering.** Grouped assets by kind (Models first, then WorldModels) and sorted by exhibit
>    footprint size (smallest to largest) with alphabetical grouping within size tiers.
> 3. **Handwriting / cursive script font.** Added full 5x7 cursive handwriting font table to `RosettaTextPainter`
>    covering uppercase ('A'-'Z'), lowercase ('a'-'z'), digits ('0'-'9'), and punctuation, baked into the 1024×1024 MCAL Layer 1 texture.
> 4. **Westfall/Westwood sand base texture.** Updated default ground texture to `tileset\westfall\westfallsand.blp` and
>    prioritized Westfall/Westwood sand in `ResolveGroundTexture`.
>
> Verified: 37 focused `RosettaTilesetGeneratorTests` pass green.
>
> **Implementation checkpoint 19 (2026-08-27).** Negative pedestal height support (sunken object viewing dips).
>
> 1. **Enabled negative pedestal heights (`PedestalHeightMeters = -10f` default).** Allowed negative
>    height values (e.g. `-10.0`, `-20.0`) in `RosettaGeneratorOptions` and `Program.cs` so object viewing
>    cells sink into the terrain mesh with beveled ramps, exposing the full bounding mass of WMOs whose
>    local origin/feet sit below ground.
> 2. **Mesh height computation (`CreateChunkHeights`).** Updated `CreateChunkHeights` to sink heights
>    toward negative values when `p.Height < 0` with proper bevel scaling.
>
> Verified: 34 focused `RosettaTilesetGeneratorTests` pass including new `Generate_PedestalHeights_NegativeSunkenDipWithBevel`.
>
> **Implementation checkpoint 18 (2026-08-27).** Alpha WDT MCAL/MCLY tile transform alignment.
>
> 1. **Aligned Alpha tile dictionary keys in `Program.cs`.** `rosetta-generate --format alpha` now passes
>    `(tile.TileY, tile.TileX)` to `AlphaWdtWriter.Build`, using the exact same coordinate transform
>    as `LkWdtWriter.Write` and `RosettaMinimapPainter` (`map{tileY}_{tileX}.blp`).
> 2. **Base writers and readers remain untouched.** No protected format readers or writers were mutated.
> 3. **MCAL/MCLY texture layers now land on the exact correct tiles.**
>
> Verified: 33 focused `RosettaTilesetGeneratorTests` pass, all 17 `LkToAlphaRoundTripTests` pass.
>
> **Implementation checkpoint 17 (2026-08-27).** Correction to checkpoint 16: the Alpha WDT bytes
> were not the proven defect. The reported mismatch was in new minimap tooling.
>
> 1. **Removed the mistaken Alpha WDT byte mutation.** `rosetta-generate --format alpha` again writes
>    the monolithic map bytes directly from `AlphaWdtWriter.Build(...)`; there is no Rosetta post-write
>    MDDF/MODF coordinate patch and no manifest field claiming alternate Alpha client file positions.
> 2. **The retained fix is minimap-only.** `RosettaMinimapPainter` places the
>    cyan model marker and amber WMO marker at the same upper-band center used by the generated
>    placement, instead of the whole text cell center. The minimap marker is still a synthetic locator,
>    not rendered MDX/WMO geometry.
> 3. **No base renderer, terrain-loading, protected writer, or generated map byte contract changed.**
>    New tooling must layer over the proven base implementation unless a separate evidence-backed spec
>    explicitly reopens that base.
> 4. **Noggit reference check did not reveal a per-object minimap denylist.** Noggit3's minimap widget
>    consumes WDL horizon data, and its horizon loader skips WDL object chunks (`MWMO`, `MWID`, `MODF`)
>    while deriving the minimap from `height_17`. The object visibility logic found in `ModelInstance`
>    is normal 3D distance/frustum/projected-size culling, not a minimap object classifier. No accessible
>    Noggit/Noggit-Red asset denylist or "do not draw on minimap" table was found in this pass.
>
> Verified: 33 focused `RosettaTilesetGeneratorTests` pass, including the minimap marker pixel
> regression. Real 0.5.3 client visual proof remains operator-owned.
>
> **Implementation checkpoint 15 (2026-08-27).** Minimap generation and runtime tile coordinate readout:
>
> 1. **Automated Minimap BLP Tile Generation (`RosettaMinimapPainter` & `Blp2Writer`).** Generates 256×256
>    DXT1-compressed BLP2 minimap tiles under `Textures/Minimap/{mapName}/map{tileY}_{tileX}.blp` with
>    rendered cell borders, pedestal plateaus with bevels, downsampled antialiased text labels, and
>    distinct asset center markers (cyan diamond for models, amber box for WMOs).
> 2. **Real-time Status Bar Tile Coordinate Display.** The viewer bottom status bar now outputs
>    `Tile: {tileY:D2}_{tileX:D2}` (matching ADT and minimap naming conventions) whenever any runtime scene
>    is active.
>
> **Implementation checkpoint 14 (2026-08-27).** Three enhancements delivering museum-grade visual
> presentation and removing artificial container caps:
>
> 1. **MCAL/MCLY texture layer text painting (`RosettaAlphaPainter`).** Because Alpha WDT lacks MCCV
>    vertex colors, text is rasterized into an uncompressed 4-bit MCAL alpha layer (Layer 1 with ink
>    texture, e.g. `tileset\generic\black.blp`). At 64×64 texels/chunk (1024×1024 per tile), resolution
>    is 0.52 m/texel (8× finer than MCCV), rendering crisp, legible labels across both Alpha and LK.
> 2. **Museum pedestal heightfields (`MCVT`).** Objects sit on a raised, beveled plateau
>    (`PedestalHeightMeters = 4f`, `PedestalBevelMeters = 12.5f`) in the terrain mesh to prevent
>    base clipping and provide clean museum-grade display plinths.
> 3. **Removal of 512-tile Alpha WDT limit.** `AlphaWdtWriter` has no architectural limit and easily
>    handles 900+ tile continent maps (like Kalimdor). The default tile cap is unified to 4096.
>
> **Implementation checkpoint 13 (2026-08-26).** Two corrections, both of the same kind: the tool
> asking the operator for something it already knew, and dropping something the operator needed.
>
> 1. **`--format` is detected from the client.** Pointing at the 0.5.3 root without the flag failed
>    with *"no post-alpha assets found ... this root offered 5545 models of the other container"* —
>    the tool had counted the evidence and still demanded to be told. It now counts `.mdx`/`.mdl`
>    against `.m2` in the listfile and picks the era, printing
>    `Client era: alpha [detected: 5545 .mdx/.mdl, 0 .m2]`. An explicit `--format` still wins but
>    warns when it contradicts the client, and the refusal message now ends with the actual fix
>    (`Drop --format (it is detected from the client) or pass --format alpha`) instead of telling the
>    operator to go find a different client.
> 2. **Filename extensions are back in painted labels.** Checkpoint 7 stripped them to buy four
>    characters, on the reasoning that the plate tint already encodes model-vs-world-model. Wrong
>    trade: the extension is the asset TYPE, which is the one thing the label must carry that the
>    name alone does not. Middle-elision puts it on the surviving tail, so it shows even on a clipped
>    name — `HumanMalePirateSwashbuckler_Ghost.mdx` paints as
>    `HUMANMALEP / IRATE-KLER / _GHOST.MDX`. The plate tint stays as a redundant cue.
>
> Verified: the operator's exact command (alpha client, no `--format`, no `--ground-texture`) runs
> clean — era detected, texture auto-selected, 400 assets across 231 tiles. The post-alpha client
> still detects as `lk` (`0 .mdx/.mdl, 25833 .m2`). 26 Rosetta tests pass; the pre-existing unrelated
> Core.Tests failures are unchanged.
>
> **Implementation checkpoint 12 (2026-08-26).** Re-running into a used output folder crashed on an
> unhandled `InvalidOperationException` after minutes of archive scanning, because occupied-tile
> collection only looked at the base map folder while multi-map output writes `{MapName}00`,
> `{MapName}01`, ... and every map re-anchors at the same start tile.
>
> 1. **Pre-flight before any expensive work.** The output root is checked for existing `{MapName}NN`
>    folders holding tiles the moment the arguments parse — instant refusal naming the folders, with
>    `--overwrite` to replace them. Self-scanning our own output for "occupied" tiles is gone; only
>    `--existing-map-dir` reserves tiles now, since a re-anchored map's tile names collide by
>    construction and treating them as occupied pushed each map off its own block.
> 2. **`map inspect --archive-root [--virtual-path]`.** Maps live inside MPQs like every other asset,
>    and until now the only way to inspect a shipped WDT was to extract it by hand, which is why
>    WMO-only maps were never measured. Without `--virtual-path` it surveys every WDT it can
>    enumerate, one line each. `PrintMapSummary` re-opened `SourcePath` from disk, which cannot work
>    for an archive read, so it now takes the already-open stream.
>
> **Measured, on the operator's report that "most wmo-only maps don't open in the viewer".** Surveying
> all 20 maps of the 0.5.3 client (each WDT is wrapped in its own `.wdt.MPQ`, and the internal
> listfile names almost no map files — hence the by-path probe):
>
> | | maps |
> |---|---|
> | terrain (tiles > 0) | 9 |
> | WMO-only **with** a MODF placement | 7 — Blackfathom, Collin, StormwindJail, StormwindPrison, SunkenTemple, Uldaman, test |
> | WMO-only with **no MODF chunk at all** | 3 — GnomeragonInstance, Monastery, WailingCaverns |
> | completely empty (no tiles, no names) | 1 — UnderMine |
>
> Those 3 name a world model in MONM and never place it: `Monastery.wdt` top-level chunks are
> `MVER, MPHD, MAIN, MDNM, MONM` against StormwindJail's `..., MONM, MODF`. There is nothing to
> render, so no viewer fix can make them appear — a data gap, not a bug. Detection is also not at
> fault on the alpha side: `AlphaTerrainAdapter` sets `IsWmoBased = _wdt.IsWmoBased ||
> _existingTiles.Count == 0`, and a WMO-only map has zero tiles, so it is always flagged.
>
> **Not confirmed:** 7 of 10 WMO-only alpha maps parse with a placement, which does not match "most
> don't open", so the failure for those is downstream of the WDT — most likely v14 WMO loading or
> map discovery — and needs a specific failing map name plus a viewer run to pin down. Recorded
> rather than guessed at.
>
> **Implementation checkpoint 11 (2026-08-26).** Checkpoint 10 turned the ground-texture problem
> into an error the operator had to solve by naming a BLP path from inside the MPQs. That is the
> wrong end of the tool. Operator: *"don't fucking put so much on the user"*, *"I'm not fucking
> around peering inside all of them to find the right tileset texture"*. Fixed, plus one real bug the
> alpha run exposed.
>
> 1. **`--ground-texture` is optional and resolved from the client.** Order: an explicit request
>    (which must exist, or the run stops), else the era-neutral default when the client ships it,
>    else a deterministic pick from the client's own listfile. The pick prefers flat, low-contrast
>    surfaces — `seafloor`, `marble`, `sand`, then dirt/rock bases — because a busy tile ruins MCCV
>    label legibility, skips `_s`/`_h` companions, and falls back to ordinal-first so it never
>    depends on enumeration order. The chosen path and why is printed:
>    `Ground texture: Tileset\Ashenvale\AshenvaleSand.blp  [auto-selected from client (sand)]`.
> 2. **Verified against the real 0.5.3 client.** `--client-root "H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft"
>    --format alpha --kit-depth 2` runs with no texture flag at all: 5,545 candidate assets, era
>    admission keeps all of them (`0 wrong-container models, 0 wrong-version world models` — the
>    alpha client is all `.mdx`), and a 1,200-asset slice produced a 432-tile alpha WDT that
>    `map inspect` reads back as Version 18 with 1,200 MDNM names.
> 3. **`WdtSummaryReader` misreported every alpha map with an odd model count as WMO-based.** The
>    alpha MPHD is not the LK flags struct — its first fields are the MDX name count, MDNM offset,
>    WMO name count, MONM offset (`AlphaWdtWriter.PatchMphd`) — so the LK `flags & 0x1` test was
>    reading the **low bit of the model count**. 1,200 models (1,201 written) reported
>    `wmoBased=True`, 71 models (72) reported False. `AlphaTerrainAdapter.IsWmoBased` forwards this,
>    so an alpha Rosetta map would have told the viewer it has no terrain. Alpha carries no such
>    flag, so it is now decided structurally — no terrain tiles plus a named world model — gated on
>    the 16-byte alpha MAIN cell that already distinguishes the layouts. Guarded by a `[Theory]` over
>    both parities, **verified to fail on the odd case against the pre-fix reader**.
>
> 26 Rosetta tests pass; the pre-existing unrelated Core.Tests failures are unchanged.
>
> **Implementation checkpoint 10 (2026-08-26).** Era-gating. Checkpoint 8 let a single client root
> feed both containers, which meant an alpha WDT could be written naming `.m2` models and v17 world
> models — assets the 0.5.3 client cannot open. Operator: *"why the hell would I want to try and
> generate data that doesn't exist?"* Correct. **One run = one era = one container.**
>
> 1. **`--format both` is gone.** `--format lk|alpha` now decides which assets are admitted at all.
>    Alpha admits `.mdx`/`.mdl` models and **v14** world models (the monolithic root-plus-groups
>    form); LK admits `.m2` and **v17+**. WMO era is read from the root's version via
>    `WmoSummaryReader` — `WmoRootReaderCommon` already expands MOMO, so v14 roots parse fine — and a
>    wrong-version root is excluded by name, not silently written.
> 2. **A mismatched client root is refused, with the evidence.** `--format alpha` against the WotLK
>    client now exits 1 with: *"no alpha-era assets found ... this root offered 25833 models of the
>    other container and 2453 world models of the other version. Point --client-root at a 0.5.3-era
>    client."* The candidate count itself moves with the era (39,862 for LK, 14,029 for alpha).
> 3. **The ground texture was the same bug, undetected.** Every tile ever generated referenced the
>    hard-coded `tileset\ocean\westfallseafloor.blp`, and **that path does not exist in
>    `C:\WoW4-data\WoW-12025`** — which is why the corpus rendered as flat untextured grey.
>    `GroundTexture` is now a generator setting (`--ground-texture`), threaded into MTEX and verified
>    against the client before any tile is written; on failure the CLI lists real candidates from the
>    client's own listfile rather than substituting one, because guessing a texture is the same
>    mistake as guessing an era. Verified end to end: a tile written with
>    `--ground-texture "TILESET\Aerie Peaks\AeriePeaksRockBase.blp"` carries exactly that in MTEX.
> 4. **Provenance.** The index and every per-map manifest record `era`,
>    `admittedModelExtensions`, `admittedWmoVersion`, and the wrong-era rejection counts, so a corpus
>    can never be read back without knowing which client era produced it.
>
> Not addressed: the alpha lane still builds each tile as `LkAdtData` and converts through
> `LkToAlphaConverter`. That intermediate carries only era-neutral data (heights, normals, placement
> coordinates) and the asset names in it are now alpha-era by construction, but it remains an LK-
> shaped hop and is worth removing if the alpha lane grows era-specific tile content.
> 24 Rosetta tests pass; the pre-existing unrelated Core.Tests failures are unchanged.
>
> **Implementation checkpoint 9 (2026-08-26).** The full-corpus run died with `Out of memory`
> immediately after enumeration. Cause and fixes, all measured.
>
> 1. **The generator materialised every tile's ADT before anything was written.** A built
>    `LkAdtData` is ~400 KB of arrays (256 chunks x 145 heights + 448 normals + 580 MCCV), and a
>    full client at kit granularity runs to five figures of tiles — several GB before the first byte
>    hit disk. `RosettaTilePlan` now carries only what a tile needs (placements, rects, labels) and
>    `RosettaTilesetGenerator.BuildTileAdt(mapName, plan)` builds one on demand; the CLI builds,
>    writes, and drops each tile. **Measured: 20,000 assets / 5,236 tiles / 2 maps peaks at 249 MB**
>    (6,000 assets / 1,532 tiles peaks at 185 MB — the residual is the MPQ layer, not the tiles).
> 2. **`LkToAlphaConverter.ConvertTile` allocated 20 MB of zeroes per tile.** `alphaPack`
>    (float[1024,1024,4] = 16 MB) and `shadowMask1024` (4 MB) were unconditional, and the alpha pack
>    was then *retained* on the returned `AlphaTileData`. Both are now allocated only when the source
>    ADT actually carries alpha or shadow data — every synthetic tile carries neither, and so do
>    plenty of real single-layer ones. `McalAlphaPack` and `McshShadowMask1024` were already nullable,
>    so no consumer contract changed.
> 3. **A kit larger than one map no longer dead-ends the run.** The alpha lane caps a map at 512
>    tiles (the WDT is serialised whole in memory), and `item\objectcomponents` alone needs 974 —
>    which the "a kit is never split across maps" rule turned into a hard failure. A kit too big for
>    a map is now split into map-sized parts that keep the kit's name, so the index lists it once per
>    map it spans. Kits that fit are still whole, and a kit's standard and oversize parts merge back
>    into **one index entry per (kit, map)**.
>
> `--format alpha|both` defaults `--max-tiles-per-map` to 512 and says so on stdout; the LK lane
> streams and keeps the 4096 default. **Measured alpha peak: 1,814 MB at 512 tiles per map**
> (~3.5 MB/tile including the whole-WDT byte[]), so lower the cap on a memory-tight machine.
> Verified run (4,000 assets, `--format alpha --kit-depth 2`): 3 maps, 12 kit entries over 11
> distinct kits, `item\objectcomponents` correctly spanning AlphaMem01+02, zero duplicate same-map
> entries. 23 Rosetta tests pass; the pre-existing unrelated Core.Tests failures are unchanged.
>
> **Implementation checkpoint 8 (2026-08-26).** The corpus is now organised the way the client
> already organises it, split across as many maps as it takes, indexed, and emittable in the alpha
> container.
>
> 1. **The source folder IS the designkit.** Assets group by their directory; a kit is laid out on
>    whole tiles of its own and **a tile never mixes kits**, which is what lets one index entry name
>    a kit's location without listing cells. `--kit-depth N` coarsens the grouping when per-folder is
>    too fine (a client has thousands of single-asset folders, and each was taking a whole tile);
>    depth 0 (default) is one kit per folder, depth 1 puts all of `creature\*` in one kit.
> 2. **Multi-map splitting.** The fourth run died on `Cannot place 7442 Rosetta tiles: a square block
>    needs 87x87` — one 64x64 map cannot hold a full client at a legible cell size. Kits now pack
>    greedily into maps (`{MapName}00`, `{MapName}01`, ... ; the bare name is kept when one map
>    suffices), a kit is never split across maps, and each map re-anchors its own square block at the
>    start tile. `--max-tiles-per-map` caps a map explicitly. A layout that still cannot fit exits
>    with a named, actionable error instead of an unhandled exception.
> 3. **`rosetta-index.json`** at the output root is the object-library lookup: every designkit with
>    the map it was stashed on, the tiles it occupies there, and its model/world-model split, plus a
>    per-map summary and the full exclusion list. Per-map `rosetta-manifest.json` keeps the
>    placement-level detail and now tags each placement with its kit.
> 4. **Alpha WDT output.** `--format lk|alpha|both` writes the 0.5.3 monolithic container via
>    `LkToAlphaConverter.ConvertTile` + `AlphaWdtWriter.Build`. With `both`, alpha lands under
>    `{output}/alpha/World/Maps/{map}/` so the two containers never collide. Verified: the written
>    WDT satisfies `AlphaWdtReader.IsAlphaWdt`, reports its tiles, and every placement read back out
>    still decodes onto its own tile (`GeneratedTiles_SurviveTheAlphaWdtRoundTrip`); `map inspect`
>    parses it as Version 18 with the full MDNM name table.
>
> **Known limitation: the alpha container has no MCCV**, so the painted labels do not survive
> `--format alpha`. `AlphaWdtWriter` writes no MCCV sub-chunk and `McnkAlpha.MccvData` is empty by
> definition — vertex colour postdates 0.5.3. Identity in the alpha lane therefore comes from the
> index and manifest, which is what the spec already designates as the machine-readable authority.
> Painting labels into an alpha tile would mean using the texture alpha layer (MCAL) instead, which
> is a larger change and is not attempted here.
>
> Smoke runs against `C:\WoW4-data\WoW-12025` (500 assets): `--format both` gave 1 map, 157 tiles,
> 465 placements across 16 designkits, LK tree 60 MB and alpha WDT 47 MB; `--format alpha
> --kit-depth 1 --max-tiles-per-map 120` split into `KitDepth00` (51 tiles, 4 kits) and `KitDepth01`
> (99 tiles, 1 kit) with every kit whole. 22 Rosetta tests pass; the pre-existing unrelated
> Core.Tests failures are unchanged.
>
> **Implementation checkpoint 7 (2026-08-26, after the fourth real run).** Four changes, and one
> corpus defect the run exposed.
>
> 1. **Tiles now form a SQUARE block** anchored at the start tile (operator: "I'm expecting more of a
>    square, not this sort of weird map that isn't predictable from the x/y starting points").
>    Row-major contiguous fill wrapped at column 63, so the block's shape and final position depended
>    on the asset count. Side is now `ceil(sqrt(tileCount))`, filled row-major from
>    (StartTileX, StartTileY); occupied tiles spill into extra rows rather than perturbing the width;
>    an origin that would run off the 64x64 grid is pulled back to the largest coordinate that fits
>    and reported. `RosettaGenerationResult` carries `BlockOriginX/Y/Side`, and the CLI prints them.
> 2. **Glyphs are antialiased analytically.** Every terrain vertex integrates the glyph over a
>    one-font-pixel box centred on itself and blends plate-to-ink by that fractional coverage, instead
>    of the binary "is this vertex inside a lit pixel" test. The quincunx makes this pay double: with
>    an on-lattice run origin, INNER vertices land on font-pixel centres (crisp stroke core) and OUTER
>    vertices on font-pixel corners (free 2x2 resolve). Measured on a real tile: MCCV now carries
>    **8 distinct levels** (28 plate, 85, 142, 198 coverage steps, 255 ink, 196 rule, 127 neutral)
>    where before it carried 2.
> 3. **Labels carry more signal per character.** The extension is dropped (every cell is a model or a
>    world model, and the plate is now tinted cool/warm to say which), and an overlong name is elided
>    in the MIDDLE rather than cut at the end — WoW names put the family up front and the
>    discriminator at the back (`ICECROWN_WALL_SEMICIRCLE_PIECE_02_LONG_HOLLOW`), so a tail cut made
>    every variant paint identically.
> 4. **The oversize class gives the object the whole cell** (its label paints under the geometry),
>    recovering ~90 assets between the old 433 m band and a full tile. Non-finite bounds (camera-path
>    M2s) are now excluded by name instead of being reported as an infinite footprint.
>
> **Corpus defect found: the enumerator only accepted `.mdx`/`.mdl`/`.wmo`, so a 3.3.5-era client
> contributed ZERO models** — the fourth run's manifest was 2,119 placements, 100% `WorldModel`.
> Adding `.m2` (read via `M2ModelReaderDispatcher`, bounds at header 0xA0) takes the candidate count
> from 14,029 to **39,862**. Smoke run (400 assets, `--client-root C:\WoW4-data\WoW-12025`):
> 12x12 block at (20,20), 124 tiles, 365 placements, **0 of 365 off-tile** when the written MDDF/MODF
> bytes are decoded with the viewer's rule, and MCCV present on 55% of chunks (untouched chunks stay
> null, which renders identically and shrinks the tiles).
>
> **Open trade-off for the full run:** ~25,600 placeable assets at the default 4 cells/tile needs
> ~6,400 tiles, and a square block of those needs 80x80 — larger than the map. `--cell-chunks 4`
> (16 cells/tile) fits but cuts the label band to 5 characters per line. Splitting models and world
> models into separate maps keeps both legible; the generator has no kind filter for that yet.
> 17 Rosetta tests pass; solution builds clean; the pre-existing unrelated Core.Tests failures are
> unchanged.
>
> **Implementation checkpoint 6 (2026-08-26, after the third real run).** Three defects fixed.
>
> 1. **Objects on the perpendicular axis (root cause of "objects horizontal, map vertical").**
>    `LkAdtWriter.BuildMddf/BuildModf` apply the MapOrigin flip THEMSELVES — `LkMddfEntry.Position`
>    is a **renderer** coordinate, and `LkAdtReader` is its exact inverse. The generator was handing
>    them `RawPosition`, already in file space, so every coordinate was flipped twice: the on-disk
>    value became `MapOrigin - raw`, and the viewer decoded it back to `raw`. With tiles marching
>    along `tileX` the terrain ran along renderer X at fixed Y while the objects ran along renderer Y
>    at fixed X — perpendicular, and ~9,000 units apart. Checkpoint 4's note blaming the naming
>    convention was wrong: the naming was already correct by then, this was a second, independent
>    bug. Placements now pass `RendererPosition`, and MODF bounds are a renderer-space AABB built
>    from a square half-extent (rotation is zero, so this is conservative for either axis mapping).
>    **This also explains "WMOs unload under the camera"**: `WorldScene.OnTileUnloaded` drops the
>    instances belonging to a tile when that tile leaves the camera window (retained radius 1-3
>    tiles), so objects sitting ~17 tiles from their owning tile could only be seen from a distance
>    and vanished on approach. Not a renderer defect — the same coordinate bug.
> 2. **Grid was not a grid.** Cell size was `clamp(footprint*1.25 + margin, 48, 512)` shelf-packed
>    row by row, so no two cells matched and rows were ragged. Replaced with a uniform grid: cell
>    edge is a chunk count that divides 16 (`--cell-chunks`, default 8 = 266.67 m, 2x2 cells per
>    tile), every cell edge lands on a chunk boundary, and assets too large for the standard cell go
>    to a separate run of whole-tile cells so **no tile ever mixes cell sizes**.
> 3. **MCCV text illegible.** The terrain vertex lattice is a **quincunx** — 9 outer + 8 inner rows,
>    so a vertex exists only where the two half-pitch indices share parity. Font pixels were being
>    computed as small as one half-pitch step and then floor-clamped, so half of every glyph fell
>    into the lattice gaps, and labels that still did not fit were truncated character by character
>    down to nonsense. Font pixel is now quantized to one **sub-cell** (`chunk/8` = 4.167 m, the
>    smallest pixel that always covers exactly one outer and one inner vertex), line origins are
>    snapped to the lattice, and names **wrap** across the cell's label band (default 3 lines x 10
>    chars at the default cell) instead of being truncated. Contrast raised from 127-vs-255 to a
>    dark plate (28) behind bright glyphs (255), and cell boundaries carry a one-sub-cell rule so the
>    grid is visibly regular. MCCV is now emitted only for chunks actually painted (untouched chunks
>    render identically with null MCCV), which also shrinks the tiles substantially.
>
> New regression test `WrittenBytes_PutEveryObjectOnItsOwnTerrain` parses the MDDF/MODF bytes out of
> the built file and decodes them with the viewer's exact rule; it was **verified to fail** against
> the pre-fix code path. Also new: `Generate_CellsAreUniformAndChunkAligned`,
> `PaintTile_GlyphPixelsLandOnTerrainVertices` (asserts an exact bright-vertex count, which only
> holds if every font pixel lands on the lattice), `PaintTile_LeavesUntouchedChunksWithoutMccv`,
> `WrapLabel_WrapsInsteadOfTruncatingToNonsense`, `Generate_RejectsCellSizesThatDoNotTileAnAdt`.
> 14 Rosetta tests pass; solution builds clean; the 10 unrelated Core.Tests failures are unchanged.
> Still owed: US2 library builder, US3 lookup, US4 companion synthesis, legibility proof on a real
> render (user-owned).
>
> **Implementation checkpoint 5 (2026-08-26).** Stride-scatter reverted: tiles are assigned
> CONTIGUOUSLY row-major from the start tile (operator: "the tiles it generates are fucking
> random" — the full ~14k-asset corpus needs hundreds of tiles, so contiguous fill naturally
> covers the map; scattering was wrong). New mechanical alignment test
> (`Generate_PlacementsDecodeOntoTheirTiles`): every placement decoded with the viewer's exact
> rule `(M − rawY, M − rawX)` must land inside its tile's ChunkCorner-derived renderer bounds —
> object-on-tile alignment is now proven in CI, not by eyeballing. Smoke run (400 assets):
> 74 contiguous tiles (row 10 full, wrapping into row 11), 345 placements, 2,578 exclusions
> (mostly WMO group files failing MOHD read — expected). 8 focused tests pass.
>
> **Implementation checkpoint 4 (2026-08-26).** Output is now a self-consistent standalone map:
> everything lands under `{output}/World/Maps/{mapName}/` and every name derives from the SAME map
> name (`{mapName}_{tileY}_{tileX}.adt`, `{mapName}.wdt`, `{mapName}.wdl`) — folder, tile prefix,
> and WDT name can no longer disagree (a mismatched mix is refused with an error). Tiles are
> distributed across the WHOLE 64×64 grid on a stride grid (measured tile count → stride → spread
> coordinates), not filled sequentially. Smoke-verified against the real client root
> `C:\WoW4-data\WoW-12025`: 14,029 candidate assets, 60 placements across 7 tiles at
> (10,10)…(60,10) stride-25, WDT parses via `map inspect` (7/4096 tiles). 7 focused tests pass.
> Note: the "horizontal objects vs vertical ADT strip" observation came from outputs generated by
> the pre-fix binary (old `{tileX}_{tileY}` naming) — under that naming the viewer transposes every
> tile, which is exactly the perpendicular offset observed.
>
> **Implementation checkpoint 3 (2026-08-26, after second real run).** Root cause of "no objects
> load" + the tile strip: the viewer loads files named `{map}_{tileY}_{tileX}.adt` — column first,
> row second ([StandardTerrainAdapter.LoadMapTile](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs)) —
> while the generator wrote `{tileX}_{tileY}`. Every tile therefore loaded transposed: terrain
> rendered as a strip and every object landed on a tile coordinate that does not exist. Fixed:
> file naming, occupied-tile parsing, and the WDT/WDL tuple convention (MAIN is read at
> `tileX*64+tileY`, so tuples pass transposed). Also: layout now spreads row-major across the whole
> 64×64 grid instead of a single strip; labels shrink to fit their cell (overflowing labels were
> the "nonsense" text); MMDX/MWMO names normalized to backslashes. 7 focused tests pass.
>
> **Implementation checkpoint 2 (2026-08-26, after first real run).** Three defects from the first
> real-client run are fixed and tested:
> 1. **Placements invisible (~95%)** — raw MDDF/MODF coords were written unswapped; the viewer
>    decodes `(rawX, rawY)` as renderer `(MapOrigin − rawY, MapOrigin − rawX)`. Generator now writes
>    `rawX = tileY*T + u`, `rawY = tileX*T + v` so objects land on their tile's terrain.
> 2. **MCCV text illegible** — the painter assumed a row-major 17×17 vertex grid; the actual layout
>    is interleaved 9-8-9-8 rows (outer corners + inner cell centers). Painter now mirrors the mesh
>    builder's exact mapping.
> 3. **LkAdtWriter MCRF size bug** — declared size omitted the two count fields (`4+…` → `8+…`),
>    desyncing any chunk that carried placement refs. Fixed; chunks now also carry per-chunk MCRF
>    refs (placement indices) so chunk-admitted pipelines see every object.
> Output is now a **standalone map**: `{map}.wdt` (via `LkWdtWriter`, MCCV flag on), flat
> `{map}.wdl` (via `WdlWriter`), optional `--pm4-dir` copy of PM4 guides beside the tiles. Existing
> files are never overwritten (layout-time skip + write-time refusal). 5 focused tests pass;
> solution builds clean (9 pre-existing unrelated Core.Tests failures + the documented
> Pm4RegionObjectGrouperTests corpus failure remain). Still owed: US2 library builder, US3 lookup,
> US4 companion synthesis, legibility proof on a real render (user-owned).

**Consumed by**: [176](../176-object-transfer/spec.md) (reconciliation matching authority),
related evidence lanes [184](../184-pm4-generation-from-geometry/spec.md) and
[185](../185-pm4-pd4-format-documentation/spec.md). Tile-creation mechanics build on
[177](../177-adt-tile-creation/spec.md) prior art but this feature is an offline pipeline, not an
editor workflow.

## Problem

PM4 object identification today is per-tile heuristic scoring (`Pm4AssetMatchScorer`): each decoded
PM4 segment is ranked against whatever placements happen to exist in the paired ADT corpus, with a
score floor and ambiguity window. Consequences:

1. **Ground truth is scarce and uneven.** The current object library maps 904 PM4 objects to 243
   source assets — a fraction of what exists. Tiles whose companion ADT is missing or thin produce
   weak or empty candidate sets.
2. **Tiles without a companion `_obj0.adt` are skipped entirely** rather than processed, so those
   PM4 files contribute nothing.
3. **Scores are relative, not absolute.** A "best candidate" among three bad candidates still looks
   like a match. There is no complete, labelled reference to say what each asset actually looks like
   to the pipeline.

Meanwhile the configured client contains every placeable object in the game. That is a complete
labelled corpus waiting to be built — we control both the writer and the reader, so we can construct
data whose ground truth is perfect by construction.

## Solution Concept

Build a **Rosetta map**: a synthetic tileset in which every placeable game object is placed exactly
once at a deterministic grid position, with a manifest recording cell → asset identity. Write it with
our own placement authoring/writers, read it back with our own readers, and run the **same**
segmentation/signature pipeline used on real PM4 data over the result. Each asset's signature is then
known perfectly. Real PM4 objects are identified by looking up their signature in this complete
reference library — a deterministic comparison against total coverage, not a per-tile popularity
contest.

This is not an invented shape — it mirrors Blizzard's own level-design practice. Official development
files contain **designkit** maps: grids of objects laid out across multiple ADT tiles with each
object's name written below it in the world. No visible grid lines — the spacing is implicit — and
the layout freely crosses tile boundaries, because the underlying map is just a canvas to paint on.
Designers never need to know how ADTs work underneath. The Rosetta map is a **regenerated
designkit**: the same continuous labelled canvas — uniform invisible spacing, tile boundaries treated
as irrelevant — rebuilt by us so that instead of in-world name labels the identity lives in a
machine-readable manifest. Harvesting the shipped kit maps is explicitly not relied upon; we
regenerate the tiles ourselves so layout, labels, and coverage stay fully under pipeline control.

The pipeline must be **offline-only**: our writers emit the tiles, our readers decode them back, and
no real client ever loads the Rosetta map.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate the Rosetta tileset with perfect labels (Priority: P1)

The operator points the tool at a configured client root. The tool enumerates every placeable object
(model and world-model) the client offers, lays them out one-per-cell across as many synthetic tiles
as needed at fixed spacing, writes the tiles plus a manifest mapping every grid cell to its asset
identity, and verifies by reading the tiles back that every placement survived the round trip with
its identity intact.

**Why this priority**: Nothing else in the feature exists without the labelled corpus. It delivers
standalone value immediately: a complete inventory of placeable assets with known geometry.

**Independent Test**: Run generation against a configured client; read every emitted tile back;
confirm the number of recovered placements equals the number of manifest entries and each recovered
placement resolves to the manifest's asset for its cell.

**Acceptance Scenarios**:

1. **Given** a configured client root, **When** generation runs, **Then** every enumerable placeable
   object appears exactly once in the synthetic tileset or appears on an explicit exclusion report
   naming the reason it could not be placed.
2. **Given** the written tileset, **When** it is decoded with the project's own readers, **Then**
   100% of written placements are recovered and each maps back to its manifest entry.
3. **Given** two runs over the same inputs, **When** generation repeats, **Then** the outputs are
   byte-stable or differ only in recorded timestamps — layout is deterministic.
4. **Given** an object too large for one grid cell, **When** it is laid out, **Then** the layout
   accounts for its footprint so no two objects' footprints overlap, and the manifest records the
   cells it spans.

### User Story 2 - Build the labelled reference library (Priority: P1)

The decoded Rosetta tiles are pushed through the same object-segmentation and signature pipeline that
processes real PM4 data. The output is a reference library: one entry per source asset, holding that
asset's measured signature(s), keyed by the manifest's identity.

**Why this priority**: This is the piece that converts "synthetic map" into "ground truth". It shares
P1 because without it the corpus is inert.

**Independent Test**: Take any N assets from the library, re-run their Rosetta segments through the
matcher, and confirm each resolves to its own identity — the library self-test.

**Acceptance Scenarios**:

1. **Given** the decoded Rosetta tileset, **When** the standard segmentation pipeline runs, **Then**
   every produced segment carries the manifest identity of the cell it came from.
2. **Given** the segmented corpus, **When** the library is built, **Then** it covers every asset that
   produced at least one segment and reports assets that produced none.
3. **Given** the completed library, **When** the self-test runs (match Rosetta segments against the
   library), **Then** top-1 identification accuracy is effectively perfect (≥99%); any miss is
   reported as a library defect with the offending pair named.

### User Story 3 - Deterministic PM4 identification by lookup (Priority: P2)

A real PM4 file is decoded and segmented as today. Instead of scoring candidates from whatever the
paired ADT happens to contain, each segment's signature is looked up in the reference library. The
result is one of: identified (with the matched asset and the comparison evidence), ambiguous
(competing near-equal references, all named), or no-reference (nothing in the library resembles it —
which is itself valuable, flagging either an unenumerated asset or a decode defect).

**Why this priority**: This is the payoff, but it depends on Stories 1–2 and must be proven against
the real corpus before it becomes the authority.

**Independent Test**: Run lookup over the existing measured PM4 corpus; compare identified/matched
rates against the current scorer baseline (904 objects / 243 assets).

**Acceptance Scenarios**:

1. **Given** a real PM4 object whose true asset is in the library, **When** lookup runs, **Then** it
   identifies the asset or reports ambiguity with the true asset among the named competitors — never
   a confident wrong answer with the truth absent from the candidate list.
2. **Given** the full measured corpus, **When** lookup runs, **Then** the identified fraction meets
   or exceeds the current scorer's, and every result carries comparable evidence (which signals
   agreed, which disagreed).
3. **Given** a segment matching nothing in the library, **When** lookup completes, **Then** the
   result is an explicit no-reference status, not a low-score best guess.
4. **Given** both lookup and legacy scorer available, **When** they disagree on an object, **Then**
   the disagreement is surfaced in the report rather than silently resolved.

### User Story 4 - Synthesize companion ADTs for ADT-less PM4 tiles (Priority: P2)

For every PM4 tile that has no companion ADT, the pipeline synthesizes a minimal companion (empty or
flat terrain, correct era form, registered in the map's tile index) so downstream steps stop skipping
the tile. The synthesis report names every synthesized file and the tile it was created for.

**Why this priority**: It removes a whole class of silently-missing coverage and is independent of
the matching mechanism.

**Independent Test**: Enumerate PM4 tiles lacking companions before and after; after synthesis the
before-set is fully covered by the synthesis report and zero tiles are skipped for a missing
companion.

**Acceptance Scenarios**:

1. **Given** a PM4 tile with no companion ADT, **When** the pipeline runs, **Then** a valid companion
   is produced in the output directory in the correct era form and the tile proceeds through the
   normal pipeline.
2. **Given** a PM4 tile whose companion already exists, **When** the pipeline runs, **Then** the
   existing file is used and nothing is overwritten.
3. **Given** synthesis of a companion, **When** complete, **Then** the run report lists it with its
   source tile and content hash so synthesized data is always distinguishable from real data.

### Edge Cases

- More placeable objects than fit one tile → layout spans multiple tiles; manifest remains complete.
- The same underlying asset reachable under multiple names/paths → deduplicated by identity with all
  aliases recorded, or placed per-alias if the pipeline treats them as distinct; decision recorded in
  the manifest.
- Assets that fail to load or contain no usable geometry → exclusion report, never silent omission.
- Name-table capacity limits in the target era's format when a tile holds many distinct assets →
  layout spreads distinct names across tiles within format limits.
- A real PM4 object composed of multiple sub-objects spanning cells → segmentation must handle
  multi-cell footprints via the manifest's span records.
- Synthesized companions must never be mistaken for authentic data in later analysis → provenance is
  machine-readable and travels with the output.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST enumerate every placeable object available from the configured client
  root at runtime (never a hardcoded path), and record the enumeration (counts, sources, exclusions)
  in the run report.
- **FR-002**: The system MUST generate synthetic map tiles placing enumerated objects at
  deterministic grid positions with non-overlapping footprints, sized to each object.
- **FR-003**: The system MUST emit a machine-readable manifest mapping every grid cell (and
  multi-cell span) to its asset identity, plus the generation parameters needed to reproduce it.
- **FR-004**: Generation MUST use only existing project writers and readers for emission and
  round-trip verification; no new format serializer may be introduced.
- **FR-005**: The system MUST verify the write→read round trip for every generated placement and fail
  the run if any placement fails to recover with its identity intact.
- **FR-006**: The reference library MUST be built by running the same segmentation/signature pipeline
  used on real PM4 data over the decoded Rosetta tiles — never a separate simplified path.
- **FR-007**: Real PM4 identification MUST be performed as lookup against the reference library,
  returning identified / ambiguous / no-reference statuses with comparison evidence; a ranked-list
  score alone MUST NOT be presented as identification.
- **FR-008**: The legacy per-tile scorer MUST be demoted to a secondary signal used for tie-breaking
  and disagreement reporting; it MUST remain available for diagnostics.
- **FR-009**: The system MUST synthesize companion ADTs for PM4 tiles that lack one, in the correct
  era form, into the output directory, leaving existing companions untouched.
- **FR-010**: All synthesized files MUST be listed in a provenance report (source tile, parameters,
  content hash) distinguishing them from authentic data.
- **FR-011**: All outputs go to the configured output directory; no game install or Blizzard
  container is ever written.
- **FR-012**: Regeneration over unchanged inputs MUST be idempotent (stable layout and library
  content), so the library can be cached and version-checked rather than rebuilt blindly.
- **FR-013**: Synthetic layout MUST follow the designkit convention: one continuous canvas with
  uniform invisible spacing that crosses tile boundaries freely — no per-tile alignment or padding is
  introduced to respect file boundaries, because the manifest, not the terrain, carries identity.

### Key Entities

- **RosettaManifest**: The authoritative cell → asset mapping for a generated tileset, including
  layout parameters, multi-cell spans, aliases, and exclusions.
- **ReferenceLibrary**: The set of per-asset signatures derived from the decoded Rosetta tiles via the
  standard pipeline, keyed by manifest identity, versioned against pipeline and manifest versions.
- **IdentificationResult**: Per real-PM4-segment outcome — matched asset + evidence, ambiguous with
  named competitors, or no-reference — plus any disagreement with the legacy scorer.
- **CompanionSynthesisReport**: Record of every synthesized companion ADT: source tile, era form,
  content hash.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Round-trip recovery is 100%: every written Rosetta placement is read back and resolves
  to its manifest identity; any failure aborts the run.
- **SC-002**: Library self-test top-1 accuracy is ≥99% on Rosetta-derived segments; every miss is
  explained and tracked as a defect.
- **SC-003**: On the existing measured PM4 corpus, lookup identifies a strictly larger fraction of
  objects than the current baseline (904 objects mapped to 243 assets), with zero confident-wrong
  results where the true asset was absent from candidates.
- **SC-004**: After companion synthesis, zero PM4 tiles are skipped due to a missing companion ADT,
  and every synthesized file appears in the provenance report.
- **SC-005**: Reference-library coverage equals the enumeration minus reported exclusions — the
  operator can see, as a number, how much of the game's object space the matcher can possibly name.
- **SC-006**: Two consecutive generations over identical inputs produce identical manifests and
  libraries (modulo recorded timestamps).

## Assumptions

- **Offline-only** (operator decision): the Rosetta map is never required to load in a real client;
  project writers/readers are both halves of the round trip.
- Target era follows the project's active Alpha 0.5.3 focus; other eras are out of scope until the
  Alpha pipeline is proven.
- "Every object" means every placeable model/world-model the client enumeration exposes; non-placeable
  or unloadable entries land on the exclusion report rather than blocking generation.
- Existing placement-authoring and tile-writing seams (per Specs 175/177) are sufficient; if a gap is
  found it is raised as a bounded extension to the owning owner, not a new serializer here.
- The legacy scorer's saved choices and corpus signals remain readable evidence (Spec 176 decision 2);
  this feature changes which mechanism is *authoritative*, not the historical record.

## Out of Scope

- Loading the Rosetta map in a real client or harvesting client-side renders of it.
- Cross-era transfer mechanics (Spec 176 P1) beyond what lookup naturally enables.
- New ADT/WDT serializers; terrain synthesis beyond flat/minimal companion tiles.
- Editing the frozen Alpha WDT writer without an explicitly reopened decision.
