# ACTIVE CONTEXT — V14 Branch (V11 Reset)

## BRANCH
`v0.5.0-dev` branched forward from the `v0.4.9-strict-guards` line on 2026-05-14.

## wow-viewer Library Completeness / Harvest Status — Resynced 2026-05-11

Phase A (terrain type system) and Phase B (harvest pipeline) are COMPLETE.
Phase C (Converters): AlphaToLk is real-data validated at 100% tile conversion across 4 maps. LkToAlpha now has **real-data MdxViewer validation** of 4.0.0 Azeroth (839/839 tiles) against staged 0.5.3 client, with asset filtering, tileset bundling, and the current alphaWDT placement/culling contract documented below.

### Landed In The May 9 Session (Phase C Alpha WDT Writer Fixup + MdxViewer Validation)

**AlphaWdtWriter structural fixes (critical):**
- MAIN grid order confirmed as Alpha client row-major (`tileY * 64 + tileX`). Ghidra `CMap::PrepareArea(x,y)` computes `index = y * 64 + x` after `CMap::LoadWdt` reads raw `MAIN` into `areaInfo`.
- Empty MCNK removal eliminated: removed `BuildEmptyMcnk`. Legacy `McnkAlpha` always reads 256 MCNK per tile; all 256 MCNKs are now emitted with full subchunk payloads.
- MCRF always emitted: legacy `McnkAlpha` unconditionally reads MCRF data. Previously omitted when empty.
- MDDF/MODF always emitted per embedded tile, even with empty payloads: 0.5.3 `CMapArea::Create` unconditionally asserts both chunk headers.
- MCLY/MCAL offsets always populated (not conditionally zeroed).
- `chunkBaseHeight` mirrored at both MCNK header offsets `0x68` and `0x6C` for compatibility across reader code paths.

### Reverted May 10 — MCNR/MCRF/Liquid Commits

**Commits `8bcb7045` and `d52bda9b` were reverted** (back to `47cbb435`). Both broke 0.5.3 client rendering:
1. `8bcb7045` — Removed MCRF FourCC wrapper (raw uint32). Ghidra confirms raw uint32 is what the client reads, but FourCC-wrapped currently works; raw-uint32 broke the client.
2. `d52bda9b` — Changed MCNR from `(X,Z,Y)` to Ghidra-verified `(-Y,Z,-X)` format and liquid flags from `0x3C` mask to `ClassifyAlphaLiquidType` switch. Both changes broke rendering.

**Current working format (matches `47cbb435`):**
- MCNR: `(X, Z, Y)` per byte — `byte[0]=X, byte[1]=Z, byte[2]=Y`
- MCRF: FourCC-wrapped chunk (`MCRF` + size + data)
- Liquid flags: `McnkFlags & 0x3C`

**Ghidra-verified MDDF/MODF transforms** (these are correct and were NOT reverted):
- Position: `file = (MapOrigin - world.Y, world.Z, MapOrigin - world.X)`
- Rotation: `file = (world.pitch, world.yaw - 180°, world.roll)`, yaw offset = π (confirmed `_DAT_00810e04 = 3.14159274`)
- Bounds: `file.t = (MapOrigin - world.min.Y, world.max.Z, MapOrigin - world.min.X)`, `file.b = (MapOrigin - world.max.Y, world.min.Z, MapOrigin - world.max.X)`

**Current alphaWDT placement contract (2026-05-11):**
- `AlphaWdtReader`, `AlphaWdtWriter`, and `LkAdtWriter` now share the same round-trip-safe raw rotation convention: `Rotation = (fileRotX, fileRotZ, fileRotY)`. Do not subtract 180 degrees on write; the client applies the yaw `+π` at load time.
- Alpha doodads stay single-owner in `MCRF` to avoid the native purge assert. The writer now chooses the chunk containing the doodad anchor by default; preserved LK source refs only win when they stay inside that chunk's local `3x3` neighborhood.
- Alpha WMOs keep overlap-based multi-chunk refs derived from bounds, with containing-chunk fallback only when no bounds overlap exists.
- Alpha `MCNK` payloads must stay below the client-side limit `chunkInfo->size < 15000` confirmed in the 0.5.3 binary (`MapChunk.cpp` assert string). `AlphaWdtWriter` now trims only duplicate non-anchor WMO refs when a chunk would exceed that ceiling and throws if a chunk still cannot be brought under budget.
- Target-client asset matching must be built from the actual target archives, Alpha wrapper scan, and loose files only. Do not treat external listfiles as proof that a 0.5.3 asset exists.
- Top-level alphaWDT chunks are contiguous. Do not odd-byte pad `MDNM`, `MONM`, or other top-level chunks between headers.

**Ownership rule:** alphaWDT file semantics now live in `wow-viewer` shared I/O (`AlphaWdtReader`, `AlphaWdtWriter`, `AlphaTerrainAdapter`, `AlphaToLkConverter`, `LkToAlphaConverter`). Future `MdxViewer` compatibility work must consume those shared contracts instead of adding another legacy alpha parser or writer.

**AlphaWdtReader fixes:**
- `ReadExistingTiles`: MAIN index `x = i % 64, y = i / 64`.
- `TryReadTile`: `mainEntryIndex = tileY * 64 + tileX`.

**Placement orientation confirmed in Ghidra:**
- 0.5.3 `CMap::CreateDoodadDef(SMDoodadDef&, C3Vector&)` and `CMap::CreateMapObjDef(SMMapObjDef&, C3Vector&)` convert file position `(x,y,z)` to world/internal `(origin - z, origin - x, y)`.
- The same functions convert file rotation `(x,y,z)` to applied Euler axes `(z, x, y + 180deg)` before rotating Z, then Y, then X.
- `AlphaWdtWriter` position/rotation and MODF bounds encoding matches the reader/client inverse. `LkAdtWriter` MODF bounds had the same axis inversion risk and is now covered by a round-trip regression.

**LkAdtWriter FourCC fix:**
- All chunk IDs (`MCNK`, `MCVT`, `MCNR`, `MCLY`, `MCAL`, `MCRF`, `MCSH`, `MCCV`, `MCLV`, top-level `MVER`/`MHDR`/`MCIN`) switched from `Encoding.ASCII.GetBytes()` to `FourCC.FromString().ToFileBytes()` to match repository-wide I/O boundary convention. This fixed the return-path test where `MapFileSummaryReader` could not detect ADT files.

**Added `WriteAlphaWdt_UsesClientMainOrderAndMcnkSubchunkContract` test:**
- Validates MAIN ordering uses Alpha client row-major (`tileY * 64 + tileX`).
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

### Landed In The May 11 Session (Object Converter Slice)

**WMO v14 <-> v17 converters:**
- `wow-viewer` now owns both shared WMO directions: `WmoV17ToV14Converter` and `WmoV14ToV17Converter` live in `WowViewer.Core.IO/Wmo` with focused tests and converter CLI wiring.
- The downgrade path is not a raw copy. It rebuilds legacy `MOHD`/`MOGI`, downgrades root and group chunk layouts, and handles the Alpha 0.5.3-era practical ceiling of `384` groups through spatial bucket merging rather than blunt overflow collapse.
- Portal refs are remapped into the merged legacy group index space so converted roots stay structurally consistent after overflow merging.

**M2 -> MDX minimal downgrade lane:**
- `wow-viewer` now owns a first minimal shared `M2ToMdxConverter` in `WowViewer.Core.IO/M2`.
- The current proof boundary is explicit: geometry, texture names, materials, sequences, bones, pivots, and skin-fed index data are mapped into structurally valid classic `MDX` output and validated by re-reading the produced bytes through existing MDX readers.
- This is not full parity yet. `convert-m2-to-mdx` is now wired into `WowViewer.Tool.Converter` and validated with a focused converter build, but broad real-data downgrade proof is still open and `MdxToM2` had not started.

**MDX -> M2 minimal upgrade lane:**
- `wow-viewer` now owns a first minimal shared `MdxToM2Converter` in `WowViewer.Core.IO/M2`.
- The current proof boundary is explicit: a strict `MD20` root plus one `00.skin` companion are emitted from classic `MDX` using the current minimal shared subset for geometry, sequences, textures, material flags/blend mode, and skin-fed submesh or batch tables.
- The lane is validated by re-reading the produced root with `M2GeometryReader` and the companion skin with `M2SkinReader`, and `convert-mdx-to-m2` is now wired into `WowViewer.Tool.Converter`.
- This is still a minimal structure lane, not a full parity pass for MDX animation tracks, helpers, lights, ribbons, particles, or deep material semantics.

**New CLI flags for `convert-lk-to-alpha`:**
- `--target-client-root <dir>` / `-tcr` — filter placements against target client
- `--terrain-only` / `-to` — strip all placements (crash-proof validation)
- `--bundle-tilesets` / `-bt` — extract textures alongside output
- `--bundle-m2s` / `-bm` — bundle doodad models as local MDX outputs, rewrite placement paths, and rewrite bundled MDX TEXS paths to colocated local BLPs
- `--limit <N>` / `-n` — limit tile count for testing

### Architecture Notes for LkToAlpha
- Legacy reference readers: `gillijimproject-csharp/WowFiles/Alpha/{WdtAlpha,AdtAlpha,McnkAlpha}.cs`
- `McnkAlphaHeader` struct has specific subchunk offset fields that must point into the MCNK data area (after 8-byte chunk header + 128-byte MCNK header)
- MCVT/MCNR are raw (no chunk headers); MCLY/MCRF/MCSH/MCAL/MCLQ are chunked (FourCC + size + data)
- When `McnkAlpha` reads a subchunk with `new Chunk(file, offset)`, the offset must point to the chunk's FourCC, not its data
- Row-major MAIN indexing: mainEntryIndex = tileY * 64 + tileX
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
- Forward `AlphaToLk` AreaID crosswalk wiring (reverse `LkToAlpha` mapping landed; the forward lane is still open)
- LkToAlpha real-data batch validation beyond the new focused round-trip regressions
- Full Alpha/LK chunk-for-chunk preservation instead of reduced terrain-domain reconstruction
- `M2ToMdx` CLI wiring and broader real-data downgrade proof
- `MdxToM2` implementation and tests
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
2. Phase C (continued): AreaID crosswalk, LkToAlpha broader real-data validation, broader real-data proof for `M2ToMdx` and `MdxToM2`
3. Phase D: Deep format readers (WDT retail flags, WDL, WMO full version range, MDX, BLP pixel decode)
4. Phase E: DBC/DB2 metadata enrichment (AreaTable, WorldSafeLocs, LiquidType, GroundEffects)
5. Phase F: Placement provenance (MCRF per-chunk arrays, PM4 SQLite, prefab detection)

## TOP-LEVEL DIRECTION UPDATE — 2026-05-14

- The program framing has been widened from "viewer migration" to "engine modernization."
- The current simple working label for this broader engine program is `museum-explorer`.
- The new top-level architecture doc is `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md`.
- The editor/modding-tool follow-up doc is `wow-viewer/docs/architecture/wow-engine-editor-and-interop-plan-2026-05-14.md`.
- `wow-viewer/docs/architecture/game-viewer-host-plan-2026-05-13.md` remains active, but only as the current app-host/viewer sub-plan inside the broader engine direction.
- The active execution skeleton is now `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/README.md` plus the `GV-00` through `GV-26` micro-plans under that folder.
- The pack now has a corrected universal foundation: `GV-00-universal-content-contracts.md`, `GV-00A-artifact-provenance-and-preservation.md`, and a rewritten `GV-01-core-constants-registry.md` that explicitly forbids WoW-style assumptions in engine core constants.
- The pack now also carries explicit future-extraction/base-architecture stories: `GV-00B-base-engine-repo-extraction-boundary.md`, `GV-00C-base-solution-and-folder-topology.md`, and `GV-06A-profile-personality-library-contract.md`. The long-range shape is an extracted engine repo with a first-class `BASE` layer plus sibling profile/personality libraries such as WoW, Warcraft 3, and Museums.
- The plan pack now also makes `.NET 10` / `C#` the default engine-orchestration story via `GV-00D-dotnet-first-engine-and-orchestration-boundary.md`, with Python kept explicit as the ML/data side rather than the owner of engine-truth orchestration.
- The pack has also been widened for forward-native content and small-model execution: `GV-04A-forward-native-glb-metadata-profile.md`, `GV-09A-raw-artifact-capture-store.md`, `GV-10A-sidecar-metadata-schema-index.md`, `GV-11A-glb-import-adapter.md`, `GV-12A-forward-native-export-package.md`, `GV-19A-profile-schema-source-routing.md`, `GV-25-small-model-story-template.md`, and `GV-26-plan-pack-audit-matrix.md`.
- `Museums` is now an explicit named supported data family in the plan pack, centered on `GV-04A-museums-forward-native-profile.md`, `GV-04B-museums-object-package-contract.md`, `GV-04C-museums-shard-and-index-store.md`, and `GV-23A-distilled-portable-model-packages.md`.
- Audio is now treated as a first-class engine subsystem in the plan pack and higher-level docs, anchored by `audio-engine-plan-2026-04-21.md` plus the micro-plans `GV-14A-audio-system-foundation.md`, `GV-14B-profile-audio-resolution-contracts.md`, `GV-14C-runtime-audio-scene-and-mixer.md`, and `GV-17A-audio-backend-bridge.md`.
- The audio lane now explicitly calls out the required asset families and their complexity split: decoded `wav`/`ogg`/`flac` are separate from `midi` sequencing plus instrument-bank families `SFP0` and `DLS`, with new micro-plans `GV-14D-audio-asset-family-support-matrix.md` and `GV-17B-midi-synth-and-instrument-bank-bridge.md`.
- Web delivery is now explicit as a component surface rather than a replacement backend: `GV-17C-webgl-component-and-web-delivery-surface.md` treats WebGL as a browser/embed output seam alongside the Vulkan-first and OpenGL-fallback native renderer story.
- Backend direction is now explicit: Vulkan first, OpenGL fallback.
- Future planning should treat `WowViewer.App` as the current `game-viewer` host product rather than the full identity of the program.
- Future app-shell planning should start from import/export-first UX, game/data-root management, metadata-driven feature gating, DBC/DB2 editing direction, and compatibility profiles rather than assuming a single-version viewer shell.
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs` is now treated as a protected complete surface on `v0.5.0-dev`; future edits should happen only when the user explicitly reopens alphaWDT writer work or when a focused validated regression fix is required.

**Full roadmap**: `wow-viewer/docs/architecture/wow-viewer-full-porting-roadmap.md`
**Current architecture**: library → dataset → trainer → CLI → viewer (bottom-up)

## Pending Merge

Commit `e5b149167bbc703e3f257566701fbc0f0380941a` (`fix: bind MdxViewer to legacy M2ToMdxConverter`) on a sibling branch resolves circular dependencies that broke cross-platform GitHub Actions compilation. Three files changed in `gillijimproject_refactor/src/MdxViewer/`: `WmoRenderer.cs`, `WorldAssetManager.cs`, `ViewerApp.cs`. Needs to be merged into `v0.5.0-dev` at the next convenient integration point. Secondary to the harvest/training lane.

## Active Development — 2026-05-15

### V15 Terrain Model (CURRENT FOCUS)

**Status**: Spec written, code ready, training not yet started.

**What**: Single model that takes a 256×256 minimap and predicts a full terrain ADT patch:
- Height (257×257), normals (257×257×3), alpha (256×256×4), holes (16×16), MCLY texture IDs (16×16×4)

**Architecture**: ConvNeXt V2 Nano encoder (15.6M) + U-Net decoder, 27.4M params total.
**No priors at inference** — just the minimap image. All supervision from NPZ shard signals.

**Files**:
- Spec: `wow-viewer/docs/architecture/v15-terrain-model-spec-2026-05-15.md`
- Model: `wow-viewer/data-harvester/src/harvester/v15_model.py`
- Dataset: `wow-viewer/data-harvester/src/harvester/v15_dataset.py`
- Training: `wow-viewer/data-harvester/scripts/train_v15.py`
- Test/inference: `wow-viewer/data-harvester/scripts/test_v15.py`

**Training command**:
```
cd wow-viewer/data-harvester
uv run python scripts/train_v15.py --device cuda --epochs 200 --batch-size 8 --val-interval 10 --max-samples 1000
```
Outputs: checkpoints at `checkpoints/v15_best.pt`, validation snapshots at `logs/val_epoch<NNNN>/tile_XX/`, training log at `logs/v15_training_log.json`.

**Dependencies**: `pyproject.toml` has CUDA torch pinned via `[[tool.uv.index]]` → PyTorch cu130. `triton-windows>=3.7` for `torch.compile`.

### C# Pipeline Fixes (Landed)

Three fixes applied to the harvest tool to eliminate signal dropouts:

| File | Fix |
|------|-----|
| `AdtTensorPackBuilder.cs:454` | When `_tex0.adt` missing, fall back to root ADT for pre-Cata builds (fixes 3.0.1) |
| `Program.cs:470` | Auto-detect `buildVersion` from client root dir name (fixes 4.0.0 profile) |
| `AlphaWdtReader.cs:527` | Remove MCNR re-normalization; clamp only. Matches `DecodeNormalComponent` |
| `Program.cs` | Added `DecodeBlpToRgbNative` and texture swatch extraction to NPZ shards |

### D1 Model (Attempted, Not Converged)

Small U-Net tileset decomposition. Trained but outputs not useful — BCE loss collapsed alphas to zero. Switched to L1 but user assessed outputs as insufficient. Deferred in favor of V15.

### Dataset State

- Re-harvested corpus at `output/datasets/d1_reharvest/shards/` — 6 builds, 23K+ shards
- 16,079 D1/V15-eligible training shards, 66 validation
- Texture swatches extracted via C# fix (requires re-harvest to populate)
- `object_mask_257` used for loss weighting (object pixels excluded)

### Python Environment

Managed by `uv` in `wow-viewer/data-harvester/`. Key deps:
- `torch==2.12.0+cu130` (CUDA 13.0)
- `timm>=1.0` (ConvNeXt backbone)
- `zarr>=2.0`, `numcodecs>=0.13`
- `triton-windows>=3.7` (for `torch.compile`)
