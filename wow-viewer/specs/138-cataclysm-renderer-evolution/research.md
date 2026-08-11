# Spec 138 Research — Archive Adapters and Cross-Era Terrain Inputs

**Date**: 2026-08-08  
**Status**: Resolved for planning; implementation still pending

## Decision 1 — Keep archive access behind the existing C# contract

**Decision**: Extend the existing `IArchiveReader`/`IArchiveCatalog` boundary in
`WowViewer.Core.IO.Files`. The terrain readers and renderer consume virtual files and profile
capabilities, never a CASC or MPQ library directly.

**Rationale**: `NativeMpqService` already owns MPQ discovery, Alpha wrapper scanning, listfiles,
and virtual-path reads. Replacing it or teaching terrain code about archive formats would violate
the library-first and one-owner rules. CASC support should be another catalog implementation or
an adapter at the same seam.

**Alternatives considered**:

- Put CASC calls in `WowViewer.Tool.Harvest`: rejected because the C# harvester would become a
  second format owner and the viewer could not reuse the source path.
- Make Python/pyCASCLib the primary reader: rejected because the viewer and C# tools already own
  the format-read path, and the Python wrapper is explicitly incomplete.

## Decision 2 — Use a capability-selected CASC adapter matrix

**Decision**: Start with three source families rather than one universal CASC dependency:

1. `NativeMpqService` for loose/MPQ-era clients, including the 0.x–5.x legacy line.
2. A narrow CascLib adapter for early CASC compatibility, beginning with 6.x fixtures.
3. A TACTSharp adapter candidate for later CASC/local/CDN sources when its build probe passes.

**Evidence**:

- [CascLib](https://github.com/ladislav-zezula/CascLib) describes itself as an open-source reader
  for Blizzard CASC storages since 2014 and provides CMake and Windows build paths.
- The [WoW-Tools/CascLib](https://github.com/WoW-Tools/CascLib) fork is already recorded as a
  submodule in the local Marlamin reference checkout.
- [TACTSharp](https://github.com/wowdev/TACTSharp) uses memory-mapped access, supports local and
  online installations plus CDN-structured folders, and can resolve by name, ID, ekey, or ckey.
  Its README still lists encrypted products, install-tag priority, all-6.0+-build testing, and
  automated tests as unfinished or future work.
- The local `libs/Marlamin/WoWTools.Minimaps/.gitmodules` records `wowdev/TACT.NET`,
  `Marlamin/Warcraft.NET`, and `WoW-Tools/CascLib` as the reference dependency family, but those
  submodule directories are empty in this checkout and are not currently integrated.

**Rationale**: The reader chosen for a build must be the one proven against that build. A profile
can prefer TACTSharp for a later build while retaining CascLib as a compatibility fallback, but a
failed probe must be visible and fail closed rather than silently changing provenance.

**Alternatives considered**:

- TACTSharp everywhere: rejected until its stated coverage and encrypted-product gaps are tested.
- CascLib everywhere: rejected as the only long-term managed boundary because the project already
  has a C# TACTSharp reference and later source modes may benefit from its online/CDN support.
- TACT.Net in the shipped viewer: rejected for now because its GitHub repository is GPL-3.0 and
  it is a distribution/repository library, not the required terrain file catalog by itself.

## Decision 3 — Keep modern tools as comparative authorities

**Decision**: Reuse the already-integrated DBCD, WoWDBDefs, and wow-listfile authorities for
database/build/listfile metadata. Use [Marlamin/WoWTools.Minimaps](https://github.com/Marlamin/WoWTools.Minimaps)
and [Kruithne/wow.export](https://github.com/Kruithne/wow.export) as reference and validation
authorities. Do not add a runtime dependency on wow.export or copy its tool ownership into the
viewer.

**Evidence**:

- WoWTools.Minimaps supports local or streamed extraction and documents both modern 512-pixel
  minimaps and older 256-pixel inputs.
- wow.export supports Retail and Classic, online streaming, legacy MPQ browsing, terrain/texture/
  object overhead-map export, and modern M2/WMO preview.
- [wowdev/wow-listfile](https://github.com/wowdev/wow-listfile) distinguishes community names from
  verified names and warns that many modern names are mutable.
- [wowdev/WoWDBDefs](https://github.com/wowdev/WoWDBDefs) provides machine- and human-readable
  database definitions beginning with broad 7.3.5 coverage and ongoing older coverage.
- [wowdev/pywowlib](https://github.com/wowdev/pywowlib) is useful format evidence across 3.3.5a,
  4.3.4, 5.4.8, 6.2.4, 7.3.5, 8.3.5, and 9.0.0, but its CASC path is a read-only pyCASCLib
  wrapper and is not the C# runtime boundary.
- The current project already references DBCD and bundles WoWDBDefs definitions in
  `WowViewer.Core.IO` and the viewer projects; it also already locates and downloads wowdev
  listfiles. Those are existing authorities, not missing implementation work for this epic.

## Decision 4 — Provenance is part of source selection

Every selected adapter must emit a common provenance record containing the configured source mode,
client root or remote product, build identity, locale/install-tag selection, adapter and dependency
versions, listfile source, capability probe results, and a content probe hash. The terrain pipeline
must reject an unproven profile instead of falling back to a different reader or treating a missing
file as a format feature.

## Resolved planning unknowns

| Unknown | Resolution |
|---|---|
| Where does 0.5.3–5.x source access live? | Existing MPQ/loose-file path; CASC is not used for the original MPQ line |
| What is the first CASC compatibility anchor? | CascLib, with a real 6.x local fixture |
| What is the later-CASC candidate? | TACTSharp, selected only after per-build probes |
| What is TACT.Net’s role? | Isolated acquisition/reference tooling only until GPL and ownership review are complete |
| What does wow.export contribute? | Modern extraction and terrain behavior comparison, not a runtime library |
| What metadata is shared? | Existing DBCD/WoWDBDefs/listfile authorities plus new build/profile provenance |

## 0.5.3 Dataset-Generation Audit — Ghidra WoWClient.exe 0.5.3.3368

**Date**: 2026-08-09  
**Program**: `WoWClient.exe`, x86, image base `0x00400000`, configured source root
`H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft`  
**Status**: Read-only binary audit. No harvest, training run, or heavy build was started.

The binary separates three things that the current v60 workstream had begun to treat as one
signal: authored minimap BLP tiles, terrain-renderer lighting/shadows, and object/icon overlays.
That distinction is now a dataset gate.

### Confirmed native contracts

| Native evidence | Finding | Current v60 consequence |
|---|---|---|
| `BuildPathName` `0x006c2514`; `SetupTextureHandles` `0x006c23d8`; strings `0x008a4ea8`, `0x008a4eb8` | Runtime minimap terrain is loaded as `Textures\\Minimap\\<map>\\map%d_%d.blp` tiles. | The current `textures/minimap/{map}/map{x}_{y}.blp` lookup is correct. Do not “fix” it into an MCSH or terrain-render path. |
| `CMap::PrepareArea` `0x00684a4a`; `CMap::PrepareChunk` `0x00684c1d` | WDT `MAIN` is row-major (`y * 64 + x`); chunks are addressed as area-local `y * 16 + x`. | The direct `AlphaWdtReader` tile lookup is correct. The shared `AlphaTerrainAdapter` `TileExists`/`ExistingTiles` mapping is transposed and can poison viewer-side validation. |
| `CMapChunk::Create` `0x00698e10`; `CreateVertices` `0x006997e0` | Alpha MCVT is 145 absolute Z samples: 9x9 outer plus interleaved 8x8 inner. No MCNK base-height addition is applied. | Current absolute-height and 257-grid construction are aligned with the client. |
| `CreateNormals` `0x00699b60` | MCNR bytes decode to client normal components in the order/signs `(-b2, -b0, +b1) / 127`. | The current raw-byte decode plus renderer-space transform is consistent; this is not a remaining normal-axis bug. |
| `CMap::LoadWdl` `0x0067fa8b` | External WDL is separately loaded, version `0x12`, with 545 `int16` heights per MARE (17x17 outer + 16x16 inner). | A WDL lattice derived from MCVT is a proxy, not native WDL supervision. It must be labelled as derived or replaced by an external-WDL reader. |
| `CMap::Load` `0x0067f898`; `LoadLightsAndFog` `0x006c4110`; `SetColors` `0x006bb5d0` | `lights.lit` is loaded after WDT and participates in day/night lighting. | The current synthetic shadow composition does not claim exact 0.5.3 lighting: it does not consume the native LIT table. |
| `CMapChunk::Create` and `CreateShadow` `0x00699fb0` | MCSH is a per-chunk packed 64x64 bitmask (512 bytes), activated by MCNK flag bit `0` and global terrain-shadow bit `0x40`; it is unpacked at a client-selected 64 or 32 mip. | A 1024-to-256 resample is a diagnostic representation of raw MCSH, not native minimap shading and not a native 256 target. |
| `CreateRefs` `0x0069a0c0`; `CMapArea::Create` `0x006aad50` | MCRF links chunk-local references to MDDF/MODF records. | Alpha object supervision currently ignores chunk-local references and paints placement heuristics, so it is auxiliary placement guidance, not a native screen-space minimap mask. |
| `MinimapUpdate` `0x006c08a0`; `CWorld::QueryMapObjMinimap` `0x00663740`; `RenderObjectBlips` `0x0052a030` | WMO minimap quads and dynamic object/POI icons are separate overlay paths. They are not baked into the terrain BLP. | Never describe MDDF/MODF masks as “the real minimap object mask” or use them to rewrite authored BLP pixels without a separate visibility contract. |
| `SetDirection` `0x006bca40` | 0.5.3 exposes a fixed native light azimuth of 225 degrees in its ray/direction convention, with explicit quarter-cycle phi values of about 127 and 110 degrees. | The existing numeric 45-degree source-bearing inverse may be usable, but its provenance is no longer “assumed from another build”; exact time interpolation remains open. |

### Blocking defects in the current 0.5.3 harvest path

1. **MCAL layer offsets are read and discarded.** `AlphaWdtReader.TryParseMcnk` reads each
   `MCLY.offsAlpha` but advances one shared `alphaSrcOffset` sequentially. The client stores and
   consumes each layer's own offset. Non-contiguous or reordered alpha blocks therefore attach the
   wrong alpha data to the wrong layer. This invalidates `mcal_alpha_pack_256`, the synthesized
   texture path, and any labels derived from it.
2. **Zero is still used as an implicit missing-height sentinel.** `FillHeightmapGaps` claims to
   use NaN but the array is initialized to zero and the fill loop tests `hm[y,x] == 0f`. Since
   native MCVT heights are absolute, sea-level and genuinely flat zero vertices are valid. The
   gap pass can overwrite real 0.0 values with neighbouring heights. A written/presence mask is
   required before gap filling.
3. **The harvested MCSH channel is semantically overclaimed.** The reader ignores the native
   MCNK shadow-enable condition, expands packed chunk masks to a 1024 grid, and the builder emits
   a 256 resample. That can remain as `mcsh_raw_resampled_diagnostic`, but it cannot be called
   `terrain_shadow_256` or compared directly with the authored minimap BLP.
4. **Harvest shadow composition is not profile-controlled.**
   `BuildHarvestShadowCompositionOptions` uses `SyntheticMinimapTuning.Default`, forces
   `ApplyCastShadows = true`, and does not consume `MinimapEraProfile.CastShadowsEnabled` or
   `lights.lit`. The Alpha profile says cast shadows are off, so current real-tile shadow output
   is a synthetic, target-derived renderer product rather than a 0.5.3 observation. The broad
   exception in `BuildEnrichedTensorPackForTile` also makes missing shadow output fail open.
5. **Alpha object masks lack native reference/visibility semantics.** The current MDDF/MODF masks
   use guessed circles/AABBs and a four-candidate projection chooser. They do not use MCRF links,
   object geometry, occlusion, or the separate WMO/dynamic minimap overlay paths. They must be
   retained only as explicitly named auxiliary placement labels until a screen-space mask contract
   is defined.

### Non-blocking but required before accepting a 0.5.3 transfer corpus

- Fix the shared `AlphaTerrainAdapter` `MAIN` index transpose before using the viewer as a visual
  validator; the direct harvester reader is already row-major.
- Read external WDL as its own signal and rename the MCVT-derived lattice so a model cannot mistake
  the proxy for native WDL data.
- Add a 0.5.3 lighting profile whose azimuth is marked binary-traced and whose elevation/time
  interpolation is explicitly approximate until LIT interpolation is recovered.
- Prove the MCLQ `SOVert` per-vertex field mapping. The current audit confirms the liquid flag
  family and the 162-entry vertex block, but not every byte field's semantic name.
- Make a missing required signal fatal in the validation contract. “Shadow unavailable” is valid
  for an explicitly diagnostic row; silently emitting a partial real row is not.

The first accepted 0.5.3 corpus therefore remains **not generated**. The safe first repair slice is
reader correctness (MCAL offsets, height presence, and explicit raw-MCSH provenance), followed by
object-label separation and only then native-light calibration.

## Dataset-construction correction — stale v50 synthesis (2026-08-09)

The active issue is simpler: the synthesized minimap arrays in the old v50 datastore were not
regenerated after the renderer's lighting fixes. The raw harvested terrain is not the thing to
re-diagnose here, and the 0.5.3 renderer remains the known-good control.

The old v50 builder coupled synthesis to a fresh client-backed build, so there was no narrow
refresh operation. That allowed the datastore to retain old synthetic RGB while the current C#
compositor had moved on. The existing `0_5_3_3368-Azeroth.zarr` also contains 43 all-zero rows in
both synthesized minimap resolutions, confirming that the old output was not cleanly refreshed or
validated.

The bounded repair is now explicit in
`data-harvester/scripts/refresh_v50_synthetic_minimaps.py`: render current 256/1024 synthetic
tiles from the existing store's tile index, require every tile to be written and non-black, copy
the old store to a new path, and replace only `minimap_rgb` and `minimap_rgb_1024`. No raw terrain,
object masks, or other harvested signals are changed. It also recomputes each refreshed signal's
content identity and coverage metadata.

`v60_build_dataset.py` accepts `--v50-store-root`, so the refreshed per-build stores can be
consolidated directly without overwriting the original v50 datastore.

The old 0.5.3 Ghidra findings remain separate future archaeology gates. They are not the
explanation for this stale-synthesis failure.

## 4.x runtime correctness correction — 2026-08-11

- The active 4.0 terrain path must resolve MH2O `LiquidTypeId` through the exact-build
  `LiquidType` DBC loaded by DBCD. Numeric ID-family tables are not an acceptable runtime owner.
- The loader now reads the DBC row's actual ID field and classifies the exact row using its DBC
  class/name data. A missing DBC row uses the documented safe water default instead of guessing
  magma from the numeric ID.
- Exact local Light* zone data remains diagnostic-only by default. Until the local-zone transform
  and falloff contract is proven, applying it to ordinary outdoor terrain can produce a dark/orange
  noon frame; the global viewer light remains the renderer identity case.

## 4.x WMO placement reuse correction — 2026-08-11

- Stormwind profiling exposed duplicate work after residency settled: a shared `WmoRenderer` was
  updating each internal doodad model once per placed WMO, re-running placement-local visibility,
  and sorting the same opaque doodad list for every placement.
- `WorldScene` now brackets each visible unique WMO renderer with `BeginWorldFrame` and
  `EndWorldFrame`. Doodad animation advances once per unique WMO asset per frame; placement
  transforms remain per-instance for rendering.
- The already-safe opaque WMO instance route skips redundant portal/frustum traversal and distance
  sorting for its placement-local doodads. Portal-aware/fallback WMO rendering remains unchanged.
- Viewer build passed with 0 errors and focused world-pass tests passed 14/14. Sustained Stormwind
  FPS and GPU timing remain user-run proof requirements.
