# Spec 197 Research: WoW 5.0.1.15464 Ghidra Reconnaissance

**Date:** 2026-08-31  
**Binary:** `Wow.exe`, Mists of Pandaria 5.0.1.15464  
**Ghidra project:** `Mists of Pandaria 5.0.1.15464`  
**Mode:** static, read-only reconnaissance; no Ghidra program edits

This is an interim evidence checkpoint for T117 in [`tasks.md`](tasks.md:26).
It records facts recovered before the full terrain/WMO/rendering extraction pass,
not a complete client architecture reconstruction. Addresses are Ghidra static
addresses in the loaded 32-bit image.

## 1. Binary and bridge provenance

- GhidraMCP 6.0.0 is serving HTTP on `http://127.0.0.1:8089`.
- The active program is PE x86, language `x86:LE:32:default`, image base
  `0x00400000`.
- Program inventory at the start of this pass: **38,405 functions**, **175,352
  symbols**, **790 data types**, and **8 memory blocks**.
- The source-anchor report contains **661 `.cpp` anchors**. The broader string
  inventory reports **766 `.cpp`-matching strings**; these are different counts
  because the anchor report groups only strings associated with function
  references.
- Requests were made directly against the healthy HTTP bridge because the
  chat-side dynamic Ghidra tool catalog did not refresh after bridge recovery.
- No bytes, symbols, labels, comments, data types, functions, or analysis
  settings were changed in Ghidra.

## 2. High-value source anchors

The following source-path strings are preserved in the 5.0.1 binary. The
associated `FUN_*` addresses are the first function-mapping targets for the next
decompilation pass.

| Native source anchor | String address | Mapped function addresses recovered |
|---|---:|---|
| `MapChunk.cpp` | `0x00def440` | `0x00ba80f0`, `0x00ba8620`, `0x00ba3050`, `0x00ba36d0`, `0x00ba37d0`, `0x00ba78d0`, `0x00ba7cb0`, `0x00ba7de0`, `0x00ba7d60`, `0x00ba7600` |
| `MapChunkLiquid.cpp` | `0x00def0a0` | `0x00b9b1b0`, `0x00b9b110` |
| `MapChunkRender.cpp` | `0x00def1ac` | `0x00b9cae0`, `0x00b9cba0`, `0x00b9c7a0` |
| `MapChunkIntersect.cpp` | `0x00def35c` | `0x00b9fe10` |
| `MapRenderChunk.cpp` | `0x00ded5dc` | `0x00b67120`, `0x00b668e0`, `0x00b66ae0`, `0x00b66bc0`, `0x00b66ea0`, `0x00b66330`, `0x00b664e0`, `0x00b66770`, `0x00b65ce0`, `0x00b64c10`, `0x00b641c0`, `0x00b64130`, `0x00b64280`, `0x00b643f0` |
| `MapRenderChunkState.cpp` | `0x00def59c` | `0x00bab8b0`, `0x00bab9c0`, `0x00babaa0`, `0x00babb70`, `0x00babed0`, `0x00ba9a00`, `0x00bae9a0`, `0x00bae0e0`, `0x00bae1e0`, `0x00badab0`, `0x00badb80`, `0x00bad4c0`, `0x00bacdc0`, `0x00bacd70` |
| `MapRenderChunkBatch.cpp` | `0x00defc48` | `0x00bb3130`, `0x00bb2ad0`, `0x00bb2bc0`, `0x00bb2d30`, `0x00bb2ee0` |
| `MapTexture.cpp` | `0x00defaf8` | `0x00bb2030`, `0x00bb21e0`, `0x00bb23c0`, `0x00bb1f90`, `0x00bb1df0` |
| `MapObjRead.cpp` | `0x00dee208` | `0x00b83080`, `0x00b83660`, `0x00b82030`, `0x00b82270`, `0x00b82750`, `0x00b81d80`, `0x00b81e90`, `0x00b81e50`, `0x00b81f80`, `0x00b81f30`, `0x00b84a90`, `0x00b84ae0`, `0x00b842e0`, `0x00b843f0` |
| `MapObj.cpp` | `0x00deb8ec` | `0x00b2bb80`, `0x00b2bea0`, `0x00b2b140`, `0x00b2b210`, `0x00b2b570`, `0x00b2b6a0`, `0x00b2a970`, `0x00b2aa30`, `0x00b2ae80`, `0x00b2afe0`, `0x00b2a130`, `0x00b298c0`, `0x00b299e0`, `0x00b29950`, `0x00b29b30`, `0x00b29500`, `0x00b29680`, `0x00b28d80`, `0x00b28ef0`, `0x00b28f70`, `0x00b2c4c0` |
| `MapWater.cpp` | `0x00dee738` | `0x00b8ab90`, `0x00b8a280`, `0x00b89a90`, `0x00b89170`, `0x00b887d0` |
| `WorldScene.cpp` | `0x00debf1c` | `0x00b3b1d0`, `0x00b3b410`, `0x00b3b770`, `0x00b3a9a0`, `0x00b3ac50`, `0x00b3a030`, `0x00b3a550`, `0x00b39dd0`, `0x00b39ea0`, `0x00b3ccf0`, `0x00b3c3f0`, `0x00b3c4b0`, `0x00b3c700`, `0x00b3aaa0` |
| `WorldMapPreload.cpp` | `0x00dee9e4` | `0x00b8e880`, `0x00b8e060`, `0x00b8db00`, `0x00b8dd00`, `0x00b8d760` |
| `LiquidRenderer.cpp` | `0x00e804c4` | `0x00d089e0`, `0x00d080e0`, `0x00d081f0`, `0x00d08350`, `0x00d07fc0`, `0x00d07f80` |
| `CSimpleRender.cpp` | `0x00d5bc28` | 25 undocumented functions; exact mapping remains to be narrowed |
| `M2Scene.cpp` | `0x00d70c10` | 41 undocumented functions; projected-texture and model-batch strings are present |
| `SFileArchives.cpp` | `0x00d51388` | 40 undocumented functions; archive/file-stack extraction lane |

Additional rendering anchors include `CGxShader.cpp`, `CGxD3dShader.cpp`,
`CGxD3dTexture.cpp`, `CGxD3d11Scene.cpp`, `CGxD3d11Texture.cpp`,
`CGxOglScene.cpp`, `CGxOglShader.cpp`, `CGxOglTexture.cpp`, `ShaderEffect.cpp`,
`ShaderEffectManager.cpp`, `ShadowCache.cpp`, `BlobShadow.cpp`, and
`DefaultShaders.cpp`.

## 3. Confirmed `MapChunk` parse contract

### 3.1 MCNK header hand-off — `0x00ba37d0`

The routine at `0x00ba37d0` first asserts the four-byte token is `MCNK`. When
called in its header-consuming mode, it copies these raw header locations into
the in-memory chunk object:

| Input location relative to the MCNK token | Destination object offset | Observed use |
|---:|---:|---|
| `+0x44` (16-bit) | `+0x78` | copied as a short |
| `+0x46` (16-bit) | `+0x7a` | copied as a short |
| `+0x70` (32-bit) | `+0x64` | copied as a 32-bit value |
| `+0x74` (32-bit) | `+0x68` | copied as a 32-bit value |
| `+0x78` (32-bit) | `+0x6c` | copied as a 32-bit value |
| `+0x08` (32-bit) | `+0x88` | flags word |
| `+0x3c` (32-bit) | `+0x8c` | area-like header word |
| `+0x18` (16-bit) | `+0x90` | layer-like count |
| `+0x40` (16-bit) | `+0x92` | doodad/reference-like count |
| `+0x14` (16-bit) | `+0x94` | map-object/reference-like count |
| `+0x48` | `+0x98` | raw header pointer |
| `+0x58` | `+0x9c` | raw header pointer |

After the 128-byte MCNK header, parsing continues at token-relative offset
`+0x88` and the remaining byte count is reduced by `0x80`. The same routine can
also be called in a mode that skips this header copy and passes an already
positioned subchunk range to the dispatcher.

### 3.2 Subchunk dispatcher — `0x00ba3050`

The routine at `0x00ba3050` walks a sequence of `{token, size, payload}`
records. Each iteration advances by `8 + payloadSize`; it asserts that the final
remaining `dataSize` is zero. The following routing was directly visible in the
decompiler output:

| FourCC | Destination/object effect |
|---|---|
| `MCMT` | Stores the first payload word at object offset `+0x13c`. |
| `MCDD` | Stores the payload pointer at `+0x9c`; accepts only payload sizes `8` or `0x20`; size `0x20` sets object flag bit `0x1` at `+0x7c`. |
| `MCAL` | Stores the payload pointer at `+0x124`. |
| `MCBB` | Stores the payload pointer at `+0x148`; computes `m_blendMeshBatchCount = payloadSize / 0x14` and stores it as a byte at `+0x147`; count is capped at `0xff`. |
| `MCCV` | Stores the payload pointer at `+0x114`. |
| `MCLQ` | Stores the payload pointer at `+0x134`. |
| `MCLV` | Stores the payload pointer at `+0x118`. |
| `MCLY` | Stores `payloadSize / 0x10` as a 16-bit layer count at `+0x94`, then copies the payload into the object buffer beginning at `+0xa0`. |
| `MCRW` | Stores `payloadSize / 4` as a 16-bit count at `+0x92` and the payload pointer at `+0x130`. |
| `MCNR` | Stores the payload pointer at `+0x11c`. |
| `MCRD` | Stores `payloadSize / 4` as a 16-bit count at `+0x90` and the payload pointer at `+0x12c`. |
| `MCRF` | Stores the payload pointer at `+0x128`. |
| `MCSH` | Stores the payload pointer at `+0x120`. |
| `MCVT` | Stores the payload pointer at `+0x110`. |

This is the strongest recovered format fact in the current pass: the 5.0.1
client has explicit `MCMT`, `MCBB`, `MCRD`, and `MCRW` handling in the
`CMapChunk` subchunk dispatcher in addition to the familiar terrain payloads.
The exact semantic names of the `MCDD`/`MCMT` fields and the relationship
between `MCBB` and the WMO/terrain seam path still require caller and renderer
decompilation.

### 3.3 World placement and bounds — `0x00ba8620` / `0x00ba7de0`

The initialization path at `0x00ba8620` performs these exact transforms before
building dependent render state:

```text
object[0x64] = -(float)object[0x74] * 33.333332 + 17066.666
object[0x68] = -(float)object[0x70] * 33.333332 + 17066.666
```

The two source fields are the chunk grid indices used by this object; the
axis naming is intentionally left unresolved until the reader/renderer
coordinate cross-check. The same scale and origin appear in
`0x00ba7de0`: local chunk edges use `33.333332`, and the transformed world
origin is `17066.666`.

`0x00ba7de0` initializes the chunk AABB, scans the MCVT payload referenced at
object offset `+0x110`, adds the chunk base value at `+0x6c` to the discovered
vertical bounds, and copies the resulting six values into an AABB block at
object offsets `+0x34` through `+0x48`. When an auxiliary linked structure is
present at `+0x10c`, it calls a secondary bounds routine and clamps the vertical
limits against values at `+0x54` and `+0x60`; otherwise the auxiliary bounds are
initialized to floating-point sentinels.

### 3.4 Liquid construction — `0x00ba78d0`

The routine at `0x00ba78d0` consumes the MCLQ payload pointer at object offset
`+0x134`. It examines four successive chunk flag bits beginning at bit value
`0x04`, creates a liquid/render entry for each enabled bit, copies the chunk
world position into the entry, and binds an 8-by-8-style render grid. It also
walks a separate liquid table reached through the parent object at `+0xc0`,
derives per-entry dimensions from compact header bytes when the entry is shorter
than `0x2a`, and computes the final grid size as:

```text
(decodedHeightCells + 1) * (decodedWidthCells + 1)
```

There is a build-specific branch for a parent value of `0x212` that remaps the
third liquid slot to slot `0xf`. The liquid routine and the dedicated
`MapChunkLiquid.cpp` functions should be decompiled together before assigning
semantic names to these fields.

### 3.5 Resource and placement paths

- `0x00ba7cb0` resolves a doodad definition through the parent map's model
  tables, constructs the associated object, and marks a placement flag. This is
  a terrain-to-doodad admission path, not yet a complete MDDF contract.
- `0x00ba7600` walks the chunk texture list at object offset `+0xa0`, releases
  texture/material references, tears down auxiliary render state, and clears
  linked entries. It confirms that the per-chunk texture table is an owned,
  reference-counted resource path in the client.

### 3.6 Native ADT file family and version gate — `0x00bb9490`, `0x00bb0850`, `0x00bb7010`, `0x00bb6f10`

The exact 5.0.1 map-area loader constructs the root ADT and two selected split
companions. The path builder at `0x00bb9490` formats the root as
`%s\\%s_%d_%d.adt`, then selects a LOD band and appends one object and one
texture suffix:

| Logical role | Band 0 path | Band 1 path | Cache file-type value observed |
|---|---|---|---:|
| Root terrain/map file | `{map}_{x}_{y}.adt` | `{map}_{x}_{y}.adt` | `0` |
| Object split | `{map}_{x}_{y}_obj0.adt` | `{map}_{x}_{y}_obj1.adt` | `1` / `2` |
| Texture split | `{map}_{x}_{y}_tex0.adt` | `{map}_{x}_{y}_tex1.adt` | `3` / `4` |

The loader asserts `lodBand <= 1`; therefore the native suffix selection in
this path is exactly `_obj0`/`_tex0` or `_obj1`/`_tex1`. A search of the same
binary found no `_lod.adt` construction in this path. The only `_lod` search
hit was the unrelated graphics extension string `GL_EXT_texture_lod_bias`.
This does **not** prove that no other subsystem or later client build ever
uses `_lod.adt`; it does prove that `_lod.adt` must not be assumed to be the
5.0.1 replacement for the band-1 pair without another native call path.

The cache key routine at `0x00bb6c70` accepts a 4-bit ADT file-type field and
packs it as:

```text
((mapId << 4 | adtFileType) << 6 | y) << 6 | x
```

It validates `x < 64`, `y < 64`, `adtFileType < 0x10`, and `mapId < 0x10000`.
This is a larger file-type namespace than the three resident pointers in
`m_adtFileData[0..2]`: the active area holds the root plus the selected object
and texture files, while the cache can distinguish at least the five values
listed above.

The file-data parser at `0x00bb7010` requires `MVER` and asserts
`adtVersion == 0x0012`. Its top-level parser at `0x00bb6f10` retains `MTXF`,
`MTXP`, and `MTEX` data and requires exactly `0x100` top-level `MCNK` records.
That 256-record requirement applies to each loaded file-data object, including
the split companions; it is not a signal that only the root contains MCNK
records.

The merge routine at `0x00bb0a30` reads the same chunk index from all three
resident file-data slots. It invokes the MCNK constructor with header
consumption enabled only for slot 0:

```text
slot 0: root MCNK wrapper + 128-byte MCNK header + subchunks
slot 1: selected object MCNK wrapper + headerless subchunk stream
slot 2: selected texture MCNK wrapper + headerless subchunk stream
```

The split files therefore still have 256 outer `MCNK` records. “Headerless”
means that the 128-byte per-MCNK header is absent from each record payload; it
does not mean that the file is a flat stream of unwrapped `MCLY`/`MCAL` data.
The native `MapArea` load loop then parses the three file-data objects and
merges their per-index chunk state before building world coordinates and render
state.

### 3.7 Tile admission is separate from ADT file discovery — `0x00bb4a70`, `0x00b8f8f0`, `0x00b8f830`

The native world-map path does not admit terrain solely because an ADT-family
file can be found. It consults a map-table entry and requires
`CMapTableEntry::Flag_Exists` (bit `0x1`). The area creation loop at
`0x00bb4a70` checks that bit before creating/loading an area, and the map-table
accessors at `0x00b8f8f0`/`0x00b8f830` assert the same invariant.

This separates three questions that the viewer currently combines in some
paths:

1. Does the WDT map table mark the tile as existing?
2. Does the archive or loose overlay contain the root ADT?
3. Does the tile have one or more split companions?

The native answer is not “yes” to question 2 or 3 by itself. A split pair is
additional data for an admitted tile, not independent proof that a terrain
area should exist. This is the leading native explanation for the symptom of
missing/empty tiles appearing in the viewer while valid MCVT and MCCV data
still render: tile admission and payload parsing can both look locally valid
while the admission source is wrong.

The client stores an area pointer per internal LOD band and refuses to attach a
second area to an occupied band slot. The suffix loader accepts two file bands,
but other internal `LOD_COUNT` assertions expose a broader client-side band
concept. Those two notions must remain distinct until the `WorldMap`/
`MapAreaBase` ownership path is extracted further.

## 4. WMO reader evidence recovered so far

The WMO source anchor `MapObjRead.cpp` maps to a large, cohesive reader family.
Two routines are especially valuable:

- `0x00b843f0` validates `MVER` and requires version `0x0011`, then validates
  `MOGP`, copies six consecutive group-bound values, group flags, count fields,
  and format-dependent metadata into the group object. It also derives a group
  format selector from the group flags, including branches that produce values
  `0xd`, `0xe`, and `0x13`.
- `0x00b82030` validates a material index against `materialCount`, requires the
  material texture list, and indexes material records with a `0x10`-byte stride.
- `0x00b84ae0` loads a group by index, rejects a group that still has raw or
  asynchronous data attached, requires a group-info record at a `0x20`-byte
  stride, and formats the group suffix as `_%03d`.
- `0x00b83080` is a top-level WMO chunk loop. It records payload pointers and
  derives table counts using at least `payloadSize / 0xc`, `payloadSize / 0x18`,
  and `payloadSize >> 4`; the full FourCC-to-field table remains to be extracted
  from the saved decompilation artifact.

These facts establish a concrete WMO parsing lane, but they do **not** yet prove
the terrain/WMO seam algorithm. The seam-specific candidates are the
`MapChunkRender.cpp`, `MapRenderChunk.cpp`, `MapRenderChunkState.cpp`, and
`MapRenderChunkBatch.cpp` families below.

## 5. Renderer and blend anchors not yet fully decompiled

String xrefs already tie the terrain blend path to specific functions:

| Evidence string | Xref source/function |
|---|---|
| `m_parentArea->m_blendMeshIndexCount` | `0x00b9c817` in `FUN_00b9c7a0` |
| `m_parentArea->m_blendMeshVertexCount` | `0x00b9c7e7` in `FUN_00b9c7a0` |
| `m_blendMeshBatchCount` | `0x00b9c7b7` in `FUN_00b9c7a0` |
| `Blend Batch` | `0x00ba97fa` and `0x00ba9816` in `FUN_00ba96d0` |
| `renderChunk->HasBlendBatches()` | `0x00badc19` in `FUN_00badb80` |
| `blendBatchList` | `0x00badb97` in `FUN_00badb80` |
| `m_blendTexture == 0` | `0x00bb31c4` in `FUN_00bb3130` |
| `CMapChunk::UnpackAlphaBits(): Bad genformat.` | `0x00b664ca` in `FUN_00b66330` |
| `CMapChunk::UnpackAlphaShadowBits(): Bad genformat.` | `0x00b65db8` in `FUN_00b65ce0` |

The broad rendering string inventory also contains `CSimpleRender.cpp`,
`CM2SceneRender::SetupTextureTransforms`, `RenderModelBatchesForProjectedTexture`,
`TerrainBlend`, `WorldSceneRender.cpp`, `MapObjRender.cpp`, shadow-map/prepass
terms, WMO water/reflection terms, `TerrainType.dbc`, `TerrainMaterial.dbc`,
and `WMOAreaTable.dbc`.

The next decompilation target is therefore not a speculative shader rewrite. It
is the small blend/render family anchored by `0x00b9c7a0`, `0x00badb80`, and
`0x00bb3130`, followed by the corresponding state/batch constructors.

## 6. Split-ADT token search and negative results

Literal string searches returned **zero** matches for `MHID`, `MDID`, `MCXH`,
`CMapTile`, and `MapTile` in the current Ghidra string table. This is a negative
string result only. It does not establish that the client lacks those chunks:

1. The `CMapChunk` dispatcher visibly uses numeric FourCC comparisons rather
   than string references.
2. The names may exist only in stripped source, in a different module, or as
   runtime/file-data identifiers.
3. The next search must use byte/FourCC patterns and callers of the late-Cata
   split-ADT loading path, not just literal string discovery.

The current repository baseline already contains split-ADT and MoP-related code
to audit, including [`MopAdtChunkParser.cs`](../../src/core/WowViewer.Core.IO/Maps/MopAdtChunkParser.cs:83),
the `_tex0.adt` path in [`StandardTerrainAdapter.cs`](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:492),
and the headerless-MCNK handling in [`Mcnk.cs`](../../src/core/WowViewer.Core.IO/Lk/Mcnk.cs:20).
Those files were not modified during this Ghidra pass and should be treated as
the comparison target, not as proof of client parity.

## 7. 5.0.1 evidence versus the current WowViewer ADT model

The native file-family result exposes concrete gaps in the current support
surface. The comparison is against the existing [ADT family model](../../src/core/WowViewer.Core/Maps/AdtTileFamily.cs:3),
[family resolver](../../src/core/WowViewer.Core.IO/Maps/AdtTileFamilyResolver.cs:5),
[file-kind enum](../../src/core/WowViewer.Core/Maps/MapFileKind.cs:3),
[file detector](../../src/core/WowViewer.Core.IO/Files/WowFileDetector.cs:94),
and [standard terrain adapter](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:225);
no production code was changed during this comparison.

- The family model exposes root, `_obj0`, `_tex0`, and `_lod`, but no `_obj1`
  or `_tex1` paths. The resolver strips only `_tex0`, `_obj0`, and `_lod` from
  an incoming stem, so a band-1 companion cannot currently resolve to its root
  family.
- The detector has distinct categories for `AdtTex`, `AdtObj`, and `AdtLod`,
  but no band-aware object/texture categories. This can preserve a generic
  “split ADT” behavior while silently losing which native LOD band supplied a
  chunk.
- `StandardTerrainAdapter.LoadMapTile` constructs only root, `_tex0`, and
  `_obj0` paths. `HasTilePayload`, `TileHasBackingFiles`, and the indexed-file
  scan likewise ignore `_obj1` and `_tex1`; the indexed scan also rejects every
  suffix other than `obj0` and `tex0`.
- The adapter reads split texture MCNKs with the existing headerless option in
  [`TryBuildMcnkIndexMap()`](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:936),
  which is directionally consistent with the native constructor flag, but the
  current implementation discovers only the first 256 scanned records and
  maps them by scan order. It has not yet been proven against the native
  wrapper/index contract for malformed, sparse, or band-1 files.
- The [MCNK reader](../../src/core/WowViewer.Core.IO/Lk/Mcnk.cs:14) already has a
  `SkipHeader` option and scans split subchunks from offset zero, but its
  headerless documentation names only `_tex0.adt`; the native evidence applies
  the same mode to the selected object and texture companions.
- The current root fallback can admit a tile from WDT flags when no indexed
  ADT files were found, while indexed coverage can add tiles from discovered
  root/`obj0`/`tex0` files. Neither path is yet a faithful implementation of
  the native `Flag_Exists` gate. This remains a likely source of false tile
  visibility, not a final diagnosis until a real WDT/MAIN sample is compared.

The existing parser scaffold still recognizes `MDID`, `MHID`, and `MCXH`, but
the exact 5.0.1 binary has not yielded native parser evidence for those tokens.
They remain **unconfirmed for this build**. Conversely, `MCMT`, `MCDD`,
`MCBB`, `MCRD`, and `MCRW` are directly confirmed by the native `CMapChunk`
dispatcher and should not be displaced by the unconfirmed token set.

## 8. Next bounded extraction sequence

1. Compare the native band/file-data matrix against every path in the [standard
   terrain adapter](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:225),
   [ADT family resolver](../../src/core/WowViewer.Core.IO/Maps/AdtTileFamilyResolver.cs:5),
   and split-file exporters before making parser changes.
2. Decompile `0x00b9c7a0`, `0x00b9cae0`, `0x00b9cba0`, `0x00badb80`, and
   `0x00bb3130`; record blend mesh counts, batch layouts, texture ownership,
   and state transitions.
3. Decompile the relevant `MapRenderChunkState.cpp` constructors and mutators
   around `0x00bab8b0`–`0x00bacd70`.
4. Extract the complete FourCC table and group-header offsets from
   `0x00b83080` and `0x00b843f0`.
5. Search for numeric `MHID`/`MDID`/`MCXH` FourCC encodings and trace their
   consumers into split-file loading.
6. Trace the WDT/MAIN map-table builder and confirm how `Flag_Exists` is
   populated for sparse, split-only, and placeholder tiles.
7. Cross-check recovered offsets against the existing standard ADT readers and
   the planned [`TerrainRenderer.cs`](../../src/core/WowViewer.Core.Renderer/Terrain/TerrainRenderer.cs:8)
   height/seam contract. Do not change implementation until the evidence table
   is complete.

## 9. Dead, dormant, and partially implemented native paths

The static pass found useful unfinished and optional paths, but they must not be
collapsed into a single “dead code” category. The focused evidence note
[`5.0.1-dead-dormant-partial-rendering.md`](evidence/5.0.1-dead-dormant-partial-rendering.md)
contains the short excerpts and classification rules.

### 9.1 Confirmed partial paths

- `FUN_00b7a230` (`LiquidGeomFactories.cpp`) has explicit constructor cases for
  `7`, `9`, `10`, and `0xd`; the default branch asserts
  `0 && "Not implemented!"` and leaves the apparent indirect call target
  unset. This is a real partial implementation. No direct caller was recovered,
  so ordinary-load reachability remains open.
- `FUN_004b16d0` (`CSimpleEditBox.cpp`) checks bit `0x2` at `this + 0x204`,
  asserts `!"FIXME: Not yet implemented"`, and returns early. This is confirmed
  unfinished UI behavior, not a terrain path.

### 9.2 Optional and capability-gated paths

- `FUN_005c5300` (`DepthCache.cpp`) has two direct callers and both supported/
  unsupported messages; `!s_depthRT` is asserted. It is implemented resource
  infrastructure with a device capability gate, not dead code.
- `FUN_005c74a0`, `FUN_005c7820`, and `FUN_005c7940` form an implemented GBuffer/
  prepass family. Strings identify GBuffer depth/shadow-mask targets and three
  prepass modes (`0`, `1`, `2`); render-target/refcount assertions guard its
  lifecycle. Native terrain ownership remains unproven.
- `FUN_004019e0` controls texture-atlas state and can forcibly clear the enable
  bit after device/context checks. `FUN_00401ab0` and `FUN_00401b10` explicitly
  toggle doodad batching bit `0x10` and particle batching bit `0x40`.
- `FUN_00b27950` reports `Unsupported water detail mode.` when its setting helper
  rejects a mode. Its lack of a direct caller is not proof of deadness because a
  setting callback may be registered indirectly.

### 9.3 Lifecycle contracts and low-reachability cautions

- `FUN_005c55140` asserts released `MapArea` diffuse/height textures and zero
  refcount under the applicable flag.
- `FUN_00bac5b0` asserts zero render-state refcount, four cleared diffuse and
  height texture slots, no environment texture, and an unlinked render-list
  node. These are resource ownership contracts relevant to split-ADT residency.
- A broad static inventory found 2,184 functions with zero direct xrefs. This is
  a candidate pool only: vtables, function-pointer factories, settings
  callbacks, exports, jump tables, and registration tables can all hide callers.
- The built-in dead-code result was not a completed reachability analysis in this
  pass. No function is promoted to “proven dead” from zero direct xrefs alone.

### 9.4 Developer/string-only leads

`TODO: Add Support!`, `ERR_BG_DEVELOPER_ONLY`, `development.MPQ`, `M2FasterDebug`,
`Hardware PCF not supported by this graphics card.`, `Doodad instancing is not
supported.`, and several navigation `NOT YET IMPLEMENTED` strings are retained
as leads. They require function xrefs or registration evidence before being
classified as developer-only, disabled, or dead in ordinary play.

## 10. Address discrepancy requiring correction

The earlier file-family notes cite `0x00bb9490`, while a later caller inventory
contains `FUN_00b94990`; an endpoint resolution around `0x00bb9490` also returned
a different function boundary. This is recorded as unresolved rather than
silently choosing one spelling. The native ADT family and MCNK facts remain
supported by the surrounding loader/parser functions, but the final path-builder
address must be rechecked before it is used as a stable symbol anchor.

## 11. Definitive-guide outputs

The consolidated guide is now [`wow-5.0.1-adt-wdt-definitive.md`](../../docs/architecture/wow-5.0.1-adt-wdt-definitive.md).
The dead/dormant/partial inventory is [`5.0.1-dead-dormant-partial-rendering.md`](evidence/5.0.1-dead-dormant-partial-rendering.md).

## 12. Confidence boundary

**High confidence:** binary identity, image metadata, source-anchor addresses,
MCNK token/header hand-off, subchunk routing, payload-size-derived counts,
chunk coordinate scale/origin arithmetic, WMO version assertion, and material
record stride.

**Medium confidence:** semantic names for the two chunk index fields, the exact
meaning of `MCDD`/`MCMT`, the identity of the auxiliary liquid table, and the
WMO group format selector values.

**Not established yet:** literal `MHID`/`MDID`/`MCXH` parser locations, complete
`CMapTile` construction/loading, whether another 5.0.1 subsystem uses
`_lod.adt`, the exact relationship between suffix bands and internal
`LOD_COUNT`, blend texture shader inputs, WMO-to-terrain seam vertex ownership,
height-texture blend equations, production visual parity, indirect reachability
of the liquid factory default branch, and the final spelling/address of the ADT
family path builder.

The implementation task remains T117. Parser/renderer changes described by
[`plan.md`](plan.md:51) and acceptance criteria AC-006/AC-007 in
[`spec.md`](spec.md:39) are still gated on this evidence pass.
