# Data Model: ADT v26 (AHDR family)

Field layouts are **v26 measurements** where marked, otherwise **hypotheses carried over from the wiki v22/v23 relatives**. Anything marked *(measure)* is not final until
Phase 0/1 evidence settles it, and the final layout is recorded in
`docs/architecture/adt-v26-format.md`.

## Core model (`WowViewer.Core/Maps/AdtAhdr/`)

### AdtAhdrTile

| Field | Type | Source | Notes |
|---|---|---|---|
| SourcePath | string | | |
| Kind | MapFileKind | AHDR.version | AdtV22 / AdtV23 / AdtV26 / AdtAhdrUnknownVersion (+ Error) |
| Version | uint | AHDR +0x00 | |
| VerticesX, VerticesY | int | AHDR +0x04, +0x08 | expected 129 |
| ChunksX, ChunksY | int | AHDR +0x0C, +0x10 | expected 16 |
| HeaderReserved | uint[11] | AHDR +0x14 | kept raw; revision 26 has `8396383` at +0x14 (unexplained) |
| AlocField0 | uint | ALOC +0x00 | constant 2869 in corpus; unexplained |
| TileX, TileY | int | ALOC +0x04, +0x08 | **measured** tile position |
| AlocTail | uint[2] | ALOC +0x0C, +0x10 | equal to TileX/TileY in corpus; meaning open |
| Aoch | byte[2048]? | AOCH | all zero in corpus; kept raw |
| Adst | uint[3][] | ADST ×n | 7/699 files (321 chunks); first field in ACDO uniqueId range; kept raw |
| OuterHeights | float[VX×VY] | AVTX first block | order *(measure R4)*, frame *(measure R5)* |
| InnerHeights | float[(VX−1)×(VY−1)] | AVTX second block | |
| OuterNormals / InnerNormals | Vector3[] | ANRM, sbyte/127 | component order *(measure R6)*; raw bytes retained |
| TextureNames | string[] | ATEX | **one chunk per name** (measured, revision 26); may be empty |
| ModelNames | string[] | ADOO | **one chunk per name** (measured); M2 and WMO mixed; looks map-global (the same ~225/285 names in every tile) |
| Chunks | AdtAhdrChunk[ChunksX×ChunksY] | ACNK in file order | |
| FlightBounds | (short[9] max, short[9] min)? | AFBO | v23 only; absent in revision 26 |
| VertexShading | byte[]? (RGBA per vertex) | ACVT | v23 **and revision 26** (every file); order same as AVTX *(measure)* |
| UnknownChunks | AdtAhdrRawChunk[] | inventory | id, offset, size, parent |
| Diagnostics | AdtAhdrDiagnostic[] | reader | per channel: Ok / Missing / SizeMismatch / Malformed + message |

### AdtAhdrChunk

| Field | Type | Source | Notes |
|---|---|---|---|
| IndexX, IndexY | int | ACNK +0x00, +0x04 | **chunk-local**: i % 16, i / 16 in file order (measured, all chunks) |
| Flags | uint | v23 ACNK +0x08 | v22: reserved DWORD; v26 *(measure)*; kept raw |
| AreaId | int | ACNK +0x0C | |
| HolesLowRes | ushort | v23 +0x10 | v22: reserved; v26 *(measure)* |
| LowDetailTextureMap | uint[4] | +0x12 | |
| HighResHoles | ulong? | v23, flags & 0x10000 | |
| HeaderRaw | byte[0x40] | | kept for unexplained fields |
| HasHeader | bool | ACNK size > 0x40 | |
| Layers | AdtAhdrLayer[] | ALYR | |
| ShadowMap | byte[64×64]? | ASHD, expanded | bit order *(measure R8)* |
| Placements | AdtAhdrPlacement[] | ACDO | |

### AdtAhdrLayer

| Field | Type | Source | Notes |
|---|---|---|---|
| TextureIndex | int | ALYR +0x00 | must be < TextureNames.Length or flagged |
| Flags | uint | ALYR +0x04 | v23: 0x100 = has AMAP |
| Reserved | uint[6] | ALYR +0x08 | |
| AlphaEncoding | enum { None, Uncompressed8, Packed4, Compressed, Unknown } | inferred *(R7)* | |
| AlphaEncodingReason | string | | e.g. "payload 2048 bytes" |
| Alpha | byte[64×64]? | AMAP decoded | |

### AdtAhdrPlacement

| Field | Type | Source | Notes |
|---|---|---|---|
| ModelIndex | int | ACDO +0x00 | into ModelNames |
| Target | enum { Doodad, WorldObject, Unresolved } | ModelNames extension | |
| Position | Vector3 | +0x04 | frame *(measure R9)*; raw kept |
| Rotation | Vector3 | +0x10 | units *(measure)* |
| Scale | Vector3 | +0x1C | uniform check *(measure)* |
| Unknown0x28 | float | +0x28 | |
| UniqueId | uint | +0x2C | |
| Trailing | uint[] | +0x30.. | record size *(measure R9)* |

### AdtAhdrInventory (corpus-level)

- **Files**: path, size, SHA-256, kind, version, parse status
- **Chunks**: (fileId, id, parentId, offset, size)
- **Aggregates**: version histogram; chunk-occurrence table (id × parent × count × size histogram); unknown chunks; documented-size disagreements; unaccounted bytes per file; distinct texture/model names

## Validation rules (from the spec)

- Every byte is covered by a chunk record or reported as a gap/overrun (SC-002).
- `AVTX` size = (VX·VY + (VX−1)(VY−1)) × 4; `ANRM` size = same count × 3.
- `TextureIndex` < `TextureNames.Length`; `ModelIndex` < `ModelNames.Length`; otherwise a diagnostic is emitted (SC-006).
- A malformed channel produces a diagnostic, never an exception (FR-010).

## Viewer mapping (`AhdrTerrainAdapter` → existing shapes)

| `TerrainChunkData` | From |
|---|---|
| TileX, TileY | `ALOC[1]`, `ALOC[2]` (measured; filenames are meaningless FileDataIDs) |
| ChunkX, ChunkY, McinIndex | chunk grid position; McinIndex = ChunkY×16+ChunkX |
| Heights (145, 9-8-9) | `AdtAhdrTileSlicer`: outer[(cy·8+r), (cx·8+c)] rows interleaved with inner[(cy·8+r), (cx·8+c)] using the R4 order, then re-framed per R5 |
| Normals (145) | same slicing over outer/inner normals |
| Layers | ALYR → TerrainLayer { TextureIndex, Flags } |
| AlphaMaps | layer index → decoded 64×64 alpha (layer 0 none) |
| ShadowMap | ASHD expanded |
| MccvColors | v23 ACVT sliced to 145 (BGRA conversion *(measure)*); present in v26 and v23; null for v22 |
| AreaId | ACNK AreaId |
| HoleMask | HolesLowRes where the revision has it *(v26: measure)*; else 0 |
| WorldPosition | tile/chunk grid math shared with StandardTerrainAdapter |
| Liquid | null (R13) |

Tile level: `TileTextures[(x,y)]` = TextureNames. ACDO placements become `MddfPlacement`/`ModfPlacement`, with names appended to `MdxModelNames`/`WmoModelNames`, deduplicated by UniqueId.
