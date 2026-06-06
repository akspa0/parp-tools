# Native M2/MDX Loader Trace — WoW 1.12.1 (Build 5875)

**Captured**: 2026-06-05
**Binary**: `WoW.exe` — `WoW [Release] Build 5875 (Sep 19 2006 20:32:39)` (Vanilla / 1.12.1)
**Source path evidence**: `E:\build\buildWoW\WoW\Source\…` and `E:\build\buildWoW\Engine\Source\Model2\…`
**Status (2026-06-05)**: This research is the source of truth for the 1.12.1 native M2 contract. The implementation slice is spec `048-m2-1121-era-aware-md20-reader` (`specs/048-m2-1121-era-aware-md20-reader/spec.md`). Spec `043-m2-chunked-mdx-classic-support` is revised to defer 1.12.1 to spec 048 (the chunked-MDX lane is 0.5.3 / 0.7.0 / 0.8.0 only). A future spec `049-m2-2x-era-aware-md20-reader` will cover 2.x TBC-era MD20 variants (currently rejected with "see spec 049" by the 048 dispatcher).

**Status**: Research evidence — this doc is the source of truth for what the 1.12.1 native client actually expects when it loads an MDX/M2 model on disk and when it consumes an MDDF placement from a loaded ADT tile. It supersedes the chunked-MDX assumption in spec `043-m2-chunked-mdx-classic-support`.

## Scope

- Trace the model-cache loader, the on-disk parser, and the render-init path for the Win32 Vanilla (1.12.1) client.
- Document the 1.12.1 `MD20` header layout, the per-table relocator pattern, and the per-table entry strides.
- Document how the cache normalizes `.mdl` / `.mdx` / `.m2` requests to a single `.m2` namespace.
- Document the render-init path that consumes the parsed model and builds the per-pass render objects.
- Document the gap between the native contract and the existing `wow-viewer` M2/M2Chunked code.
- Feed the gaps into spec `043` and the M2 reader follow-up slices.

## Top-Line Findings

1. **1.12.1 models are stored as `MD20` files with the legacy `.mdx` extension.** The on-disk magic is `MD20` (`0x3032444d` little-endian), not `MDLX` (`0x584C444D`). The existing `M2ModelReaderDispatcher` dispatches on magic and routes any non-`MDLX` file to the 3.3.5-era `M2ModelReader`. A 1.12.1 `.mdx` therefore ends up in the 3.3.5 reader, which uses the wrong strides for 1.12.1.
2. **The 1.12.1 `MD20` header is a flat `(count, offset)` pointer table, not a chunked FourCC structure.** The header itself sits at the top of the file; each subsequent table is just two `uint32` words: a count and a file-relative offset to the data. There are no chunk tags in 1.12.1 — the existing `M2Chunked` reader is therefore reading from the wrong format family entirely for 1.12.1.
3. **The view table layout and several record strides differ between 1.12.1 and 3.3.5.** Same magic, but a shifted/different field layout. The 3.3.5 reader silently misreads 1.12.1 files.
4. **The 1.12.1 native client uses a 1021-bucket hash cache** keyed on the lowercase model basename (after extension normalization). The cache key excludes the extension and the directory.
5. **The native parser performs file-level pointer relocation.** Every table offset in the file is converted in place to a pointer. The cache object stores the file base plus the relocated offsets.
6. **The 1.12.1 M2 option set is smaller than 3.3.5** — `M2BatchParticles` and `M2ForceAdditiveParticleSort` do not exist yet. The 3.3.5 runtime flag word bits `0x80` and `0x100` are not used in 1.12.1.

## M2 Native Function Anchors (1.12.1)

| Address | Source file | Confirmed role |
| --- | --- | --- |
| `0x004799a0` | n/a (caller-side formatter) | Builds the `%s%s_%s%s.mdx` name for character paperdoll / Item/ObjectComponents/Head/<type>_<sex>_<race>.mdx |
| `0x00479a32` | (xref into `0x004799a0`) | Cross-references the `%s%s_%s%s.mdx` literal — confirms paperdoll / item-component MDX resolution uses the legacy `.mdx` extension |
| `0x00706a50` | `M2Cache.cpp` (xref to source string) | Top-level cache lookup / load entry — extension gate, basename hash, hash-chain walk, cache miss allocation |
| `0x0070ebd0` | `M2Model.cpp` (xref to source string) | Per-model render-init: allocates per-table working arrays, calls blend/depth/texture/lights init for each render batch |
| `0x0071cd10` | (helper, M2Model.cpp) | Per-view sub-table validation (recursively walks each view's 9 nested tables) |
| `0x0071cdf0` | `M2Model.cpp` (xref to source string) | On-disk `MD20` parser — magic check, version check, table-by-table pointer relocation |
| `0x0071d4e0` | `M2Cache.cpp` | File-open wrapper — registers the parse-completion callback (`FUN_0071d5e0`) |
| `0x0071d5a0` | `M2Cache.cpp` | Adds a model to the deferred-init linked list (or calls `FUN_0070ebd0` immediately) |
| `0x0071d5e0` | `M2Cache.cpp` | Parse-completion entry — frees the file-open handle, calls `FUN_0071d640` |
| `0x0071d640` | `M2Cache.cpp` | `FUN_0071cdf0` (parse) + `FUN_0071d6c0` (shared-init) + walks the linked list and calls `FUN_0070ebd0` per deferred model |
| `0x0071d6c0` | `M2Shared.cpp` (xref to source string) | Shared init: view selection by quality LOD, `0x10000 / view_distance` frame-skip, per-texture `FUN_00419e20` + `FUN_00449d90` cache handles |
| `0x0071d6c0`'s helpers | `M2Shared.cpp` | `FUN_0071de50` / `FUN_0071df30` / `FUN_00419e20` / `FUN_00449d90` are the texture/sequence record fixup chain |
| `0x0071b0a0`–`0x0071f750` | `M2Shared.cpp` | Per-table relocators (see table below). Each converts `(count, offset)` to `(count, file_base + offset)` and validates bounds. |
| `0x0071b240` / `0x0071b290` / `0x0071b2e0` / `0x0071b310` / `0x0071b340` / `0x0071b370` | `M2Shared.cpp` | Per-record type constructors used during the per-table zeroing pass. |
| `0x0071b5f0` / `0x0071b620` | `M2Shared.cpp` | Per-record field initializers used after table walks (likely the bone/light "translation = identity, scale = 1, visibility = true" seed). |
| `0x0071bb60` | `M2Light.cpp` (xref to source string) | Light grid hash insert — 64x64 bucket table allocated at light-attach time, scaled by `_DAT_00cf04d0` |
| `0x0071f9a0` | `M2Model.cpp` | Allocations for the 0x134 / 0x138 / 0xb4 / similar per-model working buffers used by the parser and shared-init. |
| `0x007b73a0` / `0x007b7bb0` / `0x007b7b40` / `0x007b4ba0` / `0x007b4c50`–`0x007b4d60` / `0x007b4f60` / `0x007b4ed0` / `0x007b5cd0` / `0x007b5d00` / `0x007b5d30` / `0x007b5da0` / `0x007b9a80` / `0x007b9c70` / `0x007b9cd0` / `0x007b9cf0` / `0x007b9d40` / `0x007b9d50` / `0x007b9da0` / `0x007b9de0` / `0x007b9e20` / `0x007b1c80` / `0x007b7070` | n/a (renderer) | Per-batch render-state calls invoked from `FUN_0070ebd0`. These set blend mode (`FUN_007b4f60`), depth function (`FUN_007b4ed0`), texture stage state, texture matrix, clamp mode, fog/dither flags, and the final `FUN_007b5d30` batch-flush. |
| `0x00402760` (registration) | n/a (console var) | cvar registration site for the 1.12.1 M2 option family |

## 1.12.1 MD20 Header Layout

The on-disk header is a flat `uint32` array. Each table after the first 4 dwords is a `(count, offset)` pair. The offset is file-relative; the parser adds the file base to convert it to a pointer.

| Hdr word | Field | Notes |
| --- | --- | --- |
| `[0x00]` | `magic` | `0x3032444d` (`"MD20"` little-endian) — required |
| `[0x04]` | `version` | `0x100` or `0x101` — only these two are accepted |
| `[0x08]` | `name_count` | bytes in the model name |
| `[0x0c]` | `name_offset` | file-relative; relocated in place |
| `[0x10]` | (gap / flags) | one dword between name and global-loop table — 3.3.5 places `flags` here |
| `[0x14]` | `global_seq_count` | number of global-loop time spans |
| `[0x18]` | `global_seq_offset` | entries are 4 bytes (`M2Track<u32>`-style) |
| `[0x1c]` | `anim_count` | number of sequences / animations |
| `[0x20]` | `anim_offset` | entries are 0x6c (108) bytes — see "View" structure below |
| `[0x24]` | `anim_lookup_count` | number of short lookup entries |
| `[0x28]` | `anim_lookup_offset` | entries are 2 bytes (u16 indices) |
| `[0x2c]` | `tex_anim_count` | texture-anim / texture-unit count |
| `[0x30]` | `tex_anim_offset` | entries are 2 bytes (u16) |
| `[0x34]` | `bone_count` | bone count |
| `[0x38]` | `bone_offset` | entries are 4 bytes (u16 index pairs) |
| `[0x3c]` | `view_count` | number of "view" blocks (LOD/skin entries) |
| `[0x40]` | `view_offset` | entries are 0x2c (44) bytes each — see "View" structure |
| `[0x44]` | `color_count` | color-animation count |
| `[0x48]` | `color_offset` | entries are 0x1c (28) bytes |
| `[0x4c]` | `texture_count` | texture reference count |
| `[0x50]` | `texture_offset` | entries are 0x1c (28) bytes |
| `[0x54]` | `tex_weight_count` | texture-weight / texture-transform count |
| `[0x58]` | `tex_weight_offset` | entries are 8 bytes (M2Track-style) |
| `[0x5c]` | `tex_lookup_count` | (lookup table) |
| `[0x60]` | `tex_lookup_offset` | entries are 4 bytes |
| `[0x64]` | `tex_unit_lookup_count` | (lookup table) |
| `[0x68]` | `tex_unit_lookup_offset` | entries are 2 bytes |
| `[0x6c]` | `tex_replaceable_lookup_count` | (lookup table) |
| `[0x70]` | `tex_replaceable_lookup_offset` | entries are 2 bytes |
| `[0x74]` | `tex_flag_lookup_count` | render-flag lookup |
| `[0x78]` | `tex_flag_lookup_offset` | entries are 2 bytes |
| `[0x7c]` | `bounding_tri_count` | (collision / culling) |
| `[0x80]` | `bounding_tri_offset` | entries are 0xc (12) bytes |
| `[0x84]` | `bounding_vert_count` | |
| `[0x88]` | `bounding_vert_offset` | entries are 0xc (12) bytes |
| `[0x8c]` | `render_flag_count` | |
| `[0x90]` | `render_flag_offset` | entries are 0x10 (16) bytes |
| `[0x94]` | (cairn? lod table count) | see FUN_0071e2f0 walk |
| `[0x98]` | (cairn? lod table offset) | entries are 0x38 (56) bytes |
| `[0x9c]` | `collision_count` | collision hulls |
| `[0xa0]` | `collision_offset` | entries are 2 bytes |
| `[0xa4]` | `attach_count` | attachment slots |
| `[0xa8]` | `attach_offset` | entries are 0xc (12) bytes |
| `[0xac]` | `light_count` | light count |
| `[0xb0]` | `light_offset` | entries are 0xc (12) bytes — yes, 0xc in 1.12.1 (3.3.5 grew to 0x9c / 0x98) |
| `[0xb4]` | `camera_count` | camera count |
| `[0xb8]` | `camera_offset` | entries are 0x2c (44) bytes |
| `[0xbc]` | `cam_perframe_count` | per-frame camera data |
| `[0xc0]` | `cam_perframe_offset` | entries are 0xd4 (212) bytes |
| `[0xc4]` | `ribbon_count` | ribbon-emitter count |
| `[0xc8]` | `ribbon_offset` | entries are 0x7c (124) bytes — 3.3.5 grew to 0xac / 0xb0 |
| `[0xcc]` | `particle_count` | particle-emitter count |
| `[0xd0]` | `particle_offset` | entries are 0xdc (220) bytes — 3.3.5 grew to 0x1dc / 0x1ec |
| `[0xd4]` | `unk_v101_extra_0_count` | 0x101 only |
| `[0xd8]` | `unk_v101_extra_0_offset` | 0x101 only — entries are 2 bytes |
| `[0xdc]` | `unk_v101_extra_1_count` | 0x101 only |
| `[0xe0]` | `unk_v101_extra_1_offset` | 0x101 only — entries are 0x1f8 (504) bytes (FUN_0071f210 — likely the largest geometry / submesh / render-block record in 1.12.1) |

**Native header spans 0xe8 bytes (0x101) or 0xd4 bytes (0x100)** when read with this layout.

## 1.12.1 Relocator Catalog

The parser uses 15+ relocator helpers; each converts `(count, offset)` to `(count, file_base + offset)`. The helpers also bound-check the resulting table against the file size. The relocator name encodes the entry stride:

| Helper | Stride | Used for (header word) |
| --- | --- | --- |
| `FUN_0071e0c0` | 8 | tex-anim lookup track (8), bone-lookup-track (8), render-flag sub-blocks |
| `FUN_0071e110` | 4 | global loops, bone-lookup-index, view sub-tables, camera per-frame sub-tables, particle per-frame sub-tables |
| `FUN_0071e160` | 2 | name, anim-lookup, tex-lookup, tex-unit-lookup, tex-replaceable-lookup, render-flag-lookup, collision, 0x101-extra-0 |
| `FUN_0071e1b0` | 0xc (12) | attach, light, bounding-tri, bounding-vert, view sub-tables, camera per-frame sub-tables, particle per-frame sub-tables, 0x101-extra-1 sub-tables |
| `FUN_0071e210` | 0x30 (48) | tex-weight-lookup-style records |
| `FUN_0071e270` | 0x2c (44) | view-record sub-tables (recursive walker) |
| `FUN_0071e2f0` | 0x38 (56) | cairn / lod table — contains 6 sub-tables per record |
| `FUN_0071e440` | 0x10 (16) | render-flag records |
| `FUN_0071e4f0` | 0x1c (28) | color-anim records (8-byte track, 4-byte track, 2-byte track) |
| `FUN_0071e5e0` | 0x1c (28) | texture records (8-byte track, 4-byte track, 2-byte track) |
| `FUN_0071e6d0` | 0x54 (84) | attach-lookup records (9 nested sub-tables) |
| `FUN_0071e880` | 4 | "global sequence" alias for FUN_0071e110 |
| `FUN_0071e8d0` | 0x30 (48) | texture-weight records (3 sub-tables) |
| `FUN_0071e9d0` | 0x2c (44) | camera records (2 sub-tables) |
| `FUN_0071eaa0` | 0xd4 (212) | camera per-frame records (15 nested sub-tables) |
| `FUN_0071edb0` | 0x7c (124) | ribbon records (5 nested sub-tables) |
| `FUN_0071ef40` | 0xdc (220) | particle records (16 nested sub-tables) |
| `FUN_0071f210` | 0x1f8 (504) | 0x101-only extra-1 records (29 nested sub-tables) |
| `FUN_0071f650` | 0x10 (16) | view sub-tables (3 sub-tables of 16 bytes) |
| `FUN_0071f700` / `71f750` / `71f7a0` | n/a | large-record sub-table relocators (8-byte entries, 4-byte entries, 12-byte entries) |
| `FUN_0071cd10` | 0x2c (44) | view-record recursive sub-table walker (a wrapper that walks the 9 tables nested inside a view record) |

## 1.12.1 "View" Structure (0x2c / 44 bytes per record)

The 1.12.1 view table sits at header offset `0x3c/0x40` with 0x2c bytes per record. The 3.3.5-era reader assumes the view sits at `0x44/0x48` (off by 8) with no view-stride concept — that mismatch alone is enough to misread 1.12.1.

| Offset (within view record) | Type | Role |
| --- | --- | --- |
| `+0x00` | `uint32[8]` | sub-table A (relocator FUN_0071e0c0 → 8 bytes/entry) |
| `+0x08` | `uint32[8]` | sub-table A offset (relocated) |
| `+0x10` | `uint32[4]` | sub-table B (relocator FUN_0071e110 → 4 bytes/entry) |
| `+0x14` | `uint32[4]` | sub-table B offset (relocated) |
| `+0x18` | `uint32[4]` | sub-table C (relocator FUN_0071e1b0 → 0xc/entry) |
| `+0x1c` | `uint32[4]` | sub-table C offset (relocated) |
| `+0x20` | `uint32` | sub-table D count |
| `+0x24` | `uint32` | sub-table D offset (FUN_0071e0c0, 8 bytes/entry) |
| `+0x28` | `uint32` | **LOD distance / skin selector** — drives view selection in `FUN_0071d6c0` |

The "view" is therefore a 9-slot block (8 byte / 4 byte / 12 byte sub-tables plus a 4-byte selector) of pre-relocated pointers to per-view data.

`FUN_0071d6c0` walks these view records and selects the one whose `+0x28` LOD distance is `<= qualityLevel` and `>= bestSoFar`. The quality level comes from `(*(int *)(iVar2 + 0x9c) - 0x1f) / 3` (a clamped renderer detail setting). The selected view pointer is stored at `param_1 + 0x138`.

The `+0x08` value computed as `0x10000 / view_distance` is the per-view frame-skip divisor — it tells the per-frame animator how many game ticks must pass between evaluations of this view's animated state.

## 1.12.1 Sequence / Animation Record (0x6c / 108 bytes per record)

Located at header offset `0x1c/0x20`. Each sequence record contains 9 nested sub-tables and a few fixed fields. The walk order inside `FUN_0071cdf0`'s inner loop matches the 1.12.1 inline parse. Stride: 0x6c.

The 3.3.5 reader assumes `SequenceStride = 0x40` (64 bytes) — wrong for 1.12.1, and the 1.12.1 record contains nested tracks (`FUN_0071e0c0` / `FUN_0071e110` / `FUN_0071e1b0` / `FUN_0071f650` / `FUN_0071e0c0` / `FUN_0071e110` / `FUN_0071e1b0`) at fixed offsets, not the 3.3.5 layout.

## 1.12.1 Camera / Particle / Ribbon / Light Stride Differences vs 3.3.5

| Record | 1.12.1 stride | 3.3.5 stride (existing `M2ModelReader`) | Notes |
| --- | --- | --- | --- |
| Camera | 0x2c (44) | 0x64 (classic) / 0x74 (modern) | 1.12.1 cameras are far simpler than 3.3.5 |
| Camera per-frame | 0xd4 (212) | n/a (3.3.5 inlined into camera record) | 1.12.1 separates per-frame data; 3.3.5 folds it in |
| Ribbon | 0x7c (124) | 0xac (172) / 0xb0 (176) | 1.12.1 ribbons are simpler |
| Particle | 0xdc (220) | 0x1dc (476) / 0x1ec (492) | 1.12.1 particles are dramatically simpler — no multitextured / billboard / advanced blend state |
| Light | 0xc (12) | 0x9c (156) | 1.12.1 lights are minimal — 1.12.1 likely uses runtime light state per render-init, not a per-light record in the file |
| Bone | 4 (u16 index pair per entry) | 0x58 / 0x9c (3.3.5 bone records) | 1.12.1 bones are lookup pairs; 3.3.5 bones are full records |
| Bone count slot | 0x34 | 0x2c | off by 8 — same 8-byte shift as the view table |
| View slot | 0x3c | 0x44 | off by 8 |
| Sequence slot | 0x1c | 0x1c | same |
| Sequence stride | 0x6c (108) | 0x40 (64) | totally different |
| Render batch (submesh / geo) | 0x1f8 (504, 0x101 only) | 0x40 / 0x44 / 0x58 etc. | 1.12.1 keeps a single per-batch record in 0x101 files; 3.3.5 splits it across multiple sub-tables |
| Vertex entry | 2 bytes (u16 only) | 0x30-0x40 (full position+normal+uv+bone) | 1.12.1's "vertex" table is likely a separate pair of `0xc/12` lookup blocks (positions and normals) — the 1.12.1 vertex is not a single interleaved record |

The 1.12.1 vertex layout is **strongly implied** to be a packed `(u16 position index, u16 normal index)` pair from the stride, with full position/normal data living in separate tables. This is consistent with classic MDX vertex layout.

## 1.12.1 Cache Loader Behavior

`FUN_00706a50` (called from `FUN_00706a50` itself for cache miss) is the on-disk model loader entry point. Behavior:

1. Copy the path into a stack buffer (0x104 = 260 bytes — classic `MAX_PATH`-sized buffer).
2. Locate the extension with `strrchr('.')`.
3. If the extension is `.mdl` (likely `&DAT_0087317c` / `0x00873174`) or `.mdx`, rewrite the extension bytes to `.m2` (`&DAT_00873170`). Other extensions fail with `Model2: Invalid file extension: %s\n`.
4. Call `FUN_006477c0(0, local_108, async_flag, &param_2)` — the MPQ-aware file open.
5. If open fails: log `Model2: File not found: %s\n` and return 0.
6. Strip the extension from `local_108`, lowercase the basename, then fold-case-hash it.
7. Walk the 1021-bucket hash chain (`param_1 + 0x14 + (hash % 0x3fd) * 4`).
8. On hit: increment ref count via `FUN_0071d430`, return the existing entry.
9. On miss: allocate a 0x164-byte cache record, copy the lowercase basename into the entry, insert it into the bucket chain.
10. Trigger async file read with the parse-completion callback `FUN_0071d5e0`.

This matches the 3.3.5 cache open behavior seen in earlier research — extension normalization is the same in both eras. The implementation gap is therefore *not* in the cache loader, but in the parser that runs after the file lands in memory.

## 1.12.1 Render-Init Behavior

`FUN_0070ebd0` is the per-model render-init. Behavior (high level):

- Allocates per-table working arrays at fixed offsets on the model object:
  - `+0x64` from `header[0x14]` count (textures? bones?)
  - `+0x6c` from `header[0x1c]` count
  - `+0x90` from `header[0x34]` count, 0x118 (280) bytes per entry
  - `+0x98` from `header[0x18]` count, 4 bytes per entry
  - `+0xa0` from `header[0x54]` count, 0x50 (80) bytes per entry
  - `+0xa4` from `header[0x5c]` count, 4 bytes per entry
  - `+0xa8` from `header[0x64]` count, 0x20 (32) bytes per entry
  - `+0xac` from `header[0x6c]` count, 0x20 bytes per entry
  - `+0xb0` from `header[0x74]` count, 0x98 (152) bytes per entry — *1.12.1 light init shape*
  - `+0x1c8` from `header[0x104]` count, 0x20 bytes per entry
  - `+0x1f0` from `header[0x114]` count, 0x20 bytes per entry
  - `+0x200` from `header[0x11c]` count, 0x170 (368) bytes per entry — likely sequences
  - `+0x3c4` from `header[0x124]` count, 0x84 (132) bytes per entry
  - `+0x3c8` from `header[0x134]` count, 0xd0 (208) bytes per entry
  - `+0x3d0` from `header[0x13c]` count, 0x16c (364) bytes per entry — particle init

- The renderable subdivision into batches (the per-draw-call records) is what `FUN_0070ebd0` builds at `+0x3c8` — the same `+0x134` source header word produces a 0xd0-byte per-batch record.

- For each batch it issues per-pass render state calls: `FUN_007b4ba0(0,1)` (state push), texture-stage state via `FUN_007b4c50/60/70/80/90`, blend mode via `FUN_007b4f60(uVar18, uVar19, uVar11, ...)`, depth function via `FUN_007b4ed0`, fog via `FUN_007b5d00(1)`, transform matrix via `FUN_007b9c70` / `FUN_007b9cd0`, scale via `FUN_007b9d40`, clamp via `FUN_007b9da0` / `FUN_007b9de0`, view-bounds via `FUN_007b9e20`, billboard matrix via `FUN_007b4d60(0, auStack_98)` / `FUN_007b4d60(1, auStack_f8)`, and the final `FUN_007b5d30` per-batch materialization.

- The end of `FUN_0070ebd0` drains a deferred event list at `+0x38` via a `switch` over event IDs 0-9 that calls the per-event handlers `FUN_00710ec0` / `FUN_007110d0` / `FUN_00711230` / `FUN_00711f10` / `FUN_007121a0` / `FUN_007127f0` / `FUN_00712910` / `FUN_00712b00` / `FUN_00713430` / `FUN_00719370`. These are the per-frame state machines for the model.

## 1.12.1 cvar Option Set (smaller than 3.3.5)

The 1.12.1 client registers only these M2 cvars:

| cvar string | Address | Notes |
| --- | --- | --- |
| `M2UseZFill` | `0x0082e720` | pixel-shader z-fill toggle |
| `M2UseClipPlanes` | `0x0082e6e0` | clip-plane sort |
| `M2UseThreads` | `0x0082e6b0` | multithreaded model animation |
| `M2UseShaders` | `0x0082e67c` | enable shaders on models |
| `M2UsePixelShaders` | `0x0082e644` | enable pixel shaders (lighting) |
| `M2BatchDoodads` | `0x0082e60c` | batched doodad submission |
| `M2Faster` | `0x0082e5cc` | quality knob for M2 throughput |
| `M2FasterDebug` | `0x0082e58c` | debug variant of M2Faster |

**Missing in 1.12.1** (compared to 3.3.5):
- `M2BatchParticles`
- `M2ForceAdditiveParticleSort`

The shared runtime flag word bit assignments for the present cvars are not yet decompiled for 1.12.1; the 3.3.5 mapping (`M2UseZFill=0x1`, `M2UseClipPlanes=0x2`, `M2UseThreads=0x4`, `M2BatchDoodads=0x20`, `M2UsePixelShaders/M2UseShaders=0x8`) is a strong prior but not confirmed for 1.12.1.

## 1.12.1 Model Render-State Decisions

For each per-batch record at `+0x134`, `FUN_0070ebd0` reads:

- `*(byte *)(iVar10 + 4) & 0x10` → blend flag `0x100`
- `*(byte *)(iVar10 + 4) & 0x40` → blend flag `0x400`
- `*(byte *)(iVar10 + 4) & 0x20` → blend flag `0x200`
- `*(uint *)(iVar10 + 4) & 0x800` → blend flag `0x1000`
- `*(uint *)(iVar10 + 4) & 0x1000` → blend flag `0x2000`
- `*(char *)(iVar10 + 4) < 0` (sign bit) → `0x800` (only when `short at +0x2a == 2`)
- `*(uint *)(iVar10 + 4) & 0x100` → `0x4000` (only when `short at +0x2a == 2`)
- `*(uint *)(iVar10 + 4) & 0x200` → `0x8000`
- `*(uint *)(iVar10 + 4) & 0x2000` → `0x20000`
- `*(uint *)(iVar10 + 4) & 0x4000` → `0x40000`
- `*(char)((uint)*(undefined4 *)(iVar10 + 4) >> 8) < 0` → calls `FUN_007b4ba0(0,1)` (likely a "no-shader" fallback path)

`short at +0x2a` selects the material type (1, 2, or 3) and routes through `FUN_007b8770` / `FUN_007b8c50` / `FUN_007b91e0` (vtable factory).

`short at +0x2c` selects the texture-blend mode (0, 1, or 2) and feeds into `FUN_007b4f60`. The `uVar11` (large value) is the per-batch texture-bound or material-handle field.

`short at +0x28` selects a vertex-billboard or vertex-color mode and feeds the `uStack_28` switch with cases 1, 2, 4, 5, 6. Cases 1/2 = opaque-like; 4/5/6 = transparent modes.

The 1.12.1 per-batch record is **dramatically simpler** than 3.3.5's per-batch / per-submesh record set. The 3.3.5 reader should not be used to decode 1.12.1 records — it will silently misread the field layout.

## Gap Analysis vs Existing `wow-viewer` M2 / M2Chunked

The current `wow-viewer` M2 handling has three layers:

1. `WowViewer.Core.IO.M2.M2ModelReader` (3.3.5 era, MD20 format)
2. `WowViewer.Core.IO.M2Chunked.M2ChunkedModelReader` (1.12.1 era, MDLX chunked)
3. `WowViewer.Core.IO.M2Chunked.M2ModelReaderDispatcher` (dispatches on first-4-byte magic)

**The fundamental finding is that the dispatcher's premise is wrong for 1.12.1.** The 1.12.1 client uses `MD20` (not `MDLX`) for its `.mdx` files, and 3.3.5's `M2ModelReader` cannot read 1.12.1's `MD20` because:

| Issue | Detail | Existing code | Required fix |
| --- | --- | --- | --- |
| View table offset | 1.12.1 view sits at header `0x3c/0x40`; 3.3.5 view sits at `0x44/0x48` | `M2ModelReader.ViewCountOffset = 0x44` | Need an era-aware view-offset table; the 1.12.1 offset is `0x3c` |
| Sequence stride | 1.12.1: 0x6c (108); 3.3.5: 0x40 (64) | `SequenceStride = 0x40` | Add `SequenceStride1121 = 0x6c` |
| Light stride | 1.12.1: 0xc (12); 3.3.5: 0x9c (156) | `LightStride = 0x9c` | Add `LightStride1121 = 0xc` |
| Camera stride | 1.12.1: 0x2c (44); 3.3.5: 0x64 / 0x74 | `CameraStrideClassic = 0x64`, `CameraStrideModern = 0x74` | Add `CameraStride1121 = 0x2c` |
| Camera per-frame | 1.12.1 splits camera vs camera-per-frame across two tables; 3.3.5 inlines | n/a | Add `CameraPerFrameStride1121 = 0xd4` and a separate table walk |
| Ribbon stride | 1.12.1: 0x7c (124); 3.3.5: 0xac / 0xb0 | `RibbonStrideClassic = 0xac`, `RibbonStrideModern = 0xb0` | Add `RibbonStride1121 = 0x7c` |
| Particle stride | 1.12.1: 0xdc (220); 3.3.5: 0x1dc / 0x1ec | `ParticleStrideClassic = 0x1dc`, `ParticleStrideModern = 0x1ec` | Add `ParticleStride1121 = 0xdc` |
| Vertex layout | 1.12.1: separate position + normal tables, each 12 bytes/entry; 3.3.5: interleaved 0x30-0x40 | n/a (3.3.5 interleaves) | Add a 1.12.1 vertex decoder that walks two `0xc/entry` tables and stitches positions + normals |
| Bone layout | 1.12.1: `u16` index pair; 3.3.5: 0x58 / 0x9c record | `BoneStride = 0x58` | Add `BoneStride1121 = 0x4` (or similar; the 1.12.1 bone table is a lookup pair) |
| Dispatcher | Routes by magic; only `MDLX` and `else` | `M2ModelReaderDispatcher.Read` checks for `MdxMagic.Mdlx`, falls through to `M2ModelReader` | 1.12.1 `.mdx` files have `MD20` magic — they reach `M2ModelReader`, which uses 3.3.5 strides. The fix is to detect 1.12.1 by `version` field (0x100 / 0x101) and dispatch to a 1.12.1-aware reader |
| M2Chunked reader | Reads `MDLX` magic and walks FourCC chunks | `M2ChunkedModelReader.ValidateChunkedMagic` rejects any non-MDLX file | 1.12.1 has no MDLX chunked files in the on-disk set; the M2Chunked reader is currently a dead branch for 1.12.1 (correctly) but the M2ModelReader is silently producing wrong output. **This is the actual bug.** |
| Spec 043 assumption | Spec 043 assumes 1.12.1 uses chunked `MDLX` | spec 043 § "User Story 1" | The spec premise is wrong for 1.12.1's `.mdx` files. 1.12.1 uses `MD20`, not `MDLX`. The 1.12.1 path is the 3.3.5 reader's responsibility, but with a 1.12.1 stride set |

The user-reported symptom — "Our existing MDX/M2 handling does not account for this version of the model format properly" — is now grounded: the 1.12.1 model on disk uses `MD20` magic, the dispatch routes it to the 3.3.5 `M2ModelReader`, and the 3.3.5 reader misreads nearly every per-record stride and the view-table offset.

## Recommended Fix Plan (non-authoritative until a spec is created)

These are research-stage recommendations, not a spec. A spec must be written before any of these are implemented.

1. **Add a 1.12.1 version detector to `M2ModelReaderDispatcher`.** Check the `version` field at offset 4. If `0x100` or `0x101`, route to a new `M2_1121ModelReader` (suggested name) instead of the 3.3.5 `M2ModelReader`. The dispatcher should still fall through to 3.3.5 for `0x108+`.
2. **Create `M2_1121ModelReader` in a new namespace `WowViewer.Core.IO.M2_1121`.** It reads the 1.12.1 `MD20` format using the stride table from this doc. No modification to the existing 3.3.5 reader.
3. **Map 1.12.1 fields to the existing `M2ModelDocument` shape.** Vertices, normals, materials, bones, sequences, textures, cameras, ribbons, particles, lights all need explicit 1.12.1 → canonical mappers.
4. **Map 1.12.1 view selection to the runtime.** The 1.12.1 view at header `0x3c/0x40` (0x2c stride, with LOD distance at `+0x28`) drives a runtime view selection. The existing `M2ModelDocument` already has a `viewCount` field; a `viewLodDistances` field may need to be added.
5. **Update spec `043-m2-chunked-mdx-classic-support` to reflect the new finding.** The current spec assumes 1.12.1 `.mdx` files are chunked. The reality is the opposite: they are `MD20` with legacy extension. The spec's "Chunked" namespace is now strictly a 0.5.3 / pre-1.x lane and a research-only 2.x-pre-2.0.0 lane; the 1.12.1 lane moves to a 1.12.1 MD20 lane.
6. **Capture the cvar / runtime flag bit assignments for 1.12.1.** Not decompiled yet in this pass. The 3.3.5 mapping is a strong prior but should be confirmed for 1.12.1 before parity work begins.
7. **Add a 1.12.1 test in `WowViewer.Core.IO.Tests`.** Use a staged `1.12.1.5875` `.mdx` from the WoWArchive. Read it via the new 1.12.1 reader; assert the same `M2ModelDocument` shape as the 3.3.5 reader, and assert specific 1.12.1 strides (e.g. `Vertices.Length > 0`, `Cameras.Length >= 0`, `Particles.Length == 0` for UI models, etc.).

## Open Questions

- **OQ-1**: The 1.12.1 0x101 "extra-1" table at header `0xdc/0xe0` is the largest single record in the file (0x1f8 per record, 29 nested sub-tables). It is not present in 0x100. Best current guess: it is the consolidated per-batch / per-submesh record that 3.3.5 splits into multiple smaller tables. **Status (spec 048)**: deferred. The 048 MVP reader walks the count/offset pair with bounds checks but does not decode the per-record sub-tables — the reader is fully populated for the rest of the document. Future slice (049 or 050) decodes the sub-tables.
- **OQ-2**: The 1.12.1 `*(byte *)(iVar10 + 4)` flag word controls blend modes 0x100 / 0x200 / 0x400 / 0x800 / 0x1000 / 0x2000 / 0x4000 / 0x8000 / 0x20000 / 0x40000. The flag layout overlaps with 3.3.5 but the bit meanings are likely different. **Status (spec 048)**: deferred. The 048 reader passes the file's flags word through to `M2ModelDocument.Flags` without re-interpretation. The 1.12.1-specific blend bit mapping is a future slice.
- **OQ-3**: The 1.12.1 vertex table uses 12-byte entries (0xc/entry) instead of 3.3.5's interleaved 0x30-0x40. **Status (spec 048)**: deferred. The 048 MVP does not parse vertex or normal records (the `M2ModelDocument` schema does not yet absorb them, and the existing 1.12.1 stride 0xc lacks the field-level evidence needed to map them). The reader's view count, sequence count, color count, texture weight count, etc. are all populated, providing the seams for a future vertex-walker slice.
- **OQ-4**: The 1.12.1 light "record" at 0xc/entry. **Status (spec 048)**: confirmed as a 3-float record. The 048 reader parses `LightCount` records at `0xc/entry` with the first 8 bytes as `(float positionX, float positionY)` and the trailing 4 bytes as a `float` radius (per the spec 048 reader's `ReadLights` implementation, sourced from a re-derivation of the Ghidra constants and validated against the 1.12.1 fixture). The remaining light state (color, attenuation, etc.) is runtime-default for 1.12.1, matching the original OQ-4 hypothesis.
- **OQ-5**: The 1.12.1 M2 cvar bit mapping. **Status (spec 048)**: still open. The 048 reader does not consume cvar state. The 3.3.5 mapping is a strong prior for future work. Spec 048's `M2ModelDocument.Flags` is the file's per-batch flag word, not the cvar-derived runtime flag word.
- **OQ-6**: How 1.12.1 handles external `.anim` files. **Status (spec 048)**: confirmed out of scope. The 048 reader does not look for external `.anim` files. The 1.12.1 binary's lack of any animation cvar (see cvar table above) supports the original hypothesis: 1.12.1 either inlines animations into the sequence records or has no multithreaded animation seam at all.

## Notes for Follow-Up Sessions

- The 1.12.1 binary is rich and well-structured. The decompilation in this pass was driven entirely by string xrefs from the `M2*.cpp` source-path strings. The next pass should follow the cvar registration site at `0x00402760` (xrefs from the M2 cvar literals) to confirm the bit mapping.
- The view-record sub-table relocator `FUN_0071cd10` should be decompiled again with the view-record context to confirm the 9 nested tables.
- The render-event switch at the end of `FUN_0070ebd0` (events 0-9) is a good target for a future pass that wants to map 1.12.1 per-frame state machine calls back to canonical M2 runtime operations.
- The 1.12.1 `M2Cache.cpp` references "Corrupt skin profile data" in the 3.3.5 path. The 1.12.1 path likely has no `.skin` companion files at all — the 0.5.3 era started with inline skins; 1.12.1 may or may not have separate skins. The `FUN_0071d6c0` view-selection logic suggests 1.12.1 does have a notion of multiple skin profiles (the view table is the "skin profile" table in this era). External `.skin` files for 1.12.1 are an open question.
