# WoW 5.0.1.15464 ADT/WDT Definitive Guide

**Status:** Evidence-backed working definitive guide. Native static extraction is
complete for the file-family, MCNK wrapper, map-table admission, and initial
dead/dormant rendering inventory. The viewer's split-reader/runtime slice and
explicit cross-era serialization boundary are implemented; native blend/shader
seam semantics and a native split-family writer remain open.

**Client:** `Wow.exe`, Mists of Pandaria 5.0.1.15464, PE x86, image base
`0x00400000`.

**Primary evidence:**
[`research-ghidra-5.0.1.md`](../../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/research-ghidra-5.0.1.md)
and
[`5.0.1-dead-dormant-partial-rendering.md`](../../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/evidence/5.0.1-dead-dormant-partial-rendering.md).
This guide follows the structure of the legacy
[`ADT_WDT_Format_Specification.md`](../../../gillijimproject_refactor/docs/ADT_WDT_Format_Specification.md)
and
[`ghidra-definitive-guides.md`](../../../gillijimproject_refactor/docs/ghidra-definitive-guides.md).

This document records what the 5.0.1 binary proves. It does not turn viewer
behavior, wiki terminology, or a string-only search into native proof.

## 1. Scope and evidence

The Ghidra program contained 38,405 functions, 175,352 symbols, 790 data types,
and preserved 5.0.1 source/debug strings for terrain, WMO, liquid, renderer, and
resource paths. The pass was static and read-only: no bytes, labels, comments,
types, functions, or analysis settings were changed.

Evidence levels used here:

- **Confirmed:** direct decompiler control flow, token checks, offsets, counts,
  or assertions.
- **Strongly indicated:** multiple direct xrefs/callers and consistent field or
  resource use, but semantic naming remains incomplete.
- **Unconfirmed:** compatible with the binary or viewer but not mapped to a
  native parser/consumer in this build.
- **Negative string result:** a literal search found nothing; this never proves
  a numeric FourCC or stripped identifier is absent.

## 2. Version history and family matrix

The native path investigated in 5.0.1 uses ADT version 18:

```text
MVER payload: 0x0012
```

The map-area loader selects one object and one texture companion for one of two
LOD bands:

| Logical role | Band 0 | Band 1 | Cache type |
|---|---|---|---:|
| Root terrain | `{map}_{x}_{y}.adt` | `{map}_{x}_{y}.adt` | `0` |
| Object split | `{map}_{x}_{y}_obj0.adt` | `{map}_{x}_{y}_obj1.adt` | `1` / `2` |
| Texture split | `{map}_{x}_{y}_tex0.adt` | `{map}_{x}_{y}_tex1.adt` | `3` / `4` |

The loader asserts `lodBand <= 1`. This is direct evidence for `_obj1.adt` and
`_tex1.adt` support in this path. The investigated path does **not** construct
`_lod.adt`; the only `_lod` hit found in the relevant search was the graphics
extension `GL_EXT_texture_lod_bias`. Another subsystem or another client build
could still use that suffix, but it is not a substitute for the proven band-1
pair here.

## 3. High-level native load flow

The recovered flow is:

```text
WDT/map-table entry
        │
        ├─ require CMapTableEntry::Flag_Exists (0x1)
        │
        ▼
MapArea selects one LOD band
        │
        ├─ root {map}_{x}_{y}.adt
        ├─ selected _obj0/_obj1.adt
        └─ selected _tex0/_tex1.adt
        │
        ▼
Each file-data object: MVER + top-level records + 0x100 MCNK records
        │
        ▼
Same MCNK index merged across resident slots
        │
        ▼
World coordinates, bounds, liquid, placements, and render state
```

The crucial admission distinction is:

1. **Map-table existence:** the WDT/map table says the tile is an area.
2. **Root payload presence:** the root ADT contains terrain data.
3. **Companion presence:** object/texture files provide additional data.

The native area creation path requires item 1. A split companion by itself is
not proof that a terrain area should be created. This is the leading native
explanation for a viewer showing a tile whose local MCVT/MCCV data is valid but
whose tile admission is wrong.

## 4. WDT and tile admission

The relevant native functions are:

| Role | Function |
|---|---:|
| Area creation/admission loop | `0x00bb4a70` |
| Map-table accessor | `0x00b8f8f0` |
| Map-table accessor/validation | `0x00b8f830` |

The admission bit is:

```c
enum CMapTableEntryFlags {
    Flag_Exists = 0x00000001,
};
```

The exact native `MAIN` population algorithm for sparse, split-only, or
placeholder entries remains open. What is proven is the consumer-side gate:
the area path checks the existence bit before attaching/loading the area.

### Viewer consequence

The viewer now preserves the distinction between the root and selected companion
bands during supported split loading, but map-table existence is not yet the
uniform owner of tile admission in every fallback/conversion path. File discovery
should remain diagnostic/input data; map-table existence should own tile admission
unless separate evidence proves a deliberate fallback policy.

## 5. ADT top-level structure

The native file-data parser at `0x00bb7010` requires `MVER` and asserts version
`0x0012`. The top-level parser at `0x00bb6f10` retains at least `MTXF`, `MTXP`,
and `MTEX` data and requires exactly `0x100` outer `MCNK` records.

| Record | Native status | Notes |
|---|---|---|
| `MVER` | Confirmed | Payload must be `0x0012` |
| `MTXF` | Confirmed retained | Native top-level parser stores/retains it |
| `MTXP` | Confirmed retained | Native top-level parser stores/retains it |
| `MTEX` | Confirmed retained | Texture-name/data path |
| `MCNK` | Confirmed | Exactly 256 outer records per file-data object |
| `MHDR`, `MCIN`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF` | Existing ADT-family expectation; native 5.0.1 mapping incomplete in this pass | Do not present the viewer's full order as recovered native order |

`MHID`, `MDID`, and `MCXH` remain unconfirmed in this binary. The current
viewer recognizes them in places, but the native parser location was not proven.
The absence of literal strings is not enough to reject them because the MCNK
dispatcher compares numeric FourCC values.

## 6. File-data cache and resident slots

The cache-key routine at `0x00bb6c70` packs:

```text
key = ((mapId << 4 | adtFileType) << 6 | y) << 6 | x
```

It validates:

```text
x < 64
y < 64
adtFileType < 0x10
mapId < 0x10000
```

The cache namespace is wider than the active resident array. The merge routine
at `0x00bb0a30` uses three resident file-data slots:

| Slot | Source | MCNK treatment |
|---:|---|---|
| `0` | Root ADT | Consume the 128-byte MCNK header |
| `1` | Selected object companion | Headerless MCNK payload |
| `2` | Selected texture companion | Headerless MCNK payload |

The selected band determines whether slots 1 and 2 are the `0` or `1` suffix
pair. The cache still distinguishes at least five file types (`0` through `4`),
even though only three are resident in one merge operation.

## 7. MCNK wrapper and header contract

Every file-data object contains 256 outer `MCNK` records. Split records are not
a flat stream of bare subchunks. They retain the outer `{FourCC, size, payload}`
MCNK wrapper, but the 128-byte per-chunk header is absent from the payload
consumed by the native split constructor.

### Root/headered record

The constructor/hand-off at `0x00ba37d0` validates `MCNK`, copies the header,
then advances past 128 bytes before dispatching subchunks. Important mappings,
relative to the MCNK token, are:

| Input offset | Width | Native object offset | Observed role |
|---:|---:|---:|---|
| `0x08` | 4 | `0x88` | Flags |
| `0x14` | 2 | `0x94` | Layer-like count |
| `0x18` | 2 | `0x90` | Reference-like count |
| `0x3C` | 4 | `0x8C` | Area-like header word |
| `0x40` | 2 | `0x92` | Reference-like count |
| `0x44` | 2 | `0x78` | Chunk index-like field |
| `0x46` | 2 | `0x7A` | Chunk index-like field |
| `0x48` | pointer/raw | `0x98` | Header pointer |
| `0x58` | pointer/raw | `0x9C` | Header pointer |
| `0x70` | 4 | `0x64` | Coordinate/index source |
| `0x74` | 4 | `0x68` | Coordinate/index source |
| `0x78` | 4 | `0x6C` | Chunk base/vertical-bound source |

After the header, the subchunk stream begins at token-relative offset `0x88`
and the remaining size is reduced by `0x80`.

### Split/headerless record

For slots 1 and 2, the native merge path calls the same chunk machinery without
consuming/copying the root header. The payload begins directly with subchunk
records at offset zero:

```text
outer MCNK token + size
    └─ MCVT/MCNR/MCLY/...
```

The viewer's `SkipHeader` behavior is directionally correct, but it must be
applied to both selected object and selected texture companions, including
band-1 files.

## 8. MCNK subchunk dispatcher

The dispatcher at `0x00ba3050` walks `{FourCC, payloadSize, payload}` records.
Each iteration advances by `8 + payloadSize` and the final remaining size must
be zero.

| FourCC | Native action | Count/size rule |
|---|---|---|
| `MCVT` | Stores height payload pointer | Variable payload |
| `MCNR` | Stores normal payload pointer | Variable payload |
| `MCLY` | Copies layer records into object buffer | `payloadSize / 0x10`, 16-byte entries |
| `MCAL` | Stores alpha payload pointer | Variable payload |
| `MCSH` | Stores shadow payload pointer | Variable payload |
| `MCRF` | Stores object-reference payload pointer | Variable payload |
| `MCRD` | Stores reference payload pointer | `payloadSize / 4` |
| `MCRW` | Stores reference payload pointer | `payloadSize / 4` |
| `MCLQ` | Stores old-liquid payload pointer | Variable payload |
| `MCCV` | Stores vertex-color payload pointer | Variable payload |
| `MCLV` | Stores vertex-light payload pointer | Variable payload |
| `MCBB` | Stores blend-batch payload pointer | `payloadSize / 0x14`, capped at `0xff` |
| `MCDD` | Stores auxiliary payload pointer | Payload size `8` or `0x20`; `0x20` sets flag bit `0x1` |
| `MCMT` | Stores first payload word | Semantic name remains open |

This table is native proof for `MCBB`, `MCRD`, `MCRW`, `MCDD`, and `MCMT`; they
are not merely viewer-invented extensions. Conversely, `MHID`, `MDID`, and
`MCXH` are not promoted to native 5.0.1 chunks until numeric-token tracing finds
their consumer.

## 9. Coordinates and bounds

The terrain initialization path at `0x00ba8620` applies:

```text
world[0x64] = -(float)field[0x74] * 33.333332 + 17066.666
world[0x68] = -(float)field[0x70] * 33.333332 + 17066.666
```

The bounds path at `0x00ba7de0` uses the same `33.333332` chunk scale and
`17066.666` origin. It scans `MCVT`, adds the chunk base at object offset
`0x6C` to vertical bounds, and writes six AABB values at offsets `0x34`–`0x48`.
An auxiliary linked structure at `0x10C` can provide additional vertical bounds;
otherwise sentinel values are used.

The axis labels of the two chunk index fields remain intentionally unresolved.
The arithmetic, scale, and origin are confirmed; semantic renaming awaits the
reader/renderer coordinate cross-check.

## 10. Liquids, blend data, and WMO seams

### Liquids

`0x00ba78d0` consumes the `MCLQ` payload at object offset `0x134`, checks four
successive flag bits beginning at `0x04`, creates an entry for each enabled
liquid slot, and binds an 8-by-8-style render grid. A compact liquid table can
override dimensions; the final vertex-grid size follows:

```text
(decodedHeightCells + 1) * (decodedWidthCells + 1)
```

A build-specific parent value `0x212` remaps the third liquid slot to slot
`0xF`. The adjacent `MapChunkLiquid.cpp` and `LiquidGeomFactories.cpp` paths
show format-specific constructors and an explicit unfinished default branch; see
the focused
[`dead/dormant evidence note`](../../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/evidence/5.0.1-dead-dormant-partial-rendering.md).

### Blend batches and alpha/shadow paths

| Function | Evidence |
|---:|---|
| `0x00b9c7a0` | reads parent blend mesh index/vertex counts and batch count |
| `0x00ba96d0` | `Blend Batch` construction/logging references |
| `0x00badb80` | checks `HasBlendBatches()` and `blendBatchList` |
| `0x00bb3130` | asserts `m_blendTexture == 0` during resource handling |
| `0x00b66330` | `CMapChunk::UnpackAlphaBits(): Bad genformat.` |
| `0x00b65ce0` | `CMapChunk::UnpackAlphaShadowBits(): Bad genformat.` |

These anchors prove a native blend/batch path, but the complete alpha/height
equation and WMO-to-terrain seam vertex ownership are not established. The viewer
must not treat a generic height-blend implementation as native parity until the
batch and shader inputs are recovered.

### WMO reader boundary

The WMO reader family includes:

- `0x00b843f0`: requires WMO `MVER == 0x0011`, validates `MOGP`, copies group
  bounds/counts, and derives format selectors from group flags.
- `0x00b82030`: validates material indices and uses a `0x10`-byte material
  stride.
- `0x00b84ae0`: loads group information with a `0x20`-byte stride and formats
  group suffixes as `_%03d`.
- `0x00b83080`: top-level WMO chunk loop with table-count derivations including
  `payloadSize / 0xC`, `payloadSize / 0x18`, and `payloadSize >> 4`.

This is strong WMO parsing evidence, not yet proof of the terrain/WMO seam
algorithm. Seam ownership remains in the `MapChunkRender`, `MapRenderChunk`,
`MapRenderChunkState`, and `MapRenderChunkBatch` families.

## 11. Dead, dormant, and unsupported native paths

The binary contains real incomplete and optional subsystems, but they must not
be collapsed into one “dead code” bucket:

| Classification | Native examples | Viewer implication |
|---|---|---|
| Partial | `LiquidGeomFactories.cpp` default branch; `CSimpleEditBox.cpp` FIXME | Preserve unsupported/default behavior |
| Capability gated | DepthCache; GBuffer; hardware PCF | Separate unavailable hardware from unused code |
| Optional toggle | Texture atlas; doodad batching; particle batching | Feature state can change resource/batch shape |
| Lifecycle contract | `MapArea.h`; `MapRenderChunkState.h` assertions | Use ownership invariants for leak/residency diagnostics |
| String-only developer lead | `development.MPQ`, `ERR_BG_DEVELOPER_ONLY`, `M2FasterDebug` | Require xrefs/registration proof |
| Low direct reachability | 2,184 zero-direct-xref candidates | Search indirect tables before dead-code labels |

The complete classification table and excerpts live in
[`5.0.1-dead-dormant-partial-rendering.md`](../../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/evidence/5.0.1-dead-dormant-partial-rendering.md).

## 12. Current WowViewer delta

The first native-aligned split loading slice is now implemented:

- `AdtTileFamily` and `AdtTileFamilyResolver` retain root, `_obj0`/`_obj1`, and
  `_tex0`/`_tex1` band identity. `_lod.adt` remains an unproven native path and
  is not treated as a substitute for band 1.
- `WowFileDetector` and the shared file-kind model classify the companion suffix
  bands without collapsing them into one generic object/texture category.
- `StandardTerrainAdapter` loads the root plus the selected object/texture band
  and routes companion `MCNK` payloads through the headerless form. Root `MCNK`
  headers remain distinct from companion payloads.
- The shared texture reader and adapter preserve sparse MCIN physical-slot
  identity for the supported paths. The compact `LkAdtData.Chunks` model and
  remaining merger/transfer consumers are not yet a complete slot-aware document
  model.
- `_obj1`/`_tex1` support is present in discovery and runtime loading, but band-1
  coverage is not yet complete across every converter, merger, and texture-
  transfer path.
- The viewer recognizes `MHID`, `MDID`, and `MCXH`, but their exact native
  5.0.1 consumers remain evidence gaps. No native shader-height or WMO-seam
  parity claim is made.
- The map-table `Flag_Exists` rule remains a native admission contract that the
  fallback/conversion paths still need to apply uniformly.

These are implementation status boundaries, not a claim that the remaining
native rendering or serializer contracts have been recovered.

### 12.1 Explicit serialization target boundary

The map-converter UI and Core contract now select a target independently from the
source family:

| Source family | Target | Current behavior |
|---|---|---|
| Alpha 0.5.3 WDT | Alpha 0.5.3 monolithic WDT | Supported identity copy; non-lossy |
| Alpha 0.5.3 WDT | WotLK/LK v18 monolithic ADT | Supported normalization; lossy |
| Split ADT family | Alpha 0.5.3 monolithic WDT | Supported flattening; lossy |
| Split ADT family | WotLK/LK v18 monolithic ADT | Supported monolithic down-conversion; lossy |
| Any supported source family | Cataclysm/MoP split ADT family | Blocked: no native split writer is registered |

`LkAdtWriter` is guarded at the target-aware converter boundary and accepts only
the LK v18 target. It must not be used to label a monolithic result as native MoP
split output. The missing split writer must preserve the root plus independent
object/texture bands, 256 physical MCIN slots, root-only MCNK headers, and modern
split-only state before the MoP target can be enabled.

The UI also routes split-to-Alpha through the Alpha converter rather than the
split-to-LK command, and passes the selected loose split directory separately
from the archive client root for split-to-LK conversion.

## 13. Quick-reference constants and addresses

| Item | Value |
|---|---:|
| Image base | `0x00400000` |
| ADT `MVER` | `0x0012` |
| Native WMO `MVER` | `0x0011` |
| MCNK header | `0x80` bytes |
| Outer MCNK records per file-data object | `0x100` / 256 |
| Chunk scale | `33.333332` |
| World-origin term | `17066.666` |
| Map-table exists flag | `0x1` |
| Cache X/Y range | `< 64` |
| Cache map-id range | `< 0x10000` |
| Cache file-type range | `< 0x10` |
| `MCLY` entry stride | `0x10` |
| `MCRD`/`MCRW` element size | `4` |
| `MCBB` entry stride | `0x14` |
| `MCDD` accepted payload sizes | `8`, `0x20` |

## 14. Address discrepancy and open questions

One native path-builder address must be corrected before the function map is
called final:

- Earlier notes identify the ADT family builder as `0x00bb9490`.
- A later caller inventory contains `FUN_00b94990`.
- An endpoint resolution around `0x00bb9490` also produced a different function
  boundary.

Until this is rechecked against the exact Ghidra listing, file-family claims
should cite the loader family and surrounding confirmed functions rather than
treating one spelling as a final symbol identity.

Remaining evidence gates:

1. Exact WDT/MAIN population and sparse-tile semantics.
2. Relationship between suffix band `0`/`1` and broader internal `LOD_COUNT`.
3. Numeric FourCC consumers for `MHID`, `MDID`, and `MCXH`.
4. Complete `MCBB` record semantics and blend/shader inputs.
5. Native height-texture equations and WMO seam vertex ownership.
6. Indirect callers for liquid factories and settings callbacks.
7. Runtime confirmation of optional/capability paths on fixed hardware.

## 15. Definitive answer

For the investigated WoW 5.0.1.15464 path, the reliable terrain contract is:

```text
WDT/map-table Flag_Exists admits the area
root ADT + selected object/texture band provide data
all three file-data objects contain 256 outer MCNK records
only the root MCNK consumes the 128-byte header
split MCNK payloads are headerless subchunk streams inside MCNK wrappers
the native dispatcher includes MCBB/MCDD/MCMT/MCRD/MCRW
```

The native client also retains optional and unfinished renderer infrastructure.
The liquid factory's unsupported default branch is a real partial path; GBuffer,
DepthCache, atlas, and batching systems are not proven dead. The viewer now has a
supported, tested split-reader/runtime slice and an explicit writer-target gate,
but it does not yet have a native MoP split serializer. Split-to-LK and
split-to-Alpha are deliberate lossy down-conversions, not round-trip native MoP
output. Native blend/seam equations, WDT/MAIN population semantics, remaining
compact-slot consumers, and address-resolution questions remain open.
