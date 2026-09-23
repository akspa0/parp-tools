# Spec 245 Evidence — State of Modern Write Support (MEASURED)

Date: 2026-09-20

## Operator observation

> "we should be writing modern ADT's but we have no support for any modern chunks or writers, we just
> read the bare minimum from modern game data, and it somehow renders quite well, despite having none
> of the real shaders"

Measured below. The observation is **correct**, with one nuance that matters for the current export
lane.

## What modern write support actually exists

| Piece | State |
|---|---|
| `MapConversionTargetFormat.MopSplitAdt` | **Declared**, with display name "Cataclysm/MoP split ADT family", command value `mop-split-adt`, and a parser |
| `MapConversionFormats.HasWriter(MopSplitAdt)` | **`false`** |
| `GetUnavailableReason(MopSplitAdt)` | *"The native split ADT writer is not implemented yet; no split input may be routed through LkAdtWriter."* |
| `LkAdtWriter.EnsureTargetFormat` | **throws** `NotSupportedException` for any target that is not `LkAdtV18` |
| Any consumer of `MopSplitAdt` outside the format-description file | **none** — zero call sites |

So there is a **prepared socket with nothing plugged in**. To the project's credit this is honest
rather than dangerous: selecting the split target is refused up front, and the monolithic writer
refuses to serialize a split document by accident. Nobody gets an LK file mislabelled as modern.

But the practical position is exactly as stated: **there is no modern ADT writer.**

## What modern read support actually exists

`MopAdtChunkParser` — the file named for modern ADT chunk parsing — contains **one** public method:

```
ParseMtxpChunk(ReadOnlySpan<byte> data, int textureCount)
```

That is the whole of modern-specific chunk parsing. Everything else the renderer needs comes through
the shared legacy-shaped readers, which is why "the bare minimum, and it renders quite well" is an
accurate description.

## Unknown chunks are collected — but never written back

`AdtRawChunkBlobCollector.Collect` already walks an ADT and captures every non-structural chunk
verbatim as `TerrainRawChunkBlob`, treating only `MVER`/`MHDR`/`MCIN`/`MCNK` as structural. That is
exactly the raw material a preserving writer would need.

Its consumers are: `AdtTensorPackBuilder`, `NpzTileSerializer`, `TerrainTileTensorPack`,
`AlphaWdtReader`, `AlphaTileData` — **the ML/dataset path only**. No writer accepts raw blobs, so
nothing that is read-but-not-understood can survive a round trip through any output format. Chunks we
cannot parse are preserved *for training data* and discarded *for files*.

This is the cheapest available half-step toward a modern writer: a passthrough that re-emits unparsed
chunks would make a modern→modern round trip lossless for everything we do not yet understand,
without understanding any of it.

## The nuance: LK v18 is NOT lossy for DAT v26

The observation implies the DAT v26 → LK v18 exporter (spec 247 US3) is throwing modern data away.
Measured over the full 700-file v26 corpus, it is not.

`ALYR` per `ACNK`, all 179,200 chunks:

| Layers | 0 | 1 | 2 | 3 | 4 | **5+** |
|---|---|---|---|---|---|---|
| Chunks | 174,108 | 1,089 | 598 | 1,588 | 1,561 | **0** |

**Maximum 4 layers — exactly LK v18's limit.** Modern client ADTs go to 8; v26 does not use them.

Full accounting of what v26 carries against what LK v18 can hold:

| v26 | LK v18 | Carried by the exporter |
|---|---|---|
| `AVTX` heights | `MCVT` | yes |
| `ANRM` normals | `MCNR` | yes |
| `ACVT` vertex colours | `MCCV` | yes — 179,200 chunks |
| `ALYR` ≤4 layers | `MCLY` (max 4) | yes — 13,293 |
| `AMAP` 4096 weights | `MCAL` big alpha | yes — 8,457 |
| `ACDO` placements | `MDDF`/`MODF` | yes — 5,309 |
| `ALOC` tile location | filename | yes |
| Area ids, holes, shadows, liquids | `MCNK`/`MCSH`/`MH2O` | source carries none |
| **`ADST`** (321 rows: uniqueId + FileDataID, no position) | **no equivalent** | **no** |
| **`AOCH`** (2048 B, all-zero, unexplained) | **no equivalent** | **no** |

The only genuine losses are `ADST` and `AOCH`. `ADST` rows carry no position and cannot be placed by
construction; `AOCH` is all-zero in every file in the corpus, so nothing observable is lost even
though its meaning is unknown. Both are now named in the export manifest (they were silently dropped
until this session — a real violation of spec 247 FR-013/SC-004, fixed here).

**Conclusion for the current lane:** v26 → LK v18 loses nothing v26 contains. The modern-writer gap is
real but it does not bite this conversion.

## Where the gap does bite

Writing **modern client data** — the spec 239/243/245 lane. A modern ADT family carries up to 8 layers
plus `MH2O`, `MTXP`, `MAMP`, `MCBB`, the `_tex0`/`_obj0`/`_obj1`/`_lod` split, and more. Reducing that
to LK v18 is genuinely lossy, and spec 243 already accepts that by being explicitly one-way
modern → legacy.

The asymmetry is the real finding: **this project can read modern data and write only legacy.** Any
modern→modern workflow — round-tripping, editing, or re-emitting a modern map — is impossible today.

## Suggested sequencing (not started, needs operator direction)

1. **Spec 245's inventory first.** You cannot write a format you have not enumerated. This spec exists
   precisely to list every discarded modern chunk and classify it representable / not representable /
   not understood. It is still Draft with no tasks generated.
2. **Raw-chunk passthrough** into a writer as the cheap half-step — preserves the unparsed without
   understanding it, and is independent of (1).
3. **A native split-ADT writer** behind the `MopSplitAdt` socket that already exists, once (1) says
   what has to go in it.

## Verification

| Action | Result |
|---|---|
| Grep for `MopSplitAdt` consumers outside `MapConversionFormat.cs` | 0 |
| `HasWriter(MopSplitAdt)` | `false`, with an explicit unavailable reason |
| Public methods in `MopAdtChunkParser` | 1 (`ParseMtxpChunk`) |
| `TerrainRawChunkBlob` consumers | 5, all ML/dataset; 0 writers |
| `ALYR`-per-`ACNK` census, 700 v26 files | max 4, 0 chunks with 5+ |
| `ADST` rows / `AOCH` chunks in v26 corpus | 321 / 699 |
| Export manifest after the fix | names `ADST` 321 and `AOCH` explicitly |

**Not claimed:** no modern writer was built, no chunk inventory was performed, and the renderer's
shader fidelity was not assessed.
