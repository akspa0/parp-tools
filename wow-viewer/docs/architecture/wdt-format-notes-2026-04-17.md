# WDT Format Notes

This note records what the current `wow-viewer` codebase actually proves about WDT structure, especially the `MAIN` chunk and the `8`-byte vs `16`-byte per-tile record width.

## Short Answer

- In the current `wow-viewer` readers, the `8` and `16` values are **not pulled from an explicit field** inside the WDT.
- They are inferred from the byte length of the `MAIN` chunk payload divided by `64 * 64` tiles.
- The current evidence says this is a **WDT tile-index record layout difference**, not a proven "terrain subdivision" or "ADT mesh resolution" field.
- The current Alpha path uses `16` because Alpha monolithic WDTs appear to store embedded tile-offset records in `MAIN`.
- The current standard path uses `8` because later WDTs expose a compact per-tile flags/async-id record in `MAIN`.

## Where The Data Comes From

The current `wow-viewer` implementation reads the raw `MAIN` chunk payload and infers the entry size from its total byte length:

- `WdtTileIndexReader.ReadOccupiedTiles(...)`
- `WdtSummaryReader.Read(...)`

Current logic:

```csharp
byte[] mainData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Main) ?? [];
int mainCellSize = InferMainCellSize(mainData);
```

`InferMainCellSize(...)` currently treats these sizes as canonical:

- `8` bytes per tile for standard WDT `MAIN`
- `16` bytes per tile for Alpha WDT `MAIN`

It does **not** currently read a field like `MPHD.someValue == 8` or `MPHD.someValue == 16`.

## What `MAIN` Means In This Repo

### Standard-era WDT (`8` bytes per tile)

Current proven interpretation in `WdtSummaryReader`:

- first `uint32`: tile flags
- second `uint32`: async/load-related id or secondary per-tile value

The current reader names these flag bits:

- `0x1`: `HasAdt`
- `0x2`: `AllWater`
- `0x4`: `Loaded`

Anything else in the first `uint32` is currently treated as unknown flag bits.

This is why `WdtSummaryReader` can produce `WdtMainFlagsSummary` only when `mainCellSize == 8`.

### Alpha WDT (`16` bytes per tile)

Current proven interpretation in `AlphaEmbeddedAdtReader`:

- the first `int32` of each `16`-byte `MAIN` entry is used as an embedded ADT offset into the monolithic Alpha WDT container
- the remaining `12` bytes are **not yet semantically decoded** in this repo

Current Alpha tile resolution logic:

```csharp
int entryOffset = index * AlphaMainEntrySize;
offset = BitConverter.ToInt32(mainData, entryOffset);
return offset > 0;
```

That means the current Alpha path proves:

- `MAIN` is being used as a tile lookup table
- the first field of each Alpha entry is meaningful as an ADT offset

It does **not** prove:

- that the remaining `12` bytes are padding
- that they are terrain-resolution controls
- that they are chunk-subdivision metadata

## Why This Is Probably Not Terrain Mesh Resolution

The user hypothesis was:

- Alpha `16x16` might mean finer chunk or terrain subdivision
- later clients `8x8` might mean coarser subdivision
- modern clients might use an even smaller value to push mesh resolution higher

That is **not supported by the current code evidence**.

Why:

- WDT is a world-level map descriptor and tile index surface.
- In this repo, the `MAIN` cell width only changes how each **tile record** is decoded.
- Terrain mesh granularity is owned lower down by the ADT/MCNK/MCVT family, not by the WDT `MAIN` record width.
- The Alpha `16`-byte path is currently useful because it carries embedded tile-offset data, not because it tells the client how many terrain vertices a chunk should have.

So the safest current statement is:

- `8` vs `16` is a WDT `MAIN` record-layout distinction.
- It is **not** a proven terrain subdivision knob.

## What `MPHD` Currently Means Here

`WdtSummaryReader.IsWmoBased(...)` currently uses two signals:

- if `MPHD[8..12] == 2`, treat the WDT as WMO-based
- otherwise, if `MPHD[0..4] & 0x1 != 0`, also treat it as WMO-based

This is pragmatic behavior recovery for the formats currently seen in the repo.

Important constraint:

- the current code does **not** use `MPHD` to derive `MAIN` cell size
- the current code does **not** have a proven `MPHD` field that says "tile resolution = 8" or "tile resolution = 16"

## What Is Actually Proven By Tests

`WdtSummaryReaderTests` currently prove these cases:

- standard synthetic WDT: `MAIN` length = `64 * 64 * 8`
  - interpreted as `8`-byte records
  - standard `MAIN` flags summary is available
- Alpha synthetic WDT: `MAIN` length = `64 * 64 * 16`
  - interpreted as `16`-byte records
  - `MainFlags` is intentionally `null` because standard flag semantics do not apply
- development WDT real-data proof:
  - `MainCellSizeBytes == 8`
  - `1496` occupied tiles

## What Remains Open

The following points are still open research questions:

- the exact semantic meaning of Alpha `MAIN` bytes `4..15`
- whether Alpha `MAIN` is strictly row-major, column-major, or tolerant of both because of archive-era irregularities
- whether any later client family ever changed `MAIN` entry width away from `8` in a way that matters for WoW-era WDTs
- whether any `MPHD` field should be treated as a format discriminator for `MAIN` interpretation instead of relying only on payload length

## Practical Guidance For Future Work

- Keep treating `MAIN` cell width as a **record format signal**, not terrain tessellation metadata, unless real-file evidence proves otherwise.
- Keep Alpha and standard `MAIN` decoding separate.
- If a future client family is suspected to differ, validate with real WDT samples first and document the exact payload shape before changing the reader.
- If we later decode the rest of the Alpha `16`-byte record, update this note with a "proven" section instead of folding speculation into the standard summary path.

## Relevant Current Files

- `wow-viewer/src/core/WowViewer.Core.IO/Maps/WdtTileIndexReader.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/WdtSummaryReader.cs`
- `wow-viewer/src/core/WowViewer.Core/Maps/WdtSummary.cs`
- `wow-viewer/src/core/WowViewer.Core/Maps/WdtMainFlagsSummary.cs`
- `wow-viewer/src/viewer/WowViewer.App/AlphaEmbeddedAdtReader.cs`
- `wow-viewer/tests/WowViewer.Core.Tests/WdtSummaryReaderTests.cs`