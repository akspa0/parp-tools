# Spec 237 Evidence — v22/v23 DAT Loading via Filename Tile Location

Date: 2026-09-19

## Operator request

"We need to be able to load any version of the DAT files. I have some more in
`E:\WC2\wrath\World\Maps\Kalimdor` to be analyzed/loaded." The files are part of the Kalimdor map,
a deleted region from the early Lost Isles pre-alpha era; the `XX_YY` in the filenames is enough to
place them.

## Analysis of the corpus

`E:\WC2\wrath\World\Maps\Kalimdor\area_51_31.dat` and `area_51_32.dat` (plus byte-identical `.error`
companions) are **DAT v23** (`MVER` = 0x17 = 23), AHDR-family, with the **same vocabulary as v26**:

| Chunk | Size | Meaning |
|---|---|---|
| `MVER` | 4 | 23 |
| `AHDR` | 64 | version 23, 129×129 vertices, 16×16 chunks |
| `AVTX` | 132100 | (129² + 128²) × 4 floats |
| `ANRM` | 99075 | (129² + 128²) × 3 bytes |
| `ATEX` × 4 | 23/42/42/42 | texture names |
| `ACNK` × 256 | 4208 each | header 0x40 + one `ALYR` (4136 = 32 fixed + `AMAP` 4096) |

**No `ALOC` chunk** (and no `AOCH`/`ADOO`/`ADST`/`ACVT`).

## Root cause

[`AhdrTerrainAdapter`](../../../src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs) required
`AdtAhdrReader.TryReadTileLocation` (ALOC) to place a file, so every v22/v23 file was skipped with
"no ALOC". The reader itself is version-agnostic (walks by FourCC) and already handles the v23
vocabulary.

## Fix

- [`AdtAhdrReader.TryParseTileLocationFromName`](../../../src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs:74):
  new public helper that reads the trailing `XX_YY` integers from a file name (`area_51_31.dat` →
  X=51, Y=31), in ALOC order.
- [`AhdrTerrainAdapter`](../../../src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs:45): when ALOC is
  absent, falls back to the filename location; v26 keeps its ALOC and is unaffected.

## Verification

| Command | Exit | Output |
|---|---:|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` | 0 | `Build succeeded. 0 Error(s)` |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~AdtAhdr"` | 0 | `Passed! - Failed: 0, Passed: 13, Skipped: 0, Total: 13` |

New [`AdtAhdrV23Tests`](../../../tests/WowViewer.Core.Tests/AdtAhdrV23Tests.cs) pins the filename
parser and that a v23-shaped file (MVER 23, no ALOC) is AHDR-family, reads without throwing, and
reports the expected "no ALOC" diagnostic.

## Proof boundary

Source + unit-test proof only at the time of writing. **No runtime load witness was claimed** — the
operator was asked to open `E:\WC2\wrath\World\Maps\Kalimdor` with **File → Open DAT Terrain Folder**
and confirm the two tiles render.

**Resolved 2026-09-20 (operator):** v23 renders fine, as does v26. The runtime witness is satisfied;
see [real-v23-dat-icecrown-2026-09-20.md](real-v23-dat-icecrown-2026-09-20.md). The menu label quoted
above was renamed from "DAT v26 Height Scale" to "DAT Terrain Height Scale" in the same change.

## Operator observation recorded (not a task)

v26 objects can carry a **negative uniqueID** — the operator suspects these are objects not meant to
ship in the client or be tracked. [`AdtAhdrObjectDefinition.UniqueId`](../../../src/core/WowViewer.Core/Maps/AdtAhdr/AdtAhdrTile.cs:113)
is read as `uint`, so a negative on-disk int32 surfaces as a large unsigned value. Worth keeping in
mind if a negative id is ever encountered; no change made here.
