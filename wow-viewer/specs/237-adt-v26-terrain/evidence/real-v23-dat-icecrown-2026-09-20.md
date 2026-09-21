# Spec 237 Evidence — Real v23 DAT Sample (IcecrownCitadel) + Path-Picker Load Blocker

Date: 2026-09-20

## Operator report

> "v22 DAT files don't load, it errors out. `E:\WC2\wrat2\world\maps\IcecrownCitadel`"

Screenshot status line: `No DAT v26 (AHDR-family) terrain files with ALOC found in
I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0.`

Operator context on the corpus as a whole:

> "These are loose files all over the game in the MPQs, often mis-labelled with extensions like
> `.dat.test`, `.dat.error`, `.what` — these are all the same type of DAT file, things that never
> should have shipped in client data, because it's all developer project data files. If we scour the
> listfile for these files, we can find them. I renamed all of the weird file extensions to `.dat` by
> hand, to make the viewer try to load them at all."

## Root cause: the picker, not the format

The folder named in the status line is the **process working directory**, not the operator's path. The
files were never scanned.

[`ImGuiPathPicker.ResolveSelection`](../../../src/viewer/WoWViewer/ImGuiPathPicker.cs) returned
`_currentDirectory` and ignored `_pathInputBuffer`. The path bar commits only on **Enter** or the **Go**
button, so pasting a path and clicking "Use Folder" silently discarded it and fell back to
`Directory.GetCurrentDirectory()`. This affected **every** picker in the viewer (CASC install, exports,
overlays, loose overlays, PM4 roots), not just the DAT folder.

Corollary: the hand-renaming of extensions was **not required**. `AhdrTerrainAdapter` calls
`Directory.EnumerateFiles(folder)` with no pattern and content-sniffs each file via
`AdtAhdrReader.IsAhdrFamily`; `.dat.test`, `.what` and extensionless files are all accepted already.
The loads failed for the picker reason alone.

## Measurement: the IcecrownCitadel sample

Three files in `E:\WC2\wrat2\world\maps\IcecrownCitadel`. Version fields read directly from the bytes:

| File | Bytes | MVER | AHDR version |
|---|---|---|---|
| `area_27_25.dat` | 580,867 | 23 | 23 |
| `area_28_25.dat` | 717,735 | 23 | 23 |
| `area_29_25.dat` | 381,895 | 23 | 23 |

Raw head, identical in all three:
`5245 564d 0400 0000 1700 0000 5244 4841 4000 0000 1700 0000 8100 0000 8100 0000`
— `MVER` size 4 value `0x17` = 23, `AHDR` size 0x40, first payload dword `0x17` = 23, 129×129
vertices, 16×16 chunks.

Chunk census (top-level walk closes exactly on EOF for all three, 0 unaccounted bytes):

| Chunk | 27_25 | 28_25 | 29_25 | Size |
|---|---|---|---|---|
| `MVER` | 1 | 1 | 1 | 4 |
| `AHDR` | 1 | 1 | 1 | 64 |
| `AVTX` | 1 | 1 | 1 | 132,100 = (129² + 128²) × 4 |
| `ANRM` | 1 | 1 | 1 | 99,075 = (129² + 128²) × 3 |
| `ATEX` | 1 | 3 | **0** | 49–52 (one name each) |
| `ACNK` | 256 | 256 | 256 | 64, 4208, 8352, 12496 |
| `AFBO` | 1 | 1 | 1 | 72 |
| `ACVT` | 1 | 1 | 1 | 132,100 = 33,025 × 4 |

Absent in all three: `ALOC`, `AOCH`, `ADOO`, `ADST`, `ACDO`, `ASHD`.

`ACNK` size is `0x40 + 4144n` for n `ALYR` sub-chunks (`4144 = 8 + 4136`, and `4136 = 0x20 fixed +
AMAP 4096`): 64 (n=0), 4208 (n=1), 8352 (n=2), 12496 (n=3). This confirms R3's "0x40 header if size >
0x40" walk on real v23 data — `area_29_25.dat` has **all 256 ACNK as bare 64-byte headers and no
`ATEX` at all**, i.e. a fully untextured terrain-only tile.

### This falsifies a standing research claim

[`research.md` R2](../research.md) said:

> "**v26 measured**: no `AFBO`; `ACVT` in every file. The v22/v23 claims ('v22 has no `AFBO`/`ACVT`')
> stay unverified because no real v22/v23 files exist."

Real v23 files now exist in two independent samples, and this one carries **both** `AFBO` (72 bytes)
and `ACVT` (132,100 bytes). The wiki-derived "v22 has no `AFBO`/`ACVT`" claim does not hold for v23,
and `AFBO` is a chunk the v26 corpus never showed. `research.md` is corrected in the same change.

`AFBO` at 72 bytes is **not** decoded and not interpreted here; it is walked past. (v18 `MFBO` is 36
bytes = two 3×3 int16 planes; 72 is double that, but nothing in this sample tests why.)

## Decoder result on the real files

```
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- \
  adt-ahdr check --root "E:/WC2/wrat2/world/maps/IcecrownCitadel"
```

```
  area_27_25.dat: no ALOC chunk (tile location unknown)
  area_28_25.dat: no ALOC chunk (tile location unknown)
  area_29_25.dat: no ALOC chunk (tile location unknown)
files=3 ahdr=3 versions=23:3 unique-tiles=0 duplicate-tiles=0
files-with-diagnostics=3 acnk-index-mismatches=0
```

The only diagnostic is the expected missing `ALOC`, which
`AdtAhdrReader.TryParseTileLocationFromName` already covers from the `area_XX_YY` name (see
[v22-v23-filename-tile-location-2026-09-19.md](v22-v23-filename-tile-location-2026-09-19.md)).
`unique-tiles=0` is the CLI's ALOC-only tile count, not a load failure.

## Changes

| File | Change |
|---|---|
| [`ImGuiPathPicker.cs`](../../../src/viewer/WoWViewer/ImGuiPathPicker.cs) | New `TryCommitPathBar()`, called from `ResolveSelection()`: resolves a typed/pasted path on confirm, and shows an error instead of silently using the stale directory. SaveFile mode still accepts a new filename in an existing directory. |
| [`AdtAhdrReader.cs`](../../../src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs) | New `TryReadVersion(head, out version)`: per-file AHDR revision from a 256-byte head, no full decode. |
| [`AhdrTerrainAdapter.cs`](../../../src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs) | Added `ScannedFileCount` and `VersionCounts` / `VersionSummary`. |
| [`ViewerApp_CascAhdr.cs`](../../../src/viewer/WoWViewer/ViewerApp_CascAhdr.cs) | Zero-tile status now names the full path, the scanned-file count and the first skip reason; skip reasons are logged on the failure path too (previously only on success). Reader-side labels renamed. Model-info panel reports `Revisions:`. |
| [`ViewerApp.cs`](../../../src/viewer/WoWViewer/ViewerApp.cs) | Menu label renamed. |
| [`AdtAhdrV23Tests.cs`](../../../tests/WowViewer.Core.Tests/AdtAhdrV23Tests.cs) | Two tests for `TryReadVersion` (256-byte head; non-AHDR rejection). |

### UI naming

Reader-side surfaces no longer claim v26, because a folder can mix revisions:

- `Open DAT v26 Terrain Folder...` → `Open DAT Terrain Folder (v22/23/26)...`
- `DAT v26 Height Scale` → `DAT Terrain Height Scale`
- `Select a folder of DAT v26 terrain files (any file names)` → `...DAT terrain files (v22/23/26; any file name or extension)`
- Status/log/map-name prefixes `DAT v26` → `DAT`
- Model info gains `Revisions: v23x3` from the actual scan

**Export labels keep "v26" deliberately.** `AdtAhdrTileBuilder` hardcodes `MverVersion = 26` and
`Version = 26`, so the writer emits exactly one revision; calling that button "v22/23/26" would be
false.

## Verification

| Command | Result |
|---|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` | 0 errors (276 pre-existing warnings) |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Ahdr"` | **Passed! 15/15**, 0 failed, 0 skipped |
| `dotnet build wow-viewer/WowViewer.slnx -c Debug` | 0 errors |
| `dotnet test wow-viewer/WowViewer.slnx -c Debug` (scope gate) | 11 failures, **all pre-existing**: the identical set fails on a stashed clean tree (11 there vs 10 here; the difference is the flaky allocation test `WorldRenderFrameHistoryTests.Recording_DoesNotAllocate`). No regression from this change. |
| `adt-ahdr check --root "E:/WC2/wrat2/world/maps/IcecrownCitadel"` | 3/3 AHDR, versions 23:3, 0 ACNK index mismatches |

| Criterion | Evidence |
|---|---|
| The reported files are AHDR-family and decode | CLI `files=3 ahdr=3 ... acnk-index-mismatches=0` on the real operator path |
| Their revision is 23 | MVER and AHDR version fields read from bytes, all three files, hex quoted above |
| The load failure was the picker | Status line named the process working directory; `ResolveSelection` ignored `_pathInputBuffer` |
| Extension renaming is unnecessary | `Directory.EnumerateFiles(folder)` with no pattern + `IsAhdrFamily` content sniff |
| Revision is reported per file | `TryReadVersion` tests pass on a 256-byte head |

## Render witness (operator, 2026-09-20)

Operator confirmation: **v26 and v23 both render fine.** This supersedes the "no runtime load witness
is claimed" boundary in
[v22-v23-filename-tile-location-2026-09-19.md](v22-v23-filename-tile-location-2026-09-19.md), which was
written before the operator loaded the Kalimdor v23 tiles. The v23 read path — filename tile location,
`ALYR` texture layers, `ACVT`, the ÷36 height divisor — is visually confirmed, not merely inferred from
the v26 corpus.

**SUPERSEDED the same day:** real v22 files *were* in hand — `E:\WC2\wrat2\world\maps\Expansion01`
holds four MVER/AHDR 22 files, and they render. See
[first-v22-dat-render-2026-09-20.md](first-v22-dat-render-2026-09-20.md). The paragraph below was
written before that folder was examined and is kept for the record.

~~**v22 has never been seen.** No v22 file is in hand: both real non-v26 samples (Kalimdor and
IcecrownCitadel) are revision 23.~~ So the v22 leg of the new `DAT (v22/23/26)` label is a statement of
intended coverage, **not** a proven one, and research.md R2's "version-gated ACNK header layout" for
v22 has no test data behind it. Finding a genuine v22 file is the open item; the listfile scan below is
the obvious way to look for one.

## Proposed next step (not implemented — needs operator approval per §9.1)

Operator's idea: scour the listfile for AHDR-family files across client data, since they ship under
arbitrary extensions. A bounded version would be a `adt-ahdr scan --source <MPQ/CASC>` that
content-sniffs every listfile entry for `MVER`+`AHDR`/`AHDR` and writes a manifest of
path → revision → tile → chunk census. That is new scope and is **not** started.
