# Spec 247 Evidence — LkAdtWriter Was Emitting Incomplete ADTs (FIXED)

Date: 2026-09-20

## Operator report

> "the lk adt's you write are not valid adt's, as they are missing many chunks that the format
> requires for proper LK client loading, or noggit loading for that matter. that's not ok. You must
> include every expected chunk, or else the files are not valid files!" … "subchunks too!"

Correct. This is **not** specific to the DAT export — `LkAdtWriter` is shared by 17 call sites
(`AlphaToLkConverter`, `SplitAdtToLkCommand`, `RosettaTilesetGenerator`, `NewMapCreatorService`, the
viewer's map save…), so **every LK file this project has ever written** had these defects.

## Defects found

| # | Defect | Effect |
|---|---|---|
| 1 | **`MCSE` never written** in any MCNK | Real LK tiles carry `ofsSndEmitters` → a zero-length `MCSE` with `nSndEmitters = 0`. Ours left both the chunk and the offset absent. |
| 2 | **`ofsMCCV` (MCNK +0x74) never set** | `MCCV` *was* being written into the file with its header offset left at 0 — **written where nothing can find it**. The v23/v26 exports produced 179,200 vertex-colour chunks that no reader could locate. |
| 3 | **MCNK flag `0x40` never set** | Even had the offset been right, the flag announcing MCCV was missing. |
| 4 | **`ofsMCLV` (MCNK +0x78) never set** | Same bug as #2 for the Cata+ lighting chunk. |
| 5 | **`MHDR.ofsMFBO` (+0x24) hardcoded to 0** | `MFBO` was written when flight bounds existed, but unreachable — and MHDR flag `0x1` (`mhdr_MFBO`) was never set. |
| 6 | **`MTXF` never written**, `MHDR.ofsMTXF` (+0x2C) hardcoded to 0 | WotLK's per-texture flag table, one `uint32` per `MTEX` entry, was entirely absent. |

Defects 2, 4 and 5 are the worst kind: the data was written but orphaned. `mccvOffset` and
`mclvOffset` were computed into local variables and then **discarded** — the header patch block wrote
only `0x14`–`0x30`.

## Verified against a real ADT, not from memory

Before changing anything I checked the MCNK sub-chunk offset base against a real file
(`H:\CLIENTS\test_data\original_development\painted\development_0_0.adt`), because the reader and
writer appeared to disagree by 8 bytes:

```
ofsMCVT = 136 (0x88)   ofsSndEmitters = 1180   nSndEmitters = 0   ofsMCCV = 724
```

`136` is measured from the **MCNK chunk tag**, i.e. `8 (tag+size) + 128 (header)`, and points at the
`MCVT` tag. That is exactly what `LkAdtWriter` already emitted, and `LkAdtReader`'s
`mcnkPayloadOffset + ofsMcvt` lands on the payload because its `+8` base cancels the sub-chunk tag.
**Both were already correct — I nearly "fixed" a non-bug.** The same file also confirmed `MCSE` in all
256 MCNKs and a populated `ofsMCCV`, which is what validated defects 1 and 2.

Caveat: that file is atypical (no `MTEX`/`MDDF`/`MCIN` at all), so it is weak evidence for what is
*required*. No pristine Blizzard ADT exists loose on this machine — all 16,453 `.adt` files under
`H:\CLIENTS` are this project's own `parp2026` output or `development` files. `MTXF` is therefore
added on the format definition, not on a measured exemplar.

## Fix

[`LkAdtWriter.cs`](../../../../src/core/WowViewer.Core.IO/Maps/LkAdtWriter.cs):

- Writes an empty `MCSE` in every MCNK; sets `ofsSndEmitters` (+0x58) and `nSndEmitters` (+0x5C).
- Sets `ofsMCCV` (+0x74) and `ofsMCLV` (+0x78) from the offsets it already computed.
- ORs MCNK flag `0x40` when MCCV is present.
- Sets `MHDR.ofsMFBO` and ORs MHDR flag `0x1` when MFBO is written.
- Writes `MTXF` (4 bytes per `MTEX` entry, zero-filled) and sets `MHDR.ofsMTXF`.

## Verification — every header offset resolves to its own tag

Re-exported both corpora and walked the output, checking that each MCNK header offset points at a
chunk whose tag matches the field name:

**v23 `IcecrownCitadel_28_25.adt`**

```
MHDR: flags=0, mcin=64, mtex=4168, mmdx=4331, mmid=4339, mwmo=4347, mwid=4355,
      mddf=4363, modf=4371, mfbo=0, mh2o=0, mtxf=657419
TOP: MVER MHDR MCIN MTEX MMDX MMID MWMO MWID MDDF MODF MCNK×256 MTXF
SUB: MCVT 256, MCNR 256, MCLY 33, MCCV 256, MCSE 256, MCAL 26
walk closes on EOF: True
misaddressed offsets: none
```

**v22 `Expansion01_25_37.adt`** (has objects and shadows)

```
TOP: MVER MHDR MCIN MTEX MMDX MMID MWMO MWID MDDF MODF MCNK×256 MTXF
SUB: MCVT 256, MCNR 256, MCLY 243, MCRF 115, MCSH 207, MCSE 256
walk closes on EOF: True
misaddressed offsets: none
```

Checked fields: `MCVT` 0x14, `MCNR` 0x18, `MCLY` 0x1C, `MCRF` 0x20, `MCAL` 0x24, `MCSH` 0x2C,
`MCSE` 0x58, `MCCV` 0x74. Before the fix, `MCSE` and `MCCV` were 0 in every chunk.

| Command | Result |
|---|---|
| `dotnet build WowViewer.Core.IO.csproj -c Debug` | 0 errors |
| LK writer regression tests (`LkAdt`, `LkToAlpha`, `MapConversion`, `RosettaTileset`, `DatToLk`) | **83 passed, 1 failed** |

The single failure is `AlphaToLkRoundTripTests.AlphaToLk_FlagContract_AllowsAlphaRoundTripThroughLkBytes`:
*"Alpha round-trip drift 0.9333 exceeds one byte step"* — an alpha **quantisation** drift, unrelated to
chunk structure, and present in the pre-existing failure set baselined at the start of this session on
a stashed clean tree.

## Still not claimed

**No file has been loaded in a 3.3.5 client or in Noggit.** Structural validity is proven — every
chunk present, every offset resolving, walk closing exactly on EOF — but that is not the same as
loading. That remains operator proof.

Known remaining gaps, deliberately not invented:

- **`MH2O`** is written only when liquid data exists; the DAT sources have none, so the exports carry
  no water. Not a writer defect.
- **`MCLQ`** (deprecated pre-WotLK liquid) is never written. Correct for LK, which uses `MH2O`.
- **`MCRF`** is written only when a chunk has refs. Offset 0 is the legal "absent" encoding, matching
  the real file examined, but if Noggit turns out to require it unconditionally that is a one-line
  change.
- `MTXF` values are all zero. The chunk's presence and its length agreeing with `MTEX` is what a
  reader needs; per-texture flag values would need a real source for them.
