# Spec 247 Evidence — US3: DAT → LK v18 ADT Export (DELIVERED)

Date: 2026-09-20

## What was delivered

One-way conversion of AHDR-family DAT tiles (v22/v23/v26) to Wrath-of-the-Lich-King ADT + WDT, with a
loss manifest. The four recovered Expansion01 v22 tiles now exist as `.adt` files.

**Deviation from the chosen phase order, stated plainly:** the operator chose Phase 1 (the v22 `AMAP`
codec) to go first. It did not fall — see
[237 evidence](../../237-adt-v26-terrain/evidence/v22-amap-codec-attempt-2026-09-20.md), extended
below. Rather than deliver nothing, US3 was built, because it is only partially blocked: everything
except texture blending carries. US2 (capture) is **not started**.

## Files changed

| File | Change |
|---|---|
| [`DatToLkAdtConverter.cs`](../../../../src/core/WowViewer.Core.IO/Maps/DatToLkAdtConverter.cs) | New. `AdtAhdrTile` → `LkAdtData`, plus `DatToLkConversionOptions` and `DatToLkConversionReport`. |
| [`AdtAhdrCommandSupport.cs`](../../../../tools/inspect/WowViewer.Tool.Inspect/AdtAhdrCommandSupport.cs) | New `adt-ahdr export-lk` command: walks a folder, writes ADT per tile, the WDT, and the manifest. |
| [`DatToLkAdtConverterTests.cs`](../../../../tests/WowViewer.Core.Tests/DatToLkAdtConverterTests.cs) | New. 6 tests on a synthetic v22-shaped tile (portable; the real corpus is outside the repo). |

No existing reader or writer was modified. `LkAdtWriter`, `LkWdtWriter`, `AdtAhdrTileSlicer` and
`AdtAhdrAlpha` are reused unchanged, so no new format writer entered the codebase (FR: Dependencies).

## Command

```powershell
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  adt-ahdr export-lk --root "E:/WC2/wrat2/world/maps/Expansion01" --map Expansion01
```

Output (defaults to `output/dat-lk-export/<map>/`, AGENTS.md §9.3):

```
  area_24_38.dat (v22) -> Expansion01_24_38.adt
  area_25_37.dat (v22) -> Expansion01_25_37.adt
  area_25_38.dat (v22) -> Expansion01_25_38.adt
  area_26_38.dat (v22) -> Expansion01_26_38.adt
files=4 skipped=0 tiles=4 versions=v22:4
chunks: source=999 written=999 empty-filled=25
layers=997 alpha-maps=0 dropped-no-alpha=1386
shadows=289 area-ids=997 mccv-chunks=0 objects=849 (skipped 0)
```

## Structural validation of the real output

Independent chunk walk of all four written `.adt` files:

| File | Bytes | Walk end | MCNK | MCVT | MCNR | MCLY | MCSH | MCRF |
|---|---|---|---|---|---|---|---|---|
| `Expansion01_24_38.adt` | 350,351 | 350,351 | 256 | 256 | 256 | 254 | 60 | 30 |
| `Expansion01_25_37.adt` | 450,661 | 450,661 | 256 | 256 | 256 | 243 | 207 | 115 |
| `Expansion01_25_38.adt` | 327,424 | 327,424 | 256 | 256 | 256 | 251 | 14 | 22 |
| `Expansion01_26_38.adt` | 312,644 | 312,644 | 256 | 256 | 256 | 249 | 0 | 0 |

Every file closes **exactly** on EOF with 0 unaccounted bytes, and carries the full LK top-level set:
`MVER MHDR MCIN MTEX MMDX MMID MWMO MWID MDDF MODF` + 256 `MCNK`.

Cross-check against the converter's own counters: MCSH 60+207+22+0 = **289** = reported shadows
carried; MCLY 254+243+251+249 = **997** = reported layers written. The report is not self-asserted —
it matches an independent read of the bytes.

## Criterion → evidence

| Criterion | Evidence |
|---|---|
| FR-010 writes LK v18 ADT + WDT | 4 `.adt` + `Expansion01.wdt` written; structure table above |
| FR-011 carries heights, normals, layers, shadows, area ids, objects | 999 chunks with MCVT+MCNR; 997 MCLY; 289 MCSH; 997 area ids; 849 MDDF/MODF |
| FR-012 addresses chunks by ACNK index, not ordinal | `Convert_AddressesChunksByAcnkIndex_NotByOrdinalPosition` — a source chunk stored at ordinal 0 with index (3,2) lands at (3,2), and (0,0) stays empty |
| FR-013 writes a loss manifest | `conversion-manifest.txt`, CARRIED/DROPPED sections + notes |
| FR-015 inch→yard conversion, stated | `Convert_ConvertsInchesToYards` (3600 in → 100 yd base); manifest note |
| Full 16×16 grid despite omitted source chunks | `Convert_WritesAFull256ChunkGrid_...`; 25 empty MCNK filled across the 4 tiles |
| Output re-reads | `WrittenAdt_ReadsBackWithItsTerrainAndAreaIds` round-trips `LkAdtWriter` → `LkAdtReader` |

## Verification

| Command | Result |
|---|---|
| `dotnet build wow-viewer/WowViewer.slnx -c Debug` | **0 errors** |
| `dotnet test ... --filter "FullyQualifiedName~Ahdr\|FullyQualifiedName~DatToLk"` | **Passed! 21/21**, 0 failed, 0 skipped |
| `adt-ahdr export-lk` on the real v22 corpus | 4/4 tiles, 999/999 chunks, 0 skipped |
| Independent chunk walk of the 4 outputs | 256 MCNK each, 0 unaccounted bytes |

## NOT proven — the output has not been loaded

**No client or viewer has opened these ADTs.** FR-014 (the exported map loads in the application's LK
path) and SC-003 (visually indistinguishable from the source) are **unmet**. What is proven is that
the bytes are structurally well-formed LK v18 and survive a writer→reader round trip. Loading is
operator-owned proof.

Two specific things a load will decide, neither of which measurement here can settle:

1. **Chunk axis orientation.** The DAT grid's row axis runs along ALOC tile Y while the renderer's
   runs along its tile X. Which way a standalone ADT should be written is not established, so the
   converter keeps the source's own ACNK indices and exposes `--transpose`. If the terrain comes out
   mirrored or rotated, that flag is the fix.
2. **Normal component order.** LK `MCNR` is written as (X, Z, Y) of a Z-up normal, copied from
   `AlphaToLkConverter`. Lighting will show if it is wrong.

## Known loss on this corpus (all v22)

| Field | Status |
|---|---|
| Texture blending beyond layer 0 | **Dropped** — 1386 layers. `AMAP` codec unidentified (Phase 1). |
| Vertex colours (MCCV) | Absent in source: v22 has no `ACVT`. |
| Liquids | Absent in every DAT sample ever seen. |
| Holes | Source value is 0 everywhere; written as 0. |
| MFBO flight bounds | Not carried (v22 has no `AFBO`; v23 does). |

Upper layers are deliberately **not** emitted when their alpha cannot be decoded: an alpha-less upper
layer renders opaque and would hide layer 0 completely, which is worse than omitting it.

## v23 run — the alpha path IS exercised

`export-lk --root "E:/WC2/wrat2/world/maps/IcecrownCitadel" --map IcecrownCitadel`:

```
files=3 skipped=0 tiles=3 versions=v23:3
chunks: source=768 written=768 empty-filled=0
layers=129 alpha-maps=48 dropped-no-alpha=0
shadows=0 area-ids=768 mccv-chunks=768 objects=0 (skipped 0)
```

Independent walk of the output:

| File | Walk closes | MCNK | MCLY | MCCV | MCAL bytes |
|---|---|---|---|---|---|
| `IcecrownCitadel_27_25.adt` | yes | 256 | 48 | 256 | 0 |
| `IcecrownCitadel_28_25.adt` | yes | 256 | 33 | 256 | **196,608** |
| `IcecrownCitadel_29_25.adt` | yes | 256 | 0 | 256 | 0 |

196,608 = **exactly 48 x 4096**, matching the reported `alpha-maps=48`. So the weights->sequential
conversion into 8-bit MCAL runs end to end on real v23 data, with **0 layers dropped**. `MCCV` lands
on all 256 chunks of all three tiles (v23 has `ACVT`; v22 does not).

Also measured here: **v23 carries area ids too** (768/768 non-zero), like v22 and unlike v26.

The v26 corpus has still not been run through `export-lk`.

## v26 run — all three revisions now exported

`export-lk --root "wow-viewer/test_data/v22_adts/unknown" --map V26Corpus`:

```
files=700 skipped=0 tiles=699 versions=v26:700
chunks: source=179200 written=179200 empty-filled=0
layers=13293 alpha-maps=8457 dropped-no-alpha=0
shadows=0 area-ids=0 mccv-chunks=179200 objects=5309 (skipped 0)
```

Every count matches what spec 237 measured on this corpus independently: 179,200 chunks, **0 area
ids** and **0 shadows** (v26 `ACNK` +0x0C is zero everywhere and its `ASHD` is all-zero), `MCCV` on
every chunk, and **5,309 objects**. 0 layers dropped, so the weights->alpha path covers v26 as well as
v23.

The manifest also reports **1,213 placements with a negative DAT uniqueId** — matching the
independent count in
[237 evidence](../../237-adt-v26-terrain/evidence/acdo-negative-uniqueids-2026-09-20.md). v26 runs two
id allocators and LK `MDDF` uniqueId is unsigned, so those are written as very large unsigned values.
Legal and still unique, but the "uncommitted object" signal is lost. **Operator decision outstanding**:
pass through (current), remap to a high positive block with a recorded mapping, or drop them.

## UI entry point added (operator report: "there's nowhere to push to export!")

The first cut shipped as a CLI command only, with no way to run it from the viewer. Fixed:

- **File > Export Loaded DAT as LK ADT...**, enabled only when a DAT folder is the loaded terrain
  (`_terrainManager?.Adapter is AhdrTerrainAdapter`); the tooltip says to open one first otherwise.
  It asks for an output directory, then writes into `<chosen>/<mapname>/` and puts the result summary
  on the status line. The picker is opened inline, so this adds **no ViewerApp state field**
  (AGENTS.md §10).
- The folder walk, the WDT write and the manifest moved into
  [`DatToLkAdtFolderExporter`](../../../../src/core/WowViewer.Core.IO/Maps/DatToLkAdtFolderExporter.cs)
  so the menu item and `adt-ahdr export-lk` run the **same** conversion. The CLI lost 4,615 characters
  of duplicated walk/manifest code rather than the viewer gaining a second copy.

Behaviour preserved across the refactor — the v22 corpus re-exports to identical counts:

```
files=4 skipped=0 tiles=4 versions=v22 x4
chunks: source=999 written=999 empty-filled=25
layers=997 alpha-maps=0 dropped-no-alpha=1386
shadows=289 area-ids=997 mccv-chunks=0 objects=849 (skipped 0)
adst-dropped=0 negative-uniqueids=0
```

### Manifest defects fixed in the same pass

1. **`ADST` and `AOCH` were dropped silently** — a direct violation of FR-013 / SC-004 ("no silent
   drops"). Both are now named with counts (v26 run: `ADST` 321, `AOCH` present).
2. **Count-bearing notes de-duplicated badly**, emitting ~15 near-identical running-count lines per
   manifest. Counts are now report counters, emitted once as totals.

## Coverage

| Revision | Exported | Alpha path | Notes |
|---|---|---|---|
| v22 Expansion01 | 4 tiles | n/a (codec open) | shadows + area ids carried |
| v23 IcecrownCitadel | 3 tiles | 48 MCAL | MCCV carried |
| v26 corpus | 699 tiles | 8,457 MCAL | 1,213 negative uniqueIds |

All three revisions export. **None has been loaded in a client or the viewer.**
