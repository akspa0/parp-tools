# Spec 237 Evidence — First DAT v22 Ever Loaded: Expansion01 (Terokkar / Bone Wastes)

Date: 2026-09-20

## Summary

`E:\WC2\wrat2\world\maps\Expansion01` holds **four genuine DAT v22 files** — `MVER` 22 and `AHDR`
version 22 in all four. They load and render in the viewer. Operator witness (screenshot, v0.6.0-alpha
title bar, map name `DAT: Expansion01`, 4 tiles loaded, 999 chunks, 119 FPS).

This is the **first v22 file the project has ever had**. Every prior AHDR-family sample was v26 (the
699-file `wow_classic_beta` corpus) or v23 (Kalimdor/Lost Isles, IcecrownCitadel). It closes the
long-standing "no real v22/v23 files exist" gap in [research.md](../research.md) R2 completely, and it
retires my own statement earlier the same day in
[real-v23-dat-icecrown-2026-09-20.md](real-v23-dat-icecrown-2026-09-20.md) that "v22 has never been
seen" — it had been seen within the hour.

Operator characterization: *"We just uncovered hidden game data from ~2005 that no one has seen in
over 20 years."* The data supporting an era attribution is the tileset paths, which are all
`Expansion01` (Burning Crusade) zone art — see the texture list below. This document does not attempt
to date the files beyond that; there is no timestamp inside them.

## Corpus

| File | Bytes | MVER | AHDR ver | Grid | ACNK |
|---|---|---|---|---|---|
| `area_24_38.dat` | 1,261,561 | 22 | 22 | 129×129 / 16×16 | 255 |
| `area_25_37.dat` | 1,110,010 | 22 | 22 | 129×129 / 16×16 | 243 |
| `area_25_38.dat` | 925,691 | 22 | 22 | 129×129 / 16×16 | 252 |
| `area_26_38.dat` | 462,842 | 22 | 22 | 129×129 / 16×16 | 249 |

Top-level walk closes exactly on EOF for all four, 0 unaccounted bytes. No `ALOC`, so tile location
comes from the filename (`area_24_38` → X=24, Y=38), which the viewer resolves to renderer tile
(38, 24) — matching the Inspector's `Tile (X, Y): (38, 24)` in the witness screenshot.

## v22 is structurally distinct from v23/v26

| | **v22** (measured here) | v23 (measured) | v26 (measured) |
|---|---|---|---|
| `MVER` / `AHDR` version | 22 / 22 | 23 / 23 | 26 / 26 |
| `ALOC` | absent | absent | present |
| `ACVT` | **absent** | present | present |
| `AFBO` | **absent** | present (72) | absent |
| `AOCH`, `ADST` | absent | absent | present |
| `ADOO` | **present** (0–21 per file) | absent | present |
| `ACNK` count | **243–255, variable** | exactly 256 | exactly 256 |
| `ACNK` sub-chunks | `ALYR`, **`ASHD`**, **`ACDO`** | `ALYR` only | `ALYR`, `ASHD`, `ACDO` |
| `AMAP` size | **128–3474, variable** | 4096 | 4096 |

Two findings worth calling out:

1. **`ACNK` count is not fixed in v22.** v23 and v26 always emit all 256. v22 emits 243–255, so it
   **omits chunks** rather than writing empty ones. Anything that indexes chunks by ordinal position
   is wrong on v22; `ACNK` +0x00/+0x04 must be used.
2. **The wiki's "v22 has no `AFBO`/`ACVT`" claim is CONFIRMED** on real data, for the first time.
   research.md R2 had it filed as unverifiable.

### Corpus-wide sub-chunk census (999 ACNK across the four files)

```
ALYR 2383   ASHD 767   ACDO 851   (1 ACNK overruns its parent and is truncated by the walk)
ACNK smaller than the 0x40 header: 2 (both 16 bytes)
```

**`ACNK` header is 0x40 in v22 too**: scanning each `ACNK` payload for its first sub-chunk tag puts it
at offset 64 in 254 of 255 chunks in `area_24_38.dat`. The existing reader's `AcnkHeaderSize = 0x40`
is therefore correct across all three revisions — not an assumption inherited from v26.

### `ALYR` flag 0x100 gates `AMAP` — measured on v22

| `ALYR` flags | has trailing `AMAP` | count |
|---|---|---|
| `0x100` | yes | 1385 |
| `0` | no | 997 |
| `0x100` | **no** (anomaly) | 1 |

998 `ALYR` are exactly 0x20 bytes (fixed part only). research.md R3 recorded the wiki as saying the
`flags & 0x100` signal is "v23 only"; it holds on v22 in 2382 of 2383 cases.

## The real defect this exposes: v22 alpha is silently dropped

`AMAP` payloads in v22 are **128–3474 bytes and never 4096**, so they are encoded, not a raw 64×64
map.

[`AhdrTerrainAdapter.cs:178`](../../../src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs) builds alpha
only when **every** layer carries a 4096-byte map:

```csharp
if (source.Layers.Count > 1 && source.Layers.All(static l => l.AlphaMap is { Length: AdtAhdrAlpha.Pixels }))
```

v22 can never satisfy this: layer 0 has no `AMAP` at all, and the rest are short. The dictionary stays
empty, so **only layer 0 is drawn and all other layers are invisible**. This is exactly what the
witness screenshot reports for chunk (4, 10): `Layers: 3`, `Alpha Maps: 0`. The terrain looks correct
because layer 0 covers most ground, but the texture blending is not being applied.

The same guard means v22 `ASHD` (767 present) and `ACDO` (851 present) are read by
`AdtAhdrReader` but nothing downstream consumes them for v22 either.

### The `AMAP` encoding is NOT v18 MCAL RLE (measured, refuted)

Tested the v18 `MCAL` fill/copy RLE (`0x80` fill bit, 7-bit count) against all 1385 v22 `AMAP`
payloads, requiring **both** exactly 4096 bytes out and the whole payload consumed:

```
ok = 137   mismatched = 1248
```

137/1385 is chance level. The detector is not vacuous — it *can* succeed, and does on 137 payloads —
so this is a real negative, not an untestable one. **The v22 `AMAP` encoding is unidentified.** It is
not raw, and it is not v18 MCAL RLE.

## Witness screenshot facts (operator run, v0.6.0-alpha)

| Field | Value |
|---|---|
| Map | `DAT: Expansion01` (the renamed, version-agnostic label) |
| Tiles | 4 loaded, 7 missing or failed; camera tile (38, 23) |
| Chunks | 999 / 0 — matches the measured 999 `ACNK` exactly |
| Inspector chunk | ADT Chunk (24, 38) MCNK (4, 10); Area ID 0 (`MissingAreaId`); 3 layers; **0 alpha maps**; no shadow map; no MCCV; no liquid |
| Perf | 119 FPS, CPU 5.7 ms, MDX 839/843 |

Texture layers on that chunk — all `Expansion01` (Burning Crusade) art:

```
Layer 0: TILESET\Expansion01\BoneWastes\BoneWastesDirtHoles.blp      (tex#3)
Layer 1: Tileset\Expansion01\Terokkar\TerokkarForest_Crystal.blp     (tex#1)
Layer 2: TILESET\Expansion01\BoneWastes\BoneWastesDirtCracked02.blp  (tex#4)
```

`Area ID 0 / MissingAreaId` is a **reader gap, not missing data** — `ACNK` +0x0C is populated (3519 /
3520) in these files. See "v22 carries fields v26 does not" below.

## Verification

| Command / action | Result |
|---|---|
| Byte read of `MVER`/`AHDR` version fields, 4 files | 22 / 22 in all four |
| Top-level chunk walk, 4 files | closes exactly on EOF, 0 unaccounted bytes |
| Sub-chunk walk from `+0x40`, 999 ACNK | `ALYR` 2383, `ASHD` 767, `ACDO` 851, 1 overrun |
| First-sub-chunk offset scan | 64 in 254/255 chunks → header is 0x40 |
| `ALYR` flag ↔ `AMAP` correlation | 1385 yes / 997 no / 1 anomaly |
| v18 MCAL RLE decode of 1385 `AMAP` | 137 ok, 1248 mismatched → refuted |
| Operator viewer run | renders; screenshot recorded above |

| Criterion | Evidence |
|---|---|
| Real v22 files exist and are in hand | Four files, version fields read from bytes |
| v22 loads and renders | Operator screenshot: `DAT: Expansion01`, 4 tiles, 999 chunks, 119 FPS |
| v22 differs structurally from v23/v26 | Chunk census table; variable `ACNK` count; no `ACVT`/`AFBO` |
| v22 alpha is not rendered | `AhdrTerrainAdapter.cs:178` guard + Inspector `Layers: 3 / Alpha Maps: 0` |
| `AMAP` encoding is unknown | Sizes 128–3474, never 4096; MCAL RLE refuted at chance level |

**Not claimed:** no fix has been made. The alpha gap, `ASHD` and `ACDO` for v22 are **reported, not
addressed** — decoding an unidentified `AMAP` encoding is new scope and needs a spec task per §9.1.
Height/normal interpretation for v22 is assumed to match v23/v26 and has not been separately measured.

## v22 carries fields v26 does not (measured 2026-09-20)

Spec 241's capability table was built on the **v26** corpus and records "no area IDs", "ASHD present
but all zero". **Neither holds for v22.** Header survey over all 997 v22 `ACNK` with layers:

| Field | v26 (per spec 241) | **v22 (measured)** |
|---|---|---|
| `ACNK` +0x0C area id | 0 everywhere | **3519 (838 chunks), 3520 (159 chunks)** |
| `ACNK` +0x10 holes | 0 everywhere | 0 everywhere (agrees) |
| `ASHD` | present, all zero | **767 present, 289 NON-ZERO**, 512 bytes (5 are size 0) |

Three consequences:

1. **v22 has real area IDs.** Exactly two adjacent values appear, in the Burning Crusade `AreaTable`
   range, on a tile whose textures are `Expansion01\Terokkar` and `Expansion01\BoneWastes` (Bone
   Wastes sits inside Terokkar Forest). The pairing is consistent, but the ID→name mapping is **not
   confirmed against `AreaTable.dbc`** here and should be before it is asserted anywhere.
2. **The viewer reports Area ID 0 for these chunks** (witness screenshot: `Area ID: 0`,
   `MissingAreaId`) even though +0x0C is populated. `AdtAhdrChunk` keeps only `HeaderRaw`, so nothing
   surfaces the field. That is a reader gap, not missing data.
3. **v22 `ASHD` is live shadow data**, 512 bytes = 64×64 bits — the exact size and shape of LK `MCSH`.
   On v26 this chunk is all zero and was reasonably ignored; on v22 it must not be.

This reduces the expected loss in any DAT→ADT conversion: area IDs and shadows are carryable for v22,
where the v26-derived assumption said they were absent.

## Open items this creates

1. **Identify the v22 `AMAP` encoding.** Not raw, not MCAL RLE. Until then v22 renders layer 0 only.
2. **Relax the `AhdrTerrainAdapter` alpha guard** so a partial/absent alpha set still blends the layers
   it does have, instead of dropping all of them.
3. **v22 `ASHD` / `ACDO`** are parsed but unused for v22.
4. `ACNK` ordinal indexing is unsafe on v22 (chunks are omitted).
5. One `ACNK` overruns its parent in the corpus; the walk truncates safely but the cause is unknown.
