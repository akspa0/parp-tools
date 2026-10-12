# DAT v26 Terrain Format (raw ADT project files)

**Naming (2026-09-16)**: these are **DAT files**: the raw terrain project files that the client's ADTs are built from (the
wiki's ADT/v22 page now also describes versions 22/23/26 as DAT files). The repo paths
(`adt-v26-format.md`, `specs/archived/237-adt-v26-terrain/`, and code names such as `AdtV26`) keep "ADT" to avoid churn. The format
name is **DAT v26**.

**Status**: living document, first written 2026-09-16. Owner spec: [237 ADT v26](../../specs/archived/237-adt-v26-terrain/spec.md).
**Confidence key**: **MEASURED** = proven on the real corpus with evidence linked; *open* = observed but not explained;
*hypothesis* = carried over from the wiki v22/v23 relatives and not yet tested on v26.

## Discovery

ADT v26 is a **brand-new World of Warcraft terrain tile format**. As of this write-up, no other public documentation of it
exists.

| When (2026-09-16) | Event |
|---|---|
| ~8 hours before first analysis | The format first appears publicly in the `wow_classic_beta` build: the first WoW: Forever build on Battle.net servers, and the first build of the final WoW remaster on a new engine |
| ~1 hour before first analysis | The files are identified at a glance as ADT "v22" and passed to this project |
| First analysis | The files are measured, identified as version 26, and tile placement is decoded from the new `ALOC` chunk |

**No companions exist**: no WDT, map table entry, listfile names or CDN-side metadata accompany these files. Everything
below comes from the tile files alone.

**Referenced assets do resolve**: all 33 texture names (`.blp`) and 285 model names (`.m2`, `.wmo`) in `ATEX`/`ADOO` have
FileDataIDs in the community listfile (evidence Finding 6).

## Corpus

- 699 unique tiles (plus one renamed duplicate). Extensionless files named by FileDataID (6893600–6894299). **The names encode no position.**
- Content: Lordaeron / Eastern Kingdoms tilesets (TirisFall, SilverPine, Wetlands) and matching doodads.
- 30 tiles carry terrain. 669 are perfectly flat.
- The terrain is the **southwest corner of Tirisfal Glades**. The exact offset to the shipped ADT tile grid is not solved yet (best alignment attempt: median RMSE 41.6 yd, see evidence `align_dat_to_adt.py`).

## Units: heights are in inches

`AVTX` heights are stored in **inches**. The client's ADTs store yards, so the DAT height data is **36× finer** than the
shipped terrain (1 yd = 36 in). Dividing by 36 gives ADT units: the corpus minimum −18559.47 in ÷ 36 = −515.54 yd, which
matches the shipped ocean floor in that region. **MEASURED**

The horizontal vertex grid is the same size as an ADT tile's (129 × 129 outer + 128 × 128 inner, i.e. 16 × 16 chunks of
8 × 8 cells), so the extra precision is in height values, not in vertex count. The viewer's default display scale is ÷36
(File → DAT v26 Height Scale).
- Evidence and reproducible scripts: [specs/archived/237-adt-v26-terrain/evidence/](../../specs/archived/237-adt-v26-terrain/evidence/phase0-first-look-2026-09-16.md).

## File layout

Standard little-endian IFF-style chunks: a 4-byte id (stored reversed on disk, e.g. `REVM`), a uint32 size and the payload,
**unpadded**. 700/700 files walk with 0 unaccounted bytes. **MEASURED**

Top-level order (the same in every file; the bracketed chunks are optional):

```text
MVER  AHDR  ALOC  AOCH  AVTX  ANRM  [ATEX × n]  ADOO × n  ACNK × 256  [ADST × n]  ACVT
                                     30 files                          7 files
```

`ADST` (12 bytes) appears in 7 files as a run of 2–139 chunks between the last `ACNK` and `ACVT`. **MEASURED**

## Relationship to DAT v22/v23

**All three revisions are now MEASURED on real files** (v22 and v23 acquired 2026-09-19/20), so this
table is no longer wiki-derived:

| | **v22** | **v23** | **v26** |
|---|---|---|---|
| Corpus | Expansion01, 4 files | Kalimdor + IcecrownCitadel | `wow_classic_beta`, 699 files |
| First chunk | `MVER` (22), then `AHDR` | `MVER` (23), then `AHDR` | `MVER` (26), then `AHDR` |
| Tile location | not in file (filename) | not in file (filename) | **`ALOC` chunk** |
| `ACVT` | **absent** | present | every file |
| `AFBO` | **absent** | **present** (72 bytes) | absent |
| `ADOO` | present | absent in sample | present |
| `AOCH`, `ADST` | absent | absent | present (unexplained) |
| `ACNK` count | **243–255, variable (chunks omitted)** | exactly 256 | exactly 256 |
| `ACNK` sub-chunks | `ALYR`, `ASHD`, `ACDO` | `ALYR` only | `ALYR`, `ASHD`, `ACDO` |
| `ACNK` header size | 0x40 | 0x40 | 0x40 |
| `AMAP` | **128–3474, variable, encoding unknown** | 4096 | 4096 |
| `ACNK` index fields | chunk position | chunk position | chunk-local 0–15, row-major (tile position is in `ALOC`) |

Notes:

- The wiki's "v22 has no `AFBO`/`ACVT`" claim is **CONFIRMED**. `AFBO` turns out to be a **v23-only**
  chunk in the samples held — v26 never shows it.
- **v22 omits empty `ACNK` rather than writing all 256.** Never index v22 chunks by ordinal position;
  use the `ACNK` +0x00/+0x04 index fields.
- **v22 `AMAP` is encoded, not raw.** Sizes 128–3474, never 4096. The v18 `MCAL` fill/copy RLE is
  **refuted** (137 of 1385 payloads decode cleanly — chance level). Encoding unidentified; the viewer
  currently renders v22 layer 0 only.
- `ALYR` flag `0x100` gates the nested `AMAP` on v22 as well as v23 (measured 2382/2383).

Evidence: [first-v22-dat-render-2026-09-20.md](../../specs/archived/237-adt-v26-terrain/evidence/first-v22-dat-render-2026-09-20.md) ·
[real-v23-dat-icecrown-2026-09-20.md](../../specs/archived/237-adt-v26-terrain/evidence/real-v23-dat-icecrown-2026-09-20.md)

## Chunks

| Chunk | Size | Layout | Confidence |
|---|---|---|---|
| `MVER` | 4 | uint32 version = 26 | **MEASURED** |
| `AHDR` | 64 | uint32 version=26, verticesX=129, verticesY=129, chunksX=16, chunksY=16, then 11×uint32: `8396383`, 0×10 | sizes **MEASURED**; `8396383` *open* |
| `ALOC` | 20 | 5×uint32 `(2869, tileX, tileY, tileX, tileY)` | tileX/tileY **MEASURED**; field 0 see "ALOC[0] = 2869" below; the repeat in fields 3/4 *open* |
| `AOCH` | 2048 | all bytes zero in every file | *open* |
| `AVTX` | 132100 | float32 × (129² + 128²): outer 129×129 **row-major**, then inner 128×128 | outer order and tile axes **MEASURED**; inner order *hypothesis* |
| `ANRM` | 99075 | int8 × 3 × (129² + 128²), outer then inner; components **(column axis, vertical, row axis)**, 127 = 1.0 | **MEASURED** (mean dot 0.985 with height-derived normals; next-best order 0.658; every vector has length 127). Inner order *hypothesis* |
| `ATEX` | variable | one texture path per chunk, **no NUL terminator** (length = chunk size) | **MEASURED** |
| `ADOO` | variable | one model path per chunk, **no NUL terminator** (225 or 285 per file, near-identical across tiles, so it looks map-global) | **MEASURED** |
| `ADST` | 12 | 3×uint32 `(uniqueId, model FileDataID, 1)`, e.g. `(63420377, 190719 = World/critter/BIRDS/Bird01.m2, 1)` | **MEASURED**: 321 rows in 7 files, 321/321 FileDataIDs name models in the community listfile, last field always 1. **None of the uniqueIds match an `ACDO`** in the 699 files and the rows have no position, so they are not placements |
| `ACNK` | 64 or more | 64-byte header, then `ALYR × 0..4`, `ASHD` (512), `ACDO × n` (56 or 60 each). Header: +0x00/+0x04 chunk index; +0x08 = `0xD000`; +0x0C = 0; +0x10 uint16 = 0; **+0x12 16 bytes = 2-bit 8×8 predominant layer map** (LSB first, row-major); +0x22 uint64 sparse bitmask (725 of 179,200 chunks non-zero; position of MCNK's no-effect-doodad map); the rest 0 | predominant map **MEASURED** (95.0% of 239,808 cells match the dominant AMAP layer, control 67.9%); +0x22 *hypothesis*; others constant |
| `ALYR` | 4136 | 0x20 fixed part (flags `0x100` always) + nested `AMAP` (4096 = 8-bit 64×64) | **MEASURED** |
| `AMAP` | 4096 | per-layer **blend weight**, not ADT sequential alpha: the weights of all layers (layer 0 included) sum to 255 per pixel | **MEASURED**: per-chunk mean sum 254.6 (252.6..255.0), exactly 255 for 74% of pixels; highest-weight layer matches the ACNK predominant map in 99.93% of cells vs 95.03% when read as sequential alpha |
| `ACVT` | 132100 | 4 bytes × (129² + 128²) per vertex, outer then inner: three bytes centred on 127 (neutral, like MCCV) and a fourth always 255 | value ranges **MEASURED**; which of bytes 0/2 is red *open* (the corpus is almost entirely neutral) |

## Writer and round trip (experimental)

- `AdtAhdrWriter` serializes the decoded model; `inspect adt-ahdr roundtrip --root <folder>` decodes and re-encodes
  every file. **700/700 corpus files are byte-identical**, with `ACDO` written from its decoded fields, so the model keeps
  every byte. (This run also found that `ATEX`/`ADOO` names carry no NUL terminator.)
- `AdtAhdrTileBuilder` builds a v26 tile from ADT data through the measured frames (heights × 36, sequential alpha →
  weights, MDDF/MODF → chunk-relative `ACDO`, MCNR → `ANRM`, MCCV → `ACVT`). In the viewer:
  **File → Export Nearby Tiles as DAT v26 (experimental)** writes the loaded ADT map's tiles within 2 of the camera to
  `output/dat_v26_export/`. Checks on the output: reopen with **Open DAT v26 Terrain Folder** and compare with the ADT view;
  `inspect adt-ahdr check` (seams) and `adt-ahdr objects` (object heights vs terrain); `evidence/scripts/anrm_acvt_v26.py`
  (stored normals vs heights, which tests the MCNR → ANRM mapping against independent client normals).

## Objects: ACDO (MEASURED)

5,309 records in 18 files (5,276 M2, 33 WMO; 0 invalid model indexes). Scripts:
`evidence/scripts/objects_v26.py`, `acdo_chunk_frame_v26.py`; C# check: `inspect adt-ahdr objects --root <folder>`.

| Offset | Type | Meaning |
|---|---|---|
| 0x00 | uint32 | model index into `ADOO` (0..284) |
| 0x04 | float | column-axis offset in **inches** from the centre of the chunk that stores the record (±600) |
| 0x08 | float | vertical offset in inches from the **mean of the chunk's 145 `AVTX` heights** |
| 0x0C | float | row-axis offset in inches from the chunk centre (±600) |
| 0x10 | float ×3 | rotation in degrees, same axis order as the position (+0x14 about the vertical, 0..360) |
| 0x1C | float | scale (0.1..3.47) |
| 0x20 | float | always 1.0 |
| 0x24 | uint32 | always 0 |
| 0x28 | float | mostly 0, 298 distinct values (±1484); *open* |
| 0x2C | uint32 | uniqueId, distinct for every record |
| 0x30 | uint32 | count of trailing uint32 values: 0 (56-byte records) or 1 (60-byte records, all WMOs) |
| 0x34 | uint32 | 0 (3,143), 65536 (2,159), 1/2/3 (7); *open* |
| 0x38 | uint32 × count | 1 or 2; *open* (possibly a WMO doodad set) |

**Frame proof**: objects placed this way sit on the terrain with median |Δh| **0.31 in** (p90 2.37 in, signed median 0.00;
3,202 of 5,309 within 0.5 in). The same test with objects moved to random chunks gives 103.9 in; using the chunk-centre
vertex instead of the 145-vertex mean gives 22.3 in; treating positions as tile-relative yards or inches gives ≥ 55 yd.

## Units: horizontal span

A chunk is 1200 inches wide (8 cells × 150 in), the same 33.33 yd span as an ADT chunk. The ±600 in range of `ACDO`
horizontal offsets (half a chunk) is consistent with this.

## ALOC[0] = 2869 and AHDR +0x14 = 8396383

Both values are the same in all 700 files, so neither is a per-tile index (as a 64×64 index 2869 would be tile 53, 44).
MEASURED 2026-09-17 with `inspect casc db2 --find` and the community listfile:

- Neither value is a FileDataID in the community listfile.
- Neither appears in any column of `Map`, `AreaTable`, `MapDifficulty` or `UiMap` in `wow_classic_beta` 1.60.1.69876 or
  `wow_classic_era` 1.15.9.69722. (Control: the same search finds Map 451 `development`.)
- `Map` IDs in both builds run 0..3109 (beta) / 0..2921 (era). Around 2869 the rows present are 2853 Deadwind Pass,
  2856 Scarlet Enclave, **2868 Eastern Plaguelands (Scarlet Raid Phase)**, then **2875 Karazhan Crypts**; 2869–2874 have
  no rows. These neighbours use the map ID as their `Directory`.

*Hypothesis*: 2869 is the Map ID of the map these tiles belong to, a map with no row in either client's `Map` table.
Not proven: no file or table in these builds refers to 2869.

## Tile placement and heights (MEASURED)

- `ALOC[1]` is tile X, along the height grid's **column** axis. `ALOC[2]` is tile Y, along the **row** axis. Observed ranges are X 18–45, Y 16–40.
- Proof: for every pair of `ALOC`-adjacent tiles, the last outer column (or row) equals the neighbour's first column (or row) exactly. The median |Δh| is 0.0000 over 35 X-pairs and 36 Y-pairs. Every one of the other 15 edge pairings per axis is ≥ 650 off, so the test could not pass by accident.
- Heights are **absolute** at tile level: they are continuous across tile edges with no per-tile offset.

## Open questions

1. The inner 128×128 grid order (assumed to match the outer grid; the ACDO height fit using the 145-vertex mean is consistent with it).
2. `ACVT` red/blue byte order.
3. `ASHD` is all zero in all 15,360 chunks that carry it, so its bit layout cannot be measured from this corpus.
4. `ACDO` +0x28, +0x34 and the trailing values; `ACNK` +0x22 bitmask; whether ACDO yaw is mirrored relative to the game world.
5. `ADST`: why its uniqueIds match no `ACDO` (objects outside this corpus, or removed objects).
6. Why `ALOC[3..4]` repeat X/Y (possibly a tile range), `AHDR` +0x14 = 8396383, `AOCH` (all zero); whether `ALOC[0]` is a Map ID (see above).
7. The exact offset of these tiles against the shipped ADT tile grid.

Update this document whenever a probe settles one of these. Link the evidence and never promote a *hypothesis* without it.
