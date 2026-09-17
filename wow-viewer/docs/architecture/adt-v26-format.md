# DAT v26 Terrain Format (raw ADT project files)

**Naming (2026-09-16)**: these are **DAT files**: the raw terrain project files that the client's ADTs are built from (the
wiki's ADT/v22 page now also describes versions 22/23/26 as DAT files). The repo paths
(`adt-v26-format.md`, `specs/237-adt-v26-terrain/`, and code names such as `AdtV26`) keep "ADT" to avoid churn. The format
name is **DAT v26**.

**Status**: living document, first written 2026-09-16. Owner spec: [237 ADT v26](../../specs/237-adt-v26-terrain/spec.md).
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
- Evidence and reproducible scripts: [specs/237-adt-v26-terrain/evidence/](../../specs/237-adt-v26-terrain/evidence/phase0-first-look-2026-09-16.md).

## File layout

Standard little-endian IFF-style chunks: a 4-byte id (stored reversed on disk, e.g. `REVM`), a uint32 size and the payload,
**unpadded**. 700/700 files walk with 0 unaccounted bytes. **MEASURED**

Top-level order (the same in every file; the bracketed chunks are optional):

```text
MVER  AHDR  ALOC  AOCH  AVTX  ANRM  [ATEX × n]  ADOO × n  ACNK × 256  [ADST × n]  ACVT
                                     30 files                          7 files
```

`ADST` (12 bytes) appears in 7 files as a run of 2–139 chunks between the last `ACNK` and `ACVT`. **MEASURED**

## Relationship to ADT v22/v23

v26 reuses the chunk vocabulary of the pre-Cataclysm experimental ADT v22/v23 on wowdev.wiki. It differs as follows:

| | wiki v22/v23 | **v26** |
|---|---|---|
| First chunk | `AHDR` | `MVER` (26), then `AHDR` |
| Version | 22 / 23 | **26** (in both `MVER` and `AHDR`) |
| Tile location | not in file (filename convention) | **`ALOC` chunk** |
| `ACVT` | v23 only | every file |
| `AOCH`, `ADST` | absent | present (unexplained) |
| `ACNK` index fields | chunk position | chunk-local 0–15, row-major (tile position is in `ALOC`) |

## Chunks

| Chunk | Size | Layout | Confidence |
|---|---|---|---|
| `MVER` | 4 | uint32 version = 26 | **MEASURED** |
| `AHDR` | 64 | uint32 version=26, verticesX=129, verticesY=129, chunksX=16, chunksY=16, then 11×uint32: `8396383`, 0×10 | sizes **MEASURED**; `8396383` *open* |
| `ALOC` | 20 | 5×uint32 `(2869, tileX, tileY, tileX, tileY)` | tileX/tileY **MEASURED**; field 0 (constant 2869) and the repeat in fields 3/4 *open* |
| `AOCH` | 2048 | all bytes zero in every file | *open* |
| `AVTX` | 132100 | float32 × (129² + 128²): outer 129×129 **row-major**, then inner 128×128 | outer order and tile axes **MEASURED**; inner order *hypothesis* |
| `ANRM` | 99075 | int8 × 3 × (129² + 128²), outer then inner | size **MEASURED**; component order and scale *hypothesis* |
| `ATEX` | variable | one NUL-terminated texture path per chunk | **MEASURED** |
| `ADOO` | variable | one NUL-terminated model path per chunk (225 or 285 per file, near-identical across tiles, so it looks map-global) | **MEASURED** |
| `ADST` | 12 | 3×uint32, e.g. `(63420377, 190719, 1)`; the first field falls in the `ACDO` uniqueId range | *open* |
| `ACNK` | 64 or more | 64-byte header (+0x00/+0x04 chunk index, +0x08 = `0xD000`, +0x0C = 0), then `ALYR × 0..4`, `ASHD` (512), optional `ACDO` (56 or 60) | order and sizes **MEASURED**; most header fields *open* |
| `ALYR` | 4136 | 0x20 fixed part (flags `0x100` always) + nested `AMAP` (4096 = 8-bit 64×64) | **MEASURED** |
| `ACVT` | 132100 | 4 bytes × (129² + 128²), presumably RGBA per vertex, outer then inner | size **MEASURED**; channel order *hypothesis* |

## Tile placement and heights (MEASURED)

- `ALOC[1]` is tile X, along the height grid's **column** axis. `ALOC[2]` is tile Y, along the **row** axis. Observed ranges are X 18–45, Y 16–40.
- Proof: for every pair of `ALOC`-adjacent tiles, the last outer column (or row) equals the neighbour's first column (or row) exactly. The median |Δh| is 0.0000 over 35 X-pairs and 36 Y-pairs. Every one of the other 15 edge pairings per axis is ≥ 650 off, so the test could not pass by accident.
- Heights are **absolute** at tile level: they are continuous across tile edges with no per-tile offset.

## Open questions

1. The inner 128×128 grid order (the wireframe fast path will show a wrong order as spikes).
2. `ANRM` component order and scale; `ACVT` channel order.
3. `ACNK` sub-chunk layouts in v26 (`ALYR`, `AMAP` alpha encoding, `ASHD`, `ACDO` placements and their frame).
4. `ALOC[0]` = 2869, and why `ALOC[3..4]` repeat X/Y.
5. `AHDR` +0x14 = 8396383, `AOCH` (all zero) and `ADST`.
6. The overall height range is −18798.58..+9965.19. Check which tiles hold the extremes, and whether flat tiles sit at a sentinel height.

Update this document whenever a probe settles one of these. Link the evidence and never promote a *hypothesis* without it.
