# PM4 / PD4 — Format Reference

PM4 is the per-ADT-tile server-side navigation file; PD4 is the per-WMO equivalent. Both are
chunked binary files. This document describes the container, every chunk, the record layouts, the
measured relationships between chunks, the coordinate transforms, and the questions that remain
open. Field names follow the current wowdev.wiki articles (`PM4`, `PD4`); proposed names are
collected separately in §9.

**Corpus:** 616 PM4 files (309 non-empty) from the 4.x-era `development` map, plus 2 PD4 files
(`6or_garrison_workshop_v3_snow`, 6.0.1). All figures below were measured on that corpus. A
similarly named field in another format or build does not carry the same meaning.

---

## 1. Container

```c
struct chunk {
  char   fourcc[4];   // reversed ASCII: on-disk "REVM" is logical "MVER"
  uint32 size;        // little-endian payload length, excluding this 8-byte header
  byte   payload[size];
};
```

- Integers are little-endian; floats are 32-bit IEEE-754.
- No alignment padding between chunks.
- No enforced chunk order. The canonical order is:
  `MVER MSHD MSPV MSPI MSCN MSLK MSVT MSVI MSUR MPRL MPRR MDBH MDBI MDBF MDOS MDSF`.
- PD4 uses `MVER MCRC MSHD MSPV MSPI MSCN MSLK MSVT MSVI MSUR`.
- A reader must walk the file by `size`; it must not assume a fixed chunk sequence.
- Unknown signatures are skipped by `size` and their payload retained.

---

## 2. Chunk inventory

| Chunk | Record width | Contents |
|---|---|---|
| `MVER` | 4 | Version word |
| `MCRC` | 4 | PD4-only checksum/placeholder word |
| `MSHD` | 32 | Eight `uint32` header words |
| `MSPV` | 12 | `C3Vector` path/connection vertices |
| `MSPI` | 4 | `uint32` indices into `MSPV` |
| `MSCN` | 12 | `C3Vector` node positions |
| `MSLK` | 20 | Surface-adjacency records |
| `MSVT` | 12 | `C3Vector` surface vertices |
| `MSVI` | 4 | `uint32` indices into `MSVT` |
| `MSUR` | 32 | Surface records |
| `MPRL` | 24 | Terrain-intersection points |
| `MPRR` | 4 | Quadtree index over `MPRL` |
| `MDBH` | 4 | Destructible-building count |
| `MDBI` | 4 | Destructible-building index |
| `MDBF` | variable | NUL-terminated filename |
| `MDOS` | 8 | Destructible-building state |
| `MDSF` | 8 | Surface/state association |

---

## 3. Record layouts

### 3.1 `MVER`

```c
uint32_t version;
```

- PM4: `12304` (`0x00003010`) in all 502 structured files.
- PD4: `48` (`0x00000030`).
- The value is constant across the corpus and is a format version, not a build stamp.

### 3.2 `MCRC` (PD4 only)

```c
uint32_t _0x00;   // 0 in version 48
```

### 3.3 `MSHD`

```c
struct {
  uint32_t _0x00;
  uint32_t _0x04;
  uint32_t _0x08;
  uint32_t _0x0c[5];
} header;
```

- `_0x0c[5]` is zero in 502 of 502 files.
- `_0x00` and `_0x08` are clamped world-unit spans, capped at 534 (`ceil(533.333)`, the tile size).
  They saturate on 27.5% of tiles with geometry and 76.2% of empty tiles. Against the tile's geometry
  extents they correlate crossed, in the same axis order as the rest of PM4: `_0x00` vs Y extent
  +0.780, `_0x08` vs X extent +0.847. They are not the extent of any single point stream: against
  `MSCN`, `MSPV` or `MSVT` the median difference is +42 to +96 units.
- `_0x04 == 1` marks a tile with no surfaces: 140 of 193 empty tiles, 0 of 309 tiles with geometry.
  Otherwise it has 207 distinct values over 309 geometry tiles, 62 shared by more than one tile.
- Exact meaning of `_0x00`, `_0x04` and `_0x08` is not closed.

### 3.4 `MSPV`

```c
C3Vector msp_vertices[];
```

Path/connection geometry vertices. Same axis order as `MSVT` and `MSCN`.

### 3.5 `MSPI`

```c
uint32_t msp_indices[];   // index into MSPV
```

### 3.6 `MSCN`

```c
C3Vector mscn[];   // positions, not normals
```

- No point has unit length; lengths are coordinate magnitudes.
- `MSCN` is an ordered chain, not a point cloud: consecutive entries sit a median 4.417 apart,
  against 161.247 for a random same-file pair.
- The chain closes into short rings: 6.09% of starts return within 0.5 units in at most 64 steps,
  against 1.79% on a shuffled copy of the same points. Ring lengths decay from three
  (3×7,731; 4×3,308; 5×2,103; 6×1,431; 7×1,002); the shuffled control is flat.
- The rings lie on the mesh, held off it: median 1.245 to the nearest `MSVT` vertex (41.7% within
  1.0), against 50.188 for a random point.
- Population is 2.591 points per `MSUR` surface corpus-wide, inside 1.00–3.58 per file.
- Not a lattice: snap rates against 533.33/33.33/8.33/4.17/2.08/1.0/0.5/0.25 are
  0.287%/0.345%/0.506%/0.732%/1.167%/2.133%/4.119%/8.402%, indistinguishable from the `MSVT`
  control and from chance.
- Per-object node network: 1,886 of 1,895 object groups reference `MSCN`; 1,214 of those (64.4%)
  reference nodes that another group also references. 724 of 1,886 groups have `MSCN` bounds that
  extend beyond their own `MSVT` bounds.
- Not doodad data: the nearest `MSCN` point to a real `MDDF` row has a median of 10.5 units (15.1%
  within 1.0), against a random-point control of 25.2 units (3.3%).
- Nothing in the file indexes `MSCN`; an ordered chain is its own edge list. The leading reading is
  a navmesh contour set, not proven.

### 3.7 `MSLK` — 20 bytes

```c
struct {
  uint8_t  _0x00;              // bitfield; bit 0 = record carries no geometry
  uint8_t  _0x01;              // zero-based source-edge ordinal
  uint16_t _0x02;              // 0
  uint32_t _0x04;              // object-group id; equals MPRL._0x04
  int24_t  MSPI_first_index;   // -1 when MSPI_index_count is 0
  uint8_t  MSPI_index_count;
  uint32_t _0x0c;              // tile address; high 16 bits 0xFFFF
  uint16_t msur_index;         // target surface in the addressed tile
  uint16_t _0x12;              // 0x8000
} mslk[];
```

- `_0x00` is a bitfield. Bit 0 means the record carries no geometry: values `0x01` and `0x11` carry
  a wall window 0.0% of the time across 443,882 records; the other eight values carry one 100.0% of
  the time across 386,351. Only 10 of 32 combinations occur, using bits 0–4; bits 1 and 2 never
  co-occur.
- `_0x01` has 19 values. The counts decay like a counter (229,468; 218,771; 211,486; 131,560;
  28,663; 7,336; 1,960; 613; 190; 84; 29; 26; 18; 8; 7; 6; 5; 2; 1). Measured as a zero-based
  source-edge ordinal (§4.3).
- `_0x02` is 0 in every record.
- `_0x04` is a group key with small groups: 283,066 groups of exactly two and 250,192 singletons
  over 830,233 records. Paired records share `_0x00` 99.82% of the time; 77.09% of pairs are two
  no-geometry records, 22.91% two geometry records, 0.00% one of each.
- `_0x0c` is a tile address packed in reverse filename order: for `<map>_<first>_<second>` the
  value is `0xFFFF0000 | (second << 8) | first`. Scored over 1,248,682 links in 305 files with
  diagonal tiles excluded: 98.74%, against a mismatched-tile control of 0.31%; filename order
  scores 0.00%. The high 16 bits are `0xFFFF` in 305 of 305 files. 82.5% of links target the same
  tile, 17.5% an adjacent tile, 0% a more distant tile.
- `_0x12` is constant `0x8000` across 486,819 records.

### 3.8 `MSVT`

```c
C3Vector msvt[];   // surface vertices
```

Axis order is YXZ. The wiki transform to world space is:

```text
worldPos.y = 17066.666 - position.y
worldPos.x = 17066.666 - position.x
worldPos.z = position.z / 36.0
```

The `/36` converts the internal inch height to yards. The placement join in this repository uses
`z` unscaled and matches authored placements; apply `/36` only when the source data is in internal
inches.

### 3.9 `MSVI`

```c
uint32_t msv_indices[];   // index into MSVT
```

### 3.10 `MSUR` — 32 bytes

```c
struct {
  uint8_t  _0x00;              // surface class; 0x03 = doodad, 0x10..0x15 = placed
  uint8_t  _0x01;              // count of indices in MSVI
  uint8_t  _0x02;              // count of MSLK records owned by this surface
  uint8_t  _0x03;              // 0
  float    _0x04;              // normal X
  float    _0x08;              // normal Y
  float    _0x0c;              // normal Z
  float    _0x10;              // signed plane distance
  uint32_t MSVI_first_index;   // first index into MSVI
  uint32_t _0x18;              // first index into MSLK
  uint32_t _0x1c;              // IEEE-754 float, placement Z
} msur[];
```

A surface is a polygon fan over `MSVI[MSVI_first_index .. MSVI_first_index + _0x01)` into `MSVT`.
Polygon sizes are not restricted to triangles; the garrison PD4 is 77% quads.

`_0x00` is a surface class. `0x03` marks doodad surfaces: 100.0% of them carry no placement height
(`_0x1c == 0`). `0x10`–`0x15` are placed surfaces, stratified by height inside the object. In the
M2/WMO populations, `0x03` covers 14,489 of 16,338 M2-only surfaces (88.7%) and 57 of 1,961
WMO-only surfaces (2.9%); WMO-only surfaces are dominated by `0x10`–`0x13` (16–19).

`_0x1c` is a float. Measured against `MODF`/`MDDF` placement records, it equals the placement's
`Position.Z` bit-exactly for 904 of 966 joined objects (93.58%), median `|Δ| = 0.000000`; a rotated
control matches 2.10%. `_0x1c == 0.0f` is the unattributed bucket (no placement join).

The bytes of `_0x1c` are sometimes read as a packed key. Because the field is a float, its top byte
is the exponent band: `0x00` for `0.0f`, `0x40`–`0x43` (64–67) for typical placement heights. That
is why the byte view appears to have only a few "types".

### 3.11 `MPRL` — 24 bytes

```c
struct {
  uint16_t _0x00;      // per-tile region value; 265 distinct in this corpus
  int16_t  _0x02;      // -1 for every point
  uint16_t _0x04;      // object-group id; equals MSLK._0x04
  uint16_t _0x06;      // 0x8000 for every point
  C3Vector position;   // terrain-contact point; height is Y
  int16_t  _0x14;      // counter; 0xFFFF marks the same records as _0x16 == 0x3FFF
  uint16_t _0x16;      // 0x0000 or 0x3FFF
} mprl[];
```

- `position` is the only permuted chunk. Height is component Y; the horizontal pair is Z then X, so
  ADT placement space is `(17066.666 - p.Z, 17066.666 - p.X, p.Y)`. Of the six component orders,
  only this one puts the height inside the terrain range of the cell it sits over: 89.38% of 69,659
  points, median miss 0.000, against a wrong-cell control of 19.27%; the other five score
  0.00%–0.25%.
- `_0x00` averages 1.0 distinct value per file, so it is a per-tile value rather than a per-point
  one. Every `_0x04` group has exactly one `_0x00`.
- `_0x04` is a footprint group id. Points sharing a value sit a median 1.62 world units apart,
  against 170.99 for random groups of the same size in the same file. It is not an angle: the
  observed range is 0–38,317, while a 16-bit heading scaled by `2π/65536` would reach ~65,535.
- `_0x14` and `_0x16` mark the same records: `_0x16` is `0x0000` (101,851) or `0x3FFF` (39,367),
  and `_0x14`'s `0xFFFF` bucket is also 39,367. Excluding that sentinel, `_0x14` is a small counter
  decaying from zero (31,950; 29,916; 24,357; 12,712; 2,016; 653; 165; 47; 12; …).

### 3.12 `MPRR` — 4 bytes

```c
struct {
  uint16_t _0x00;
  uint16_t _0x02;
} mprr[];
```

`0xFFFF` in `_0x00` is a run sentinel. 13,978,231 non-sentinel entries form 3,171,410 runs across
502 files. 99.9843% of runs have length ≡ 3 (mod 4), over 246 distinct run lengths (maximum 5,019);
residues 0 and 2 are empty, and the only 497 exceptions are residue 1. A run plus its sentinel
therefore occupies a whole number of 4-entry (16-byte) blocks, and 75.5% are the minimal single
block. See §4.7 for the run structure.

### 3.13 Destructible-building family

```c
uint32_t m_destructible_building_count;          // MDBH
uint32_t m_destructible_building_index;          // MDBI
char     m_destructible_building_filename[];     // MDBF, NUL-terminated
struct { uint32_t m_destructible_building_index; uint32_t destruction_state; } mdos[];  // MDOS
struct { uint32_t msur_index; uint32_t mdos_index; } mdsf[];                            // MDSF
```

`MDBH` is followed by `count` `MDBI` records and three `MDBF` records per `MDBI`. `MDSF` links a
surface to a destruction state: 2,684 fits, 0 misses. `MDSF` + `MDOS` together give a per-surface
destruction state, i.e. a mechanism for swapping which surfaces are walkable as world state changes.
The family is present on essentially one tile in this corpus, so treat it as unrepresentative.

---

## 4. Measured relationships

### 4.1 `MSUR` owns two windows

```text
MSVI window: MSVI[MSUR.MSVI_first_index .. + MSUR._0x01)  -> MSVT polygon
MSLK window: MSLK[MSUR._0x18          .. + MSUR._0x02)    -> adjacency records
```

- All 517,783 consecutive surface pairs satisfy `_0x18[i] + _0x02[i] == _0x18[i+1]`.
- The window lengths sum to all 1,273,335 `MSLK` records: 100% coverage, zero overruns.
- Reading `_0x18` as an index into `MSCN` overruns 6,240 times and covers 93.52%.
- Wrong lengths taken from `_0x01` or `_0x00` fit only 22.25% and 8.46%.

### 4.2 `MSLK` addresses a target tile and surface

```text
targetTileX      = MSLK._0x0c & 0xFF
targetTileY      = (MSLK._0x0c >> 8) & 0xFF
targetSurface    = MSLK.msur_index
target           = map[targetTileX, targetTileY].MSUR[targetSurface]
```

- Same-tile addressed links: 1,257,562 of 1,257,562 are reciprocal.
- Cross-tile links: 15,773, of which 14,659 resolve; 1,114 address files absent from the corpus.
- References outside a present addressed tile: 0.
- The upper half of `_0x0c` is unknown.

### 4.3 `MSLK._0x01` is a source-edge ordinal

`_0x01` is a zero-based source-polygon edge ordinal for all 1,273,335 records. With the source
vertices in `MSVI`-window order, ordinal `e` selects vertices `e` and `(e + 1) mod count`.

- The selected edge matches the addressed boundary 674,013 times, against 137 for a shifted control
  and 96 for a one-based control.
- For path-bearing records it selects the auxiliary geometry's vertical plane 536,142 of 598,882,
  against 3,120 for a shifted control.

### 4.4 `MSLK._0x00` separates connection roles

- `0x01` and `0x11` carry negative `MSPI_first_index` and track shared polygon boundaries.
- Other values carry nonnegative geometry windows and different boundary behaviour.
- Path-bearing population: 598,882 records — 587,229 four-index windows, 11,024 six-index, 629
  five-index.
- The additional bits are unknown.

### 4.5 `MSLK._0x04` is an object-group id

`MSLK._0x04` equals `MPRL._0x04` for 176,748 of 178,588 points (98.97%). Alternatives are
negligible: `MSUR._0x00` 289 (0.16%), the low 16 bits of `MSUR._0x1c` 111 (0.06%).

- The value is reused across placements; it is a group/model id, not a unique placement id.
- 889,594 values are reused across the corpus.
- Type-1 reciprocal records share the value 98.92% of the time; type 17, 100%; types
  4/10/12/20/26/28 share it at most 1.5%.

### 4.6 `MPRL` is the terrain-intersection point layer

- 178,588 points; 177,525 (99.4%) lie inside the PM4 mesh under the measured mapping.
- `_0x02 = -1` and `_0x06 = 0x8000` for every point.
- The varying payload is `_0x00`, `_0x04`, `_0x14`, `_0x16`.
- The fields form a region → object → point hierarchy:

| Level | Field | Distinct values | Behaviour |
|---|---|---|---|
| Region | `_0x00` | 265 | every `_0x04` group has exactly one `_0x00` (42,092 / 0) |
| Object | `_0x04` | 42,092 | equals `MSLK._0x04`; groups points spatially |
| Point | `_0x14` | 1,603 | varies within an object group |
| Point | `_0x16` | 507 | mostly constant per object |

- The direct group→surface link is not containment (0.4%): the linked surfaces are the object's
  connection/wall surfaces, while the points sit on the ground.
- The ground surfaces containing the points are M2 66.7%, WMO 29.0%, other 6.7%.
- `MPRL` points are not `MSPV` path vertices: 0.8% within 0.1 and 13.1% within 1.0. The path
  geometry is the separate `MSPV`/`MSPI` stream.

### 4.7 `MPRR` is a quadtree index over `MPRL`

17,150,640 records at exactly 4 bytes each. Each run is a quadtree node:

```text
record 0        : node bound (minX, minY) in a 768x768 grid over the tile
record 1        : node bound (maxX, maxY) in the same grid
records 2..n-2  : MPRL indices inside the node
record n-1      : terminator
```

- Run lengths are `4k + 3`.
- The first two records are ordered spatial bounds in 3,171,133 of 3,171,912 runs (99.98%).
- The middle values are valid `MPRL` indices in 1,966,930 runs (62%).
- The last record is `(0, 768)` in 1,366,697 runs (43%); the other runs use a different terminator.
- Across all 616 files the node width/height is a power of two:

| Node width/height | Runs |
|---|---|
| 2 | 1,205,258 |
| 1 | 872,798 |
| 4 | 740,137 |
| 8 | 216,751 |
| 32 | 75,101 |
| 16 | 61,482 |

- The first run of a file is a special header, not a node.
- 98.8% of runs group `MPRL` points sharing one `_0x04`; 100% share one `_0x00`.
- `MPRR` is a quadtree index over `MPRL`; only 39.5%/62.4% of values fall in `MPRL` range, so the
  index is not a direct value list.

### 4.8 M2 and WMO surface populations

Working backwards from `_obj0.adt` placements (`MDDF` = M2, `MODF` = WMO), a surface covers a
placement when the placement origin is inside its projected polygon or within 0.05 of its boundary.

| Measure | M2 (`MDDF`) | WMO (`MODF`) |
|---|---|---|
| Authored placements | 59,544 | 1,116 |
| Covered by at least one surface | 12,147 (20.4%) | 883 (79.1%) |

Of 308,892 surfaces: 16,338 cover an M2 origin only, 1,961 a WMO origin only, 218 both, 290,375
neither.

| Surface population | Surfaces | `_0x1c == 0` | `_0x00 == 3` | Dominant `_0x00` |
|---|---|---|---|---|
| M2-only | 16,338 | 14,489 (88.7%) | 14,489 (88.7%) | 3 |
| WMO-only | 1,961 | 57 (2.9%) | 51 (2.6%) | 16–19 |
| Both | 218 | 110 | 110 | mixed |
| Neither | 290,375 | 98,162 (33.8%) | 96,516 (33.3%) | 18 |

`_0x1c == 0` is the unattributed bucket, and it is predominantly M2-only surfaces. It is not a
per-object identity: only 12.8% of zero surfaces cover an authored M2 origin.

Connection kinds also separate: `_0x00` values 17/18/20/26/28 occur on WMO-only surfaces and on
zero M2-only surfaces; 1/2/4/12 are M2-leaning. `MSUR._0x02 == 0` is M2-leaning; 9/10 are
WMO-leaning.

### 4.9 The merged M2 mesh is not separable by decoded fields

The M2 population has 112,818 surfaces in 170 tiles. No decoded structural feature partitions it
into individual placed objects:

| Test | Result |
|---|---|
| Topological components | 22,286; 14,777 cover 0 origins, 4,775 cover 1, 2,734 cover 2+ |
| Multi-origin components | 734 same-model, 2,000 mixed-model |
| Largest component | 1,023 surfaces, 133 origins, 7 model paths |
| `MSVT` vertex sets | 4,289 disjoint, 6,576 share vertices |
| `MSUR` index ranges | 1,295 disjoint, 9,570 overlap |
| `MSUR._0x03` | 0 for all 112,818 M2 surfaces |
| `MSLK._0x04` | 1,895 placements single-valued, 7,315 multi-valued |

The merge is genuine. Per-object M2 ownership inside the merged mesh is unknown.

### 4.10 Object groups join to authored models

Joining each `_0x04` group to the nearest authored ADT placement (411 files with placements, 29,164
groups):

| Nearest placement distance | Groups |
|---|---|
| ≤ 1 | 518 (1.8%) |
| ≤ 5 | 9,402 (32.2%) |
| ≤ 10 | 8,019 (27.5%) |
| ≤ 25 | 7,415 (25.4%) |
| > 25 | 3,662 (12.6%) |

Adding a class-consistency check (the ground-surface class must match the model's `.M2`/`.WMO`
class):

| Result | Groups |
|---|---|
| Consistent (≤ 25 units, class match) | 19,991 (68.5%) |
| Inconsistent (≤ 25 units, class mismatch) | 5,363 (18.4%) |
| No match within 25 units | 3,662 (12.6%) |

Aggregating consistent groups per model gives a ranked list of 895 models. The top entries:

| Model | Groups | Mean points | Mean surfaces | Distance range |
|---|---|---|---|---|
| `WINTERSPRINGROCK01.M2` | 931 | 3.1 | 2.2 | 0.4–19.3 |
| `SHADOWMOON_OVERHANGROCK_SMALL_02.M2` | 753 | 5.2 | 3.1 | 2.8–25.0 |
| `STONETREELOG02.M2` | 618 | 2.7 | 6.8 | 0.6–15.6 |
| `OM_BONES_02.M2` | 567 | 3.0 | 3.0 | 1.4–21.1 |
| `BONEWALL_01.M2` | 479 | 2.4 | 2.1 | 3.9–24.1 |
| `HUMANTWOSTORY.WMO` | 394 | 4.3 | 4.7 | 0.8–24.4 |
| `FARM.WMO` | 296 | 4.7 | 5.7 | 3.2–24.1 |
| `STABLE.WMO` | 283 | 2.1 | 3.4 | 2.2–13.4 |

### 4.11 Terrain and placement recovery

`MPRL` points are terrain contacts, so they are terrain height samples at known positions. Scored
against 105 tiles that still have mesh:

| Source | n | Median error | p90 | Within 1.0 |
|---|---|---|---|---|
| `MPRL` height | 69,648 | 0.202 | 1.457 | 80.8% |
| Object placement Z | 517 | 5.267 | 109.2 | 20.5% |
| Control: terrain elsewhere in tile | 69,648 | 28.192 | 131.4 | 7.7% |

Coverage is the limit: `MPRL` samples 9.51% of a tile's 256 cells, and objects 1.59%, both clustered
where structures are.

Because `MSUR._0x1c` is bit-identical to `MODF.Position.Z`, an object names a specific placement
row. Over 132 paired tiles and 1,061 objects, 99.6% matched one. Recoverability:

| Field | Recoverability |
|---|---|
| `Position.Z` | exact (join key) |
| `Position.X` / `.Y` | median 1.4–1.6 units, p90 ~21 (control 76/63) |
| `BoundsMax.Z` | median 0.028 units, 62.5% within 1 |
| Other bounds axes | median 2.7–3.7 units |
| `Rotation`, `NameId`, `UniqueId`, `Flags`, `DoodadSet` | not recovered |

`MODF`'s bounding box is the world-space AABB of the already-placed, already-rotated model, so a
valid box can be written without knowing the rotation. The horizontal axes miss by a couple of units
because PM4 holds only collision-relevant geometry. `MODF.Position` is the model origin, not the box
centre, and that offset is a per-asset constant.

`MDDF` is harder: doodads live in the `_0x1c == 0` bucket, which is not split per instance by the
key, so recovering doodad rows requires splitting that bucket by adjacency component first.

### 4.12 The three point streams are not copies

`MSVT`, `MSPV` and `MSCN` are near-identical in size (373,517 / 368,323 / 387,163 over 60 files,
within 5%), but they do not share their points at 0.25 units:

| Pair | Coincidence |
|---|---|
| `MSVT` on `MSPV` | 32.58% |
| `MSPV` on `MSVT` | 43.15% |
| `MSVT` on `MSCN` | 25.05% |
| `MSCN` on `MSVT` | 15.76% |
| `MSPV` on `MSCN` | 19.46% |
| `MSCN` on `MSPV` | 11.84% |

The strongest pair follows from the wall-quad construction: a wall quad stands on a blocked
adjacency edge between two floor surfaces, so 43% of wall vertices land on floor vertices by
construction.

### 4.13 Open-question results (2026-10-20)

Corpus-wide tests for the questions in §7. Counts are over 616 files (502 structured).

| Question | Result | Status |
|---|---|---|
| Q01 `MVER` | 502 files at 12304; 114 zero-byte files at 0 | answered |
| Q02 `MSHD` spans | `_0x00`/`_0x08` vs point-stream spans: r = 0.13–0.26; vs raw placement spans: r = 0.39–0.58 | partial |
| Q03 `MSUR._0x00` | 7 classes: 3 = 184,356 surfaces, all `_0x1c == 0`, mean area 45.9; 16 = 87,243 (mean Z 97.1); 17 = 13,493 (83.9); 18 = 161,359 (81.1); 19 = 71,293 (114.0); 20 = 105; 21 = 243 | answered |
| Q04 normal | 506,626 / 518,092 surfaces match the polygon normal within 5°, median angle 0° | answered |
| Q04 plane distance | `_0x10 = -dot(normal, v0)` in raw MSVT space: 518,080 / 518,092 (99.998%), median error 0 | answered |
| Q05 `_0x1c` | 332,032 nonzero; 196,016 (59.0%) match a placement Z in the same file; 231,383 (69.7%) match one anywhere in the map | partial |
| Q06 `MSLK._0x04` | 5,155 groups one model, 20,543 multi-model, 3,466 none | answered: finer group, not a model id |
| Q07 `MSLK._0x00` | bit 0 = no geometry (flags 1, 17); bit 4 = local/same-tile (flags 17/18/20/26/28 have 0 cross-tile); low family 1/2/4/10/12 is cross-tile | partial |
| Q08 `MSLK._0x0c` upper half | 1,273,335 / 1,273,335 are `0xFFFF` | answered |
| Q09 path kinds | 4-index 587,229 (99.86% coplanar); 5-index 629 (63.8%); 6-index 11,024 (83.8%); 0 Z-dominant in all kinds | answered: all are vertical wall quads |
| Q10 `MSCN` contour | 378,614 closer to boundary vs 281,007 closer to interior; median boundary 0.304, interior 0.554 | partial: supported, not decisive |
| Q11 `MPRL._0x14` | 20,834 match the run index, 53,007 the node level, 8,425 the group size, 29,369 the link count, 1,079 the run length, 292,509 none | open |
| Q12 `MPRR` | 3,171,912 runs; first two are ordered spatial bounds in 3,171,133 (99.98%); middle values are valid MPRL indices in 1,966,930 (62%); last record is `(0, 768)` in 1,366,697 (43%) | answered: spatial node bounds |
| Q13/Q14 destructible | MDBH 502, MDBI 10, MDBF 30, MDOS 526, MDSF 2,684; all MDSF indices in range | answered: sparse but consistent |
| Q16 unattributed groups | 17,965 groups: 2,478 one model, 14,053 multi-model, 1,434 none | answered: mostly multi-model |
| Q17 chunk presence | 14 core chunks in 502 files; MDBI/MDBF in 1 file | answered |
| Q18 tall surfaces | 32,567 tall, 9,789 non-Z-dominant (30.1%); 485,525 short, 55,700 non-Z-dominant (11.5%) | answered: 2.6× enrichment, 70% still Z-dominant |
| Q19 no-geometry `MSLK` | 674,453 records; 673,910 (99.9%) resolve to a present target surface; 543 absent | answered |
| Q20 `MSUR` as terrain source | 1,930,146 vertices; 180,588 (9.4%) within 1.0 of an `MPRL` sample; median height error 1.08 | partial: sparse |

---

## 5. Coordinates

`MSVT`, `MSPV` and `MSCN` share the same axis order (YXZ). `MPRL` is the only permuted chunk.

```text
MSVT/MSPV/MSCN -> placement: (17066.666 - x, 17066.666 - y, z)
MPRL           -> placement: (17066.666 - z, 17066.666 - x, y)
```

The origin constant is `17066.666`. The `/36` height factor from §3.8 applies when the source data
is in internal inches.

---

## 6. Reader and renderer recipe

1. Walk chunks by `fourcc` + `size`; do not assume order.
2. Read `MVER` and reject unknown versions only if you cannot skip unknown chunks.
3. Read `MSVT` and `MSVI`. A surface polygon is
   `MSVI[MSUR.MSVI_first_index .. + MSUR._0x01)` into `MSVT`.
4. Read `MSUR`. For each surface, the adjacency window is
   `MSLK[MSUR._0x18 .. + MSUR._0x02)`.
5. Read `MSLK`. Resolve `msur_index` in the tile named by `_0x0c` (low byte X, next byte Y). If the
   addressed tile is absent, keep the record unresolved; do not substitute the current file.
6. Read `MSPV`/`MSPI`. A record with `MSPI_index_count > 0` owns
   `MSPI[MSPI_first_index .. + MSPI_index_count)` into `MSPV`; this is the vertical connection
   geometry. `MSPI_first_index == -1` means open passage.
7. Read `MPRL` and `MPRR`. Group points by `MPRL._0x04`; use `MPRR` as the spatial index.
8. Read `MSCN` as positions, not normals.
9. Read the destructible-building family as `MDBH` count, `MDBI` indices, `MDBF` filenames, `MDOS`
   state, `MDSF` associations.
10. Apply the coordinate transform in §5 before placing geometry in world space.

---

## 7. Open questions

Results for many of these are in §4.13; the table below keeps the next discriminating test for the
ones that remain open.

| ID | Question | Next test |
|---|---|---|
| Q01 | `MVER` high byte `0x30` on PM4 | Compare independently identified builds |
| Q02 | `MSHD._0x00`/`_0x04`/`_0x08` exact meaning | Header-only controls; span vs content tests |
| Q03 | `MSUR._0x00` finer classes beyond 0x03/0x10–0x15 | Partition consistency against addressed topology |
| Q04 | `MSUR._0x04`/`_0x08`/`_0x0c` normal convention and `_0x10` sign | Compare to raw polygon geometry and rival plane formulas |
| Q05 | `MSUR._0x1c` when it is not a placement Z | Independent placements with equal-height/scaled counterexamples |
| Q06 | `MSLK._0x04` as a model id versus a finer group | Join to model paths with wrong-object controls |
| Q07 | Remaining `MSLK._0x00` bits | Relate each stratum to addressed geometry/back-links |
| Q08 | `MSLK._0x0c` upper half | Inventory all bits; compare present/absent addresses |
| Q09 | `MSPV`/`MSPI` path kinds | Addressed-target contact/plane tests per raw kind |
| Q10 | `MSCN` contour hypothesis | Distance to nearest `MSUR` edge vs interior |
| Q11 | `MPRL._0x14` counter meaning | Test against node type, connection count, or flags |
| Q12 | `MPRR` referent and terminator | Test under the 4k+3 constraint |
| Q13 | `MDBH`/`MDBI`/`MDBF` payloads | Inspect complete raw payloads and cross-record relations |
| Q14 | `MDOS`/`MDSF` association rules | Validate both directions with wrong-domain controls |
| Q15 | Authored editor node identity | Identify the screenshot scene and compare point placement/connectivity |
| Q16 | Per-object M2 ownership inside a merged mesh | Test candidate separators against Museum assets and wrong-object controls |
| Q17 | Build/family variation and PD4 differences | Compare fingerprinted independent builds |
| Q18 | What sets the vertical extent of stretched `_0x1c == 0` surfaces | Per-vertex over-water test |
| Q19 | The 34.4% of `MSLK` entries with no adjacency component link | Cross-tile neighbour test |
| Q20 | Whether `MSUR` floor surfaces are a further terrain-height source | Compare to `MPRL` samples |

---

## 8. Parser behaviour

- Every walked chunk's full raw payload is retained alongside typed projections.
- Fixed-record stride remainders produce diagnostics; typed arrays omit trailing bytes.
- `MVER`/`MSHD` project initial words; extra bytes survive only in raw payloads.
- `MDBF` retains one string plus raw length; further strings are not decoded.
- Duplicate scalar chunks use the last value; list chunks append.
- An overrun stops the walk; trailing bytes and parse failures are diagnostics, not exceptions.
- Cross-chunk address/domain validation belongs to a validation layer, not the raw reader.
- PD4 has a separate reader; similar names do not transfer this corpus's semantics.

---

## 9. Proposed names

These are suggestions for the wiki editor. They are not part of the format and are not used as
primary names above.

| Chunk | Wiki name | Proposed name | Basis |
|---|---|---|---|
| `MSUR` | `_0x00` | `surface_class` | measured 3 for most M2-only, 16–19 for WMO-only |
| `MSUR` | `_0x01` | `msvi_index_count` | window length into `MSVI` |
| `MSUR` | `_0x02` | `mslk_index_count` | window length into `MSLK` |
| `MSUR` | `_0x03` | `padding` | always 0 |
| `MSUR` | `_0x04`/`_0x08`/`_0x0c` | `normal` | three floats |
| `MSUR` | `_0x10` | `plane_distance` | float |
| `MSUR` | `_0x18` | `mslk_first_index` | window start into `MSLK` |
| `MSUR` | `_0x1c` | `placement_z` | float, bit-exact placement Z |
| `MSLK` | `_0x00` | `flags` | observed bits |
| `MSLK` | `_0x01` | `source_edge_index` | zero-based source-edge ordinal |
| `MSLK` | `_0x04` | `object_group_id` | equals `MPRL._0x04` |
| `MSLK` | `_0x0c` | `target_tile` | low byte X, next byte Y |
| `MSLK` | `msur_index` | `target_msur_index` | target surface in the addressed tile |
| `MSLK` | `_0x12` | `system_flags` | `0x8000` in PD4 v48 |
| `MPRL` | `_0x00` | `region_id` | 265 values, one per object group |
| `MPRL` | `_0x04` | `object_group_id` | equals `MSLK._0x04` |
| `MPRL` | `_0x14` | `point_type` | 1,603 values |
| `MPRL` | `_0x16` | `point_flags` | 507 values |
| `MPRR` | `_0x00`/`_0x02` | `quadtree_node_min` / `quadtree_node_max` | first two records of a run |
| `MPRR` | `_0x00` | `mprl_index` | middle records of a run |
| `MPRR` | `_0x00` | `run_terminator` | last record of a run |

---

## 10. Notes for the wiki editor

- `MPRL._0x00` has 265 distinct values in this corpus, one per object group.
- `MPRL._0x02` is `-1` and `_0x06` is `0x8000` for every point in this corpus.
- `MSUR._0x18` is a window start into `MSLK`, with length `MSUR._0x02`; the `MSCN` reading
  overruns 6,240 times.
- `MSUR._0x1c` is an IEEE-754 float holding the placement Z.
- `MSLK.msur_index` is resolved in the tile named by `MSLK._0x0c`.
- `MSLK._0x01` is a zero-based source-edge ordinal.
- `MSLK._0x04` equals `MPRL._0x04`; it is a group/model id.
- `MPRR` is a quadtree index over `MPRL`.
- `MSCN` holds positions.

---

## 11. Sources

- Reader: `src/core/WowViewer.Core.PM4/Services/Pm4ResearchReader.cs`
- Models: `src/core/WowViewer.Core.PM4/Models/Pm4ResearchChunkModels.cs`
- Coordinates: `src/core/WowViewer.Core.PM4/Services/Pm4CoordinateService.cs`
- Structural measurements: `specs/270-pm4-object-association-ledger-and-placement/evidence/`
- Format measurements: `specs/279-decode-pm4-format/evidence/`
- Corpus fingerprint (SHA-256 of sorted `filename:sha256` lines):
  `d6460882861d2ffa0e13db323a8ddf9a8e52705ce8d3c23e74fa1e6a86b8d709`
