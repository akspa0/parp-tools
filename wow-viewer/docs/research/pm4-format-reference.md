# PM4 / PD4 — Format Reference

PM4 is the per-ADT-tile server-side navigation file; PD4 is the per-WMO equivalent. Both are
chunked binary files. This document describes the container, every chunk, the record layouts, the
measured relationships between chunks, the coordinate transforms, and the questions that remain
open. Field names follow the current wowdev.wiki articles (`PM4`, `PD4`); proposed names are
collected separately in §9.

**Corpus:** 616 PM4 files (309 non-empty) from the 4.x-era `development` map, plus 2 PD4 files
(`6or_garrison_workshop_v3_snow`, 6.0.1). All figures below were measured on that corpus with the
repository reader. A similarly named field in another format or build does not carry the same
meaning.

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

- `_0x0c[5]` is zero in every file.
- `_0x00` and `_0x08` carry data (maximum 534) and correlate with the `MDBH` count at ~0.76–0.79.
- `_0x04` is smaller and more variable.
- Exact meaning of all three is unknown.

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
C3Vector mscn[];   // not normals
```

- Positions, not direction vectors: no point has unit length; lengths are coordinate magnitudes.
- Per-object node network: 1,886 of 1,895 object groups reference `MSCN`; 1,214 of those (64.4%)
  reference nodes that another group also references.
- 724 of 1,886 groups have `MSCN` bounds that extend beyond their own `MSVT` bounds.
- The role is consistent with a per-object exterior boundary; the exact rule is unknown.

### 3.7 `MSLK` — 20 bytes

```c
struct {
  uint8_t  _0x00;              // flags; observed bits &1, &2, &4, &8, &16
  uint8_t  _0x01;              // 0..11-ish; measured zero-based source-edge ordinal
  uint16_t _0x02;              // 0
  uint32_t _0x04;              // index; measured equal to MPRL._0x04
  int24_t  MSPI_first_index;   // -1 when MSPI_index_count is 0
  uint8_t  MSPI_index_count;
  uint32_t _0x0c;              // measured: low byte target tile X, next byte target tile Y
  uint16_t msur_index;         // target surface in the addressed tile
  uint16_t _0x12;              // 0x8000 in PD4 version 48
} mslk[];
```

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
  uint8_t  _0x00;              // unknown; measured 3 for most M2-only surfaces, 16..19 for WMO-only
  uint8_t  _0x01;              // count of indices in MSVI
  uint8_t  _0x02;              // count of MSLK records owned by this surface
  uint8_t  _0x03;              // 0
  float    _0x04;              // normal X
  float    _0x08;              // normal Y
  float    _0x0c;              // normal Z
  float    _0x10;              // plane distance
  uint32_t MSVI_first_index;   // first index into MSVI
  uint32_t _0x18;              // first index into MSLK
  uint32_t _0x1c;              // measured: IEEE-754 float, placement Z
} msur[];
```

A surface is a polygon fan over `MSVI[MSVI_first_index .. MSVI_first_index + _0x01)` into `MSVT`.
Polygon sizes are not restricted to triangles; the garrison PD4 is 77% quads.

`_0x1c` is a float. Measured against `MODF`/`MDDF` placement records, it equals the placement's
`Position.Z` bit-exactly for 904 of 966 joined objects (93.58%), median `|Δ| = 0.000000`; a rotated
control matches 2.10%. `_0x1c == 0.0f` is the unattributed bucket (no placement join).

The bytes of `_0x1c` are sometimes read as a packed key. Because the field is a float, its top byte
is the exponent band: `0x00` for `0.0f`, `0x40`–`0x43` (64–67) for typical placement heights. That
is why the byte view appears to have only a few "types".

### 3.11 `MPRL` — 24 bytes

```c
struct {
  uint16_t _0x00;      // 265 distinct values in this corpus
  int16_t  _0x02;      // -1 for every point
  uint16_t _0x04;      // measured equal to MSLK._0x04
  uint16_t _0x06;      // 0x8000 for every point
  C3Vector position;   // terrain-intersection point
  int16_t  _0x14;      // 1,603 distinct values
  uint16_t _0x16;      // 507 distinct values
} mprl[];
```

### 3.12 `MPRR` — 4 bytes

```c
struct {
  uint16_t _0x00;
  uint16_t _0x02;
} mprr[];
```

`0xFFFF` in `_0x00` is a run sentinel. See §4.7 for the run structure.

### 3.13 Destructible-building family

```c
uint32_t m_destructible_building_count;          // MDBH
uint32_t m_destructible_building_index;          // MDBI
char     m_destructible_building_filename[];     // MDBF, NUL-terminated
struct { uint32_t m_destructible_building_index; uint32_t destruction_state; } mdos[];  // MDOS
struct { uint32_t msur_index; uint32_t mdos_index; } mdsf[];                            // MDSF
```

`MDBH` is followed by `count` `MDBI` records and three `MDBF` records per `MDBI`. The association
rules for `MDOS`/`MDSF` are not verified.

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
record 0        : node bound (min)
record 1        : node bound (max)
records 2..n-2  : MPRL indices inside the node
record n-1      : terminator
```

- Run lengths are `4k + 3`.
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
- `MPRR` is not an edge list over `MPRL`: only 39.5%/62.4% of values fall in `MPRL` range.

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

| ID | Question |
|---|---|
| Q01 | `MVER` subfields beyond the version word |
| Q02 | `MSHD._0x00`, `_0x04`, `_0x08` |
| Q03 | `MSUR._0x00` and `_0x03` |
| Q04 | `MSUR._0x04`/`_0x08`/`_0x0c` normal convention and `_0x10` plane distance |
| Q05 | `MSUR._0x1c` when it is not a placement Z |
| Q06 | `MSLK._0x04` as a model id versus a finer group |
| Q07 | Remaining `MSLK._0x00` bits |
| Q08 | `MSLK._0x0c` upper half |
| Q09 | `MSPV`/`MSPI` path kinds |
| Q10 | `MSCN` role and boundary rule |
| Q11 | `MPRL._0x14` and `_0x16` |
| Q12 | `MPRR` terminator `_0x00` |
| Q13 | `MDBH`/`MDBI`/`MDBF` payloads |
| Q14 | `MDOS`/`MDSF` association rules |
| Q15 | Authored editor node identity |
| Q16 | Per-object M2 ownership inside a merged mesh |
| Q17 | Build/family variation and PD4 differences |

---

## 8. Parser behaviour

- Every walked chunk's full raw payload is retained alongside typed projections.
- Fixed-record stride remainders produce diagnostics; typed arrays omit trailing bytes.
- `MVER`/`MSHD` project initial words; extra bytes survive only in raw payloads.
- `MDBF` retains one string plus raw length; further strings are not decoded.
- Duplicate scalar chunks use the last value; list chunks append.
- An overrun stops the walk; trailing bytes and parse failures are diagnostics, not exceptions.
- Cross-chunk address/domain validation belongs to the research layer, not the raw reader.
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
