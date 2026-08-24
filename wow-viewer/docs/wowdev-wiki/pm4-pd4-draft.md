# PM4 / PD4 — wowdev.wiki draft

**Status**: draft for wiki submission. Every claim below cites the command that produced it and the
figure it produced, so an editor can re-run it.

**Corpus**: 616 PM4 files (309 non-empty) of the 4.x-era `development` map, plus 2 PD4 files
(`6or_garrison_workshop_v3_snow`, WoD). Placement ground truth is the companion split `_obj0.adt`
set. All figures measured 2026-08-23/24 through this project's canonical reader — no hand-parsing.

**Scope note**: this draft *corrects* two things the wiki currently states. Both corrections are
measured, and both are called out explicitly rather than silently substituted.

---

## 1. What these files are

PM4 and PD4 are **navigation-mesh** files produced by Blizzard's toolchain alongside map and model
data. They contain no renderable object geometry; they describe the surfaces characters can stand
on, how those surfaces connect, and what blocks movement between them.

| | PM4 | PD4 |
|---|---|---|
| unit | one ADT tile | one WMO |
| coordinates | distance-from-origin, tile-independent | object-local, centred near origin |
| chunks | `MVER MSHD MSPV MSPI MSCN MSLK MSVT MSVI MSUR MPRL MPRR MDBH MDBI MDBF MDOS MDSF` | `MVER MCRC MSPV MSPI MSCN MSLK MSVI MSVT MSUR` |

PD4 is the per-model core; PM4 is that core placed into tile space plus the tile-level structures
(placement lists, destructible-building payloads). The garrison PD4's `MSVT` spans
`(-29.83, -28.16, -2.10) .. (29.79, 23.94, 30.79)` — centred on the model origin, confirming the
object-local frame.

---

## 2. `MVER` is a format version, not a build

The PM4 `MVER` payload is `10 30 00 00` (12304 / `0x3010`); PD4 is `30 00 00 00` (48 / `0x0030`).

The PM4 value is **constant across the corpus** — verified on files spanning a 20× size range
(63,628 / 164,952 / 1,267,923 bytes), each with an identical 32-byte `MSHD`. It therefore encodes
neither a size nor any content-derived quantity.

Read consistently from byte 0, that is **PM4 = version 16**, **PD4 = version 48**. PM4's `0x30` high
byte is **undecoded**; it should not be assigned a meaning yet. Note that 12304 resembles the real
client build `4.0.1.12304` — this is a coincidence of digits and the word must not be read as a
build number.

---

## 3. `MSUR` — walkable surfaces

32 bytes per entry.

| offset | type | meaning |
|---|---|---|
| `0x00` | uint8 | unknown (no measured semantic) |
| `0x01` | uint8 | **vertex count** of this surface |
| `0x02` | uint8 | **count of this surface's `MSLK` entries** — see §4 |
| `0x03` | uint8 | padding |
| `0x04` | float[3] | surface normal |
| `0x10` | float | plane distance |
| `0x14` | uint32 | **first index into `MSVI`**, window length = `0x01` |
| `0x18` | uint32 | **first index into `MSLK`**, window length = `0x02` — see §4 |
| `0x1C` | float | **Z of the placement that produced this object** — see §5 |

A surface is a **polygon fan** over `MSVI[0x14 .. 0x14 + 0x01)` → `MSVT`. Faces = Σ(count − 2).

Polygon sizes are **not** triangles. In the garrison PD4: 4×2640, 5×385, 3×257, 6×151, 7×7 —
**77% quads**.

### 3.1 Correction: `0x18` indexes `MSLK`, not `MSCN`

The wiki describes `_0x18` as an index into `MSCN`. Measured over 309 non-empty files
(`pm4 msur-window`):

| target | chain fits | Σ window lengths | target count | out of range | coverage |
|---|---|---|---|---|---|
| **`MSLK`** | **517,783 / 517,783 (100.0000%)** | **1,273,335** | **1,273,335** | **0** | **100%** |
| `MSCN` | 100% | 1,273,335 | 1,342,410 | **6,240** | 93.52% |

Against `MSLK` the windows tile the stream exactly, with zero overruns. Against `MSCN` they overrun
6,240 times and cover only 93.52%. The `MSCN` reading is eliminated.

Detector power was established before the claim: a positive control
(`0x14 + 0x01` → `MSVI`, known-true) fits 100%, and two negative controls reusing the same start
field with a wrong length fit 22.25% and 8.46%.

The earlier reading measured `_0x18` as a *single index* and concluded "MSCN is not one boundary
vertex per surface." That conclusion was correct from the wrong model: a single index per surface
cannot reach a stream larger than the surface count. It is a **window**, and its length lives at
`0x02`.

---

## 4. `MSLK` is the surface-adjacency graph

Each `MSUR` owns the contiguous run `MSLK[0x18 .. 0x18 + 0x02)`. Within that run, `MSLK.RefIndex`
names a **neighbouring surface** — not the owning one.

- Owner round trip (`MSLK[j].RefIndex == owning surface`): **18 / 1,273,335 (0.0014%)** — systematically zero.
- Reciprocity (`j ∈ N(i) ⇒ i ∈ N(j)`): **1,257,562 / 1,273,301 = 98.76%**, with 18 self-edges.
- The delta histogram of `RefIndex − owningSurface` is **symmetric** (+1/−1: 1418/1418; +2/−2: 368/368; +3/−3: 286/286).

A directed or accidental relation does not reciprocate. `MSLK` is therefore the **undirected
adjacency graph** of the navmesh.

**47.03%** of adjacency records carry an `MSPI`/`MSPV` path window (598,882 of 1,273,319) — a
**vertical quad** standing on that connection. The other 53% (`MspiFirstIndex < 0`) are **open
passage**, not missing data. Independently, the path windows are 98.05% size-4, 99.6% coplanar, and
**0 of 598,790 have Z as their dominant normal axis** — walls. `MSUR` normals are 91.7% Z-dominant —
floors.

Composed: *walkable polygons, each owning a run of adjacency records naming reciprocal neighbours,
with a vertical quad erected on the blocked subset.*

Connected components of this graph are **finer than objects** — tile `00_00` has 299 components
against 16 objects, 99.00% pure but only **0.33% distinct**. Components are walkable islands inside
an object. **1.19%** of edges cross an object boundary, so the mesh is genuinely stitched between
objects.

---

## 5. Correction: `0x1C` is a placement Z, not a packed key

`MSUR._0x1C` is widely treated as a packed identity field (locally nicknamed "CK24", decomposed into
type/high/low bytes). **It is an IEEE-754 float** holding the **Z coordinate of the ADT placement
that produced the object.**

Measured against `MODF`/`MDDF` placement records:

| \|float − placement Z\| | matched | control (correspondence rotated) |
|---|---|---|
| ≤ 0.001 | **904 / 966 = 93.58%** | 2.10% |
| ≤ 5.0 | 97.00% | 67.61% |
| median \|Δ\| | **0.000000** | 1.963654 |

Bit-exact. The match also yields the **source model filename** —
`0x40AA0A7E = 5.3138 → GuardTower_intact.wmo`.

Three independent confirmations that do not rely on the ADT join:

1. Correlation with each object's bbox floor Z = **0.995227**; controls −0.125 (min X) and 0.011
   (surface count). Subtracting each object's own floor removes **90%** of the variance.
2. The high byte populates **only** float exponent bands `0x3D–0x43` and their **sign-set**
   counterparts `0xBD–0xC3`. **Negative values exist** — impossible for an object id. On one tile,
   16 values confined to three bands is ~10⁻³¹ under a uniform key.
3. Instances of one model share an identical `value − bboxMinZ` offset to three decimals
   (6.03/6.03; −10.11/−10.11; 0.372/0.374/0.374) — the model's origin-to-floor distance.

**It is not a bounding box.** It coincides with the mesh bbox min in 3 of 904 objects, with max in
**0**, and with centre in **0**. It sits *below* the bbox 57.41% of the time and *inside* it 42.59%,
**never above** — the signature of a model origin at or beneath the walkable floor (median
normalised position −0.037).

**Consequences.** Every "CK24" slice is a slice of a float; the nickname's "type" byte is the float's
**exponent band**, which is why a tile appears to have only ~4 "types". Grouping by the value works
only because distinct placements sit at distinct heights, and it **must collide for two objects at
equal height**. `0x1C == 0` (`0.0f`) is the per-tile remainder, not an object.

The field carries **no X or Y**. A full per-object placement is `(placement.X, placement.Y, 0x1C)`,
where X and Y come from the joined `MODF`/`MDDF` record and `0x1C` is both the join key and the
authored Z.

---

## 6. `MSCN` is not normals

**0 of 1,342,410** points have unit length. Lengths run 238.7 – 44,773.7, mean 26,566.9 — the
magnitudes of distance-from-origin coordinates, not direction vectors. `MSCN` holds **positions**.

With the `_0x18 → MSCN` reading eliminated (§3.1), `MSCN` currently has **no decoded index
relationship**. Its role is open.

---

## 7. `MPRR` — undecoded, but structurally constrained

`MPRR` is a flat array of `uint16` pairs delimited by a sentinel (`Value1 == 0xFFFF`): 13,978,231
non-sentinel entries in **3,171,410** runs across 502 files.

**99.9843% of runs have length ≡ 3 (mod 4)** — measured over all **246** distinct run lengths, max
5019. Residues 0 and 2 are **empty**; the only 497 exceptions are residue 1. So a run plus its
terminating sentinel always occupies a whole number of 4-entry (16-byte) blocks, and 75.5% are the
minimal single block.

Any candidate reading must satisfy that quantisation. Ruled out so far: the run count matches **no**
chunk's entry count (best 4/502 = 0.8%), so it is not a per-entry list for any known chunk. Value
range tests are weak and non-discriminating (best non-self domain `MSVI` at 67.6% / 79.0%) and are
**bound tests only** — a value in range does not prove ownership.

---

## 8. Reproducing these figures

```
pm4 msur-window        --input <dir> [--describe]     # §3.1, §4
pm4 adjacency-components --input <dir> [--packed] [--surface-z]   # §4, §5, §6
pm4 placement-z        --input <dir>                  # §5
pm4 object-library     --input <dir> --output x.json  # §5
pm4 mprr               --input <dir>                  # §7
pm4 merge-bias         --input <dir>                  # §6
```

---

## 9. Open questions

- `MSUR._0x00`, and `MSUR._0x10`'s exact convention.
- `MSCN`'s role, now that its only claimed index consumer is gone.
- `MPRR` entirely, under the 4n+3 constraint.
- `MVER`'s `0x30` high byte on PM4.
- The 34.4% of `MSLK` entries with no adjacency component link, and the 1.24% of edges that do not
  reciprocate (cross-tile neighbours are the obvious candidate, untested).
