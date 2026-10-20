# PM4: A Measured Format Reference

**Updated:** 2026-10-20 · **Research owner:** [Spec 279](../../specs/279-decode-pm4-format/spec.md)
· **Status:** substantial structural comprehension; **not a complete semantic decode**.

This is the single current account of PM4 in this repository. It supersedes the older
[PM4/PD4 draft](../wowdev-wiki/pm4-pd4-draft.md), which remains historical research only.

## 1. What PM4 is

PM4 contains **surface geometry**, a **tiled connection graph**, a **terrain-intersection point
layer** (MPRL) with a **quadtree spatial index** (MPRR), and auxiliary records. It is **not** a
scene-object list: a connection can join surfaces from different assets, and one asset can
contribute several disconnected or merged fragments.

Three layers must stay separate:

1. **Generated surface layer** — what geometry exists and which surfaces connect. Largely measured.
2. **Point layer** — MPRL terrain-intersection points, grouped per object by `Unk04`. Measured.
3. **Source-object layer** — which authored WMO/M2 model a group belongs to. Measured as a ranked
   join; per-object M2 ownership inside a merged mesh remains unresolved.

### Evidence legend

| Label | Meaning |
|---|---|
| **L** | Layout/code behavior; not proof of semantic intent |
| **M** | Measured on the real corpus with controls |
| **H** | Hypothesis requiring a discriminating test |
| **O** | Operator observation, not independently re-certified |
| **R** | Refuted or materially limited by current evidence |
| **U** | Unknown; retain raw values, do not guess in consumers |

All raw field tables are **L** unless a row says otherwise. Legacy names (“type”, “group”, “height”,
“attribute”, “node”) are aliases, not semantic authority.

## 2. Corpus and evidence identity

Measured baseline: `test_data/original_development/World/Maps/development` — **616 files, 169.08
MiB**, including 114 zero-byte inputs. **309 files contain surfaces.** Fingerprinted map:
**518,092 surface records**, **1,273,335 connection records**. SHA-256 of sorted
`filename:sha256` lines:

```text
d6460882861d2ffa0e13db323a8ddf9a8e52705ce8d3c23e74fa1e6a86b8d709
```

| Key | Source |
|---|---|
| **E-LAYOUT** | [Reader](../../src/core/WowViewer.Core.PM4/Services/Pm4ResearchReader.cs), [models](../../src/core/WowViewer.Core.PM4/Models/Pm4ResearchChunkModels.cs) |
| **E-STRUCT** | [Structural receipt](../../specs/270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md) |
| **E-WINDOW** | [Window controls](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-msur-window-2026-10-20.json) |
| **E-ADDRESS** | [Cross-tile report](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-cross-tile-paired-2026-10-20.json) |
| **E-CONNECT** | [Path contact](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-path-contact-2026-10-20.json) |
| **E-EDGE** | [Edge ordinal](../../specs/279-decode-pm4-format/evidence/phase-2a-edge-ordinal-2026-10-20.md) |
| **E-EDGE2** | [Reciprocal +0x04](../../specs/279-decode-pm4-format/evidence/phase-2b-edge-identity-2026-10-20.md) |
| **E-PLACE** | [Placement ground truth](../../specs/279-decode-pm4-format/evidence/phase-3-placement-ground-truth-2026-10-20.md) |
| **E-M2PART** | [M2 partition](../../specs/279-decode-pm4-format/evidence/phase-3-m2-partition-2026-10-20.md) |
| **E-PATH** | [MPRL/MPRR](../../specs/279-decode-pm4-format/evidence/phase-3-path-nodes-2026-10-20.md) |
| **E-MSHDP** | [MSHD words](../../specs/279-decode-pm4-format/evidence/phase-3-mshd-2026-10-20.md) |

These are corpus-scoped measurements. A similarly named field in another format or build does not
carry the same meaning.

## 3. The measured relationships

```text
MSUR surface
  +0x14 / +0x01 ── index/count ──> MSVI ──> MSVT polygon
  +0x18 / +0x02 ── start/count ──> MSLK connection records
                                      +0x0C low bytes ──> target tile X/Y
                                      +0x10 ───────────> target tile's MSUR
                                      +0x08/+0x0B ─────> MSPI ──> MSPV path geometry

MPRL point (Unk04 = MSLK.GroupObjectId) ──> object group
MPRR ──> quadtree index over MPRL points
```

### 3.1 Surface and connection windows — M

`MSUR +0x02` is the length of the connection window starting at `MSUR +0x18`:

```text
connections(surface i) = MSLK[start_i .. start_i + count_i)
start_i = MSUR[i].raw_0x18
count_i = MSUR[i].raw_0x02
```

- All **517,783 consecutive surface pairs** satisfy `start[i] + count[i] == start[i+1]`.
- The lengths sum to all **1,273,335** MSLK entries: 100% coverage, zero overruns.
- Wrong lengths from the index-count and group-byte fields fit only **22.25%** and **8.46%**.
- Aimed at MSCN, the proposed windows overrun **6,240** times.

The alias “attribute mask” is misleading for this relationship. Source: **E-WINDOW / E-STRUCT**.

### 3.2 A connection addresses a target tile — M

```text
targetX = MSLK.raw_0x0C & 0xFF
targetY = (MSLK.raw_0x0C >> 8) & 0xFF
targetSurfaceIndex = MSLK.raw_0x10
target = map[targetX, targetY].MSUR[targetSurfaceIndex]
```

The upper half of `+0x0C` is **U**. Never substitute the current file for an absent addressed tile.

| Population / test | Observation |
|---|---|
| Correctly addressed same-tile links | **1,257,562 / 1,257,562 reciprocal** |
| Cross-tile links | **15,773**, of which 14,659 resolve; 1,114 address absent files |
| References outside a present addressed tile | **0** |
| Paired 10,222 records: addressed vs local boundary | **5,026 vs 28** |
| Same paired records: addressed vs local reciprocal pair | **9,748 vs 15** |

The blanket “RefIndex is always local” is **R**. Source: **E-ADDRESS / E-STRUCT**.

### 3.3 Connection kinds differ — M, meanings incomplete

Raw `TypeFlags` 0x01/0x11 carry negative geometry starts and track shared polygon boundaries.
Other values have nonnegative geometry windows and different boundary behavior. This is evidence of
**different connection roles**, not of WMO/M2/tree/building identity. The additional bits are **U**.

The path-bearing population is 598,882 records: 587,229 four-index windows, 11,024 six-index, 629
five-index. Source: **E-CONNECT / E-STRUCT**.

### 3.4 `MSLK +0x01` is a source-edge ordinal — M

`+0x01` is a **zero-based source-polygon edge ordinal** for all **1,273,335** records. With source
vertices in MSVI-window order, ordinal `e` selects vertices `e` and `(e+1) mod count`.

- Selected edge matches the addressed boundary **674,013** times vs **137** shifted control and
  **96** one-based.
- For path-bearing records it selects the auxiliary geometry's vertical plane **536,142 / 598,882**
  vs 3,120 shifted control.

This is a source-boundary address, not an object subtype. Source: **E-EDGE**.

### 3.5 Authored placements separate M2 and WMO surfaces — M

Working backwards from `_obj0.adt` placements (MDDF = M2, MODF = WMO), a surface covers a placement
when the placement origin is inside its projected polygon or within 0.05 of its boundary.

| Measure | M2 (MDDF) | WMO (MODF) |
|---|---|---|
| Authored placements | 59,544 | 1,116 |
| Covered by ≥1 surface | 12,147 (20.4%) | 883 (79.1%) |

Of 308,892 surfaces: 16,338 cover an M2 origin only, 1,961 a WMO origin only, 218 both, 290,375
neither.

| Surface class | Surfaces | `Ck24Type == 0` | `GroupKey == 3` | Dominant type | Dominant group |
|---|---|---|---|---|---|
| M2-only | 16,338 | 14,489 (88.7%) | 14,489 (88.7%) | 0 | 3 |
| WMO-only | 1,961 | 57 (2.9%) | 51 (2.6%) | 64–67 | 16–19 |
| Both | 218 | 110 | 110 | mixed | mixed |
| Neither | 290,375 | 98,162 (33.8%) | 96,516 (33.3%) | 66/67 | 18 |

`Ck24Type == 0` and `GroupKey == 3` are redundant encodings of the **M2 class**. The bucket is a
superset, not a per-object identity: only 12.8% of zero surfaces cover an authored M2 origin.

Connection kinds also separate: `TypeFlags` 17/18/20/26/28 occur on WMO-only surfaces and on zero
M2-only surfaces; 1/2/4/12 are M2-leaning. `AttributeMask` 0 is M2-leaning; 9/10 are WMO-leaning.
Source: **E-PLACE**.

### 3.6 The merged M2 mesh is not separable by decoded fields — M (negative)

The M2 bucket has 112,818 surfaces in 170 tiles. No decoded structural feature partitions it into
individual placed objects:

| Test | Result |
|---|---|
| Topological components | 22,286; 14,777 cover 0 origins, 4,775 cover 1, 2,734 cover 2+ |
| Multi-origin components | 734 same-model, **2,000 mixed-model** |
| Largest component | 1,023 surfaces, 133 origins, 7 model paths |
| MSVT vertex sets | 4,289 disjoint, **6,576 share vertices** |
| MSUR index ranges | 1,295 disjoint, **9,570 overlap** |
| `MSUR.Padding` (+0x03) | **0 for all 112,818 M2 surfaces** |
| `MSLK.GroupObjectId` (+0x04) | 1,895 placements single-valued, **7,315 multi-valued** |

The merge is genuine. Per-object M2 ownership inside the merged mesh is **U**. Source: **E-M2PART**.

### 3.7 MPRL is the point layer; MPRR is its quadtree index — M

**MPRL** holds 178,588 points; **177,525 (99.4%)** lie inside the PM4 mesh under the measured
mapping. `Unk02 = −1` and `Unk06 = 0x8000` for every point. The varying payload is `Unk00`, `Unk04`,
`Unk14`, `Unk16`.

**MPRR** holds 17,150,640 records at exactly **4 bytes each**. Each run is a quadtree node: the
first two records are the node's `(min, max)` bounds, the middle records are the MPRL indices inside
the node, and the last record is a terminator. Across all 616 files the node width/height is a power
of two:

| Node width/height | Runs |
|---|---|
| 2 | 1,205,258 |
| 1 | 872,798 |
| 4 | 740,137 |
| 8 | 216,751 |
| 32 | 75,101 |
| 16 | 61,482 |

The `4k+3` run lengths and ~4:1 decay are the quadtree hierarchy. The first run of a file is a
special header, not a node. MPRR is **not** an edge list over MPRL (only 39.5%/62.4% of values fall
in MPRL range). Source: **E-PATH**.

### 3.8 The object partition is `MPRL.Unk04` — M

`MPRL.Unk04` is a spatial group id and equals **`MSLK.GroupObjectId` (+0x04)** for **176,748 of
178,588 points (98.97%)**. Alternatives are negligible: `MSUR.Ck24ObjectId` 111 (0.06%),
`MSUR.GroupKey` 289 (0.16%).

- **98.8%** of MPRR runs group MPRL points sharing one `Unk04`; **100%** share one `Unk00`.
- The id is a **group/model id, not a unique placement id** — it is reused across placements.

The MPRL fields form a **region → object → point** hierarchy:

| Level | Field | Distinct values | Behaviour |
|---|---|---|---|
| Region | `Unk00` | 265 | every `Unk04` group has exactly one `Unk00` (42,092 / 0) |
| Object | `Unk04` | 42,092 | equals `MSLK.GroupObjectId`; groups points spatially |
| Point | `Unk14` | 1,603 | varies within an object group |
| Point | `Unk16` | 507 | mostly constant per object |

The direct group→surface link is **not** containment (0.4%): the linked surfaces are the object's
connection/wall surfaces while the points are on the ground. The ground surfaces containing the
points are M2 66.7%, WMO 29.0%, other 6.7%. MPRL points are **not** MSPV path vertices (0.8% within
0.1, 13.1% within 1.0); the path geometry is the separate MSPV/MSPI stream. Source: **E-PATH**.

### 3.9 Object groups join to authored models — M

Joining each `Unk04` group to the nearest authored ADT placement (411 files with placements, 29,164
groups):

| Nearest placement distance | Groups |
|---|---|
| ≤ 1 | 518 (1.8%) |
| ≤ 5 | 9,402 (32.2%) |
| ≤ 10 | 8,019 (27.5%) |
| ≤ 25 | 7,415 (25.4%) |
| > 25 | 3,662 (12.6%) |

Adding a **class-consistency check** (ground-surface class must match the model's `.M2`/`.WMO`
class) gives a precision figure:

| Result | Groups |
|---|---|
| Consistent (≤ 25 units, class match) | **19,991 (68.5%)** |
| Inconsistent (≤ 25 units, class mismatch) | 5,363 (18.4%) |
| No match within 25 units | 3,662 (12.6%) |

Aggregating consistent groups per model gives a **ranked matcher** (895 models). Top models:

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

The ranking recovers the map's common objects and gives a per-model confidence from group count and
mean footprint. Source: **E-PATH**.

## 4. Binary envelope and parser contract

| Header offset | Width | Reader interpretation |
|---|---|---|
| +0x00 | 4 bytes | Reversed ASCII signature; on-disk `REVM` becomes logical `MVER` |
| +0x04 | 4 bytes | Little-endian payload length, excluding the eight-byte header |
| +0x08 | length bytes | Payload; next chunk begins immediately after it |

Integers are little-endian; floats are 32-bit IEEE-754. No alignment padding and no enforced chunk
order. Unknown signatures retain raw payloads. This is **E-LAYOUT**, not every file variant.

### Supported chunk inventory

These **16** signatures have typed handling:

| Chunk | Record width | Current understanding |
|---|---|---|
| MVER | u32 | Version word; taxonomy incomplete |
| MSHD | 8 × u32 | Header metadata; mostly **U** |
| MSVT | 12 bytes | Surface geometry positions |
| MSVI | 4 bytes | Surface indices |
| MSUR | 32 bytes | Polygon and connection windows |
| MSLK | 20 bytes | Tile-addressed surface connections |
| MSPV | 12 bytes | Connection/path geometry |
| MSPI | 4 bytes | Connection geometry indices |
| MSCN | 12 bytes | Point stream; role **U** |
| MPRL | 24 bytes | Terrain-intersection point + object group |
| MPRR | 4 bytes | Quadtree index over MPRL |
| MDBH | u32 | Count-like auxiliary header |
| MDBI | 4 bytes | Auxiliary index-like words |
| MDBF | variable | Filename-bearing payload |
| MDOS | 8 bytes | Auxiliary state-like records |
| MDSF | 8 bytes | Proposed surface/state associations |

“Supported” means parsed into a model, not semantically understood.

## 5. Record layouts

### MVER

| Offset | Type | Meaning |
|---|---|---|
| 0x00 | u32 | Version word; retain all 32 bits |

All **502 structured files** carry **0x00003010 (12304)**; the 114 zero-byte inputs are not a
separate version. Subfield meaning is **U**. Source: **E-04**.

### MSHD

Eight u32 words. `Field0C`, `Field10`, `Field14`, `Field18`, `Field1C` are **always zero** across
616 files. `Field00`/`Field08` carry data (max 534) and correlate with the destructible-building
count at ~0.76–0.79; `Field04` is smaller and more variable. Exact meaning is **U**. Source:
**E-MSHDP**.

### MSUR — 32 bytes

| Offset | Type | Alias | Meaning |
|---|---|---|---|
| 0x00 | u8 | GroupKey | M: 3 = M2 class, 16–19 = WMO class; finer meaning **U** |
| 0x01 | u8 | IndexCount | M: MSVI window length |
| 0x02 | u8 | AttributeMask | M: MSLK window length |
| 0x03 | u8 | Padding | **U**; 0 for all M2 surfaces |
| 0x04 | f32[3] | Normal | Normal-like; convention needs checking |
| 0x10 | f32 | Height | **U**; plane/height-like scalar |
| 0x14 | u32 | MsviFirstIndex | M: MSVI window start |
| 0x18 | u32 | _0x18 | M: MSLK window start |
| 0x1C | u32 | PackedParams | M: top byte (`Ck24Type`) is 0 for M2, 64–67 for WMO |

Polygon vertices come from the MSVI window into MSVT. Current extractors triangulate as fans; that
is a tooling convention, not proof of source triangulation.

**CK24** is a derived accessor, not a chunk:

```text
CK24       = (raw_0x1C >> 8) & 0x00FFFFFF
Ck24Type   = (raw_0x1C >> 24) & 0xFF
ObjectId16 = CK24 & 0xFFFF
```

The zero bucket is the **M2 surface class** (see §3.5), not a per-object identity.

### MSLK — 20 bytes

| Offset | Type | Alias | Meaning |
|---|---|---|---|
| 0x00 | u8 | TypeFlags | M: connection-role strata; full meaning **U** |
| 0x01 | u8 | Subtype | M: zero-based source-edge ordinal |
| 0x02 | u16 | Padding | **U** |
| 0x04 | u32 | GroupObjectId | M: equals `MPRL.Unk04`; group/model id, not unique placement |
| 0x08 | i24 | MspiFirstIndex | Signed start of MSPI window; sign-extend |
| 0x0B | u8 | MspiIndexCount | Count for the geometry window |
| 0x0C | u32 | LinkId | M: low byte target X, next byte target Y; upper half **U** |
| 0x10 | u16 | RefIndex | M: target surface in the addressed tile |
| 0x12 | u16 | SystemFlag | **U** |

`+0x04` is **not** a unique edge id: 889,594 values are reused. Type-1 reciprocal records share the
value 98.92%, type 17 100%; types 4/10/12/20/26/28 share it ≤1.5%. Source: **E-EDGE2**.

### MSVT / MSPV / MSCN — 12 bytes

Three f32 components, no axis permutation by the reader. MSVT's link to surfaces is measured;
MSCN's role is **U**.

### MSVI / MSPI — 4 bytes

One u32 index word. MSVI indexes surface vertices; MSPI feeds connection geometry.

### MPRL — 24 bytes

| Offset | Type | Alias | Meaning |
|---|---|---|---|
| 0x00 | u16 | Unk00 | M: region/zone parent |
| 0x02 | i16 | Unk02 | M: −1 for every point |
| 0x04 | u16 | Unk04 | M: object group = `MSLK.GroupObjectId` |
| 0x06 | u16 | Unk06 | M: 0x8000 for every point |
| 0x08 | f32[3] | Position | Terrain-intersection point |
| 0x14 | i16 | Unk14 | **U**; per-point type/flag |
| 0x16 | u16 | Unk16 | **U**; per-point flag |

### MPRR — 4 bytes

| Offset | Type | Alias | Meaning |
|---|---|---|---|
| 0x00 | u16 | Value1 | Quadtree node bound / MPRL index |
| 0x02 | u16 | Value2 | Quadtree node bound / terminator |

`0xFFFF` in `Value1` is a run sentinel. The first two records of a run are the node bounds; the
middle records are MPRL indices; the last record is a terminator.

### Auxiliary family

| Chunk | Offset | Type | Alias |
|---|---|---|---|
| MDBH | 0x00 | u32 | DestructibleBuildingCount; count-like, meaning **U** |
| MDBI | 0x00 | u32 | DestructibleBuildingIndex; domain **U** |
| MDBF | 0x00 | bytes | First NUL-delimited ASCII string; raw length retained |
| MDOS | 0x00/0x04 | u32 | Building index / destruction state; meaning **U** |
| MDSF | 0x00/0x04 | u32 | MsurIndex / MdosIndex; candidate associations |

## 6. Coordinate conventions

The [coordinate service](../../src/core/WowViewer.Core.PM4/Services/Pm4CoordinateService.cs)
implements:

```text
MSVT-to-placement: (origin - raw.X, origin - raw.Y, raw.Z)
MPRL-to-placement: (origin - raw.Z, origin - raw.X, raw.Y)
origin constant:   17066.666
```

These are **L** code behavior, validated against real placements. Do not change established
camera/terrain/render transforms because a synthetic export disagrees; revalidate with wrong-axis
and reflection controls in a scoped phase.

## 7. Viewer and matching implications

- **Read graph addresses before drawing conclusions.** File-local-only lookup loses real seams.
- **Keep raw data and provenance.** Filtering to CK24 zero before collecting relations removes
  context.
- **Do not turn navigation connectivity into object identity.** Merged scaled copies are real.
- **Use the object-group join as a prior.** `Unk04` → nearest same-class model is measured at 68.5%
  consistency; prefer high-confidence models over generic shape search.
- **Preserve the baseline.** Existing WMO matching is ~70% accurate (operator-reported). New
  interpretations stay in opt-in research layers until their consumer changes have regression and
  real behavior proof.

## 8. Unknown register

| ID | Open question | Next discriminating test |
|---|---|---|
| U01 | MVER subfields | Compare independently identified builds |
| U02 | MSHD words | Count/domain tests with header-only controls |
| U03 | MSUR +0x00/+0x03 | Partition consistency against addressed topology |
| U04 | MSUR normal/height | Compare to raw polygon geometry and rival plane formulas |
| U05 | MSUR +0x1C slices | Independent placements with equal-height/scaled counterexamples |
| U06 | MSLK +0x04 | Now identified as `MPRL.Unk04`; test whether it is a model id or a finer group |
| U07 | Remaining MSLK type bits | Relate each stratum to addressed geometry/back-links |
| U08 | MSLK +0x0C upper half | Inventory all bits; compare present/absent addresses |
| U09 | MSPV/MSPI path kinds | Addressed-target contact/plane tests per raw kind |
| U10 | MSCN role | Link-domain and geometry/spacing tests |
| U11 | MPRL `Unk14`/`Unk16` | Test against node type, connection count, or flags |
| U12 | MPRR terminator `Value1` | Test whether it is a count or a next-node pointer |
| U13 | MDBH/MDBI/MDBF | Inspect complete raw payloads and cross-record relations |
| U14 | MDOS/MDSF associations | Validate both directions with wrong-domain controls |
| U15 | Authored editor node identity | Identify the screenshot scene and compare point placement/connectivity |
| U16 | Per-object M2 ownership inside a merged mesh | Test candidate separators against Museum assets and wrong-object controls |
| U17 | Build/family variation, PD4 | Compare fingerprinted independent builds; do not generalize one corpus |

## 9. Parser preservation and limitations

The reader retains every walked chunk's **full raw payload** alongside typed projections:

- Fixed-record stride remainders produce diagnostics; typed arrays omit trailing bytes.
- MVER/MSHD project initial words; extra bytes survive only in raw payloads.
- MDBF retains one string plus raw length; further strings are not decoded.
- Duplicate scalar chunks use the last value; list chunks append.
- An overrun stops the walk; trailing bytes and parse failures are diagnostics, not exceptions.
- Cross-chunk address/domain validation belongs to the research layer, not the raw reader.
- PD4 has a separate [reader](../../src/core/WowViewer.Core.PM4/Services/Pd4ResearchReader.cs);
  similar names do not transfer this corpus's semantics.

## 10. Conclusion

We understand the **tiled surface/connection structure**, the **M2/WMO surface partition**, the
**MPRL point layer and its quadtree index**, and a **measured object-to-model join**. That is enough
to correct research interpretations and to drive a ranked matcher, but not enough to call PM4 fully
decoded. Per-object M2 ownership inside a merged mesh, the MPRL per-point flags, the MPRR terminator,
and the auxiliary family remain open and are listed in §8.
