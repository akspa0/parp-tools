# PM4: A Measured Format Reference

**Updated:** 2026-10-20 · **Research owner:** [Spec 279](../../specs/279-decode-pm4-format/spec.md)
· **Status:** partial structural comprehension, **not a complete semantic decode**.

This is the current reading route for PM4 in this repository. It replaces the older combined
[PM4/PD4 draft](../wowdev-wiki/pm4-pd4-draft.md) as an account of what we presently know.
That draft remains historical research, not a second current specification.

## 1. The short explanation

PM4 contains **geometry and a tiled connection graph**, together with point/reference streams
and auxiliary records whose meanings are not all understood. We can now follow a surface's
connections into another PM4 tile and verify that the addressed polygons really meet.

It is **not established to be a scene-object list**. A navigation connection can join surfaces
that originated in different assets. Conversely, one asset can contribute several disconnected
or merged fragments, and many M2s contribute little or no usable collision geometry. A connected
component therefore cannot automatically be called “one placed M2”.

Three questions must remain separate:

1. **Generated surface layer:** what geometry exists, and which surfaces connect? Substantial
   relationships are measured below.
2. **Authored path-node layer:** which points correspond to NPC route/control nodes visible in
   WoWEdit? The operator reports screenshot evidence, but the PM4 chunk mapping is not established.
3. **Source-object layer:** which surfaces came from which WMO/M2 instance and scale? WMO matching
   already works imperfectly; a reliable general M2 ownership partition is still unresolved.

These layers may be related. The distinction is not a claim that they are separate files or that
we know exactly which tool generated each chunk.

### Evidence legend

| Label | Meaning |
|---|---|
| **L — layout/code** | What the existing reader stores or current code does; not proof of semantic intent |
| **M — measured** | Relationship supported by real corpus output and controls, within the stated scope |
| **H — hypothesis** | Plausible interpretation requiring a discriminating test |
| **O — operator evidence** | Attributed operator observation, not independently re-certified here |
| **R — refuted/limited** | An interpretation contradicted or materially limited by current evidence |
| **U — unknown** | Meaning not established; retain raw values and do not guess in consumers |

All raw field tables below are **L** unless a row explicitly says otherwise. “Type”, “group”,
“height”, “attribute” and “node” in legacy code are aliases, not automatic semantic authority.

## 2. Scope and evidence identity

The measured baseline is the extracted development map under repository-relative
`test_data/original_development/World/Maps/development`: **616 files, 169.08 MiB**, including
114 zero-byte inputs. **309 files contain surfaces**; other files are still accounted for.
Exact client/source-build identity has not been independently established in this pass.

The fingerprinted map contains **518,092 surface records** and **1,273,335 connection records**.
SHA-256 of sorted `filename:sha256` lines:

```text
d6460882861d2ffa0e13db323a8ddf9a8e52705ce8d3c23e74fa1e6a86b8d709
```

Evidence keys used throughout this reference:

| Key | Source |
|---|---|
| **E-LAYOUT** | [Unchanged reader](../../src/core/WowViewer.Core.PM4/Services/Pm4ResearchReader.cs), [raw models](../../src/core/WowViewer.Core.PM4/Models/Pm4ResearchChunkModels.cs) |
| **E-STRUCT** | [Whole-map structural receipt](../../specs/270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md) |
| **E-WINDOW** | [Window/length controls](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-msur-window-2026-10-20.json) |
| **E-ADDRESS** | [Paired addressed/local-target report](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-cross-tile-paired-2026-10-20.json), [summary](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-cross-tile-paired-summary-2026-10-20.txt) |
| **E-CONNECT** | [Geometry/path contact and raw exceptions](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-path-contact-2026-10-20.json) |
| **E-LOCAL** | [Earlier five-pair diagnostic and its limits](../../specs/270-pm4-object-association-ledger-and-placement/evidence/investigation-2026-10-20.md) |
| **E-POOLS** | [Operator pools/WL hint tooling receipt](../../specs/270-pm4-object-association-ledger-and-placement/evidence/map-context-instrument-2026-10-20.md) |
| **E-04** | [Raw +0x04 domain/reverse-link report](../../specs/279-decode-pm4-format/evidence/raw-04-domain-2026-10-20.json), [summary](../../specs/279-decode-pm4-format/evidence/raw-04-summary-2026-10-20.txt) |
| **E-EDGE** | [Source-edge ordinal/spatial receipt](../../specs/279-decode-pm4-format/evidence/phase-2a-edge-ordinal-2026-10-20.md), [full-map report](../../specs/279-decode-pm4-format/evidence/edge-ordinal-geometry-2026-10-20.json) |

These are corpus-scoped measurements. A source comment, unit test or similarly named field in a
different format does not establish the same meaning for every PM4/PD4 build.

## 3. The relationships we can currently follow

```text
MSUR surface record
  +0x14 / +0x01 ── index/count ──> MSVI ── indices ──> MSVT polygon geometry
  +0x18 / +0x02 ── start/count ──> MSLK connection records
                                      │
                                      +0x0C low bytes ──> target tile X/Y
                                      +0x10 ushort ────> target tile's MSUR record
                                      │
                                      some kinds: +0x08/+0x0B ──> MSPI ──> MSPV geometry

MSCN / MPRL / MPRR ── point/reference roles still under investigation
MDBH / MDBI / MDBF / MDOS / MDSF ── auxiliary state/name relationships not fully validated
```

### 3.1 Surface and connection windows — M

The raw surface byte at **+0x02**, historically called attrMask, functions as the length of the
connection window beginning at **+0x18** in this corpus:

```text
connections(surface i) = MSLK[start_i .. start_i + count_i)
start_i = MSUR[i].raw_0x18
count_i = MSUR[i].raw_0x02
```

This is not based only on values fitting an array:

- All **517,783 consecutive surface pairs** satisfy `start[i] + count[i] == start[i+1]`.
- The lengths sum to all **1,273,335** MSLK entries: 100% coverage, zero overruns.
- The independent MSVI index/count positive control also chains exactly.
- Wrong lengths from the index-count and group-byte fields fit only **22.25%** and **8.46%**.
- Aimed at MSCN instead, the proposed windows overrun **6,240** times.

**Conclusion:** a link-window count is measured here; the alias “attribute mask” is misleading for
this relationship. This does not decode every other field or identify an M2 ownership partition.
Source: **E-WINDOW**, interpreted with **E-STRUCT**.

### 3.2 A connection reference includes its target tile — M

For the measured map:

```text
targetX = MSLK.raw_0x0C & 0xFF
targetY = (MSLK.raw_0x0C >> 8) & 0xFF
targetSurfaceIndex = MSLK.raw_0x10
target = map[targetX, targetY].MSUR[targetSurfaceIndex]
```

The high half of +0x0C is **not decoded by this rule**. Do not silently replace an absent addressed
tile with the current file, even when the numeric surface index happens to fit locally.

**Worked example:** a link in `development_01_01.pm4` has packed value `0xFFFF0201` and surface
reference 753. The byte pair addresses tile **(1,2)**, not (2,1). The source file has only 189
surfaces, while the addressed tile contains that referenced record. A range failure in the source
file is not sufficient evidence of corruption.

The stronger map-wide evidence also includes references that fit both alternatives:

| Population / test | Observation |
|---|---|
| Correctly addressed same-tile links | **1,257,562 / 1,257,562 reciprocal**, no self-references |
| Cross-tile links | **15,773**, of which 14,659 resolve; 1,114 address absent files |
| References outside a present addressed tile | **0** |
| Resolved cross-tile raw-type-0x01 links | 7,190 / 7,211 share a boundary and have a back-link |
| Paired 10,222 records: addressed vs local boundary | **5,026 vs 28** |
| Same paired records: addressed vs local reciprocal pair | **9,748 vs 15** |

Reciprocity means a B→A connection exists; it does not mean every raw field is duplicated or that
the graph is an object scene graph. Wrong-byte-order, wrong-target and local-file controls are in
**E-ADDRESS / E-STRUCT**. The blanket “RefIndex is always local to the current file” is **R**.

### 3.3 Connection kinds have different geometry behavior — M, meanings incomplete

In the observed data, raw type values **0x01 / 0x11** correspond to negative connection-geometry
starts and strongly track shared polygon boundaries. Other observed values have nonnegative
geometry windows and very different direct-boundary behavior.

This is evidence of **different connection roles**, not evidence that a type value means WMO,
M2, tree, building, object instance or design kit. The meaning of the additional bits is **U**.

The path-bearing population contains 598,882 records: **587,229 four-index windows**, 11,024
six-index windows and 629 five-index windows. A four-index window is common, not universal.
Raw MSPV/MSPI geometry shows meaningful endpoint contact, far above wrong-target controls,
but the local-target contact experiment must not be promoted into a complete classification of
walls, portals, drops, jumps or traversability. Source: **E-CONNECT / E-STRUCT**.

### 3.4 A connection selects an edge of its source polygon — M

MSLK byte **+0x01**, historically called Subtype, is strongly supported as a **zero-based source
edge ordinal** in this corpus. It is valid for every one of the **1,273,335** records. With source
polygon vertices in MSVI-window order, ordinal `e` selects vertices `e` and `(e+1) mod count`.
That wrap closes the polygon; it does not coerce an out-of-range field value into validity.

The selected edge matches the addressed polygon boundary **674,013** times, versus **137** for
a shifted-edge control and **96** under a one-based source interpretation. For direct raw kinds
0x01/0x11, the ordinal selects every full shared edge found by the independent geometry test.
For path-bearing records it also strongly selects the auxiliary geometry's vertical plane:
**536,142 / 598,882**, versus 3,120 under the shifted-edge control.

This gives a source-boundary address, not a generic object subtype or M2 group. It does not prove
equal heights, movement direction, jump/drop/portal meaning or authored path-node identity.
Source: **E-EDGE**. The current code alias is retained for compatibility, not silently renamed.

## 4. Binary envelope and parser contract

PM4 is read as consecutive chunks:

| Header offset | Width | Reader interpretation |
|---|---:|---|
| +0x00 | 4 bytes | Reversed ASCII chunk signature; on-disk `REVM` becomes logical `MVER` |
| +0x04 | 4 bytes | Unsigned little-endian payload length, excluding the eight-byte header |
| +0x08 | length bytes | Payload; next chunk begins immediately after it |

Integers are little-endian. Floats are 32-bit IEEE-754 values as read by existing code. There is
no alignment padding added by this reader and no enforced universal chunk order. Unknown chunk
signatures retain their raw payloads. This describes **E-LAYOUT**, not every possible file variant.

### Supported logical chunk inventory

These **16** signatures have typed handling in the current PM4 reader:

| Chunk | Typed payload layout / record width | Current understanding |
|---|---|---|
| MVER | First u32; minimum 4 bytes | Version word; interpretation/version taxonomy incomplete |
| MSHD | First eight u32 words; minimum 32 bytes | Header metadata; exact meanings mostly unknown |
| MSVT | f32 triple array; 12 bytes | Surface geometry positions |
| MSVI | u32 array; 4 bytes | Indices used by surface geometry |
| MSUR | 32-byte records | Polygon windows and measured connection windows, plus unresolved fields |
| MSLK | 20-byte records | Tile-addressed surface connections with multiple raw kinds |
| MSPV | f32 triple array; 12 bytes | Geometry referenced by connection windows; exact roles vary/unknown |
| MSPI | u32 array; 4 bytes | Indices used by connection geometry |
| MSCN | f32 triple array; 12 bytes | Point stream; semantic role unresolved |
| MPRL | 24-byte records | Point plus raw fields; authored-node identity unresolved |
| MPRR | Two u16 words; 4 bytes | Reference-like records; structure/meaning unresolved |
| MDBH | First u32; minimum 4 bytes | Count-like auxiliary header; interpretation incomplete |
| MDBI | u32 array; 4 bytes | Auxiliary index-like words; target/domain unresolved |
| MDBF | Variable bytes; typed first ASCII string | Filename-bearing payload; complete structure unresolved |
| MDOS | Two u32 words; 8 bytes | Auxiliary state-like records; field names are not proof |
| MDSF | Two u32 words; 8 bytes | Proposed auxiliary/surface associations; require validation |

“Supported” means parsed into a current model, not semantically understood. The raw envelope
also supports retaining other signatures without assigning a typed meaning.

## 5. Record-by-record layouts

Offsets are relative to each record/payload, not the file. Alias spelling permits source lookup;
unknown meanings are intentionally not replaced by more persuasive invented names.

### MVER

| Offset | Type | Alias / meaning |
|---|---|---|
| 0x00 | u32 | Version word; retain all 32 bits. Do not truncate to a low-byte “version 16” claim. |

The reader accepts a payload of at least four bytes and projects only the first word. Extra bytes
remain in the raw payload. No complete version-bit taxonomy is asserted here.

**M — E-04:** all **502 structured/nonzero-size files** carry the full word **0x00003010 (12304)**.
The other **114 files are zero-byte inputs**, whose reader-default version zero is not evidence
of a separate version-0 format. The significance of the word's subfields remains unknown.

### MSHD

| Offset | Type | Current alias | Semantic status |
|---|---|---|---|
| 0x00 | u32 | Field00 | U |
| 0x04 | u32 | Field04 / RegionId | Region-like grouping is a hypothesis, not a proven global object/zone key |
| 0x08 | u32 | Field08 | U |
| 0x0C | u32 | Field0C | U |
| 0x10 | u32 | Field10 | U |
| 0x14 | u32 | Field14 | U |
| 0x18 | u32 | Field18 | U |
| 0x1C | u32 | Field1C | U |

An `IsEmptyStubRegion` helper exists for one header value. That is current implementation behavior,
not proof that the word is an empty-tile flag or a complete explanation of header-only files.

### MSUR — 32 bytes

| Offset | Type | Current alias | Meaning / confidence |
|---|---|---|---|
| 0x00 | u8 | GroupKey | U: group/class-like raw value, not a confirmed object class |
| 0x01 | u8 | IndexCount | M: surface's MSVI window length |
| 0x02 | u8 | AttributeMask / attrMask | M: MSLK window length in this corpus |
| 0x03 | u8 | Padding | U: preserved raw byte; name does not prove unused padding |
| 0x04 | f32[3] | Normal | Normal-like vector; exact convention/authority needs independent checking |
| 0x10 | f32 | Height / PlaneDistance | U: plane/height-like scalar; sign/equation not closed here |
| 0x14 | u32 | MsviFirstIndex | M: start of the surface's MSVI window |
| 0x18 | u32 | _0x18 / MscnIndex alias | M: start of its MSLK window; MSCN alias is not current authority |
| 0x1C | u32 raw bits | PackedParams | Float-Z correlation is H with local support; universal meaning/ownership unresolved |

Polygon vertices are obtained by reading the MSVI window and looking up those indices in MSVT.
Current extractors triangulate these windows as fans. This is a tooling convention; the present
reference does not prove universal convexity, source triangulation or complete source-asset geometry.

#### What “CK24” currently means

CK24 is a **derived code accessor**, not a separately parsed chunk or proven asset identifier:

```text
CK24       = (raw_0x1C >> 8) & 0x00FFFFFF
Ck24Type   = (raw_0x1C >> 24) & 0xFF
ObjectId16 = CK24 & 0xFFFF
```

Bit-reinterpreting the same word as float has matched authored WMO placement heights in several
local observations, including the [five-pair diagnostic](../../specs/270-pm4-object-association-ledger-and-placement/evidence/investigation-2026-10-20.md).
That does **not** establish a global object ID, a reliable asset-kind byte or M2 ownership. Earlier
height-correlation fractions used different subsets and are not interchangeable receipts.

The current whole-map census observes **186,060** surfaces in derived CK24 zero, all also raw-word
zero. The two predicates are not identical by bit definition; retain the distinction in new data.
The zero bucket is not one object and is not established to be exclusively M2 geometry.

### MSLK — 20 bytes

| Offset | Type | Current alias | Meaning / confidence |
|---|---|---|---|
| 0x00 | u8 | TypeFlags | M: observed connection-role strata; complete flag meaning U |
| 0x01 | u8 | Subtype | M: zero-based source-polygon edge ordinal in this corpus (E-EDGE) |
| 0x02 | u16 | Padding | U; preserve the word |
| 0x04 | u32 | GroupObjectId | U: object identity is not established; index/reverse-link alternatives under test |
| 0x08 | signed i24 | MspiFirstIndex | L/M: signed start used for MSPI geometry windows; negative marker in direct-neighbor population |
| 0x0B | u8 | MspiIndexCount | L/M: count used with the geometry-window start |
| 0x0C | u32 | LinkId | M: low byte target tile X, next byte target tile Y; upper half meaning U |
| 0x10 | u16 | RefIndex | M: target surface index in the **addressed tile** |
| 0x12 | u16 | SystemFlag | U; constant-looking values do not prove padding or an external-link flag |

The signed 24-bit start must be sign-extended. Treating it as an unsigned 32-bit index would lose
the negative-marker population. A negative start does not imply “no relationship”: those records
carry the strongest direct-boundary adjacency signal.

#### Raw +0x04: tested interpretation, still unknown

The legacy `GroupObjectId` alias is not established semantics. A complete-map test found all
**1,273,335** values in range of **each** source-file MSCN, MSLK and MSVI array, so range fit alone
cannot identify the target domain.

The proposed addressed reverse-link index is **R as a general interpretation**: among 1,269,146
in-range candidates, only **1,603** both lie in the referenced surface's link window and point
back to the source surface/tile. Wrong-index +1 and +17 controls produce 1,092 and 833 such hits.
Those small partial overlaps do not establish a universal reverse-link pointer or object ID.
Only 90,033 of 469,744 nonempty surface windows have a constant +0x04 value; this observation
also must not be inflated into source-ownership semantics. Source: **E-04**.

A later spatial test also fails to identify it as a general direct boundary-point index under the
tested interpretations: MSCN[raw] hits 26 source boundaries versus 827 for a shuffled control;
MSVI[raw]→MSVT hits 3,272 versus 3,175 for index+17. Other source/addressed point alternatives and
raw/permuted MPRL do not establish a useful boundary relation. These are scoped negative results,
not proof those streams are non-spatial or that another transform/indirection is impossible.
Source: **E-EDGE**.

Never discard a connection merely because its surface index is outside the source file; resolve
the target tile first. Conversely, an in-range source-file index is not enough to justify local use.
An absent target remains explicitly unresolved, not silently redirected.

### MSVT / MSPV / MSCN — 12 bytes per entry

| Offset | Type | Current projection |
|---|---|---|
| 0x00 | f32 | Raw component 0 |
| 0x04 | f32 | Raw component 1 |
| 0x08 | f32 | Raw component 2 |

The reader applies no axis permutation to these triples. Shared positional layout is not evidence
that all three streams represent the same kind of point. MSVT's link to surface polygons is
measured; MSCN's semantic identity remains open.

### MSVI / MSPI — 4 bytes per entry

| Offset | Type | Current projection |
|---|---|---|
| 0x00 | u32 | Index word |

MSVI indexes surface vertices; MSPI is consumed by connection-geometry windows. Reference ranges
must be checked against their actual target streams; no object ownership follows from index reuse.

### MPRL — 24 bytes

| Offset | Type | Current alias | Meaning |
|---|---|---|---|
| 0x00 | u16 | Unk00 | U |
| 0x02 | i16 | Unk02 | U |
| 0x04 | u16 | Unk04 | U |
| 0x06 | u16 | Unk06 | U |
| 0x08 | f32[3] | Position | Raw point triple; exact semantic/coordinate role not closed |
| 0x14 | i16 | Unk14 | U |
| 0x16 | u16 | Unk16 | U |

Current coordinate helpers implement a particular MPRL axis permutation, based on earlier research.
The reader itself stores the raw order. A point resembling a ground contact or editor waypoint is
not sufficient to identify this stream as the authored path-node layer.

### MPRR — 4 bytes

| Offset | Type | Current alias | Meaning |
|---|---|---|---|
| 0x00 | u16 | Value1 | U; current helper labels 0xFFFF as a sentinel |
| 0x02 | u16 | Value2 | U |

The reader does not decode runs, adjacency or routes from these words. Sentinel/run hypotheses
must explain all retained words and beat wrong-domain controls. A plausible graph picture is not proof.

### Auxiliary family: MDBH / MDBI / MDBF / MDOS / MDSF

“Destructible building/state” names are existing interpretations, not a complete validated schema.

| Chunk | Offset | Type | Current alias / projection |
|---|---|---|---|
| MDBH | 0x00 | u32 | DestructibleBuildingCount; count-like value, full meaning U |
| MDBI | 0x00 per record | u32 | DestructibleBuildingIndex; target domain U |
| MDBF | 0x00 onward | bytes | First NUL-delimited ASCII string as Filename, or entire payload if no NUL; raw length retained |
| MDOS | 0x00 | u32 | DestructibleBuildingIndex; meaning/domain U |
| MDOS | 0x04 | u32 | DestructionState; meaning U |
| MDSF | 0x00 | u32 | MsurIndex; candidate surface association, not automatically validated |
| MDSF | 0x04 | u32 | MdosIndex; candidate state association, not automatically validated |

MDBH projects the first word. MDBI has four-byte records; MDOS/MDSF have eight-byte records.
MDBF projects only the first string (the full ASCII payload if unterminated), not an inferred
string table or nested structure. Bytes after
the first NUL remain accessible in the raw payload and must not be dismissed as meaningless padding.

## 6. Coordinate conventions: raw bytes versus current tooling

All geometry comparisons in **E-STRUCT** use raw positions in a common hypothesis frame and
0.01-yard endpoint tolerance. They support boundary/contact relationships, not a new viewer transform.

The current [coordinate service](../../src/core/WowViewer.Core.PM4/Services/Pm4CoordinateService.cs)
implements the following **L — code behavior**:

```text
MSVT-to-placement: (origin - raw.X, origin - raw.Y, raw.Z)
MPRL-to-placement: (origin - raw.Z, origin - raw.X, raw.Y)
origin constant:   17066.666
```

Earlier comments describe supporting terrain/placement experiments; those old claims are not
re-certified by this reference. Do not change established camera/terrain/render transformations
because a new synthetic export disagrees. Revalidate a coordinate hypothesis against independent
real placements/geometry, with wrong-axis/reflection controls, in an explicitly scoped phase.

## 7. What this means for viewer and matching tooling

- **Read graph addresses before drawing conclusions.** File-local-only lookup loses real seams and
  can invent self-links or unrelated local neighbors.
- **Keep raw data and provenance.** Filtering to CK24 zero before collecting relations can remove
  context; retaining only the first link type can erase competing roles.
- **Do not turn navigation connectivity into object identity.** Merged scaled copies are a real
  operator-observed difficulty; geometric islands and link components are candidates, not truth.
- **Separate source evidence from priors.** Museum paths/transforms/scales are independent authored
  evidence. Explicit tile pools and WL design-kit hints narrow search, but do not establish matches.
- **Treat scores honestly.** Coarse triangle histograms lose arrangements; finer bins can separate
  different observations while also splitting instances/scales of the same asset. No accuracy gain
  follows merely from more distinct fingerprints.
- **Preserve the baseline.** Existing WMO matching is ~70% accurate according to the operator;
  this reference does not remeasure it. New interpretations remain in opt-in research layers until
  their exact consumer changes have regression and real behavior proof.

## 8. Unknown register and decisive next tests

“Unknown” is a work item, not a license to fill in an attractive name. This register covers all
currently supported chunk families; the [Spec Kit plan/tasks](../../specs/279-decode-pm4-format/tasks.md)
orders the experiments and gates consumer integration.

| ID | Open question / elements | Candidate explanations | Next discriminating test |
|---|---|---|---|
| U01 | MVER word/version bits | Whole version, flags/subversion, other taxonomy | 0x3010 observed in 502 structured files (E-04); compare independently identified builds before interpreting bits |
| U02 | MSHD eight words / RegionId alias | Region key, build metadata, counters, references | Count/value/domain tests with empty/header-only and neighboring-file controls |
| U03 | MSUR +0x00/+0x03 | Surface class, flags, grouping, opaque metadata | Partition consistency against addressed topology and independent source copies; constant-field controls |
| U04 | MSUR vector/+0x10 scalar | Plane normal/distance, height convention, other encoding | Compare to raw polygon geometry and rival plane sign/axis formulas on real data |
| U05 | MSUR +0x1C / CK24 slices | Height-like metadata, source association, other packed value | Independent placements with equal-height/different-kind/scaled counterexamples; no exponent-byte typing |
| U06 | MSLK +0x04 | Other index, recurring value, ownership metadata; general reverse-link/direct boundary-point readings rejected under tested conditions | Test structural relations to now-identified source edges and reciprocal records; do not choose a domain by bounds |
| U07 | Remaining MSLK type bits/+0x02/+0x12 | Connection roles, directionality, modifiers, other metadata; +0x01 is measured source-edge ordinal | Relate each raw stratum to correctly addressed geometry/back-links; do not infer flags from one dominant value |
| U08 | MSLK +0x0C upper half / missing targets | Map/context part, sentinel, other address metadata; incomplete corpus | Inventory all bits; compare present/absent addresses and independent source maps/builds |
| U09 | MSPV/MSPI path-bearing kinds | Connection/boundary geometry with differing roles | Addressed-target contact/overlap/plane tests per raw kind, including non-quad windows and negative controls |
| U10 | MSCN point role/order | Derived geometric/reference stream, control nodes, other points | Link-domain tests and geometry/spacing relations against the addressed graph; rival indices/order controls |
| U11 | MPRL fields and coordinate role | Authored route nodes, contacts/anchors, other records | Validate all fields/domain relationships and coordinate alternatives; operator screenshot correspondence separately |
| U12 | MPRR pair/run structure | Point/reference relations, compact records, routes, other encoding | Test complete records/sentinels against independent point/graph domains; reject mere count coincidence |
| U13 | MDBH/MDBI/MDBF | Auxiliary object/state/name structure | Inspect complete raw payloads, strings/counts and cross-record relations; retain typed-loss counterexamples |
| U14 | MDOS/MDSF associations | State-to-surface mapping or another domain | Validate both directions and real auxiliary payload instances with wrong-domain controls |
| U15 | Authored editor node identity | One of the point streams or another representation | Identify the actual screenshot scene/data and independently compare point placement/connectivity |
| U16 | Source ownership / merged M2s | Explicit metadata, derived grouping, ownership absent after merge | Test candidate separators against Museum asset paths/transforms/scales and wrong-object controls |
| U17 | Build/family variation, PD4 relationship | Shared layouts with differences, different conventions | Compare fingerprinted independent builds/families; do not generalize one corpus to all |

## 9. Parser preservation and limitations

The existing reader retains every walked chunk's **full raw payload**, including unknown
signatures, alongside typed projections. The projections alone are not a lossless format model:

- Fixed-record stride remainders produce diagnostics; typed record arrays omit those trailing bytes.
- MVER/MSHD project initial words; extra bytes survive only in raw payloads. MDBH size deviations
  are diagnosed. A parseable projection does not prove complete payload interpretation.
- MDBF retains one string plus raw length, falling back to the full payload if no NUL exists;
  further strings or nested data are not decoded.
- Duplicate scalar chunks use the last value; list chunks append. Universal ordering/uniqueness is
  not enforced or proven by that implementation.
- An overrun stops the walk; trailing bytes and per-chunk parse failures are diagnostics, not an
  automatic exception. Callers must inspect diagnostics before declaring a complete evidence pass.
- Cross-chunk address/domain validation belongs to the research layer, not the raw reader.
- The PM4 reader does not type-decode every possible signature or validate a checksum. PD4 has
  separate [reader code](../../src/core/WowViewer.Core.PM4/Services/Pd4ResearchReader.cs); similar
  names are not permission to apply this PM4 corpus's semantics to it.

## 10. Current conclusion

We understand a substantial **tiled surface/connection structure**, including a previously missed
target-tile reference domain. That is enough to correct research interpretations, not enough to
call PM4 fully decoded. Authored route nodes, remaining metadata and source-object ownership are
still research questions. This reference and its linked unknown register are intended to make
those gaps explicit and testable rather than hide them behind familiar field names.
