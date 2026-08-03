# Phase 0 Research: PM4 Remaining Decode

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-08-03

Every finding below is derived either from the project's own analyzer output (already recorded in
the spec and epic, produced by `pm4 unknowns` / `pm4 cross-tile` over the 616-file corpus) or from
reading the source that produces those numbers. **Nothing here is a new measurement made outside the
canonical stack** — FR-001 forbids that, and the spec's "what must not be repeated" section exists
because it happened once already.

Three of the six findings change what later phases must do. They are R1, R3, and R5.

---

## R1 — The headline baseline does not measure grouping

**Decision**: Report two metrics side by side. Keep the existing
`MSLK.GroupObjectId → MPRL.Unk04` edge emitting identical numbers forever (it is the continuity
baseline SC-001 names), and additionally measure a *grouping* metric that is actually about
partitioning surfaces into objects. Never conflate them.

**Rationale**: `Pm4ResearchUnknownsAnalyzer.cs:185-195` computes that edge as:

```csharp
if (link.GroupObjectId != 0 && link.GroupObjectId <= ushort.MaxValue
    && mprlKeySet.Contains((ushort)link.GroupObjectId))
    groupObjectToMprlFits++;
else if (link.GroupObjectId != 0)
    groupObjectToMprlMisses++;
```

That asks: *does this link's group id also happen to be a valid `MPRL.Unk04` value in the same
file?* It is a **reference-resolution** test. It never touches MSUR and never partitions anything.
A rule could group every surface in the corpus perfectly and still score 65,819 / 1,206,977 on this
edge, and a change that pushed this edge to 100% would not, by itself, group a single surface.

The spec's SC-001 reads "surface-to-object grouping explains substantially more of the corpus than
the current baseline of 65,819 fits against 1,206,977 misses, with the improvement measured on the
same corpus." The *corpus* is the same; the *quantity* cannot be, because the baseline quantity is
not a grouping measurement. Resolving this in the plan rather than in the middle of Phase 3 is the
point of recording it here.

**Alternatives considered**:

- *Redefine the baseline edge to be a grouping metric.* Rejected — it is published in the spec, the
  epic, and the memory bank as a specific number; silently changing what it counts would invalidate
  three documents and make every future comparison ambiguous.
- *Report only the new grouping metric.* Rejected — SC-001 names the old number explicitly, and
  dropping it would make it impossible to tell whether a later regression came from the rule or from
  the harness.

**Consequence for the plan**: Phase 2's gate is that the harness reproduces the old number exactly
when the old rule is run through it. That proves the harness is faithful before any new rule is
trusted.

---

## R2 — The leading grouping candidate is already in the tree, unmeasured

**Decision**: Evaluate this candidate set, corpus-wide, in Phase 3. Measure; do not assume a winner.

| id | rule | why it is a candidate |
|---|---|---|
| G0 | baseline: `MSLK.GroupObjectId → MPRL.Unk04` | continuity reference (R1) |
| G1 | `MSUR.Ck24` (from `PackedParams >> 8`) | current de-facto object key; 1,229 distinct, 266 span 2+ tiles |
| G2 | `MSHD.Field04` (region) × `MSUR.Ck24` | region-scoped variant; `Pm4RegionObjectGrouper` already builds this |
| G3 | surface → its MSLK entries → `GroupObjectId` → all surfaces sharing it | **the lead**; needs no MPRL at all |
| G4 | transitive closure of G1 ∪ G3 | a surface joins if *either* rule links it |
| G5 | G3 partitioned further by `MSLK.TypeFlags` family | tests whether TypeFlags splits or merges objects |
| G6 | MSPV/MSPI shared-vertex connectivity | tests the second stream as the binding agent (depends on Phase 6) |

**Rationale for G3 being the lead**: `MSLK.RefIndex → MSUR` is measured at **1,268,782 fits /
4,553 misses — 99.64%**. That is the surface↔link edge and it is nearly total. `GroupObjectId` is a
field *on MSLK*. So the path surface → link → group id → sibling surfaces is available for
essentially the whole corpus without MPRL participating.

The mechanism is already implemented twice, but never as a rule in its own right and never measured:

- `Pm4RegionObjectGrouper.PartitionByGroupObjectId` (Research/Pm4RegionObjectGrouper.cs:155) uses it
  to **subdivide** a CK24 object into sub-objects.
- `WorldScene.SplitSurfaceGroupByMslk` uses it the same way, to **split** a CK24 seed group into
  parts for rendering.

In both places it is a subdivider *inside* CK24, so it can only ever make objects smaller. As a
primary key it could instead join surfaces that CK24 separates. That is a different rule with a
different outcome, and it has never been run.

**Alternatives considered**:

- *Skip straight to implementing G3.* Rejected — FR-002, and the project's own recorded lesson that
  a confident conclusion from an unmeasured mechanism is exactly how the last PM4 error happened.
- *Geometric clustering as the primary rule.* Rejected as a *primary* — it would group two adjacent
  buildings into one object and cannot express an object that is spatially disjoint. Retained inside
  G6 as a binding test, not as an identity source.

---

## R3 — `trianglesOnly = 0` is zero by construction, not by evidence

**Decision**: Keep the existing counters (they are a published baseline) and add a genuinely
discriminating geometric test in Phase 6. Gate Phase 6 on demonstrating that the new discriminator
*can* separate the interpretations before making any corpus claim.

**Rationale**: `Pm4ResearchUnknownsAnalyzer.cs:201-217` computes:

```csharp
bool indicesMode   = link.MspiFirstIndex >= 0 && (first + count)     <= mspiCount;
long trianglesEnd  = ((long)link.MspiFirstIndex * 3) + (link.MspiIndexCount * 3L);
bool trianglesMode = link.MspiFirstIndex >= 0 && trianglesEnd        <= mspiCount;
```

For any `first ≥ 0` and `count ≥ 0`, `3·first + 3·count ≥ first + count`. Therefore
`trianglesMode ⟹ indicesMode`, always. The `trianglesOnly` bucket — `trianglesMode && !indicesMode`
— **cannot be non-zero for any input whatsoever**. Its reported value of 0 is a property of the
inequality, not a property of the format.

`both = 199,699` likewise carries less than it appears to: it counts windows small enough that the
×3 bound still fits inside the chunk. That is a statement about window size, not about topology.

This is precisely the failure mode the project has already recorded twice — a null result from a
test structurally incapable of finding the thing. It is worth being explicit that the *numbers in
the spec are real analyzer output*; the defect is in what the test can distinguish, not in the
reporting.

**What a real discriminator needs to look at** (Phase 6):

- The **window-size histogram**, bucketed by `MSLK.TypeFlags` family. A spike on multiples of 3 says
  triangle list. A spike at 2 says line segments. A flat spread says polyline.
- Whether a window's **first and last index are equal** — a closed polygon.
- Whether consecutive index triples form **degenerate** triangles (collinear / zero-area), which a
  polyline read as triangles would produce constantly and a real triangle list would not.
- Whether a window's MSPV points are **collinear or coplanar** — a path versus a surface patch.

**A size constraint worth knowing before designing the test**: `MspiIndexCount` is a **single byte**
at offset 11 of the 20-byte MSLK record (`Pm4ResearchReader.ParseMslk`), so a window holds at most
255 indices. `MspiFirstIndex` is a **signed int24** read by `ReadSignedInt24`, so negative values are
representable and `indicesMode` already excludes them — a negative first index plausibly means "no
path", which the histogram should count separately rather than discard.

Across the corpus, 2,418,205 MSPI entries over 598,882 active windows is a mean of ~4.04 indices per
window. A mean near 4 is *suggestive* of quads or short polylines rather than a triangle list, but a
mean cannot distinguish a distribution — the histogram is what settles it, and that is why the
histogram, not the mean, is the deliverable.

**Alternatives considered**:

- *Fix the inequality in place.* Rejected — it would change published baseline numbers, and the
  corrected test would still only compare two bounds, which is not what decides the question.
- *Assume polyline from the mean.* Rejected — that is exactly the inference-from-one-statistic error
  the spec forbids.

---

## R4 — Cross-tile identity needs a new key; the geometric merge that exists is not identity

**Decision**: The Phase 4 object id is **tile-independent**. Sentinel exclusion is an explicit named
policy, not an accident of iteration order.

**Rationale**: The viewer's selection key is `(int tileX, int tileY, uint ck24, int objectPart)` and
its group key is `(tileX, tileY, ck24)` (`WorldScene.cs:1133-1134`). Both embed the tile. FR-006
("object selection MUST include parts of the object residing in other tiles") is therefore
unimplementable with the current key *regardless of how good the decode gets* — this is a genuine
structural blocker in Phase 5, not a consequence of the 5% grouping.

A cross-tile merge does already exist — `Pm4PlacementMath.BuildMergedGroupMap` (line 720) — but it
is **geometric**: union-find over groups in the four neighbouring tiles, joined when connector keys
are close enough (`ShouldMergeConnectorGroups`). That merges things that touch. It cannot merge an
object whose parts are separated, and it will happily merge two distinct objects that abut. It is a
rendering convenience, not an identity source, and Phase 4 must not be built on it.

`Pm4ResearchCrossTileAnalyzer` already does the identity-shaped version, keyed on CK24: 1,229
distinct keys, 266 (21.6%) spanning 2+ tiles, genuine spans of 3–8 tiles. That is the right shape
and the right thing to reconcile Phase 4's output against.

**Sentinel policy**: `CK24 = 0` spans 291 tiles and is a null key, not an object. It must be excluded
by name. `Pm4ObjectSegmentBuilder` currently handles zero-CK24 by falling back to grouping on
`(GroupKey, AttributeMask)` and flagging `ZeroCk24Seed` — a reasonable fallback that must be
preserved as an explicit, reported policy rather than reimplemented differently in the new service.
`MSLK.LinkId` has its own sentinel form (`0xFFFF` high half, tile coords in the low half —
1,273,335 entries), already decoded and verified; it is a tile reference, not an object id.

**Alternatives considered**:

- *Keep the tile in the key and merge at selection time.* Rejected — it pushes identity into the UI,
  which is where the problem already lives, and Spec 129's row layout is object-primary so it would
  need the tile-independent id anyway.
- *Reuse `BuildMergedGroupMap` for cross-tile identity.* Rejected — geometric adjacency is not
  identity; see above.

---

## R5 — MPRR's candidate domains are under-sampled, and its structure is untested

**Decision**: Phase 8 sweeps `MPRR.Value1` and `Value2` against **every** chunk domain using the
sweep helper that already exists in the same file, and additionally tests the **sentinel-delimited
run** hypothesis that the current per-entry test cannot see.

**Rationale**: The analyzer tests `Value1` against exactly two domains, MPRL and MSVT
(`Pm4ResearchUnknownsAnalyzer.cs:286-294`), and neither explains it — 6,778,712 / 7,200,518 for MPRL
and 8,740,189 / 5,239,041 for MSVT. Meanwhile `AddMismatchDomainFits` (line 456) already sweeps
`MSLK.RefIndex` against eight domains — MSLK, MSPI, MSVI, MSCN, MPRL, MSPV, MSVT, MPRR. MPRR simply
never got the same treatment. `Value2` (566 distinct values) is not tested against any domain at all.

The structural hypothesis is the more interesting one. `Pm4MprrEntry.IsSentinel` already models
`Value1 == 0xFFFF`, and the analyzer skips sentinels rather than using them:

```csharp
if (entry.IsSentinel) continue;
```

A `0xFFFF` sentinel scattered through a large flat array is the classic shape of a **run-delimited
list**. If MPRR is a sequence of runs, then `Value1` may index a domain that is *local to its run* —
in which case a global bound test against a chunk's total count is testing the wrong thing entirely,
and would produce exactly the observed pattern of "most values fit, a large minority do not, no
domain explains it."

Scale note for whoever implements it: tile 0_0 holds 327,744 bytes of MPRR at stride 4 = **81,936
entries in one tile**. Per-run statistics on that tile alone will be substantial — and tile 0_0 is
explicitly unrepresentative, so run-structure claims need the same corpus-wide treatment as
everything else.

**Alternatives considered**:

- *Drop MPRR from scope.* Rejected — SC-004 requires each of the nine questions to be resolved,
  narrowed, or documented as unresolvable. "Not attempted" is none of those. A negative result that
  eliminates six domains with evidence satisfies it.
- *Test only run structure.* Rejected — the domain sweep is nearly free once the corpus is read and
  eliminating domains is itself a recorded deliverable under FR-008.

---

## R6 — Nesting is a hypothesis to test, and the machinery to test it exists

**Decision**: Treat the nested-frame hypothesis as a *test to run through `Pm4PlacementMath`*, not
as a premise to build on. No new coordinate code.

**Rationale**: The spec's motivating observation is real and already measured — in
`development_00_00.pm4`, MSVT spans (168–501, 31–450, −12–133) while MPRL spans (31–364, 5–40,
168–499): MPRL's third axis is MSVT's first. Two chunks of one file, different axis order.

What matters for planning is that the stack already models this rather than assuming it away:

- `Pm4PlacementMath.DetectAxisConventionBySurfaceNormals` derives the convention per group from the
  data.
- `ResolveCoordinateMode` / `ResolvePlacementSolution` return a resolution, not a constant, and
  `IsLikelyTileLocal` is a *fallback* rather than an assumption.
- `Pm4CoordinateService.Pm4LocalToAdtPlacement` maps `(X, Y, Z)` to `(tileX·533.33 + X, Y,
  tileY·533.33 + Z)` — the canonical transform whose absence caused the earlier wrong conclusion.

So the correct implementation of "is the connective geometry under a different frame?" is to run the
existing detection over MSPV windows in Phase 6 and report what it resolves to per TypeFlags family
— not to write a new transform. If MSPV resolves to a different convention than MSVT within the same
file, that is a measured finding. If it resolves the same, the nesting hypothesis is weakened for
this stream and that too is recorded.

**Alternatives considered**:

- *Hand-derive the MSPV frame from bounds inspection.* Rejected explicitly and by name — this is the
  exact procedure that produced the earlier confidently-wrong conclusion about tiles stacked above
  and below the map.

---

---

## R7 — The April 2025 mesh is decoded exactly, and nothing was lost from it

**Context**: The user reported that April 2025 Python-era work "had what looks like complete
geometry figured out" and that it was lost in the first C# refactor, supplying a screenshot of
`output_development_00_00.obj` — **15,096 vertices, 7,382 faces**.

**Decision**: Treat the surface mesh as *not* lost. Do not spend a phase recovering it. The prior art
is still valuable, but for different reasons than expected (R9, R10).

**Rationale** — the screenshot's two counts were reproduced exactly, from
`development_00_00.pm4` measured through `pm4 audit`:

| chunk | entries |
|---|---|
| MSPV | 8,778 |
| MSPI | 26,458 |
| MSVT | 6,318 |
| MSVI | 15,602 |
| MSUR | 4,110 |

- **Vertices**: MSPV + MSVT = 8,778 + 6,318 = **15,096** ✓ exact
- **Faces**: Σ(MSUR.IndexCount − 2) = 15,602 − 2×4,110 = **7,382** ✓ exact

Two independent exact matches. So the April 2025 export was: **MSPV and MSVT concatenated into one
vertex list, and every MSUR surface emitted as a triangle fan over its MSVI window.**

Three things follow, and the third is the important one:

1. **The fan triangulation is still in the tree.** `WorldScene.BuildCk24ObjectTriangles` carries the
   comment *"Most PM4 surfaces are listed as loops; use a fan from the first vertex."* Same
   interpretation, plus CK24 grouping, TypeFlags, coordinate-mode resolution and cross-tile handling
   that the old export had none of. The current stack is strictly ahead on the surface mesh.
2. **The candidate counts that do *not* match** rule out the alternatives: MSPI/3 = 8,819 and
   MSVI/3 = 5,200, neither of which is 7,382. So the mesh was not a flat triangle list of either
   stream.
3. **MSPI contributed no faces at all.** 26,458 indices were present and produced nothing. MSPV
   entered the file only as loose vertices. **The connective geometry was never decoded, in any
   era.** The April 2025 result looks complete because the walkable surface mesh *is* complete —
   which spec 130 already records as zero-miss. The prior art does not shortcut US3.

**Alternatives considered**: *That the screenshot proves a lost MSPV/MSPI decode.* Rejected by the
arithmetic — a decoded MSPI would have produced faces, and the face count is fully accounted for by
MSUR fans alone.

**Note on the sibling repo**: `akspa0/parp` `scripts/pm4tool` and `scripts/old/pm4_to_obj` are
JSON/SQLite analysis tools that emit no faces; `scripts/bulkParser/v3/WarcraftAnalyzer` is the first
C# port and is *behind* the current stack (MSUR fields are unnamed `Value0x04`…, no CK24 derivation).
That repo's last commit is 2025-03-10, predating the screenshot. Its
`json_to_obj_converter.py` reads MSPV/MSCN/MSPI and would yield 8,778 verts / 8,819 faces — not the
screenshot. One detail is worth keeping: its MPRL decoder swaps the Y and Z labels with the comment
`# Swapping y and z labels`, so the MPRL axis permutation was known in the Python era too.

---

## R8 — MSPV shares MSVT's coordinate frame (measured); the nesting hypothesis does not apply to it

**Decision**: Phase 6 does **not** need to hunt for a nested frame for the second geometry stream.
The question is purely topological — what the windows mean — not spatial. Record the elimination.

**Rationale**: measured on `development_00_00.pm4` via `pm4 export-json`, through the canonical
reader:

| chunk | min | max |
|---|---|---|
| MSPV | (169.60, 31.84, 0.85) | (498.79, 363.85, 134.55) |
| MSVT | (168.11, 31.00, −12.08) | (501.55, 450.70, 133.74) |
| MSCN | (168.84, 31.42, −12.08) | (499.38, 450.40, 133.00) |
| **MPRL** | **(31.00, 5.00, 168.18)** | **(364.86, 40.20, 499.77)** |

MSPV, MSVT and MSCN share axis order and overlapping ranges. MPRL is the odd one out — its third
axis (168–499) is MSVT's first (168–501), exactly as the epic records.

So the "different axis order between chunks of one file" hazard is real but is an **MPRL** property,
not a general one. It does not implicate MSPV/MSPI. This is independently corroborated by R7: the
April 2025 export concatenated MSPV and MSVT into one vertex list and rendered a coherent scene,
which a frame mismatch would have scattered.

**Alternatives considered**: *Assume nesting because the spec raised it.* Rejected — it is
answerable by measurement, and it was answered.

---

## R9 — The prior art is not lost; it is in this repository, unported

**Decision**: Add a **prior-art harvest phase** before writing new analyzers. Harvest hypotheses and
implementations, then validate each corpus-wide through the canonical stack.

**Rationale**: `parpToolbox/src/parpToolbox/Services/PM4/` on `main` holds ~60 files of PM4
object-assembly work, committed 2025-07-25 to 2025-07-31 and never ported into `wow-viewer`:

| file | bytes | relevance |
|---|---|---|
| `Pm4CrossTileObjectAssembler.cs` | 21,058 | **FR-006** — cross-tile object assembly |
| `Pm4MsurObjectAssembler.cs` | 41,441 | surface→object assembly |
| `Pm4GroupingTester.cs` | 56,246 | **a grouping-rule harness already exists** |
| `Pm4SpatialClusteringAssembler.cs` | 39,692 | spatial clustering candidate |
| `Pm4RefinedHierarchicalObjectAssembler.cs` | 23,039 | hierarchical assembly |
| `Pm4HierarchicalContainerDecoder.cs` | 23,028 | container decode |
| `Pm4MslkPatternAnalyzer.cs` | 16,776 | MSLK semantics |
| `Pm4MprlObjectGrouper.cs`, `Pm4SmartGrouper.cs`, `MslkHierarchyAnalyzer.cs`, `MscnRemapper.cs`, `MSUR_FIELDS.md` | — | grouping and field semantics |

`PM4Tool/` holds the matching documentation (`pm4-analysis-findings.md`,
`pm4-mesh-extraction.md`, `pm4-chunk-structure.md`) and — relevant to the epic's "PM4 is the negative
BSP" thesis — a vendored copy of **MirrorMachine** including `bsptreegenerator.cpp` and
`WMO_exporter.cpp`.

**This is already sanctioned policy.** `AGENTS.md` line 345: *"Treat `Pm4Research`, `MdxViewer`,
`PM4Tool`, `parpToolbox`, and `WoWRollback.PM4Module` as extraction or reference inputs, not as the
default owners of PM4 behavior."* The harvest phase is the extraction that line anticipates.

**Constraint that does not relax**: harvested code is a source of **hypotheses**, not of conclusions.
Every claim still gets corpus-wide validation through `WowViewer.Core.PM4` (FR-001, FR-002). Prior
art that was "semi-functional" and "broken" by its own commit messages is a lead, not an authority.

**Alternatives considered**: *Port `parpToolbox` wholesale.* Rejected — it is a separate stack with
its own reader, and FR-001 forbids a second decoder. Harvest the interpretations, reimplement against
`Pm4ResearchReader`.

---

## R10 — Two prior-art claims that directly contradict or extend current assumptions

Both come from `PM4Tool/docs/pm4/pm4-analysis-findings.md`. Both are **testable corpus-wide** and
neither is accepted here as fact.

### R10a — `MSLK.MspiFirstIndex == -1` marks a doodad placement, not geometry

> "MSLK entries (with MspiFirstIndex == -1) represent doodad placements, with group/object IDs and
> anchor points."

This explains why the reader uses `ReadSignedInt24` for a field that would otherwise be unsigned, and
it gives the negative-first-index population a *meaning* rather than a discard reason. If it holds,
those records are object-placement rows — directly useful to US2 grouping and to Spec 128 — and they
must be counted and routed, never dropped. Phase 6 already counts them separately; Phase 3 should
test whether they carry grouping signal.

### R10b — MSCN is the per-object *exterior boundary*, which makes it a co-equal lead with MSPV/MSPI

> "**MSCN = Exterior Vertices:** Visual inspection confirms MSCN chunk represents exterior (boundary)
> vertices for each object in PM4 files."

The spec names MSPV/MSPI as *the* leading candidate for connective geometry. This says MSCN is the
exterior boundary — which is a more literal description of "what closes a surface set into a sealed
negative volume" than a second index stream is.

Three facts already in the tree support taking it seriously:

- `MSUR._0x18` **already indexes into MSCN**, per the current terminology catalog: *"wowdev.wiki
  calls this `_0x18`. It indexes into MSCN (scene nodes), NOT MDOS."* That is a per-surface link to
  a boundary vertex — exactly the shape needed.
- MSCN (9,990) **exceeds** MSVT (6,318) in tile 0_0, so it is not a subset of the mesh.
- MSCN shares MSVT's frame (R8), so the two are directly co-locatable.

**Consequence**: Phase 6 evaluates MSCN as a candidate connective-geometry source **alongside**
MSPV/MSPI, not after it. The spec permits this explicitly — MSPV/MSPI is recorded as "a lead, not a
conclusion" that "may be eliminated." Note the prior art also flags MSCN↔MSLK as its own unresolved
question, so this is a lead too.

### R10c — a live contradiction about `MSLK.RefIndex` that Phase 3 must resolve

The prior art says MSLK's `0x10` field is *"an anchor index (Unknown_0x10) into **MSVI**"*. The
current stack models the same offset as `RefIndex` and measures `RefIndex → MSUR` at 1,268,782 fits /
4,553 misses.

These are different claims about one field, and the 99.64% figure does not settle it: it is a
**bounds test** (`RefIndex < msurCount`), not a semantic one. A value can be in range for MSUR and
still mean an MSVI index.

This matters because **grouping rule G3 routes surface → MSLK via `RefIndex == surfaceIndex`**. If
RefIndex is an MSVI anchor, G3 is built on the wrong edge. Phase 3 must evaluate both readings and
report which one the corpus supports, rather than inheriting the current assumption.

---

## Summary of what changed because of research

| finding | effect on the plan |
|---|---|
| R1 | Phase 3 must report two metrics, and its gate is exact reproduction of the old one |
| R2 | Phase 4 gets a named 7-rule candidate set; G3 is the lead but must be measured |
| R3 | Phase 7 needs a new discriminator and must prove its power before claiming a result |
| R4 | Phase 5's object id must be tile-independent; the existing geometric merge is not identity |
| R5 | Phase 9 reuses an existing sweep helper and adds the run-structure hypothesis |
| R6 | No new coordinate code anywhere; frame questions go through `Pm4PlacementMath` |
| **R7** | **The surface mesh was never lost — no recovery phase. MSPI produced no faces in any era.** |
| **R8** | **MSPV shares MSVT's frame — the nesting hypothesis is eliminated for the second stream** |
| **R9** | **New Phase 1: harvest ~60 prior-art files already on `main`, per AGENTS.md line 345** |
| **R10** | **Phase 7 gains MSCN as a co-equal lead; Phase 3 must resolve the RefIndex contradiction** |

**No NEEDS CLARIFICATION remains.**

### Provenance of the new measurements

All produced through the canonical stack on 2026-08-03, no hand-parsing:

```powershell
dotnet ...\WowViewer.Tool.Inspect.dll pm4 audit       -i ...\development_00_00.pm4
dotnet ...\WowViewer.Tool.Inspect.dll pm4 export-json -i ...\development_00_00.pm4 -o doc_0000.json
```

**Incidental CLI defect found while doing this**: `pm4 inspect` and `pm4 audit` accept `--output`
silently and ignore it — `RunPm4Inspect`/`RunPm4Audit` only call their `Print…` method. `pm4 unknowns`,
`pm4 mshd`, `pm4 cross-tile` and `pm4 export-json` do honour it. Worth fixing when those files are
touched; recorded here so the next person does not lose an hour to it.
