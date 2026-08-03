# Data Model: PM4 Remaining Decode

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-08-03

Entities are C# records in `WowViewer.Core.PM4/Models/`, following the existing convention in
`Pm4ResearchChunkModels.cs`: `sealed record`, positional parameters, file-scoped namespace,
serialized to JSON by the CLI with `WriteIndented = true`.

**Nothing here replaces an existing chunk model.** `Pm4MslkEntry`, `Pm4MsurEntry`, `Pm4MprlEntry`,
`Pm4MprrEntry` and the rest of `Pm4KnownChunkSet` stay exactly as they are — FR-001. These entities
sit *above* them and describe interpretations, groupings, and evidence.

---

## 1. Evidence register *(Phase 1 — FR-007, FR-008, FR-012)*

### `Pm4DecodeFinding`

One proposed or settled meaning for one undecoded field.

| field | type | notes |
|---|---|---|
| `Key` | `string` | stable identifier, e.g. `MSLK.RefIndex`, `MPRR.Value1`. Never renamed once published. |
| `Status` | `Pm4FindingStatus` | see below |
| `Confidence` | `Pm4Confidence` | `none` / `low` / `medium` / `high` / `verified` |
| `Statement` | `string` | the interpretation in one sentence, or the question if still open |
| `Evidence` | `IReadOnlyList<Pm4EvidenceItem>` | corpus-wide measurements supporting the status |
| `Eliminations` | `IReadOnlyList<Pm4Elimination>` | candidates ruled out, each with what ruled it out |
| `NextStep` | `string?` | null when `Status` is terminal |
| `CorpusSignature` | `string` | which corpus and how many files produced this; guards stale evidence |

### `Pm4FindingStatus`

`Open` · `Partial` · `Resolved` · `Eliminated` · `NoSemanticMeaning`

`Eliminated` and `NoSemanticMeaning` are terminal and are what make FR-008 and FR-012 real. A
finding may leave `Open` only with at least one `Pm4EvidenceItem` attached; a status change with no
evidence is rejected at write time rather than silently accepted.

### `Pm4EvidenceItem`

| field | type | notes |
|---|---|---|
| `Claim` | `string` | what this measurement shows |
| `Fits` | `int` | corpus-wide |
| `Misses` | `int` | corpus-wide |
| `FileCount` | `int` | how many files contributed — a `1` here is a red flag under FR-002 |
| `Source` | `string` | the analyzer + command that produced it |

### `Pm4Elimination`

| field | type | notes |
|---|---|---|
| `Candidate` | `string` | the hypothesis that was ruled out, e.g. `MPRR.Value1 → MSCN` |
| `Reason` | `string` | why the evidence rules it out, not merely that it does |
| `Evidence` | `Pm4EvidenceItem` | the measurement that killed it |

### `Pm4DecodeEvidenceRegister`

The document: `SchemaVersion`, `GeneratedUtc`, `CorpusSignature`, `IReadOnlyList<Pm4DecodeFinding>`.
Round-trips to `output/pm4-decode/evidence-register.json`.

**Validation rules**

- `Key` is unique within a register.
- A finding in a terminal status has `NextStep == null` and at least one `Evidence` item.
- An `Eliminated` finding has at least one `Elimination`.
- Merging a new run into an existing register never deletes an elimination — eliminations
  accumulate. That is the whole point of the entity.

---

## 2. Grouping *(Phases 2–3 — US2, FR-002, FR-003, FR-004)*

### `Pm4GroupingRule`

A rule is a named function from a corpus to a surface→object assignment. Modelled as a descriptor
plus a delegate so the harness can evaluate a set of them in one corpus pass.

| field | type | notes |
|---|---|---|
| `Id` | `string` | `G0`…`G6` per research.md R2 |
| `Name` | `string` | human-readable |
| `Description` | `string` | what it keys on |
| `IsBaseline` | `bool` | exactly one rule (`G0`) sets this |

### `Pm4SurfaceKey`

`(string SourcePath, int SurfaceIndex)` — the corpus-unique address of one MSUR record. Tile
coordinates are derivable from `SourcePath` via `Pm4CoordinateService.TryParseTileCoordinates`, and
are **not** part of the key, because an object may span tiles (research.md R4).

### `Pm4RuleEvaluationReport`

Produced per rule, per corpus run.

| field | type | notes |
|---|---|---|
| `RuleId` | `string` | |
| `SurfacesTotal` | `int` | |
| `SurfacesGrouped` | `int` | assigned to an object of size ≥ 2 |
| `SurfacesSingleton` | `int` | assigned, but alone |
| `SurfacesUngrouped` | `int` | membership undetermined — SC-003 requires this be counted, never absorbed |
| `ObjectCount` | `int` | |
| `CrossTileObjectCount` | `int` | objects with surfaces in ≥ 2 files |
| `ObjectSizeHistogram` | `IReadOnlyList<Pm4ValueFrequency>` | **the guard against a degenerate winner** |
| `LargestObjectSurfaceCount` | `int` | one object holding most of the corpus is a failure, not a win |
| `BaselineEdgeFits` / `BaselineEdgeMisses` | `int` | the R1 continuity metric, reported alongside always |
| `PerFile` | `IReadOnlyList<Pm4RuleFileResult>` | FR-002 — per file *and* total |

`Pm4ValueFrequency` already exists in `Pm4ResearchChunkModels.cs` and is reused rather than
duplicated.

### `Pm4RuleFileResult`

`(string SourcePath, int TileX, int TileY, int SurfacesTotal, int SurfacesGrouped, int
SurfacesUngrouped, int ObjectCount)`.

Per-file rows are what make a tile-0_0-only effect visible as one instead of hiding inside a corpus
total.

### `Pm4GroupingComparisonReport`

`(string InputDirectory, int FileCount, string CorpusSignature,
IReadOnlyList<Pm4RuleEvaluationReport> Rules, IReadOnlyList<string> Notes)` — all rules from one
corpus read, so they are directly comparable.

---

## 3. Object identity *(Phase 4 — FR-003, FR-006, FR-011)*

This is the contract Spec 129 consumes. See [contracts/object-identity.md](./contracts/object-identity.md).

### `Pm4ObjectId`

A **tile-independent** identity. Not a tuple containing a tile.

| field | type | notes |
|---|---|---|
| `Value` | `string` | `pm4obj-<16 hex>` — SHA-256 prefix over the canonical key, matching the existing `pm4seg-` convention in `Pm4ObjectSegmentBuilder.BuildSegmentId` |
| `RuleId` | `string` | which rule minted it — an id is only meaningful relative to its rule |
| `CanonicalKey` | `string` | the pre-hash string, kept for debuggability |

### `Pm4SurfaceAssignment`

One row per MSUR surface in the corpus. **Every surface gets a row** — that is how FR-003 and SC-003
are enforced structurally rather than by discipline.

| field | type | notes |
|---|---|---|
| `Surface` | `Pm4SurfaceKey` | |
| `ObjectId` | `Pm4ObjectId?` | `null` iff `Status == Ungrouped` |
| `Status` | `Pm4AssignmentStatus` | `Assigned` / `Ungrouped` / `SentinelExcluded` |
| `Confidence` | `Pm4Confidence` | travels with the assignment, per FR-007 |
| `Reason` | `string?` | why ungrouped or excluded — required when not `Assigned` |

`SentinelExcluded` is distinct from `Ungrouped` on purpose: CK24 = 0 spanning 291 tiles is a known
null key, not an unsolved case, and collapsing the two would hide a solved problem inside an
unsolved one.

### `Pm4ObjectRecord`

| field | type | notes |
|---|---|---|
| `ObjectId` | `Pm4ObjectId` | |
| `Surfaces` | `IReadOnlyList<Pm4SurfaceKey>` | |
| `TileCoordinates` | `IReadOnlyList<string>` | `"x_y"`, sorted; length > 1 means cross-tile |
| `SurfaceCount` / `TotalIndexCount` | `int` | |
| `Confidence` | `Pm4Confidence` | the weakest confidence among its assignments |
| `Flags` | `Pm4ObjectFlags` | reuses the spirit of the existing `Pm4SegmentConfidenceFlags` |

### `Pm4ObjectIdentityReport`

`(string InputDirectory, string CorpusSignature, string RuleId, int FileCount,
IReadOnlyList<Pm4ObjectRecord> Objects, IReadOnlyList<Pm4SurfaceAssignment> Assignments,
Pm4IdentityCoverage Coverage, IReadOnlyList<string> Notes)`.

**Determinism requirement**: objects sorted by `ObjectId`, assignments by `(SourcePath,
SurfaceIndex)`, tile lists sorted. Re-running on an unchanged corpus must produce a byte-identical
file — that is the Phase 4 gate and it is what makes the artifact safe for 129 to cache.

---

## 4. Second geometry stream *(Phase 6 — US3, FR-009)*

### `Pm4GeometryWindow`

One `MSLK` record's view into MSPI/MSPV.

| field | type | notes |
|---|---|---|
| `Link` | `Pm4SurfaceKey`-shaped `(SourcePath, MslkIndex)` | |
| `FirstIndex` | `int` | signed int24 from the reader; negative is meaningful, not an error |
| `IndexCount` | `byte` | single byte — max 255 (research.md R3) |
| `TypeFlags` / `Subtype` | `byte` | the family this window belongs to |
| `IsNegativeFirstIndex` | `bool` | counted separately, never silently dropped |

### `Pm4WindowGeometryTraits`

The measured traits that actually discriminate — the replacement for the counters that cannot
(research.md R3).

| field | type | notes |
|---|---|---|
| `IsClosed` | `bool` | first index == last index |
| `DistinctVertexCount` | `int` | vs `IndexCount`, exposes degenerate reuse |
| `DegenerateTripleCount` | `int` | zero-area triangles if read as a triangle list |
| `CollinearityScore` | `double` | path-like vs patch-like |
| `PlanarityScore` | `double` | |
| `AxisConvention` | `Pm4AxisConvention` | resolved via `Pm4PlacementMath`, never hand-derived (R6) |

### `Pm4WindowInterpretationResult`

Per candidate interpretation — `Polyline`, `ClosedPolygon`, `TriangleList`, `TriangleFan`,
`TriangleStrip` — its corpus-wide `Fits`, `Misses`, `FileCount`, and the traits that support or
refute it. FR-009 requires every candidate be reported, including the losers.

### `Pm4GeometryStreamReport`

`(string InputDirectory, string CorpusSignature, Pm4MspiInterpretationSummary LegacyCounters,
IReadOnlyList<Pm4WindowSizeHistogramBucket> SizeHistogramByFamily,
IReadOnlyList<Pm4WindowInterpretationResult> Interpretations, IReadOnlyList<string> Notes)`.

`LegacyCounters` reuses the **existing** `Pm4MspiInterpretationSummary` record unchanged, so the
published baseline keeps appearing verbatim next to the finding that explains its limits.

---

## 5. Reconstruction *(Phase 7 — FR-010, SC-006)*

### `Pm4NegativeVolume`

`(Pm4ObjectId ObjectId, bool IncludesSecondStream, int VertexCount, int TriangleCount, Pm4Bounds3
Bounds, double EnclosedVolume, int BoundaryEdgeCount, double SealednessRatio)`.

`SealednessRatio` = fraction of edges shared by exactly two triangles. 1.0 is a closed manifold. This
is the number SC-006 asks to be reported "as a measurement" rather than asserted. `Pm4Bounds3`
already exists and is reused.

### `Pm4ReconstructionComparison`

| field | type | notes |
|---|---|---|
| `ObjectId` | `Pm4ObjectId` | |
| `WithoutSecondStream` / `WithSecondStream` | `Pm4NegativeVolume` | the with/without pair FR-010 requires |
| `AssetPath` | `string` | the real WMO/M2 compared against |
| `AssetProvenance` | `string` | configured client root + build identity (Constitution III/VI) |
| `BoundsAgreement` | `double` | |
| `VolumeRatio` | `double` | reconstructed vs real |
| `Verdict` | `string` | stated with its confidence, never as bare fact |

---

## 6. Relationships

```text
Pm4DecodeEvidenceRegister
  └── Pm4DecodeFinding (Key unique)
        ├── Pm4EvidenceItem      (corpus-wide, FileCount > 1 under FR-002)
        └── Pm4Elimination       (accumulates, never deleted)

Pm4GroupingComparisonReport
  └── Pm4RuleEvaluationReport (one per Pm4GroupingRule, one corpus read)
        └── Pm4RuleFileResult  (per file — FR-002)

Pm4ObjectIdentityReport            ← produced by the winning rule
  ├── Pm4ObjectRecord   ──┐
  └── Pm4SurfaceAssignment┘  every MSUR surface appears exactly once
        └── Pm4ObjectId (null iff not Assigned)

Pm4GeometryStreamReport
  ├── Pm4WindowSizeHistogramBucket (per TypeFlags family)
  └── Pm4WindowInterpretationResult
        └── Pm4WindowGeometryTraits ← Pm4GeometryWindow

Pm4ReconstructionComparison → Pm4NegativeVolume ×2 → Pm4ObjectRecord

Consumers:
  Spec 129 (zarr dataset)  reads Pm4ObjectIdentityReport  (object-primary rows)
  Spec 128 (matching)      reads Pm4ObjectSegmentBuilder, re-keyed onto Pm4ObjectId in Phase 4
  Viewer   (Phase 5)       reads Pm4ObjectIdentityReport  (pick → object)
```

## 7. What is deliberately not modelled

- **No new chunk record.** Every raw field stays on the existing `Pm4*Entry` records.
- **No coordinate type.** `Pm4PlacementSolution`, `Pm4AxisConvention`, `Pm4CoordinateMode` and
  `Pm4PlanarTransform` already exist and are reused as-is (research.md R6).
- **No persistent store.** JSON on disk. Spec 129 owns durable storage; duplicating it here would
  create a second source of truth for object identity, which is the exact problem Phase 4 exists to
  end.
