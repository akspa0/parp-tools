# Contract: Grouping Rule and Evaluation Report

**Phase**: 2–3 | **Satisfies**: US2, FR-002, FR-003, FR-004, SC-001, SC-003

**Path**: `output/pm4-decode/grouping-comparison.json`

## The rule interface

A rule maps a corpus to a surface→object assignment. It is a delegate plus a descriptor so the
harness can evaluate every rule in **one corpus read** rather than one read per rule.

```csharp
public sealed record Pm4GroupingRule(
    string Id,
    string Name,
    string Description,
    bool IsBaseline,
    Func<Pm4GroupingContext, IReadOnlyDictionary<Pm4SurfaceKey, string?>> Assign);
```

`Assign` returns a canonical group key per surface, or `null` for "this rule cannot determine
membership". **`null` is a first-class answer** — a rule that guesses instead of returning `null`
violates FR-003 and will look better than it is.

`Pm4GroupingContext` carries the already-read corpus (`IReadOnlyList<Pm4ResearchDocument>` plus
tile coordinates) so no rule re-reads a file. Rules do not perform I/O.

## The candidate set

Per [research.md](../research.md) R2:

| id | rule | note |
|---|---|---|
| `G0` | `MSLK.GroupObjectId → MPRL.Unk04` | baseline; `IsBaseline = true` |
| `G1` | `MSUR.Ck24` | current de-facto key |
| `G2` | `MSHD.Field04` × `MSUR.Ck24` | region-scoped |
| `G3` | surface → MSLK → `GroupObjectId` → siblings | the lead — needs no MPRL |
| `G4` | transitive closure of G1 ∪ G3 | union-find over both |
| `G5` | G3 partitioned by `MSLK.TypeFlags` family | tests whether TypeFlags splits objects |
| `G6` | MSPV/MSPI shared-vertex connectivity | depends on Phase 6; may be dropped if the stream is eliminated |

## Evaluation report

```jsonc
{
  "schemaVersion": 1,
  "inputDirectory": "test_data/development/World/Maps/development",
  "corpusSignature": "test_data/development/World/Maps/development@616",
  "fileCount": 616,
  "rules": [
    {
      "ruleId": "G3",
      "surfacesTotal": 518092,
      "surfacesGrouped": 0,
      "surfacesSingleton": 0,
      "surfacesUngrouped": 0,
      "objectCount": 0,
      "crossTileObjectCount": 0,
      "largestObjectSurfaceCount": 0,
      "objectSizeHistogram": [ { "value": "1", "count": 0 } ],
      "baselineEdgeFits": 65819,
      "baselineEdgeMisses": 1206977,
      "perFile": [
        {
          "sourcePath": "development_00_00.pm4",
          "tileX": 0, "tileY": 0,
          "surfacesTotal": 4110,
          "surfacesGrouped": 0,
          "surfacesUngrouped": 0,
          "objectCount": 0
        }
      ]
    }
  ],
  "notes": []
}
```

Counts are shown as `0` above because they are the **output** of Phase 3, not an expectation. The
only pre-filled numbers are `baselineEdgeFits` / `baselineEdgeMisses`, which are the recorded
baseline and must come back unchanged.

## Two metrics, always both

Established in [research.md](../research.md) R1: the published 65,819 / 1,206,977 figure measures
whether a link's group id resolves as an `MPRL.Unk04` value. It is **reference resolution, not
partitioning**. A rule could group the corpus perfectly and leave that number untouched.

So every report carries both:

- `baselineEdgeFits` / `baselineEdgeMisses` — continuity with the published number, unchanged.
- `surfacesGrouped` / `surfacesUngrouped` / `objectSizeHistogram` — the grouping metric SC-001
  actually asks about.

A phase report that improves one and quietly drops the other is not acceptable.

## Guard against a degenerate winner

`objectSizeHistogram` and `largestObjectSurfaceCount` exist because coverage alone is gameable: a
rule assigning every surface to one object scores 100% grouped and is worthless. The spec names this
as an edge case ("a grouping rule that fits the corpus statistically but produces objects that are
physically absurd"), and Phase 7 is the physical backstop. The histogram is the cheap statistical
one.

**Reporting requirement**: a rule whose `largestObjectSurfaceCount` exceeds 10% of
`surfacesTotal` is reported with that fact stated in `notes`, not left for the reader to notice.

## Per-file rows are mandatory

FR-002. `perFile` covers every file including empty ones (`surfacesTotal: 0`) so an absence is
distinguishable from an omission. Tile 0_0 holds 4,110 surfaces across 16 CK24 groups and is
explicitly unrepresentative; without per-file rows a 0_0-only effect is invisible inside a corpus
total.

## Ungrouped is counted, never absorbed

SC-003. `surfacesUngrouped` counts surfaces whose rule returned `null`. It is never folded into
singletons — a surface known to be alone and a surface whose membership is unknown are different
facts, and Phase 5 renders them differently.
