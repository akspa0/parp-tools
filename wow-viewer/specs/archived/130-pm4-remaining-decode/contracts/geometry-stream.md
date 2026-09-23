# Contract: Connective Geometry (MSPV/MSPI and MSCN)

**Phase**: 7 | **Satisfies**: US3, FR-009, SC-005

**Path**: `output/pm4-decode/geometry-stream.json`

## Two candidate sources, evaluated as co-equals

The spec names **MSPV/MSPI** as the lead. Research finding R10b promotes **MSCN** to a co-equal
candidate: prior art in `PM4Tool/docs/pm4/pm4-analysis-findings.md` states MSCN holds the per-object
*exterior boundary* vertices, which describes "what closes a surface set into a sealed negative
volume" more literally than a second index stream does.

Three facts already in the tree justify testing it seriously:

- `MSUR._0x18` **already indexes into MSCN** — a per-surface link to a boundary vertex.
- MSCN exceeds MSVT in count (9,990 vs 6,318 in tile 0_0), so it is not a subset of the mesh.
- MSCN shares MSVT's coordinate frame (R8), so the two are directly co-locatable.

Both sources are reported. Either may be eliminated. Neither is assumed.

**Frame handling is settled** (R8, measured): MSPV, MSVT and MSCN share one coordinate frame; MPRL is
the permuted chunk. No frame resolution is required here — the question is purely topological.

## What is being decided for MSPV/MSPI

`MSLK` records carry a window into `MSPI`, which indexes `MSPV`. The windows resolve perfectly —
`MSLK.Mspi window → MSPI` is 598,882 fits / 0 misses and `MSPI → MSPV` is 2,418,205 / 0. What they
*mean* does not. This stream is **larger than the decoded surface mesh** (2,418,205 index fits vs
MSVI→MSVT's 1,930,146) and attaches to the same link records — the right shape for the connective
geometry that would close a surface set into a sealed negative volume.

**It is a lead, not a premise.** Elimination is a valid, recordable outcome.

## Why a new discriminator is required

The existing counters cannot answer this. From `Pm4ResearchUnknownsAnalyzer.cs:201-217`:

```csharp
bool indicesMode   = first >= 0 && (first + count)         <= mspiCount;
bool trianglesMode = first >= 0 && (first*3 + count*3)     <= mspiCount;
```

For every `first ≥ 0`, `count ≥ 0`: `3·first + 3·count ≥ first + count`. So `trianglesMode` implies
`indicesMode`, and the `trianglesOnly` bucket **cannot be non-zero for any input**. The published
`trianglesOnly = 0` is a property of that inequality, not of the format. `both = 199,699` counts
windows small enough to survive the ×3 bound — a statement about size, not topology.

The published numbers are real analyzer output. The defect is that the test cannot distinguish the
things it is being asked to distinguish. This contract replaces it with tests that can.

## Detector-power gate

**Before any corpus claim**, the discriminator must be shown to separate the interpretations on
constructed cases: a known polyline, a known closed polygon, a known triangle list, a known fan.
A discriminator that returns the same verdict for all four is not evidence of anything.

This is a hard gate on Phase 6, not a suggestion.

## Schema

```jsonc
{
  "schemaVersion": 1,
  "inputDirectory": "test_data/development/World/Maps/development",
  "corpusSignature": "test_data/development/World/Maps/development@616",

  "legacyCounters": {
    "activeLinkCount": 598882,
    "indicesModeOnlyCount": 399183,
    "trianglesModeOnlyCount": 0,
    "bothModesCount": 199699,
    "neitherModeCount": 0
  },
  "legacyCountersNote": "Retained verbatim as the published baseline. trianglesModeOnlyCount is 0 by construction: trianglesMode implies indicesMode. It carries no information about topology.",

  "windowPopulation": {
    "activeWindows": 0,
    "negativeFirstIndexWindows": 0,
    "zeroCountWindows": 0,
    "meanIndicesPerWindow": 0.0
  },

  "sizeHistogramByFamily": [
    {
      "familyKey": "type=0x12 subtype=0 system=0x8000",
      "windowCount": 0,
      "buckets": [ { "value": "4", "count": 0 } ],
      "multipleOfThreeFraction": 0.0,
      "modalSize": 0
    }
  ],

  "interpretations": [
    {
      "name": "Polyline",
      "fits": 0,
      "misses": 0,
      "fileCount": 0,
      "confidence": "None",
      "supportingTraits": [],
      "refutingTraits": []
    }
  ],

  "detectorPower": {
    "constructedCasesPassed": 0,
    "constructedCasesTotal": 4,
    "note": "Polyline / ClosedPolygon / TriangleList / TriangleFan must be separable before any corpus claim."
  },

  "notes": []
}
```

Zeroes are outputs, not expectations. `legacyCounters` is pre-filled because it is the recorded
baseline and must come back unchanged.

## Candidate interpretations

Every one is reported with its own corpus-wide fits and misses, including the losers (FR-009).

| interpretation | signature |
|---|---|
| `Polyline` | open path; `n` indices → `n-1` segments; first ≠ last; sizes spread, no multiple-of-3 preference |
| `ClosedPolygon` | first == last, or an implied closing edge; sizes spread |
| `TriangleList` | sizes cluster on multiples of 3; few degenerate triples |
| `TriangleFan` | shared first index across triples |
| `TriangleStrip` | consecutive triples share two vertices |

## Discriminating traits

Per window, measured — not inferred from a summary statistic:

| trait | what it separates |
|---|---|
| window-size histogram **by TypeFlags family** | multiple-of-3 spike → triangles; spike at 2 → segments; flat → polyline |
| `isClosed` (first index == last index) | closed polygon vs open path |
| `degenerateTripleCount` (zero-area if read as triangles) | a polyline read as triangles produces these constantly; a real triangle list does not |
| `distinctVertexCount` vs `indexCount` | degenerate reuse |
| `collinearityScore` / `planarityScore` over MSPV points | path-like vs patch-like |
| `axisConvention` via `Pm4PlacementMath` | whether the stream lives in a different frame (research.md R6) |

## Constraints from the record layout

Read from `Pm4ResearchReader.ParseMslk` (stride 20), not assumed:

- `MspiIndexCount` is a **single byte** at offset 11 → a window holds at most **255** indices.
- `MspiFirstIndex` is a **signed int24** at offset 8, read by `ReadSignedInt24` → negative values are
  representable and `indicesMode` already excludes them. They are **counted as
  `negativeFirstIndexWindows`, never silently dropped**.

  **Prior art gives them a meaning** (research.md R10a): *"MSLK entries (with MspiFirstIndex == -1)
  represent doodad placements, with group/object IDs and anchor points."* If that holds, these are
  object-placement records — grouping signal for Phase 4 and input for Spec 128, not "no path"
  padding. Test it; do not assume it. Either way they are counted.

Corpus-wide, 2,418,205 MSPI entries over 598,882 active windows gives a mean of ~4.04 indices per
window. A mean near 4 is *suggestive* of quads or short polylines rather than a triangle list — but a
mean cannot distinguish a distribution. **The histogram is the deliverable; the mean is not evidence.**

## Frame handling — settled, not open

Measured on `development_00_00.pm4` through `pm4 export-json` (research.md R8):

| chunk | min | max |
|---|---|---|
| MSPV | (169.60, 31.84, 0.85) | (498.79, 363.85, 134.55) |
| MSVT | (168.11, 31.00, −12.08) | (501.55, 450.70, 133.74) |
| MSCN | (168.84, 31.42, −12.08) | (499.38, 450.40, 133.00) |
| MPRL | (31.00, 5.00, 168.18) | (364.86, 40.20, 499.77) |

MSPV, MSVT and MSCN share axis order and overlapping ranges. **MPRL is the permuted chunk** — its
third axis is MSVT's first. So the "different axis order between chunks of one file" hazard is an
MPRL property, not a general one, and it does not implicate either connective-geometry candidate.

The nesting hypothesis is therefore **eliminated for this contract** and recorded as such. Any
residual frame check runs through the existing `Pm4PlacementMath` axis-convention detection — never a
hand-derivation from bounds, which is the procedure that produced the earlier confidently wrong
conclusion about tiles stacked above and below the map (FR-001).

## MSCN-specific evaluation

| question | how it is measured |
|---|---|
| Is MSCN the per-object exterior boundary? | group MSCN points by the objects whose surfaces reference them via `MSUR._0x18`; test whether each group forms a closed hull around that object's surfaces |
| Does `MSUR._0x18 → MSCN` resolve corpus-wide? | bounds fit + miss counts, per file and total, like every other edge |
| Why does MSCN exceed MSVT? | count MSCN points not reachable from any `MSUR._0x18`, and characterise them |
| Is MSCN↔MSLK real? | prior art flags this as unresolved; test and record either way |

## Outcome recording

Whatever Phase 6 concludes lands in the evidence register under `MSLK.MspiIndexCount` and
`MSPV/MSPI semantics`, with confidence. If MSPV/MSPI is eliminated as the connective geometry, the
elimination is recorded with the evidence that killed it (FR-008) so the search is not repeated.
