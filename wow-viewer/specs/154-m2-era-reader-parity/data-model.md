# Phase 1 Data Model: M2 Reader Era Parity

Entities the feature introduces or constrains. Field names are indicative; the binding contracts are
in `contracts/`.

## BuildIdentity

The build a file came from. Travels with every measurement and every support claim.

| Field | Meaning |
|---|---|
| `Version` | Release version as staged, e.g. `3.0.1` |
| `BuildNumber` | Build number, e.g. `8303` |
| `RootLabel` | Configured root the read used, recorded — never a source-baked path |

**Rules**

- Two records with equal `Version` but different `BuildNumber` are **different builds** and are never
  merged, summarised together, or used to justify one another (FR-011).
- No entity in this feature may carry a layout claim without a `BuildIdentity`.

## LayoutSelection

Why a reader treated a file the way it did.

| Field | Meaning |
|---|---|
| `DeclaredMagic` | `MDLX` or `MD20` |
| `DeclaredVersion` | Version word as read from the file |
| `SelectedLayout` | The layout actually applied |
| `SelectionEvidence` | What in *this file* selected it — the probe result, not the version alone |

**Rules**

- `SelectionEvidence` is mandatory. "Because the version word said so" is only valid where the version
  word is genuinely sufficient, and `0x100` already proves it is not universally (FR-002).
- `SelectedLayout` is never inferred from another build's selection.

## SectionOutcome

The result of attempting one section of a model.

| Field | Meaning |
|---|---|
| `Section` | Identity, skeleton, sequences, geometry, cameras, … |
| `State` | `NotPresent`, `Succeeded`, `Failed` |
| `ElementIndex` | Element position where a failure occurred, when applicable |
| `Detail` | What was not understood |

**Rules**

- `NotPresent` and `Failed` are distinct and must never be collapsed. "This model has no bones" and
  "this model's bones were not read" are different facts (FR-005) — conflating them is the current
  `bones=0` defect.
- A `Failed` state carries `ElementIndex` whenever the failure occurred inside an indexed array
  (FR-006). "Failed" without a position is an incomplete record.
- One section failing must not prevent other sections from being attempted and recorded.

## SurveyRecord

One build/model row. The unit of evidence for the whole feature.

| Field | Meaning |
|---|---|
| `Build` | `BuildIdentity` |
| `ModelPath` | Path within the archive |
| `Layout` | `LayoutSelection` |
| `Sections` | `SectionOutcome` per section attempted |
| `ReadAt` | When the read happened |

**Rules**

- A record with any section in an unknown state is incomplete and does not satisfy SC-001.
- Records are per build **and** per model. Neither dimension may be summarised away.

## Skeleton

A model's bone set, in one shape across all routes.

| Field | Meaning |
|---|---|
| `Bones` | Ordered bones |
| `Bone.Identity` | Stable identifier or key, where the format carries one |
| `Bone.Parent` | Parent bone, or "no parent" |
| `Bone.Pivot` | Pivot position |

**Validation rules** (FR-003, FR-004)

- Bone count matches what the file declares. A mismatch is a failed read, not a truncated skeleton.
- Every pivot component is finite.
- Every parent is either "no parent" or an in-range index.
- Walking parents from any bone reaches a root without revisiting a bone.
- A skeleton failing any rule is **rejected**, never returned partially populated. A partial skeleton
  that reads as valid is the defect this feature exists to remove.

## SequenceTable

| Field | Meaning |
|---|---|
| `Entries` | Ordered sequences |
| `Entry.Identity` | Sequence identifier |
| `Entry.Duration` | Duration |
| `Entry.IsAlias` | Whether it stands alone or refers elsewhere |

## RigProjection

A `Skeleton` plus a `SequenceTable` in the shared cross-route shape, with provenance.

| Field | Meaning |
|---|---|
| `Build` | `BuildIdentity` |
| `ModelPath` | Source model |
| `Route` | Which reading route produced it |
| `Skeleton` | As above |
| `Sequences` | As above |

**Rules**

- Field meanings are identical across routes, or the projection has failed its purpose (FR-010).
- Both sides of any comparison carry their `BuildIdentity` (SC-006).
- **Comparison is structural, never by count.** The working routes yield 54 and 151 bones for related
  rigs; count equality would report difference where correspondence exists. The question is whether the
  smaller bone set appears within the larger with corresponding parents and pivots.
