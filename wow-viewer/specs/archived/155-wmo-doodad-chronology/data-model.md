# Phase 1 Data Model: Asset Reference Inventory

Entities the feature introduces. Field names are indicative; binding contracts are in `contracts/`.

## BuildIdentity

| Field | Meaning |
|---|---|
| `Version` | Release version as staged |
| `BuildNumber` | Build number |
| `RootLabel` | Configured root the observation used — never a source-baked path |

**Rules**

- Equal `Version` with different `BuildNumber` are different builds; findings are never merged across
  them, including patch-adjacent builds.
- Every record in this feature carries one.

## ReferencingAsset

One world object or model, in one build, that makes claims.

| Field | Meaning |
|---|---|
| `Path` | Logical asset path within the build |
| `Kind` | World object, or model |
| `ReadState` | Read, or unreadable |
| `RouteBlocked` | Whether this asset's format route is known not to read in this build |

**Rules**

- `RouteBlocked` is distinct from `unreadable`. The first means "this build's format route does not
  work yet"; the second means "this particular asset failed". Both are distinct from "read, found
  nothing".

## Reference

One claim that an asset exists.

| Field | Meaning |
|---|---|
| `Source` | The `ReferencingAsset` making the claim |
| `Kind` | Placed doodad, world-object texture, or model texture |
| `TargetPath` | The asset path claimed |
| `Resolution` | `Present`, `Absent`, or `Unreadable` |

**Rules**

- `Resolution` has three states and they never collapse to two. `Unreadable` becoming `Absent` would
  manufacture missing assets (FR-006).
- Resolution is determined against what the build contains, independently of any catalogue.

## CatalogueEntry

An asset named by a listfile. **Naming is not existence.**

| Field | Meaning |
|---|---|
| `Path` | The path the listfile names |
| `Source` | Which listfile named it |

## ReferenceClassification

The three-set comparison result for one referenced asset.

| Field | Meaning |
|---|---|
| `TargetPath` | The asset |
| `Catalogued` | Whether a listfile names it |
| `Present` | Whether it is readable from the build |
| `Category` | Derived: working, catalogue-claims-but-absent, catalogue-gap, or missing |

**Derivation**

| `Catalogued` | `Present` | `Category` |
|---|---|---|
| yes | yes | Working |
| yes | no | CatalogueClaimsButAbsent |
| no | yes | CatalogueGap |
| no | no | Missing |

**Rules**

- `Catalogued` and `Present` are stored, not just the derived category, so a disputed classification can
  be re-derived from its inputs.
- A referenced asset lands in exactly one category (SC-004).
- **Nothing is `Missing` on the strength of `Catalogued = no` alone** (SC-005). `Present` is an
  independent determination.

## Orphan

An asset present in a build and referenced by nothing swept.

| Field | Meaning |
|---|---|
| `Path` | The asset |
| `SweepCoverage` | Which reference sources were swept when this was concluded |

**Rules**

- `SweepCoverage` is mandatory. An orphan is only ever "unreferenced by what was examined", and the
  record must carry that limit rather than implying global disuse.

## CandidateMatch

A present asset proposed for a missing reference.

| Field | Meaning |
|---|---|
| `MissingPath` | The unresolved reference |
| `CandidatePath` | A present asset in the same build |
| `DifferenceKind` | Spelling, punctuation, casing, extension, path, or combination |

**Rules**

- `CandidatePath` MUST be verified present in the **same build** (FR-008). Cross-build and invented
  candidates are rejected.
- All candidates are listed; none is chosen (FR-008).

## SweepReport

The header for one build's sweep. Its job is to make incompleteness visible.

| Field | Meaning |
|---|---|
| `Build` | `BuildIdentity` |
| `WorldObjectsExamined` | Count |
| `ModelsExamined` | Count |
| `AssetsUnreadable` | Count, with paths |
| `RoutesBlocked` | Which format routes were not swept, and why |
| `ReferenceSourcesSwept` | Which kinds of reference were collected |

**Rules**

- A report with `RoutesBlocked` non-empty MUST NOT be read as a complete picture, and consumers must
  surface that (FR-005).
- Examined counts are reported even when nothing was found, so an under-counted sweep is visible.

## IntroductionWindow

| Field | Meaning |
|---|---|
| `Path` | The asset |
| `AbsentIn` | The latest build known not to contain it |
| `PresentIn` | The earliest build known to contain it |
| `Granularity` | Always between-build |

**Rules**

- Bounded by two named builds; never a point estimate (FR-010).
- A rename appears as a disappearance plus an introduction; where the system cannot distinguish that
  from a genuine introduction, the record says so.

## RepairRecord

| Field | Meaning |
|---|---|
| `OriginalReference` | What it was |
| `Replacement` | What it became |
| `Evidence` | The candidate match that justified it |
| `Reversal` | What is needed to restore the prior state |

**Rules**

- Written only when repair was explicitly requested (FR-012).
- `Reversal` must restore the exact prior state (FR-013).
- Never written for an ambiguous or candidate-less reference (FR-014).

## ConversionCapabilityRecord

| Field | Meaning |
|---|---|
| `Operation` | Which conversion |
| `Build` | `BuildIdentity` it ran against |
| `Outcome` | Succeeded, or failed with what failed where |
