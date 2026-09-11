# Phase 1 Data Model: Legacy MDX/M2 Rendering & Fuckported-Asset Compatibility

Entities this feature introduces, reuses, or extends. Reused entities are referenced, not
re-specified in full — see the linked source for the authoritative field-level definition.

## Reused verbatim from Spec 104 (`specs/104-legacy-m2-rendering/data-model.md`)

- **M2 Header** (relevant fields: version discriminator at `0x04`, view/skin-profile count+offset).
- **Embedded Skin Profile (View)** — index list, triangle list, vertex-property list, submeshes,
  texture units. The data whose absence produces the "empty box" symptom.
- **Submesh / Geoset** — vertexStart/Count, triangleStart/Count, the unit of rendering and material
  binding.
- **Texture Unit / Batch** — submesh-to-texture/material binding, render flags/blend.

Validation rules carried forward unchanged: view count/offset must be within file bounds (zero is
only valid for ≥264/external-skin versions); every submesh range must lie within its arrays or be
rejected individually (never crash the whole model); triangle count must be a multiple of 3; a
skin profile covering zero triangles falls back to bounding box, not a false "success."

## Reused verbatim from Spec 154 (`specs/154-m2-era-reader-parity/data-model.md`)

- **BuildIdentity** — `Version`, `BuildNumber`, `RootLabel`. Two records with equal `Version` but
  different `BuildNumber` are different builds and are never merged or used to justify one another.
- **LayoutSelection** — `DeclaredMagic`, `DeclaredVersion`, `SelectedLayout`, `SelectionEvidence`
  (mandatory — "because the version word said so" is insufficient given the measured `0x100`
  ambiguity).
- **SectionOutcome** — `Section`, `State` (`NotPresent`/`Succeeded`/`Failed`), `ElementIndex`,
  `Detail`. `NotPresent` and `Failed` must never be collapsed (this is the fix for the current
  `bones=0` defect, which conflates "no bones" with "bones not read").
- **SurveyRecord** — one build/model row: `Build`, `ModelPath`, `Layout`, `Sections`, `ReadAt`.
- **Skeleton** — ordered bones with identity/parent/pivot, plus its validation rules (finite pivots,
  in-range-or-none parents, acyclic parent walk, reject-not-partial on any rule failure).

**Not reused**: Spec 154's `SequenceTable` and `RigProjection` entities exist to serve its US4
(cross-era rig comparison), which spec 235 explicitly did not carry forward (see spec.md Out of
Scope). If a future spec revives that use case, these remain valid and available in Spec 154's own
data-model.md.

## Extended for this feature

### SectionOutcome.Section — new section: `lightEffect`

Spec 154's `Section` enumeration (identity, skeleton, sequences, geometry, cameras, …) gains
`lightEffect` for MDX assets, reported the same way as every other section — `NotPresent` when the
model defines no light node, `Succeeded` when a defined light node's visual effect renders,
`Failed` with `Detail` naming the specific unmodeled light type when it does not (spec.md FR-010
acceptance scenario 2).

### FuckportedAssetCheck (NEW — US4)

The record of one non-standard-rewritten asset's compatibility check.

| Field | Meaning |
|---|---|
| `AssetPath` | The file checked. |
| `DeclaredFormat` | What the file claims to be (magic + version). |
| `ThisReaderResult` | `Succeeded` / `Failed` (with detail) using this project's own reader. |
| `ExternalReferenceResult` | `Succeeded` / `Failed`, and which external reference was used
  (Warcraft.NET or Benilla). |
| `DivergenceDetail` | What specifically differs from a standard file of the declared
  format/version — populated only when `ThisReaderResult` failed but `ExternalReferenceResult`
  succeeded. |

**Rules**:

- A `FuckportedAssetCheck` where both readers fail is reported as a specific per-asset failure
  (FR-009) — never silently dropped.
- A check where this reader fails but the external reference succeeds is the case FR-008 requires
  fixing; `DivergenceDetail` is what Phase 4 (plan.md) diagnoses and closes.
- This is a diagnostic/validation record, not a new file format or converter — it does not itself
  repair anything.

### LightEmitterEffect (NEW — US5)

The runtime-visible outcome for one MDX light node, distinct from the parse-side
`MdxLightSummary`/`MdxLightType` it's built from.

| Field | Meaning |
|---|---|
| `NodeType` | The MDX light type (from `MdxLightType`). |
| `Scope` | `ModelLocal` only for this feature — global cross-object light transport is explicitly
  out of scope (spec.md, inherited from Spec 104 Phase 4's own boundary). |
| `EffectState` | `Rendered` / `NotYetSupported` (named type, logged explicitly — never silently
  skipped). |

**Rules**:

- `Scope` is never anything but `ModelLocal` in this feature. Promoting a light to affect
  surrounding geometry/other objects is a separate, not-yet-approved ownership slice.
- An `EffectState` of `NotYetSupported` must still be logged/reported (FR-010 acceptance scenario
  2) — it is a named gap, not a silent omission.

## Relationships

```text
M2/MDX file
├── Header ─────────────────────────▶ LayoutSelection (evidence-based, never inferred from
│                                      another build's version word)
├── view offset/count ──────────────▶ Embedded Skin Profile[LOD 0] (Spec 104 shape)
│                                      └─▶ Submesh/TextureUnit ─▶ existing render path
├── bone table ──────────────────────▶ Skeleton (Spec 154 shape, validated, reject-not-partial)
├── (MDX) light node ────────────────▶ LightEmitterEffect (NEW, model-local only)
└── (any) chunk content ─────────────▶ FuckportedAssetCheck (NEW, only when non-standard)

SurveyRecord ties one BuildIdentity + ModelPath to all of the above sections' outcomes, including
the new `lightEffect` section — this is the single evidence artifact every acceptance scenario in
spec.md traces back to.
```
