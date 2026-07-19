# Feature Specification: Direct Minimap-to-Terrain Reconstruction

**Feature Branch**: `114-direct-terrain-reconstruction`

**Created**: 2026-07-19

**Status**: Draft

**Input**: User description: "Pair corrected synthetic minimaps with real authored minimaps and
train direct image-to-terrain models without requiring a WDL prior. Use object masks to clean real
minimaps, treat high-resolution synthetic renders as visual-detail truth, and reconstruct geometry,
land-feature classes, texture families, and alpha blending through separate small models."

## Governing Principle

The deployment input is an authored minimap image. Every additional inference signal must be
predicted from that image by an independently trained model or derived deterministically from a
predicted signal. Ground-truth height, masks, texture IDs, alpha layers, or synthetic renders are
training/evaluation evidence only and never inference inputs.

The direct geometry lane has no mandatory WDL prior. It predicts one relative-height signal from
the image stack in one forward pass. Object cleanup, geometry, land-feature classification,
texture-family selection, alpha reconstruction, and detail enhancement remain separate replaceable
models with separate checkpoints and promotion gates.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reconstruct Relative Terrain Directly (Priority: P1)

A model operator trains one direct image-to-relative-height model from the corrected dual-view
corpus: clean synthetic minimaps and authored minimaps for the same terrain rows, paired with exact
numeric height truth. The model does not consume or predict a WDL lattice.

**Why this priority**: Geometry is the core reconstruction product, and the corrected corpus now
contains stronger direct supervision than the old coarse-prior route.

**Independent Test**: On a frozen held-out map split, image-only inference beats both a flat/tile-
mean baseline and the strongest recorded Spec 112 direct-height baseline while producing seam-safe
relative height at tile borders.

**Acceptance Scenarios**:

1. **Given** a clean synthetic minimap with exact numeric terrain truth, **When** direct geometry
   inference runs, **Then** it emits one relative-height field without reading WDL, ground-truth
   normals, or any other ground-truth signal.
2. **Given** authored and synthetic views of one source tile, **When** partitions are assigned,
   **Then** both views stay in one source group and cannot leak across train/validation.
3. **Given** an authored minimap with object contamination, **When** geometry inference runs with
   the generated cleanup signal, **Then** the output remains a continuous terrain surface and the
   provenance names the generated mask/cleaning checkpoint.

---

### User Story 2 - Detect and Remove Authored Objects (Priority: P2)

A model operator trains a dedicated object-mask model from authored minimaps and trusted rendered
object-visibility labels. The mask becomes an explicit generated signal that can guide cleaning or
geometry inference without pretending object-free synthetic imagery is pixel-identical to authored
imagery.

**Why this priority**: Objects are the largest known deployment-domain contaminant, but their
handling must not be entangled with geometry weights or derived from unreliable RGB differencing.

**Independent Test**: On object-bearing held-out tiles, the predicted mask beats empty/all-object
baselines and materially reduces object-shaped terrain artifacts relative to the same geometry
model run without the generated cleanup signal.

**Acceptance Scenarios**:

1. **Given** an authored minimap and a trusted object-visibility label, **When** object-mask
   training/evaluation runs, **Then** coverage and boundary metrics are reported separately from
   terrain reconstruction metrics.
2. **Given** authored and terrain-only synthetic images with different lighting/material treatment,
   **When** mask labels are prepared, **Then** raw image difference is not accepted as object truth.
3. **Given** no trusted object label for a row, **When** the corpus is built, **Then** the row is
   excluded from mask supervision rather than assigned an empty mask.

---

### User Story 3 - Classify Reusable Land Features (Priority: P3)

A dataset/model operator builds a stable terrain-feature library from numeric geometry, liquid,
material, and full-map pattern evidence, then trains one image-to-feature-class model. The labels
describe reusable terrain semantics rather than memorized map or texture filenames.

**Why this priority**: A feature library gives downstream texture and alpha models a compact,
interpretable conditioning signal and makes failures inspectable.

**Independent Test**: A frozen family-safe split reports per-class coverage and macro metrics, and
no canonical feature family appears in both training and validation.

**Acceptance Scenarios**:

1. **Given** height, slope/curvature, liquid, alpha, and material evidence for one tile, **When**
   labels are derived, **Then** every pixel/region is traceable to the exact source signals and
   library revision.
2. **Given** rare or unknown terrain patterns, **When** classification runs, **Then** the system
   emits an explicit unknown/low-confidence state rather than forcing a known family.

---

### User Story 4 - Reconstruct Texture Families and Alpha Layers (Priority: P4)

A model operator reconstructs texturing in two independent steps: first select ordered canonical
texture families, then predict their alpha/blend field. The result is evaluated by recompositing a
minimap through the existing renderer as well as by comparing numeric layer evidence.

**Why this priority**: Layer identity and blend shape are different problems; separating them keeps
the models small and makes texture-library or alpha improvements independently replaceable.

**Independent Test**: On a frozen family-safe split, ordered texture-family selection and alpha
reconstruction each beat their trivial baselines, and the recomposited terrain image improves over
base-only/uniform-blend composition without changing the geometry checkpoint.

**Acceptance Scenarios**:

1. **Given** a generated land-feature map and image-derived inputs, **When** texture-family
   selection runs, **Then** it emits ordered library identities with confidence and no ground-truth
   texture ID at inference.
2. **Given** selected texture families, **When** alpha reconstruction runs, **Then** it emits one
   bounded ordered blend stack that respects layer presence and sums/composes according to the
   existing renderer contract.
3. **Given** a missing or unknown texture family, **When** composition is requested, **Then** the
   result uses an explicit fallback state and is not labeled exact reconstruction.

### Edge Cases

- Authored minimaps can contain objects, icons, baked effects, different water, and different color
  treatment from terrain-only synthetic images; same-tile lineage does not imply pixel equality.
- Some rows have authored RGB but no usable synthetic RGB, or vice versa; every model corpus uses
  honest per-signal coverage and never zero-fills a missing source.
- Absolute world altitude is not identifiable from one minimap alone. The geometry target is
  relative height; a future independent offset model may be added only if a deployment-available
  cue and separate proof exist.
- Texture paths vary by build and map. Model labels use versioned canonical families with explicit
  unknowns, not raw path IDs as universal semantics.
- Tiles crossing map borders, oceans, holes, missing MCAL, or incomplete normal coverage require
  explicit exclusion/fallback states and may not silently enter a clean target set.
- Predicted upstream signals differ from ground truth. Every downstream model is trained/evaluated
  on generated upstream outputs or a documented scheduled mixture, never only teacher-forced truth.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The geometry model MUST consume only an authored/synthetic minimap image plus
  generated image-derived signals and MUST emit exactly one relative-height field.
- **FR-002**: The direct geometry path MUST NOT require, derive, or teacher-force a WDL prior.
- **FR-003**: Authored and synthetic views of one tile MUST share a source-group identity and split.
- **FR-004**: Synthetic 256/1024 inputs MUST carry the corrected fixed-noon-white provenance before
  entering any corpus governed by this spec.
- **FR-005**: Object cleanup MUST be a separate model with trusted object-visibility supervision;
  authored-minus-synthetic RGB difference MUST NOT be accepted as the mask label.
- **FR-006**: Any downstream consumer of an object mask or land-feature map MUST be trained and
  evaluated with the upstream model's generated output represented in the input distribution.
- **FR-007**: Land-feature labels MUST be deterministic, versioned, family-safe across partitions,
  and traceable to numeric geometry/liquid/material/alpha evidence.
- **FR-008**: Texture-family selection and alpha reconstruction MUST be separate models and
  separately promotable checkpoints.
- **FR-009**: The alpha model MUST emit one bounded ordered blend-stack signal compatible with the
  existing compositor contract; it MUST NOT rewrite MCAL decoding or `AlphaWdtWriter`.
- **FR-010**: Every training/evaluation corpus MUST record real per-signal coverage, exclusions,
  source store identities, row lineages, split groups, and model-input origin.
- **FR-011**: Every model MUST have its own architecture/config, checkpoint, run summary, baseline,
  and visual/numeric promotion gate; no shared weights or multi-task head is allowed.
- **FR-012**: High-resolution synthetic RGB MAY supervise visual detail/SR, but numeric height,
  normals, alpha, and texture-family signals remain their own ground truth and MUST NOT be replaced
  by RGB similarity.
- **FR-013**: Pretrained or Hub-sourced weights MUST be optional, license-recorded, hash-pinned, and
  compared against a small from-scratch baseline on the same split.
- **FR-014**: Training, heavy rendering, and data rebuilds MUST remain user-executed from exact
  documented commands with time/VRAM estimates.
- **FR-015**: Deployment inference MUST be auditable from authored minimap through every generated
  signal and checkpoint to the final reconstruction.

### Key Entities

- **Paired Terrain Row**: Same-tile authored RGB, corrected synthetic RGB/detail, exact numeric
  terrain signals, per-signal availability, and one leak-safe source-group identity.
- **Object Visibility Label**: Trusted binary/instance surface coverage rendered from verified
  placement geometry, with renderer/build/hash evidence.
- **Relative Height Target**: Altitude-offset-invariant terrain field and decode metadata.
- **Terrain Feature Library**: Versioned reusable landform/material-context families with unknowns,
  label rules, provenance, and group-safe identities.
- **Texture Family Selection**: Ordered canonical family identities and confidence for one region.
- **Alpha Blend Stack**: Ordered layer-presence and bounded blend weights compatible with the
  production compositor.
- **Model Stage Record**: One input/output contract, checkpoint identity, upstream dependencies,
  metrics, baselines, and promotion verdict.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Direct image-to-relative-height inference beats the flat/tile-mean baseline and the
  strongest recorded Spec 112 result on the same frozen split by at least 5% relative validation
  MAE, with best epoch after epoch 1.
- **SC-002**: Held-out adjacent-tile border error is no worse than the interior error distribution's
  95th percentile, with no visible seam promoted by the user review.
- **SC-003**: The object-mask model beats empty/all-object baselines and reduces geometry error
  inside trusted object regions by at least 10% relative without degrading clean-region MAE by more
  than 2%.
- **SC-004**: The land-feature classifier beats the majority-family baseline by at least 10 macro-
  F1 points and reports every unknown/unsupported family separately.
- **SC-005**: Ordered texture-family selection beats the per-map majority baseline, and no canonical
  family leaks across train/validation.
- **SC-006**: Alpha reconstruction improves both blend-field error and recomposited-image error by
  at least 10% relative to base-only/uniform-blend baselines on the same held-out rows.
- **SC-007**: An audit proves that every inference input is either the authored minimap or a generated
  upstream signal; zero ground-truth training signals enter deployment inference.
- **SC-008**: Each stage can be replaced and reevaluated without retraining unrelated stages, and
  each promoted checkpoint has a complete identity/run summary.

## Assumptions

- The corrected fixed-noon synthetic minimap rerender and visual validation from Spec 113 complete
  before any real corpus in this spec is promoted.
- Current v50.1 stores provide height, normals, alpha, liquid, material IDs, authored minimaps, and
  partial synthetic minimaps. Trusted object-visibility masks are not yet a frozen v50.1 signal and
  require a separate foundational proof.
- The first geometry target remains relative height because a single top-down image does not encode
  absolute world altitude reliably.
- The existing Spec 112 direct relative-height model is the mandatory small baseline, not discarded
  history. This spec determines whether a different dense encoder improves it.
- Spec 113 remains the owner of image super-resolution. This spec may consume its validated output
  or feature evidence but does not create a second SR trainer.
- All game-client-derived corpora and outputs remain private BYOD artifacts.
