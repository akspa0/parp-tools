# Feature Specification: Universal Image-to-Terrain Reconstruction

**Feature Branch**: `114-direct-terrain-reconstruction`

**Created**: 2026-07-19

**Status**: Draft

**Input**: User description: "Take any raster image as input and create proper terrain from it.
WoW authored and synthetic minimaps are high-quality paired supervision, not the deployment-domain
boundary. Reconstruct geometry first, then handle editable texture/material data through separate
small models. Do not require a WDL prior."

## Governing Principle

The deployment input is **any decodable raster image**, including but not limited to authored game
minimaps, synthetic maps, satellite/aerial imagery, drawings, paintings, grayscale images, and
ordinary photographs. WoW minimaps are one supervised training family; they do not define or limit
the accepted input domain. Image dimensions, aspect ratio, color mode, or visual source may change
preprocessing, but must not cause semantic-domain refusal.

The geometry product is a plausible, continuous image-conditioned relief field plus a deterministic
terrain mesh. For a top-down image, relief corresponds directly to terrain height. For a perspective
or non-geographic image, the result is an image-conditioned terrain relief rather than a claim that
the unknowable original three-dimensional scene was recovered exactly. The source image can be
projected onto that mesh immediately; editable material families and blend fields remain separate
replaceable model stages.

Every additional inference signal must be predicted from the input image by an independently trained
model or derived deterministically from a predicted signal. Ground-truth height, masks, texture IDs,
alpha layers, synthetic renders, map identity, and client data are training/evaluation evidence only
and never required inference inputs.

The universal geometry lane has no mandatory WDL prior. It predicts one normalized-relief signal
from the source raster in one forward pass. Object cleanup, geometry, land-feature classification,
texture-family selection, alpha reconstruction, and detail enhancement remain separate replaceable
models with separate checkpoints and promotion gates.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Turn Any Raster Image into Terrain (Priority: P1)

A user supplies any raster image and receives a normalized relief field and usable terrain mesh whose
major spatial structures follow the image. The first trainable checkpoint uses only project-owned
v50 authored minimaps and exact numeric terrain, with one entire map held out. Broader project-owned
image families may be added later without changing the raster-only deployment contract. The model
does not consume or predict a WDL lattice.

**Why this priority**: Universal image-to-terrain conversion is the product. An excellent
WoW-minimap-only estimator still fails the product contract if it cannot terrainify an unfamiliar
image.

**Independent Test**: Hold out an entire v50 map from training. Every valid raster still produces a
finite continuous mesh; held-out exact pairs beat constant and direct-luminance terrainification
baselines; and arbitrary-image sheets remain a separate user-reviewed compatibility surface rather
than mislabeled numeric truth.

**Acceptance Scenarios**:

1. **Given** any decodable RGB, RGBA, or grayscale raster at any practical dimensions/aspect ratio,
   **When** geometry inference runs, **Then** it emits one finite normalized relief field and one
   continuous terrain mesh without client files, WDL, ground-truth normals, or map-specific inputs.
2. **Given** an image from a visual/source family absent from training, **When** inference runs,
   **Then** it is processed under the same contract and is not refused or silently labeled
   in-domain.
3. **Given** multiple rendered or styled views derived from one underlying terrain, **When**
   partitions are assigned, **Then** every view stays in one source group and cannot leak across
   train/validation.
4. **Given** an image that cannot uniquely determine physical height, **When** terrain is generated,
   **Then** the result is labeled plausible image-conditioned relief and exposes user-controlled
   horizontal extent and vertical scale rather than claiming exact world elevation.

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

- Very wide, tall, tiny, large, grayscale, transparent, low-contrast, or high-dynamic-range inputs
  must receive documented normalization/padding/tiling behavior without changing the universal
  semantic input contract.
- A blank or constant image must produce a stable finite low-relief mesh, not NaNs, spikes, or an
  invented exactness claim.
- Perspective photographs and non-geographic artwork have no unique terrain solution. They produce
  view-axis relief that preserves major image structure; they are not mislabeled as physical scene
  reconstruction.
- Authored minimaps can contain objects, icons, baked effects, different water, and different color
  treatment from terrain-only synthetic images; same-tile lineage does not imply pixel equality.
- Some rows have authored RGB but no usable synthetic RGB, or vice versa; every model corpus uses
  honest per-signal coverage and never zero-fills a missing source.
- Absolute world altitude and metric scale are not identifiable from one arbitrary image. Geometry
  remains normalized relief; placement extent, vertical scale, and offset are explicit output
  parameters or later independent predictions.
- Texture paths vary by build and map. Model labels use versioned canonical families with explicit
  unknowns, not raw path IDs as universal semantics.
- Tiles crossing map borders, oceans, holes, missing MCAL, or incomplete normal coverage require
  explicit exclusion/fallback states and may not silently enter a clean target set.
- Predicted upstream signals differ from ground truth. Every downstream model is trained/evaluated
  on generated upstream outputs or a documented scheduled mixture, never only teacher-forced truth.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The geometry path MUST accept any decodable RGB, RGBA, or grayscale raster regardless
  of dimensions, aspect ratio, source application, visual style, or whether it is a minimap, and
  MUST emit exactly one normalized relief field plus its deterministic terrain mesh representation.
- **FR-002**: The direct geometry path MUST NOT require, derive, or teacher-force a WDL prior.
- **FR-003**: All views, crops, renderings, and style variants derived from one underlying terrain or
  source image MUST share a source-group identity and split.
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
- **FR-013**: Any pretrained or Hub-sourced visual initialization MUST be license-recorded,
  hash-pinned, and compared against the frozen small from-scratch baseline on the same split. A
  general visual initialization MAY be required for promotion when the from-scratch model fails the
  held-out-domain gate.
- **FR-014**: Training, heavy rendering, and data rebuilds MUST remain user-executed from exact
  documented commands with time/VRAM estimates.
- **FR-015**: Deployment inference MUST be auditable from arbitrary source image through every
  generated signal and checkpoint to the final reconstruction.
- **FR-016**: Geometry training MUST include more than the WoW minimap family and MUST hold out whole
  visual/source families for evaluation; a random within-map split alone cannot promote a universal
  image-to-terrain model.
- **FR-017**: The geometry model MUST be evaluated against both a constant-relief baseline and a
  deterministic image-luminance-to-relief baseline so that merely embossing brightness cannot be
  mistaken for learned terrain interpretation.
- **FR-018**: A mesh export MUST preserve the complete source-image coverage through documented
  padding/cropping/tiling, contain only finite vertices, and expose deterministic UV coordinates for
  immediate source-image projection.

### Key Entities

- **Universal Image Sample**: A raster, its source/visual-family identity, transform lineage, and—if
  available—paired relief/depth truth. Unpaired samples may be used only for qualitative review or
  explicitly documented self-supervision.
- **Paired Terrain Row**: Same-terrain authored RGB, corrected synthetic RGB/detail, exact numeric
  terrain signals, per-signal availability, and one leak-safe source-group identity. It is one
  universal-curriculum family, not the deployment boundary.
- **Object Visibility Label**: Trusted binary/instance surface coverage rendered from verified
  placement geometry, with renderer/build/hash evidence.
- **Relative Relief Target**: Offset-invariant view-axis displacement field plus output extent,
  scale, and mesh-construction metadata. Top-down terrain height is a strict subset of this target.
- **Terrain Feature Library**: Versioned reusable landform/material-context families with unknowns,
  label rules, provenance, and group-safe identities.
- **Texture Family Selection**: Ordered canonical family identities and confidence for one region.
- **Alpha Blend Stack**: Ordered layer-presence and bounded blend weights compatible with the
  production compositor.
- **Model Stage Record**: One input/output contract, checkpoint identity, upstream dependencies,
  metrics, baselines, and promotion verdict.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every image in a frozen compatibility suite of at least 100 valid rasters spanning at
  least five visual/source families, RGB/RGBA/grayscale modes, and five aspect-ratio/resolution
  classes produces a finite relief field and valid continuous mesh without semantic-domain refusal.
- **SC-002**: On paired truth from whole maps wholly absent from training, direct
  image-to-relief inference beats both constant-relief and direct-luminance baselines by at least 5%
  relative validation MAE and gradient MAE.
- **SC-003**: On a frozen review sheet of at least 30 unpaired arbitrary images spanning the same
  visual breadth, the user accepts at least 80% as useful terrain interpretations with major spatial
  structures visibly preserved.
- **SC-004**: Held-out adjacent-tile border error is no worse than the interior error distribution's
  95th percentile, with no visible seam promoted by the user review.
- **SC-005**: The object-mask model beats empty/all-object baselines and reduces geometry error
  inside trusted object regions by at least 10% relative without degrading clean-region MAE by more
  than 2%.
- **SC-006**: The land-feature classifier beats the majority-family baseline by at least 10 macro-
  F1 points and reports every unknown/unsupported family separately.
- **SC-007**: Ordered texture-family selection beats the per-map majority baseline, and no canonical
  family leaks across train/validation.
- **SC-008**: Alpha reconstruction improves both blend-field error and recomposited-image error by
  at least 10% relative to base-only/uniform-blend baselines on the same held-out rows.
- **SC-009**: An audit proves that every inference input is either the source raster or a generated
  upstream signal; zero ground-truth training signals enter deployment inference.
- **SC-010**: Each stage can be replaced and reevaluated without retraining unrelated stages, and
  each promoted checkpoint has a complete identity/run summary.

## Assumptions

- The corrected fixed-noon synthetic minimap rerender and visual validation from Spec 113 complete
  before any real corpus in this spec is promoted.
- The completed authored-only `direct_cnn_v112` run is retained as a narrow historical baseline. It
  failed even its tile-mean gate and is not a universal-model candidate or deployment-domain proof.
- Current v50.1 stores provide height, normals, alpha, liquid, material IDs, authored minimaps, and
  partial synthetic minimaps. Trusted object-visibility masks are not yet a frozen v50.1 signal and
  require a separate foundational proof.
- The first geometry target is normalized view-axis relief because an arbitrary image does not encode
  absolute world altitude or scale reliably. A top-down terrain image maps this relief directly to
  relative terrain height.
- The existing Spec 112 direct relative-height model remains reproducible negative evidence. It is
  not sufficient to define the universal training corpus, model initialization, or promotion gate.
- Spec 113 remains the owner of image super-resolution. This spec may consume its validated output
  or feature evidence but does not create a second SR trainer.
- All game-client-derived corpora and outputs remain private BYOD artifacts.
