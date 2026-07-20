# Feature Specification: Terrain Feature Classification for Geometry Deconfounding

**Feature Branch**: `115-terrain-feature-classifier`

**Created**: 2026-07-20

**Status**: Draft

**Input**: User description: "The direct-geometry model (Spec 114) leans on RGB texture/color as a proxy
for depth. Real-world testing against an out-of-distribution image showed roads and paths decoded
as sloping terrain, when they are flat features painted a visually distinct color. Train a separate
image-to-feature-class model that learns to recognize road/path/water/building-adjacent regions from
RGB alone (supervised by real per-chunk texture-family ground truth, never consumed at inference),
then retrain the geometry model with this classifier's generated prediction as an added input
channel, so height prediction gets an explicit texture-vs-terrain signal on any image, including
images with no client-derived ground truth at all."

## Governing Principle

This feature extends Spec 114's direct-terrain-reconstruction lineage and inherits its governing
principle without exception: the deployment input is a minimap image, and every additional
inference signal must be predicted by an independently trained model or derived deterministically
from a predicted signal. Ground-truth texture-family/tileset identity is training and evaluation
evidence only and is never an inference input, on real client tiles or on arbitrary out-of-distribution
images.

The terrain-feature classifier and the retrained direct-geometry model are separate, independently
checkpointed, independently promotable models with no shared weights and no multi-task head (Spec
114 FR-011 / constitution IV carries forward unchanged). The classifier's only deployment product is
a generated feature-map; the geometry model's only new deployment dependency is that generated
feature-map, never the classifier's training labels.

**Relationship to Spec 114 User Story 3**: Spec 114 already specifies an unbuilt "Classify Reusable
Land Features" story (US3), scoped to feed texture-family selection and alpha reconstruction (Spec
114 US4). This feature shares the same underlying idea -- a versioned, image-derived terrain-feature
library -- but its specific and urgent consumer is the geometry model itself, motivated by a
concrete, observed failure (roads decoded as hills) rather than a downstream texturing need. It is
specified as an adjacent feature rather than an edit to Spec 114 so its own promotion gate can center
on the geometry deconfounding evidence, while keeping the feature-family taxonomy reusable by Spec
114 US3/US4 later rather than inventing a second, incompatible one.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Classify Terrain Features From an Image Alone (Priority: P1)

A model operator trains one image-to-feature-class model that labels each region of a minimap tile
as real terrain, road/path, water, building/object-adjacent, or unknown/low-confidence, using only
the RGB image as input. Training labels are derived automatically from real per-chunk texture-family
ground truth already present in the v50 store; no manual annotation is required.

**Why this priority**: Without a model that can recognize road-like regions from appearance alone,
there is nothing to feed the geometry model that would generalize to images with no client-derived
ground truth. This is the foundational, independently valuable capability everything else depends on.

**Independent Test**: On a frozen held-out real-tile split, the classifier's predicted feature map is
compared against the same split's ground-truth labels and beats a majority-class baseline. Separately,
run against the out-of-distribution image that originally exposed the roads-as-hills failure (no
ground truth available for this image) and visually confirm the model flags the visibly road-like
regions as non-terrain.

**Acceptance Scenarios**:

1. **Given** a real WoW client tile with populated per-chunk texture-family data, **When** training
   labels are derived, **Then** each chunk receives exactly one canonical feature-family label or an
   explicit unknown label, with no chunk silently defaulted or zero-filled.
2. **Given** a trained classifier and a held-out real tile, **When** inference runs, **Then** it
   emits one feature-class (with confidence) per prediction unit using only the tile's RGB image, with
   no ground-truth texture-family ID read at inference.
3. **Given** an arbitrary image with no client-derived ground truth (e.g. the motivating
   out-of-distribution failure case), **When** inference runs, **Then** the classifier still produces
   a best-effort prediction across the whole image rather than refusing, and the visibly road-like
   regions are not classified as real-terrain with high confidence.

---

### User Story 2 - Retrain Geometry With the Generated Feature Map (Priority: P2)

A model operator retrains the direct-geometry model with the terrain-feature classifier's generated
predicted feature-map concatenated as an additional input channel alongside the existing minimap RGB,
and proves the retrained model makes smaller height errors specifically in road/path regions than the
current RGB-only baseline, without materially degrading its performance elsewhere.

**Why this priority**: This is where the classifier's value is actually realized -- a classifier that
never influences geometry prediction fixes nothing. It depends on User Story 1 producing a promoted
checkpoint first.

**Independent Test**: On the same frozen held-out split used to promote the current RGB-only
direct-geometry checkpoint, compare the retrained (RGB + generated-feature-map) model's height error
inside classifier-flagged road/path regions against the frozen baseline's error in the same regions,
and confirm overall (non-road) error does not regress beyond a small tolerance.

**Acceptance Scenarios**:

1. **Given** a promoted terrain-feature classifier checkpoint, **When** the geometry model is
   retrained, **Then** its input contract is minimap RGB plus the classifier's generated feature-map
   only -- never ground-truth texture-family IDs, at training or at inference.
2. **Given** the retrained geometry checkpoint and the frozen RGB-only baseline, **When** both are
   evaluated on the same held-out split, **Then** the retrained model's road/path-region height error
   is measurably lower and its non-road-region error is not materially worse.
3. **Given** the out-of-distribution image that originally showed roads decoded as hills, **When** the
   retrained model runs end to end (classifier then geometry, both generated, no ground truth for this
   image at any step), **Then** a user's visual review no longer shows roads rendered as pronounced
   sloping ridges.

---

### User Story 3 - Carry the Fix Through the Residual Detailer (Priority: P3)

A model operator re-materializes the frozen coarse model's outputs against the retrained (Story 2)
checkpoint and retrains the residual detailer against it, so the full deployed coarse-plus-detailer
chain reflects the texture-deconfounded geometry rather than mixing an old coarse checkpoint with a
new one.

**Why this priority**: The currently promoted-pending checkpoint chain is coarse-plus-detailer, not
coarse alone; leaving the detailer trained against the old RGB-only coarse output would silently
reintroduce the confound (or produce an inconsistent, unaudited pairing) in the model actually used
for deployment inference.

**Independent Test**: The detailer's own promotion gate (≥5% relative improvement over its
coarse-only baseline, SC-002 border check) is re-run against the Story 2 coarse checkpoint, and the
resulting run record's `upstream_models` binds the new coarse checkpoint's hash, not the old one.

**Acceptance Scenarios**:

1. **Given** a promoted Story 2 geometry checkpoint, **When** its outputs are re-materialized for
   detailer training, **Then** the resulting coarse store is clearly distinguished from the prior
   RGB-only coarse store's materialization (distinct output path/identity).
2. **Given** the retrained detailer, **When** its run record is written, **Then** it names the exact
   Story 2 coarse checkpoint hash it was trained against, so the chain is independently auditable.

### Edge Cases

- Texture families that do not clearly match a canonical class (rare, historical, or ambiguous
  tilesets) resolve to the explicit unknown/low-confidence class rather than being forced into
  real-terrain, road/path, water, or building-adjacent.
- Chunks with multiple blended texture layers (e.g. a road blended into grass at its edge) require an
  explicit, documented policy (such as dominant-layer-by-alpha-weight) rather than an undocumented
  default; the policy must be stated in the label-derivation rules, not left implicit in code.
- Tiles or chunks with no usable texture-family ground truth are excluded from classifier training
  supervision entirely; they are never zero-filled or assigned a default class.
- Arbitrary out-of-distribution input images (no client-derived ground truth at all) must still
  receive a full-tile classifier prediction; the classifier may report low confidence but must not
  refuse to run, and the downstream geometry model must remain numerically stable when fed a
  low-confidence or noisy generated feature-map rather than failing closed.
- The canonical feature-family taxonomy introduced here is shared, versioned vocabulary; if Spec 114
  US3/US4 are built later, they must reuse this taxonomy's revision rather than defining a second,
  incompatible one for the same underlying concept.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The terrain-feature classifier MUST consume only the minimap RGB tile (and any
  already-promoted generated upstream signal) and MUST emit one terrain-feature-family class with a
  confidence per prediction unit, using no ground-truth texture-family or tileset ID at inference.
- **FR-002**: Classifier training labels MUST be derived deterministically from real per-chunk
  texture-family/tileset-ID ground truth against a versioned canonical family lookup, and MUST NOT
  require manual annotation.
- **FR-003**: The canonical feature-family library MUST include at minimum real-terrain, road/path,
  water, building/object-adjacent, and an explicit unknown/low-confidence class; unrecognized or
  ambiguous texture families MUST map to unknown, never forced into a known class.
- **FR-004**: Tiles or chunks with no usable texture-family ground truth MUST be excluded from
  classifier training supervision, never zero-filled or defaulted to a specific class.
- **FR-005**: The classifier MUST be its own independently trained, checkpointed, and promotable
  model; it MUST NOT share weights or a multi-task head with the direct-geometry or detailer models.
- **FR-006**: The classifier's promotion gate MUST include evaluation against a frozen held-out real
  split AND a qualitative review against at least one out-of-distribution image with no
  client-derived ground truth, to verify the model generalizes from appearance rather than
  memorizing per-map texture statistics.
- **FR-007**: Retraining the direct-geometry model MUST add the classifier's generated predicted
  feature-map as an additional input channel alongside the existing minimap RGB; it MUST NOT consume
  ground-truth texture-family IDs at any point in its own input path, at training or at inference.
- **FR-008**: The retrained geometry model's promotion gate MUST compare against the frozen RGB-only
  baseline checkpoint on the same held-out split using both the existing standard error metric and a
  road/path-region-specific error metric, to directly evidence that the texture confound is reduced
  rather than relying on an aggregate metric alone.
- **FR-009**: Every stage's run/promotion record MUST bind the exact upstream checkpoint hash it
  consumed (classifier hash for the retrained geometry model; retrained geometry hash for the
  re-materialized detailer), so any stage can be independently replaced and re-evaluated without
  silently invalidating downstream provenance.
- **FR-010**: Deployment inference for the retrained chain MUST remain auditable end to end: input
  minimap image, classifier checkpoint, generated feature-map, geometry checkpoint, detailer
  checkpoint, final relative height -- with zero ground-truth signals entering any inference step.
- **FR-011**: Training, label-pipeline runs, re-materialization, and any GPU-heavy execution MUST
  remain user-executed from exact documented commands with time/VRAM estimates; the assistant never
  launches them.

### Key Entities

- **Terrain Feature Family Library**: the versioned canonical class list (real-terrain, road/path,
  water, building-adjacent, unknown, and any later additions), the texture-family/tileset-ID lookup
  rule that derives labels from it, and the blended-layer resolution policy.
- **Terrain Feature Label**: one ground-truth class per chunk (or finer unit), derived from real
  texture-family IDs, with source-row lineage; used only for classifier training and evaluation,
  never for geometry training directly.
- **Predicted Feature Map**: the classifier's generated per-tile output (class plus confidence per
  prediction unit); the only form of this signal any downstream model may consume.
- **Deconfounded Geometry Checkpoint**: the retrained direct-geometry model whose input contract adds
  the predicted feature-map channel, with a Model Stage Record binding the exact upstream classifier
  checkpoint hash consumed.
- **Re-paired Detailer Checkpoint**: the residual detailer retrained against the deconfounded coarse
  checkpoint's re-materialized outputs, with its own Model Stage Record binding that exact upstream
  hash.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a frozen held-out real-tile split, the terrain-feature classifier identifies
  road/path regions with materially higher accuracy than a majority-class baseline.
- **SC-002**: On the out-of-distribution image that originally exposed the roads-as-hills failure (or
  an equivalent held-out non-client image), a visual and quantitative review confirms the classifier
  flags the visibly road-like regions as non-terrain, using no ground truth for that image.
- **SC-003**: The retrained geometry model's height error inside classifier-flagged road/path regions
  improves by a defined relative margin over the current RGB-only baseline on the same held-out
  split, while non-road-region error does not regress beyond a small defined tolerance.
- **SC-004**: A user visually reviewing the retrained chain's output against the motivating
  out-of-distribution image no longer sees roads rendered as pronounced sloping ridges or hills.
- **SC-005**: Every promoted checkpoint in this feature (classifier, retrained geometry, re-paired
  detailer) has a complete, audit-traceable run/promotion record, and zero ground-truth signals are
  present in the deployment inference path end to end.

## Assumptions

- The v50.1 store's real per-chunk texture-family/tileset-ID signals (mcly_texture_ids,
  mcly_tileset_ids) are accurate and populated on the existing Kalimdor/Azeroth 0.5.3.3368 corpus
  and are sufficient to derive training labels without a new harvest pass.
- A small, hand-curated canonical road/path (and water, building-adjacent) texture-family name list
  is an acceptable starting point for the label lookup; broadening coverage later is a refinement,
  not a blocker to an initial promotable classifier.
- The currently promoted-pending checkpoints (`mit_b0-authored-v1`,
  `detailer-mit_b0-authored-v2-bandsplit-continued`) remain the correct RGB-only baseline this
  feature must beat; they are superseded only if the retrained checkpoints clear this feature's own
  gates, and are not discarded before that.
- This feature does not build Spec 114 US3's full land-feature classification or US4's
  texture-family/alpha reconstruction; it introduces the shared feature-family library concept those
  stories can later reuse, but does not implement their consumers.
- GPU training for the classifier, the retrained geometry model, and the re-paired detailer remains
  user-executed, matching every other training stage in Spec 114 and Spec 111.
