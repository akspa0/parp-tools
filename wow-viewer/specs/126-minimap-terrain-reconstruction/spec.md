# Feature Specification: Minimap-to-Terrain Reconstruction Stack

**Feature Branch**: `126-minimap-terrain-reconstruction`

**Created**: 2026-08-02

**Status**: Draft

**Input**: User description: "Feed one minimap tile, several tiles, or a single large stitched image into a large model and get back heightmap data exportable as a heightmap / terrain mesh, plus a best-effort decode of the texture layers."

## Why This Is Now Possible

Every prior attempt at this inverted a black box. This one does not, and that is the entire
justification for revisiting a problem that has failed before:

1. **We hold the forward model.** The synthesizer composites minimaps the way the original process
   did. For any tile with terrain data we can generate the minimap, the textureless terrain-shadow
   residual, and (via User Story 1) the unlit albedo — exact, unlimited, free training pairs.
2. **The deployment domain is matched.** Authored 0.5.3 minimaps are DXT1; the dataset degrades
   synthetic minimaps through a round-trip that is bit-exact with the C# codec, so training input
   and authored input carry the same codec noise.
3. **The signals are trustworthy now.** The tiny-model constraint that governed this work until
   2026-08-02 was a response to a corpus where roughly 90% of signals were wrong or useless. Spec 109
   rebuilt the corpus clean-room and Spec 122 gave it curation and bucketing. Constitution v2.0.0
   retired the constraint and replaced it with a per-signal evidence requirement.

What has **not** changed: inverting an image cannot recover information the image does not contain.
This spec is explicit about which outputs are recoverable, which are best-effort, and which are
supplied from elsewhere.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Separate albedo from shading with ground truth for both halves (Priority: P1)

A reconstruction engineer needs to know how much of a minimap's appearance is *texture colour* and
how much is *terrain shading*, because only the shading half carries terrain shape. Today the
synthesizer can render shading without texture (the textureless residual) but has no symmetric pass
that renders texture without shading. Adding that unlit-albedo pass yields exact
`(minimap, albedo, shading)` triples for every tile, which converts an ill-posed decomposition
problem into two directly supervised ones plus a consistency check.

**Why this priority**: It is the prerequisite for everything else and it settles an open
architectural question by measurement rather than argument — whether albedo removal is on the
critical path to height, or merely a refinement. It also delivers standalone value: the first honest
measurement of the albedo/shading variance split in the corpus.

**Independent Test**: Render the unlit-albedo pass for a set of tiles, verify that recombining
albedo with shading reproduces the full minimap within a stated tolerance, and read the reported
variance split. Delivers a decision and a dataset regardless of whether any model is trained.

**Acceptance Scenarios**:

1. **Given** a tile with terrain data, **When** the unlit-albedo pass runs, **Then** it emits a
   256x256 image containing the composited terrain texture with lighting flat and no objects.
2. **Given** the albedo and shading passes for the same tile, **When** they are recombined under the
   compositor's own blend, **Then** the result matches the full synthetic minimap within a stated
   per-pixel tolerance, and tiles exceeding it are reported rather than dropped.
3. **Given** a corpus sample, **When** the decomposition is measured, **Then** the report states what
   fraction of minimap variance is attributable to albedo versus shading, per tile and in aggregate.
4. **Given** a tile whose albedo is near-uniform (a single-texture tile), **When** the split is
   reported, **Then** that tile is identified as uninformative for the decomposition rather than
   counted as a clean result.

---

### User Story 2 - Recover relative terrain relief from a single minimap tile (Priority: P1)

A reconstruction engineer feeds one minimap tile and receives a height field on the 257x257 world
grid whose *relief* matches the real terrain. This is the core capability; every other story either
feeds it or consumes it.

**Why this priority**: It is the feature. If relief cannot be recovered from a single tile, nothing
downstream matters, and the failure should be discovered on the cheapest possible configuration.

**Independent Test**: Train on synthetic pairs, run on held-out tiles never seen in training, and
score the recovered height against real MCVT ground truth. Passes or fails on its own.

**Acceptance Scenarios**:

1. **Given** a held-out minimap tile, **When** reconstruction runs, **Then** it outputs a 257x257
   relative height field under the versioned relative-height contract.
2. **Given** held-out tiles, **When** results are scored, **Then** the model beats the tile-mean
   baseline on relief error by a stated margin, reported per tile rather than only in aggregate.
3. **Given** a tile whose terrain is genuinely flat, **When** reconstruction runs, **Then** it
   reports low confidence rather than inventing relief.
4. **Given** the best epoch is epoch 1, **When** the run summary is written, **Then** the run is
   recorded as a structural failure, not a success.
5. **Given** an authored (DXT1) tile rather than a synthetic one, **When** reconstruction runs,
   **Then** accuracy is reported separately for authored and synthetic input so codec-domain
   degradation is visible.

---

### User Story 3 - Supervise only the terrain the minimap actually shows (Priority: P1)

Minimap tiles have objects — trees, buildings, rocks — composited on top of the terrain. Where an
object covers terrain, the image contains no evidence of the ground beneath it, and training the
height loss on those pixels teaches the model to hallucinate. The loss must be masked to the
terrain that is genuinely visible.

**Why this priority**: It determines whether User Story 2's numbers mean anything on authored tiles.
An unmasked loss produces a model that confidently invents ground under every tree, and the error
will not show up in aggregate metrics because occluded pixels are a minority of each tile. This has
been requested repeatedly across prior specs and repeatedly deferred; it is a first-class story here.

**Independent Test**: Build occlusion masks for a tile set, verify masked coverage against the
rendered object footprints, and train two otherwise-identical runs with and without masking. The
masked run's error under object footprints is the measurement.

**Acceptance Scenarios**:

1. **Given** a tile containing objects, **When** the occlusion mask is built, **Then** it marks only
   the terrain pixels actually hidden by an object in the rendered view — not each object's full
   ground footprint.
2. **Given** a tile, **When** masks are produced, **Then** they are per-object and attributable, so
   an individual object's contribution can be inspected.
3. **Given** a training run, **When** the height loss is computed, **Then** occluded pixels are
   excluded, and the fraction of each tile excluded is recorded.
4. **Given** a tile that is almost entirely occluded, **When** it is selected for training, **Then**
   it is routed to a bucket rather than silently contributing a near-empty loss.
5. **Given** scored results, **When** error is reported, **Then** error under object footprints is
   reported separately from error on open terrain.
6. **Given** the same tiles, **When** coverage is measured under the occlusion-aware visible mask and
   under the full-footprint mask, **Then** the retained-terrain fraction is reported for both — the
   full-footprint mask is what made masking unusable previously, and the gap between them is the
   reason this approach is now viable.

---

### User Story 4 - Export a usable heightmap and mesh (Priority: P2)

A reconstruction engineer takes the recovered relief, composes it with absolute elevation from the
WDL lattice prior, and exports height data that opens in external tooling and meshes into a terrain
surface.

**Why this priority**: Without export the feature produces a tensor nobody can use. It is P2 rather
than P1 only because User Story 2 can be evaluated before export exists.

**Independent Test**: Export a reconstructed tile, open the result in external tooling, and mesh it.
Verifiable without any accuracy claim.

**Acceptance Scenarios**:

1. **Given** a reconstructed relative height field and a WDL lattice prior for the same tile,
   **When** they are composed, **Then** the output carries absolute elevation in world units.
2. **Given** no WDL prior is available for a tile, **When** export runs, **Then** it emits
   relative-only height and states plainly that absolute elevation is absent — never a fabricated
   datum.
3. **Given** an exported heightmap, **When** it is meshed, **Then** the surface has no inverted
   normals and no disconnected spikes.
4. **Given** an exported heightmap, **When** it is re-lit with the synthesizer's lighting model,
   **Then** the result visually corresponds to the input minimap's shading.

---

### User Story 5 - Reconstruct across many tiles and large stitched images (Priority: P2)

A reconstruction engineer submits several adjacent tiles, or one large stitched image covering many
tiles, and receives terrain that is continuous across tile boundaries rather than a patchwork with
visible steps at every seam.

**Why this priority**: Per-tile relative normalization makes each tile's height independent by
construction, so seams are the *expected* failure mode and must be addressed explicitly. Region-scale
output is what makes the feature useful beyond a demo.

**Independent Test**: Reconstruct a block of adjacent tiles, measure height discontinuity along
shared edges, and compare against the discontinuity present in the ground truth for those edges.

**Acceptance Scenarios**:

1. **Given** a set of adjacent tiles, **When** reconstruction runs, **Then** height along each shared
   edge agrees between neighbours within a stated tolerance.
2. **Given** a single large image spanning a known tile grid, **When** it is submitted, **Then** it
   is decomposed into tiles, reconstructed, and recomposed into one continuous height field.
3. **Given** an image whose tile grid or world scale is unknown, **When** it is submitted, **Then**
   the system reconstructs relative relief and reports that absolute scale and elevation are
   unavailable, rather than assuming a scale.
4. **Given** a tile with no neighbours supplied, **When** reconstruction runs, **Then** it succeeds
   and marks its edges as unconstrained.

---

### User Story 6 - Decode texture and layer information in graded tiers (Priority: P2)

A reconstruction engineer recovers what the terrain was painted with. Perfection is explicitly not
the goal: partial recovery is valuable, and the system reports which tier it achieved rather than
passing or failing as a whole.

- **Tier 1 — Area colour**: the general albedo colour of each region.
- **Tier 2 — Dominant layer**: a single dominant texture layer with best-guess tileset identities.
- **Tier 3 — Two layers plus base**: base texture plus two blended layers. Known achievable — this
  level was reached previously.
- **Tier 4 — Per-layer reconstruction**: full paint-by-numbers MCAL/MCLY per alpha layer.
  See User Story 8.

**Why this priority**: Tier 1 alone is useful output, and texture information feeds back into height
reconstruction — a prior spec measured a 21.35% reduction in road-region height error from supplying
a terrain-feature map to the geometry model. Graded tiers let partial success be reported as partial
success.

**Independent Test**: Run decode on held-out tiles and score each tier separately against real MCLY
and MCAL ground truth. Each tier is independently reportable.

**Acceptance Scenarios**:

1. **Given** a held-out tile, **When** decode runs, **Then** the output states which tier it
   achieved and reports a per-tier score.
2. **Given** tier 2 or above, **When** texture identities are reported, **Then** each carries a
   confidence, and low-confidence identities are surfaced as guesses rather than asserted.
3. **Given** two textures with near-identical average colour, **When** decode runs, **Then** their
   confusion is reported rather than silently resolved — at minimap resolution a tiled texture is
   reduced toward its mean colour, so this is expected.
4. **Given** decoded layer structure, **When** it is supplied to height reconstruction, **Then** the
   effect on height error is measured and reported, positive or negative.
5. **Given** the decode fails entirely for a tile, **When** results are reported, **Then** that tile
   is recorded as tier 0 rather than omitted from the denominator.

---

### User Story 7 - Refine iteratively using the forward model as referee (Priority: P3)

Rather than one feed-forward pass, the system refines: estimate albedo, divide it out, estimate
shading, estimate height, then re-estimate albedo *given* the height, and repeat. Because the forward
model is known, each iteration's proposal can be rendered and compared against the input.

**Why this priority**: It is a refinement over a working single pass, so it must not be attempted
first. It is also the mechanism most likely to close the gap between "beats the baseline" and
"actually usable", because it exploits the one asset that makes this problem tractable at all.

**Independent Test**: Run the same trained components with 1, 2, and N refinement passes on the same
held-out set and compare. Any improvement is attributable to refinement alone.

**Acceptance Scenarios**:

1. **Given** a reconstructed height field, **When** it is rendered through the forward model,
   **Then** the render-versus-input difference is reported as a per-tile consistency score.
2. **Given** N refinement passes, **When** results are compared to a single pass, **Then** the change
   in height error is reported per pass count, including when it is negative.
3. **Given** refinement diverges on a tile, **When** iteration halts, **Then** the last stable
   estimate is returned and divergence is recorded.

---

### User Story 8 - Per-layer paint-by-numbers reconstruction (Priority: P3)

A reconstruction engineer recovers each alpha layer individually — which texture occupies each layer
slot, and the per-layer alpha map that blends them — sufficient to rebuild the tile's texture
chunks rather than approximate its appearance.

**Why this priority**: This is tier 4, the strongest texture claim, and it depends on tier 3 working
first. Gating it prevents the most speculative output from consuming effort before the tiers beneath
it are proven. It is specified here rather than deferred to a separate feature so that the data and
evaluation built for tiers 1-3 are designed to support it.

**Independent Test**: Score recovered per-layer texture identity and alpha maps against real MCLY and
MCAL ground truth on held-out tiles.

**Acceptance Scenarios**:

1. **Given** a held-out tile, **When** per-layer decode runs, **Then** it emits a texture identity per
   layer slot and an alpha map per layer.
2. **Given** recovered layers, **When** they are composited through the forward model, **Then** the
   resulting albedo is compared against the true albedo and the difference reported.
3. **Given** tier 3 has not met its acceptance criteria, **When** this story is evaluated, **Then**
   it is reported as blocked rather than attempted and scored.

---

### Edge Cases

- **Flat terrain**: no relief signal exists in the shading. Must report low confidence, never invent
  ridges.
- **Fully occluded tiles**: a tile whose terrain is almost entirely hidden by objects has almost no
  usable supervision. Routed to a bucket, not silently trained on.
- **Back-facing slopes**: shading saturates at zero, destroying slope information in shadow. Affected
  regions must be identifiable in the output confidence.
- **Water**: liquid surfaces are flat and shade differently from terrain; liquid regions must not be
  read as terrain relief.
- **Empty / unrendered tiles**: tiles with a single colour are not terrain and must be excluded from
  aggregates rather than averaged in.
- **Codec floor**: DXT1 quantisation sets a hard error floor on colour-derived signals. Reported
  accuracy must never be presented as if that floor did not exist.
- **Tiles with no WDL prior**: relative-only output, absolute elevation explicitly absent.
- **Unknown-provenance input images**: an image that is not a known-era minimap may not follow the
  lighting model this stack inverts; provenance must be carried, not assumed.

## Requirements *(mandatory)*

### Functional Requirements

#### Decomposition and data

- **FR-001**: The synthesizer MUST emit an unlit-albedo pass — composited terrain texture with flat
  lighting and no objects — for any tile it can already render.
- **FR-002**: The system MUST verify that albedo recombined with shading reproduces the full minimap,
  and MUST report tiles that fail that check rather than discarding them.
- **FR-003**: The system MUST report the albedo-versus-shading variance split across a corpus sample,
  per tile and in aggregate.
- **FR-004**: All model input derived from synthetic minimaps MUST carry the DXT1 codec degradation
  that authored tiles carry; the pristine variant MUST remain available for ablation.
- **FR-005**: Every dataset partition — including tiles rejected for any reason — MUST remain
  queryable. Filtering that discards rows is prohibited.

#### Height reconstruction

- **FR-006**: The system MUST accept one tile, several tiles, or a single large image and produce
  height on the 257x257 per-tile world grid.
- **FR-007**: Height output MUST use the versioned relative-height contract, so absolute elevation
  cannot leak into supervision.
- **FR-008**: Absolute elevation MUST be supplied by composition with the WDL lattice prior, never
  predicted from imagery.
- **FR-009**: When no WDL prior exists for a tile, the system MUST emit relative-only height and
  state that absolute elevation is unavailable.
- **FR-010**: Adjacent tiles MUST agree along shared edges within a stated tolerance, and edge
  disagreement MUST be reported.
- **FR-011**: The system MUST emit a per-pixel confidence, and MUST report low confidence on flat
  terrain and in shadow-saturated regions rather than asserting relief.

#### Supervision honesty

- **FR-012**: The height loss MUST exclude terrain pixels occluded by objects, using masks derived
  from actual rendered occlusion — not full object ground footprints, which over-mask heavily.
- **FR-013**: Occlusion masks MUST be per-object and attributable to the object that produced them.
- **FR-014**: The excluded fraction of each tile MUST be recorded, and tiles exceeding an occlusion
  threshold MUST be bucketed rather than trained on silently.
- **FR-015**: Error MUST be reported separately for open terrain and for regions under object
  footprints.
- **FR-016**: Held-out evaluation MUST use tiles spatially disjoint from training tiles, and the
  evaluation set MUST be identified in every reported result.
- **FR-017**: Evaluation MUST use Kalimdor and Azeroth. PVPZone02 and Kalidar MUST NOT be used for
  validation.
- **FR-018**: Accuracy on authored input MUST be reported separately from accuracy on synthetic
  input.

#### Texture and layer decode

- **FR-019**: Texture decode MUST report which tier (0-4) it achieved per tile, with a per-tier score.
- **FR-020**: Reported texture identities MUST carry confidence, and confusable identities MUST be
  reported as confused rather than resolved arbitrarily.
- **FR-021**: The measured effect of supplying decoded layer structure to height reconstruction MUST
  be reported, including when it is negative.
- **FR-022**: Tier 4 MUST be reported as blocked if tier 3 has not met its criteria.

#### Model and evidence

- **FR-023**: A model producing multiple signals MUST report per-signal validation metrics, each
  against that signal's own trivial baseline. An aggregate improvement alongside a signal stuck at
  baseline MUST be reported as a partial failure.
- **FR-024**: Every output head MUST be independently ablatable — droppable or freezable without
  retraining the others from scratch.
- **FR-025**: Per-head loss weighting MUST be stated and justified, because the signals occupy
  different grids and scales and an unweighted sum lets the dominant signal swamp the others.
- **FR-026**: The system MUST support refinement over multiple passes, and MUST report the effect of
  pass count on error.
- **FR-027**: A reconstruction MUST be renderable back through the forward model, and the
  render-versus-input consistency MUST be reported per tile.
- **FR-028**: Every trained artefact MUST record its dataset identity, evaluation set identity, and
  the release it was built against.
- **FR-029**: Training and harvest execution MUST remain user-initiated. No workflow may launch them
  automatically.

#### Export

- **FR-030**: Reconstructed height MUST be exportable as heightmap data readable by external tooling.
- **FR-031**: Reconstructed height MUST be meshable into a terrain surface with no inverted normals
  and no disconnected spikes.

### Key Entities

- **Minimap tile**: a 256x256 image of one map tile, authored (DXT1, from the client) or synthetic
  (from the forward model). Carries map name, tile coordinates, era, and provenance.
- **Albedo field**: per-pixel terrain texture colour with lighting removed.
- **Shading field**: per-pixel terrain illumination with texture removed — Lambert response plus cast
  shadows and ambient. The half of the minimap that carries terrain shape.
- **Height field**: 257x257 per-tile elevation. Relative by contract; absolute only after composition
  with a WDL prior.
- **Occlusion mask**: per-object record of which terrain pixels an object hides in the rendered view.
- **Layer structure**: the texture identities occupying a tile's layer slots and the alpha maps that
  blend them.
- **WDL lattice prior**: coarse absolute elevation for a tile, the source of absolute datum.
- **Reconstruction report**: per-tile record of every output, its confidence, its tier where
  applicable, and its consistency against the forward model.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The albedo/shading variance split is reported for a corpus sample, and the decision on
  whether albedo removal is on the critical path to height is recorded before the main model is
  trained.
- **SC-002**: Albedo recombined with shading reproduces the synthetic minimap for at least 95% of
  sampled tiles within the stated tolerance; the remainder are reported individually.
- **SC-003**: Relief recovered from a single held-out tile beats the tile-mean baseline, with the
  margin reported per tile and the fraction of tiles that fail to beat it stated explicitly.
- **SC-004**: Reconstruction accuracy is reported separately for authored and synthetic input, and the
  gap between them is stated as a number.
- **SC-005**: Training with occlusion masking reduces height error under object footprints relative to
  an identical unmasked run, and the improvement is reported as a number.
- **SC-005a**: Retained-terrain fraction is reported for the occlusion-aware visible mask and for the
  full-footprint mask over the same tiles. The visible mask must retain substantially more terrain —
  this is the measurement that distinguishes viable masking from the earlier approach, which removed
  so much terrain that masking and hallucinating were the only two options.
- **SC-006**: Across a block of adjacent tiles, height disagreement at shared edges is no worse than a
  stated multiple of the disagreement present in ground truth at those same edges.
- **SC-007**: A reconstructed tile exports to heightmap data that opens in external tooling and meshes
  without inverted normals or disconnected spikes.
- **SC-008**: Texture decode reaches tier 3 on a stated fraction of held-out tiles, with per-tier
  scores reported and tier 0 tiles counted in the denominator.
- **SC-009**: The measured effect of decoded layer structure on height error is reported, whatever its
  sign.
- **SC-010**: Refinement passes are compared at 1, 2, and N passes on the same held-out set, and the
  error change per pass count is reported.
- **SC-011**: Flat and shadow-saturated regions are reported at low confidence, and no reconstruction
  asserts relief in a region where the input contains no relief evidence.
- **SC-012**: Every multi-signal run reports a per-signal metric against a per-signal baseline; no run
  is reported as a success while any of its signals sits at baseline.
- **SC-013**: The trained stack stays at or under 200 million parameters.
- **SC-014**: Every reported result identifies its evaluation set, so results from different held-out
  sets are never compared as if equivalent.

## Assumptions

- **Era scope is 0.5.3.** The lighting model, minimap generation behaviour, and DXT1 encoding this
  stack inverts are era-specific. Other builds require their own profile and are out of scope.
- **The primary large-image case is a stitched map of known tiles.** An image whose tile grid and
  world scale are known can be decomposed, reconstructed, and recomposed. Arbitrary or hand-painted
  images are supported as relative-relief-only, with scale and elevation reported as unavailable.
- **Absolute elevation is never predicted from imagery.** It comes from the WDL lattice prior. This is
  a property of the relative-height contract, not a limitation to be engineered around.
- **DXT1 sets an error floor** on every colour-derived signal. Reported accuracy is understood to sit
  above that floor.
- **Texture identity degrades toward mean colour at minimap resolution.** A tile spans 533.333 world
  units across 256 pixels, so a tiled terrain texture is minified toward its average colour. Recovering
  the albedo *field* is therefore expected to be substantially easier than recovering texture
  *identity*, and the tiers are graded accordingly.
- **Prior work is reused, not redone.** The archived relational-layers spec already defined layer
  structure prediction from minimaps and the question of whether layer masks derive from terrain shape;
  its unrun gating measurements are inputs to User Story 6, not work to re-specify.
- **Objects are removed from supervision, not from the input.** Authored tiles contain objects at
  inference time; the model must tolerate them. Masking applies to the loss.
- **Curation partitions rather than filters.** Every bucket stays queryable, per the established
  curation principle.
- **Depth-foundation-model families are excluded.** They are blacklisted for this project, and they
  solve a different problem — camera depth under unknown lighting — where here the lighting is known.
- **The user executes all training and harvest runs.** Commands are handed over, never launched.

## Dependencies

- The forward model / synthesizer and its lighting calibration (Spec 111, Spec 125).
- The DXT1 round-trip codec and its C#/Python parity lock (Spec 125).
- The clean-room corpus (Spec 109) and its curation and bucketing layer (Spec 122).
- The WDL lattice prior for absolute elevation (Spec 117).
- Object occlusion mask sources (Spec 118).
- Prior findings on layer structure prediction (archived relational-layers spec) and on terrain-feature
  maps improving height error (Spec 115).

## Out of Scope

- Recovering absolute world elevation from imagery alone.
- Perfect reconstruction. This stack targets useful relief and best-effort texturing; exactness is
  precluded by the codec floor, by shading saturation, and by genuinely flat terrain.
- Object geometry reconstruction. Objects are handled as occluders, not as reconstruction targets.
- Writing BLP containers. Codec work is in-memory only.
- Eras other than 0.5.3.
- Super-resolution beyond native 256x256, which has no client-side ground truth.
