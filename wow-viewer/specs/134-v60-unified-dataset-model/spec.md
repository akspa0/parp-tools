# Feature Specification: V60 Controlled Terrain Reconstruction Experiment

**Feature Branch**: `134-v60-unified-dataset-model`
**Created**: 2026-08-08
**Status**: Draft — terrain-only control learning is active; object lanes are parked

**Input**: Use project-owned synthetic terrain controls to test a small reconstruction experiment.
First normalize authored minimap albedo, admit only textureless results, and then test transfer to a
tiny 0.x/1.x sample before processing anything broader.

### Current execution route (2026-08-10)

The immediate experiment is only:

```text
terrain_shadow_256 -> height_257
```

It reads the validated `control-v1` NPZ corpus, preserves its complete-family validation holdout,
and compares limited training sizes against a tile-mean baseline. Object contamination, object
identity, object markers, real-client rows, and albedo normalization are not inputs or gates for
this experiment. Those artifacts remain parked for a later, separately authorized phase.

## Context

The re-baked minimap path is already working. The previous v50/v60 direction treated a large
harvested corpus as the primary product, which created avoidable problems: mixed-era inputs,
metadata trust failures, finalize/retry hazards, and an assumption that thousands of authored tiles
were required before the first useful model experiment.

The next boundary is smaller and controlled. Existing terrain tooling can generate genuine-looking
minimap/shadow signals from known terrain and can therefore provide exact targets. The first model
lane should answer one question:

```text
canonical textureless minimap 256x256 -> terrain height 257x257
```

The canonical synthetic input is `terrain_shadow_256`. For real authored minimaps, an explicit
versioned albedo-normalization operation must first produce a comparable textureless input. A
synthetic control score is necessary evidence, but is not by itself proof that the real operation
works; a tiny transfer gate is required.

### Initial source policy

- Client-backed source seeds initially come only from explicitly classified `0.x` and `1.x` roots.
- Procedural control families are valid v60-control-v1 seeds and require no client.
- The default control atlas MUST cover the existing complexity vocabulary (`easy`, `medium`, `hard`,
  `pathological`) with explicit terrain families: flat/slope, dome/basin/plateau/rolling, ridge/valley/
  terrace/cliff, chunk-grid discontinuities, island/archipelago, crater/canyon composition,
  mountainous relief, arbitrary-angle sheer drop-offs, zone-style blends, fractal/fBm and
  ridged-fractal terrain, dendritic lightning-burn strokes, and mixed/noisy pathological cases.
- Cross-tile pattern families MUST be generated from one global 2x2 pattern coordinate system so
  each tile contains only a partial motif and the four tiles can be stitched to prove continuity.
- Non-grid families MUST receive deterministic sub-cell field offsets unrelated to the 16x16 chunk
  lattice. Explicit `chunk_grid` controls remain aligned only as a diagnostic family; all other
  families must record `subcell_shifted` or `mixed_alignment` provenance.
- A client source supplies a small number of distinct terrain seeds; it is not recursively harvested
  as a full training corpus.
- Later client builds remain a planned extension point, but no later-era processing is part of the
  initial transfer route.
- Authored minimap pixels are not accepted directly into the first model lane. They require albedo
  normalization and the textureless quality gate.
- The initial object-removal route is `normalized textureless minimap -> optional object sieve ->
  terrain reconstruction`. Object identification/marking is a separate specialist and is not
  hidden inside the sieve. Synthetic object controls start from the canonical terrain shadow and
  add controlled object contamination, so object marking and object removal are tested separately
  from albedo removal.
  The object controls MUST use the existing v50 object-library `capture_rgb`/`capture_mask` pairs
  for the placed silhouettes; procedural geometric stamps are retained only as a comparison
  baseline.
- The existing v50.1 `0_5_3_3368` curriculum is an allowed supervision source for the object-mask
  experiment only. It may be read by configured path; it is not copied into v60 and does not bypass
  the later albedo/textureless gate for height reconstruction.

### Object-lane correction (2026-08-09)

The earlier `real-object-masks-v1` run is not a valid precision-object result. It used the
curriculum's tile-level placement projections (`object_mask`/`object_precise_mask`), which appear as
placement dots at minimap scale, and did not read the v50 object library. The corrected v60 lane is
`v60-object-library-sieve-v1`: it reads the 5,349-entry 0.5.3 object library, composites real
top-down object captures onto project-owned clean terrain controls, and emits the exact library
silhouette union plus a per-instance ID map as loss-side targets. The old run remains on disk as a
diagnostic artifact but MUST NOT be used to claim object segmentation quality.

### Object identification-marker pivot (2026-08-09, parked)

The sieve-only experiment answers whether object pixels can be removed, but it cannot answer which
known library object produced those pixels. The next object slice therefore consumes a minimap image
and one candidate footprint at a time. It predicts whether the footprint contains a known library
object and emits an embedding used to retrieve the best matching v50 library entry. Accepted
candidates are rasterized into a dense marker map; the library ID, match score, asset path, and
candidate geometry remain in a sidecar identity table rather than being encoded as a fragile
5,349-way pixel value.

The marker model is deliberately not an end-to-end proposal detector in this slice. Candidate
footprints come from an explicit upstream source (for the first control run, the precise synthetic
instance masks from `v60-object-library-sieve-v1`). Later proposal sources may include a real
renderer/object pass or a separate footprint detector. This keeps identity evidence separate from
proposal recall and makes a marked input minimap directly inspectable.

This lane is now explicitly deferred. Its user-run result was not useful identity evidence: held-out
retrieval top-1 was effectively zero, negatives were frequently accepted as known, and the input
corpus did not preserve the real object RGB signal needed for identity. Do not feed this checkpoint
or any object marker into the terrain experiment.

## User Stories & Testing

### User Story 1 — Small synthetic control corpus (Priority: P1)

A dataset operator can select a small set of distinct terrain families and generate deterministic
control data containing exact synthesized inputs, targets, variant parameters, and provenance.

**Independent Test**: Generate the corpus twice from the same configuration and compare manifests,
arrays, and per-row hashes. They are identical, with complete-family holdouts and no client-wide
harvest output.

**Acceptance Scenarios**:

1. **Given** an explicit control family list, **when** the control builder runs, **then** it writes
   only the requested families and variants.
2. **Given** a generated terrain variant, **when** the compositor runs, **then** the input and exact
   height target are emitted together with their parameters and hashes.
3. **Given** a client-backed seed outside the `0.x`/`1.x` policy, **when** it is supplied, **then** it
   is rejected or recorded as excluded and never silently included.
4. **Given** the same configuration and generation seed, **when** the builder runs twice, **then**
   output bytes and manifest hashes match.
5. **Given** the default family set, **when** visual review runs, **then** it writes family and
   variant atlases showing height, textureless shadow, normals, and height-edge structure, plus a
   report identifying missing expected families or complexity buckets.
6. **Given** a cross-tile fractal or lightning family, **when** visual review stitches its 2x2 rows,
   **then** the pattern crosses tile seams instead of restarting at each tile.

### User Story 2 — Object identification, marking, and optional sieve supervision (Priority: P2, deferred)

A researcher can provide a minimap image plus candidate object footprints, identify known objects
against the real v50 object library, and write a dense known-object marker map and identity table.
The existing sieve remains an optional downstream consumer for removing the marked pixels before
terrain reconstruction.

**Independent Test**: A held-out marker report evaluates known/unknown detection and library
retrieval per candidate, then writes a marker map whose nonzero regions exactly correspond to the
accepted candidate footprints. Sieve clean-terrain and contamination-mask metrics remain separate.

**Acceptance Scenarios**:

1. **Given** a canonical terrain control, **when** objects are placed on it, **then** the row carries
   the contaminated input, exact clean terrain-shadow target, object-contamination mask, and object
   placement metadata.
2. **Given** no-object, sparse, dense, overlapping, and tile-boundary placements, **when** the sieve
   controls are reviewed, **then** each regime is represented and object patterns do not always fit
   inside one tile.
3. **Given** the object-contamination mask, **when** the sieve model trains, **then** the mask is a
   separate supervised output and is not silently mixed into the height target.
4. **Given** mask-guided and non-guided sieve variants, **when** the ablation runs, **then** the
   guidance variant consumes its own predicted mask at training and inference; ground-truth masks
   are never supplied as an inference channel.
5. **Given** a sieve output, **when** it is evaluated, **then** clean-terrain error and mask metrics
   are reported independently by object-density and placement family.
6. **Given** the existing v50 object library, **when** the library-sieve corpus is built, **then**
   each placed object comes from a real `capture_rgb`/`capture_mask` pair and carries its library ID
   and transform provenance.
7. **Given** a derived library-sieve row, **when** its targets are inspected, **then** the mask is
   a visible silhouette union with a per-instance ID map, not a tile-level placement dot.
8. **Given** the old curriculum `object_precise_mask`/`object_mask` arrays, **when** the historical
   diagnostic is reviewed, **then** its dot-like targets are reported as rejected evidence and are
   not promoted to the precision object lane.
9. **Given** a minimap image and a candidate footprint, **when** the marker model evaluates it,
   **then** it emits known/unknown confidence, a retrieval embedding, and the best matching library
   ID without receiving the ground-truth library ID as an input.
10. **Given** accepted candidate results, **when** marker export runs, **then** it writes a dense
    `known_object_marker_256` map and a sidecar identity table mapping each marker instance to its
    library ID, asset path, footprint, confidence, and retrieval score.
11. **Given** an unknown object or a shifted/empty candidate footprint, **when** marker inference
    runs, **then** it can reject the candidate instead of forcing a nearest-library identity.
12. **Given** a marker map, **when** the optional sieve consumes it, **then** the sieve sees only
    the exported prediction and never the marker target or library identity target.

### User Story 3 — Limited control-data model experiment (Priority: P1)

A researcher can run a deliberately small learning-curve experiment using only
`terrain_shadow_256` as input and `height_257` as target.

**Independent Test**: The report evaluates fixed held-out terrain families, compares a tile-mean
baseline, and reports metrics for each limited training size.

**Acceptance Scenarios**:

1. **Given** a control row, **when** loaded by the evaluator, **then** the input is one deterministic
   256x256 channel and the target is its matching 257x257 height field.
2. **Given** limited training sizes such as 8, 16, and 32 rows, **when** evaluation runs, **then**
   the learning curve is recorded without changing the held-out families.
3. **Given** retextured or relit variants with unchanged terrain, **when** evaluated, **then** the
   report distinguishes terrain generalization from colour or lighting memorization.
4. **Given** flat or weakly informative controls, **when** evaluated, **then** the report marks
   ambiguity instead of presenting a confident reconstruction as proof.

### User Story 4 — Albedo normalization and textureless gate (Priority: P1)

A researcher can process a tiny explicit 0.x/1.x real sample through an albedo-removal operation and
admit only outputs that are demonstrably close enough to the canonical textureless input contract.

**Independent Test**: Deliberately textured, failed, missing, and valid inputs produce accepted,
rejected, or quarantined decisions with persisted metrics and reasons; no failed row is zero-filled
or silently passed onward.

**Acceptance Scenarios**:

1. **Given** a real authored minimap, **when** albedo normalization runs, **then** it writes a
   versioned normalized artifact and measurable residual metrics.
2. **Given** a missing, non-finite, or visibly texture-bearing result, **when** the gate runs, **then**
   the row is rejected or quarantined and remains visible in the report.
3. **Given** positive synthetic controls and negative textured controls, **when** thresholds are
   calibrated, **then** the gate records the threshold version and calibration evidence.
4. **Given** an accepted result, **when** it enters the model lane, **then** it uses the same input
   shape/range contract as the control input or is explicitly versioned as a new contract.

### User Story 5 — Tiny real-data transfer and expansion decision (Priority: P2)

A researcher can compare the accepted tiny real sample with the control result and make an evidence-
based decision to hold, diagnose, or expand.

**Independent Test**: The transfer report includes the control run, accepted sample count, input
distribution comparison, failure cases, baseline-relative metrics, and an explicit decision.

**Acceptance Scenarios**:

1. **Given** a passing control experiment, **when** a tiny accepted 0.x/1.x sample is evaluated,
   **then** transfer metrics and domain differences are reported separately.
2. **Given** a transfer failure, **when** expansion is requested, **then** the plan remains held and
   identifies albedo normalization or domain shift as the next diagnosis.
3. **Given** a passing transfer gate, **when** broader processing is authorized, **then** the input
   route remains gated and provenance-preserving.

### User Story 6 — Later client support (Priority: P3)

A maintainer can add later client eras as new source adapters after the control and initial transfer
contracts are proven.

**Independent Test**: A later-era adapter produces the same manifest and signal contracts without
changing the control generator or silently mixing source-era behavior.

## Requirements

### Functional Requirements

- **FR-001**: v60-control-v1 MUST build a small synthetic control corpus, not a complete historical
  per-client signal harvest or unified archive.
- **FR-002**: Client-backed initial sources MUST accept only explicitly classified `0.x` and `1.x`
  roots and MUST report rejected later or unknown roots.
- **FR-003**: The operator MUST provide an explicit small family/seed configuration. The builder MUST
  NOT recursively harvest every map or every client by default.
- **FR-004**: Each control row MUST contain source identity, family identity, deterministic variant
  parameters, generation version, split membership, and hashes for every emitted array.
- **FR-005**: The first control contract MUST emit an albedo-stripped `terrain_shadow_256` input and
  the exact matching `height_257` target.
- **FR-006**: The control generator MUST support the default terrain taxonomy: flat, slope, dome,
  basin, plateau, ridge, valley, rolling, terrace, cliff, chunk-grid, chunk-grid-mixed, island-sea,
  archipelago, crater-field, canyon-fan, mountainous, sheer-dropoff, zone-style-blend, fractal-fBm,
  fractal-ridged, lightning-burn,
  cross-tile-lightning, cross-tile-burn, noise, mixed, and pathological variants while retaining a
  baseline control.
- **FR-007**: Lighting and albedo variation MUST be independently parameterized so the model cannot
  win by memorizing a colour-to-height shortcut.
- **FR-008**: Splits MUST hold out whole source families or terrain archetypes; variants of a held-out
  family MUST NOT leak into training.
- **FR-009**: The initial control corpus MUST target approximately 32–128 rows, with row count
  controlled by configuration rather than a hard-coded quota.
- **FR-010**: The builder MUST fail closed when a row lacks its declared input or exact target. Missing
  signals MUST be generation errors, not zero-filled arrays.
- **FR-011**: The existing deterministic compositor/synthesis path MUST remain the signal authority;
  Python may orchestrate and validate but MUST NOT invent a second lighting implementation.
- **FR-012**: The corpus MUST include a machine-readable manifest consumable by Python evaluation. A
  historical v50 store is not a required input.
- **FR-013**: The first model experiment MUST report limited-size learning curves, family metrics,
  a trivial baseline, and ambiguity cases.
- **FR-014**: Real authored minimaps MUST pass through a versioned albedo-normalization operation
  before entering the first model lane.
- **FR-015**: The textureless gate MUST persist thresholds, metrics, and accepted/rejected/quarantined
  decisions. It MUST fail closed on missing, non-finite, or uncalibrated outputs.
- **FR-016**: Initial real transfer MUST use only a tiny explicit `0.x`/`1.x` sample and MUST compare
  its input distribution and failure modes with the controls.
- **FR-017**: Broader real-data processing MUST be blocked until the transfer report explicitly says
  `expand`; synthetic success alone MUST NOT authorize expansion.
- **FR-018**: Additional signals and later client adapters MUST be versioned extensions and MUST NOT
  block the first height proof.
- **FR-019**: Every control row MUST carry one of the four established complexity buckets, and the
  manifest MUST summarize bucket counts and family membership.
- **FR-020**: The visual-review tool MUST fail closed on an invalid corpus or missing required visual
  signals; it MUST NOT fabricate a preview from absent arrays.
- **FR-021**: The visual-review output MUST include both one representative panel per family and a
  variant strip so a human can inspect cross-family coverage and within-family variation before
  training.
- **FR-022**: Cross-tile families MUST carry a stable pattern ID, 2x2 tile coordinates, and
  continuity metadata; validation MUST fail if any of the four positions is absent.
- **FR-023**: The visual-review output MUST include a stitched cross-tile atlas for every configured
  cross-tile family.
- **FR-024**: The object-control extension MUST emit an `objectified_terrain_shadow_256` input,
  exact clean `terrain_shadow_256` target, and `object_contamination_mask_256` target for every
  object-control row.
- **FR-025**: Object controls MUST include no-object, sparse, dense, overlapping, and tile-boundary
  placement regimes with deterministic placement metadata. Broad object families may be labeled, but
  exact client object identity is out of scope for the first sieve experiment.
- **FR-026**: `object_contamination_mask_256` MUST remain a distinct exported signal and auxiliary
  loss target. It MUST NOT be treated as a height loss term or silently substituted for the existing
  `object_geometry_visible_mask_257` numeric geometry target.
- **FR-027**: The sieve experiment MUST compare clean-output-only, auxiliary-mask-loss, and
  predicted-mask-guided variants. A guided variant MUST consume its predicted mask at both training
  and inference; ground-truth masks are loss-side supervision only.
- **FR-028**: Sieve reports MUST provide separate clean-terrain metrics and mask metrics by object
  density, placement regime, and held-out object family.
- **FR-029**: A passing sieve result MUST NOT authorize real-client processing by itself. Real inputs
  still require the versioned albedo-normalization and textureless gates.
- **FR-030**: Non-grid control rows MUST persist deterministic field offsets and alignment mode. The
  validator MUST reject a default corpus whose non-grid families are all chunk-aligned or whose
  multi-variant family has no offset variation.
- **FR-031**: The promoted object-sieve lane MUST read the configured v50 object-library Zarr at
  runtime, MUST leave it unchanged, and MUST record its absolute path, library schema, release,
  eligible row count, `assets.parquet` hash, and `index.parquet` hash.
- **FR-032**: The object-library builder MUST keep library families isolated between train and
  validation and MUST preserve terrain-family holdouts; no library family may cross the derived
  split.
- **FR-033**: Each derived row MUST emit the exact transformed library-mask union as
  `object_contamination_mask_256` and a deterministic `object_instance_id_256` map, with one
  metadata record per placed library object.
- **FR-034**: Library-sieve reports MUST provide clean-terrain error, mask IoU/Dice or equivalent,
  precision, recall, coverage, and regime/family breakdowns. A good aggregate score MUST NOT hide a
  dead non-empty-mask signal.
- **FR-035**: The old curriculum `object_precise_mask`/`object_mask` projections and empty
  `object_geometry_visible_mask_257` evidence MUST remain diagnostic only; neither may be promoted
  as the precision object-library target.
- **FR-036**: The sieve's clean-terrain head MUST preserve the contaminated-input identity baseline
  at initialization and MUST report whether its best clean error beats that baseline. A mask score
  MUST NOT promote a model whose clean output damages uncontaminated terrain relative to identity.
- **FR-036a**: The real-mask lane MUST train mask prediction only until a clean terrain target exists;
  it MUST NOT claim that the v50 mask arrays provide an exact clean-minimap supervision pair.
- **FR-037**: The paired validation lane MUST select authored and synthetic minimap rows from the
  same v50 `source_group_id` and MUST verify identical map/tile identity before pairing them.
- **FR-038**: The v50 synthetic row MUST be labeled as a legacy flat fake-maptexture diagnostic;
  it MUST NOT be labeled or consumed as a `terrain_shadow_256` target.
- **FR-039**: The pair report MUST preserve the train/validation group split and MUST report the
  authored-vs-flat-synthetic absolute difference independently from object-mask metrics.
- **FR-040**: A terrain-shadow comparison MUST accept only a fresh NPZ containing
  `terrain_shadow_256` emitted by the post-fix C# compositor, with finite/range checks and explicit
  producer provenance. Missing fresh shadow artifacts MUST fail closed.
- **FR-041**: The promoted marker corpus MUST derive positive candidate examples from the real v50
  object-library `capture_rgb`/`capture_mask` pairs and MUST preserve the corresponding `library_id`
  and transform/footprint provenance. The rejected v50 curriculum dot projections MUST NOT be used
  as precision identity targets.
- **FR-042**: The marker model MUST consume a 256x256 minimap image plus one 256x256 candidate
  footprint channel and MUST emit at least a known-object confidence and a fixed-length retrieval
  embedding. The ground-truth mask, library ID, or gallery match MUST NOT be an input channel.
- **FR-043**: Exact identity MUST be resolved by nearest-neighbour retrieval against a frozen,
  provenance-bound v50 library gallery; a flat 5,349-way classifier is not the primary identity
  contract.
- **FR-044**: Marker export MUST write `known_object_marker_256` as an integer instance map with
  zero meaning background/unaccepted candidate, plus an identity table with marker instance ID,
  library ID, asset path, candidate footprint bounds/coverage, known confidence, retrieval score,
  and rejection reason when applicable.
- **FR-045**: Marker train/validation splits MUST isolate library families and terrain control
  families. Positive, shifted/empty, and unknown candidates MUST be represented and reported
  separately so identity quality cannot be hidden by candidate prevalence.
- **FR-046**: Marker evaluation MUST report known/unknown precision, recall, and coverage together
  with top-1/top-k library retrieval accuracy and per-family breakdowns. A marker model is not
  promoted from aggregate loss alone.
- **FR-047**: Marker inference MUST fail closed when the input image, candidate footprint, gallery,
  or checkpoint contract is missing, non-finite, shape-incompatible, or provenance-mismatched.
- **FR-048**: The sieve MAY consume the marker map as an optional predicted guidance artifact, but
  marker identity and sieve cleaning MUST remain independently checkpointed and independently
  ablatable.
- **FR-049**: A candidate instance with no visible pixels in the source per-instance map MUST be
  excluded from marker training, recorded as `occluded_or_overwritten_in_instance_id_map`, and
  MUST NOT cause the entire marker corpus build to fail. A failed build MUST NOT present a partial
  output directory as a valid corpus.

### Non-Functional Requirements

- **NFR-001**: Generation and normalization MUST be deterministic for fixed inputs, versions, and
  seeds.
- **NFR-002**: The control corpus MUST be buildable and inspectable before any GPU run.
- **NFR-003**: Manifests and reports MUST make provenance, thresholds, split membership, decisions,
  and array hashes auditable without opening binary arrays.
- **NFR-004**: Heavy synthesis, real-client processing, and training remain user-owned operations;
  repository work prepares commands and performs lightweight validation only.

## Success Criteria

1. A deterministic 32–128-row control corpus exists with exact `terrain_shadow_256` and `height_257`
   pairs and held-out families.
2. A limited control experiment shows whether the input/height relationship beats a tile-mean
   baseline, with metrics by family and training size.
3. The albedo operation distinguishes valid textureless outputs from textured or failed outputs and
   persists a fail-closed gate report.
4. A tiny accepted `0.x`/`1.x` transfer sample produces a separate domain/metric report before any
   expansion decision.
5. No full v50/v60 historical harvest is required to decide whether the first reconstruction lane
   is viable.
6. The default control run produces an inspectable atlas with complete expected-family and
   complexity-bucket coverage, or the report explicitly names what is missing.
7. The cross-tile atlas shows the lightning/burn motifs continuing across all four tile seams.
8. The parked object-sieve control report shows separate clean-terrain and contamination-mask metrics,
   including boundary-crossing and dense-object holdouts, with no ground-truth mask input leakage.
9. The real-mask report shows provenance-backed authored v50 rows, independent precise/coarse mask
   metrics, and no train/validation source-group leakage before any user-run GPU training.
10. A small same-tile authored/flat-synthetic validation report exists with pair identity, absolute
    difference statistics, visual atlas, and an optional post-fix terrain-shadow comparison; it must
    explicitly state that the legacy synthetic image is not a shadow target.
11. A marker control corpus and held-out report exist in which each candidate has known/unknown
    labels, library provenance, and family-isolated retrieval metrics.
12. Marker export produces a nonzero instance map only for accepted candidates and a sidecar table
    that resolves every nonzero marker instance to a known library ID and score.
13. An unknown/shifted-candidate control is rejectable, and no marker result depends on the old
    dot-like v50 curriculum projections.

## Key Entities

See [data-model.md](./data-model.md) for `ControlSourceManifest`, `SyntheticControlRow`,
`ObjectSieveControlRow`, `ObjectLibrarySieveRow`, `ObjectMarkerCandidate`, `ObjectMarkerMap`,
`ObjectSieveExperiment`, `RealObjectMaskDataset`,
`RealObjectMaskExperiment`, `RealSyntheticValidationPair`, `RealSyntheticPairReport`,
`AlbedoOperationRun`, `TexturelessGateDecision`, `ExperimentRun`, and `TransferGate`.

## Assumptions

1. The current compositor can synthesize the canonical textureless input and exact height target
   from known terrain.
2. Real authored minimap albedo removal is a separate operation whose quality must be measured; it
   is not assumed to exist merely because synthetic textureless rendering exists.
3. The existing v50 object library captures/masks are valid supervision for object appearance and
   identity-marker controls, while the
   empty v50 geometry-visible mask and absence of a clean minimap target are explicit limitations.
4. The v50 mixed curriculum's `minimap_source=synthetic` row is a same-terrain flat fake maptexture
   useful for absolute-difference diagnostics only. It is not a post-fix terrain-shadow render.
5. A small number of distinct terrain families is more useful for the first proof than a large,
   mixed historical corpus.
6. Control success is necessary but not sufficient for real-domain transfer.
7. Later client builds may eventually be added through adapters, but they are out of scope for the
   initial transfer route.
8. The user runs full synthesis, client processing, and training commands; Codex prepares them and
   validates lightweight code paths.
9. The first marker slice receives candidate footprints explicitly. Proposal recall and automatic
   footprint discovery are separate later work, not silently claimed by marker precision/recall.
