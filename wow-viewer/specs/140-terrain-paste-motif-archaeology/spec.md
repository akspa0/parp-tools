# Feature Specification: Terrain Paste and Fractal Motif Archaeology

**Feature Branch**: `140-terrain-paste-motif-archaeology`

**Created**: 2026-08-10

**Status**: Draft

**Input**: User description: "Treat recurring terrain pastes, fractal brushes, cross-tile patterns, tileset auxiliary signals, and object placement evidence as separate signals in a multi-stage minimap-to-terrain pipeline. Use the synthetic FBM/fractal controls and a small amount of real data to learn the workflow without asking one model to regenerate every signal at once."

## Context

The reconstruction problem is not one image-to-image task. A minimap can contain albedo, terrain lighting, height-dependent shape, texture-layer boundaries, repeated terrain motifs, and placed-object contamination. These signals have different observability and should not be forced into one monolithic target.

The leaked modern workflow map is evidence that terrain production used reusable biome clusters, brush-like terrain patterns, and pasted assemblies. It is useful workflow evidence, but it does not prove that 0.x or 1.x stored the same artifacts or used the same editor. The initial system therefore tests recurrence in the actual client data and synthetic controls rather than assuming a historical implementation.

This feature is the archaeology and guidance lane alongside Spec 139. Spec 139 remains the owner of the clean-signal terrain reconstruction model. This feature owns the evidence that can guide it: recurring motifs, cross-tile context, tileset profiles, and confidence-bearing scaffolds.

### Authoring-order hypothesis

The initial authoring order is now treated as a falsifiable hypothesis:

1. establish a base tileset or “brain” texture;
2. paste or stamp recurring rocky and mountain motifs;
3. paint those motifs into the alpha stack, especially concentrated rock and slope regions;
4. sculpt the heightmap to refine the painted plan;
5. repaint or refine grass, highlights, transitions, and objects.

Under this hypothesis, the ordered alpha stack is an upstream latent description of terrain intent.
Layer 0 is the opaque base tileset and is not a paste signal. Layer 1 is the first alpha-bearing
layer and the first candidate for pasted/painted rock or mountain intent. Source-side alpha/MCAL
data is supervision and archaeological evidence; a minimap-only deployment must predict a
confidence-bearing paint scaffold before geometry reconstruction. The implementation must preserve
the distinction between opaque layer 0 and alpha-bearing layer 1 rather than inventing an
`alpha_0` tensor.

### Complementary brush and paste scales

The early Python brush extractor and the later C# full-map segmentation are complementary
representations of the same authored terrain, not competing implementations:

- **Atomic brush** is a localized connected alpha component or hand-painted rock-surface patch.
  It is useful for shape-level descriptors, small brush candidates, and fine correspondence.
- **Paste block** is a middle-scale grouped arrangement, such as a blocky 16x16-cell region or a
  transformed local assembly. It preserves nearby atomic members and is a candidate reusable unit.
- **Macro prefab context** is a broad full-map parent region that preserves cross-tile placement,
  neighboring relationships, mesh response, and tileset context. It is not an atomic brush label.

The pipeline MUST retain all three scales and their parent/child links. Python atomic components
must not be promoted to complete prefabs solely because they are connected, and C# macro or blocky
regions must not be discarded as bugs merely because they contain multiple atomic components. Each
scale receives separate descriptors, retrieval metrics, review labels, and confidence. A guidance
bundle may carry a macro parent, paste-block children, and atomic evidence together.

### Alpha-first evidence rule

Alpha masks are the primary authored-terrain evidence surface for this lane. The extractor MUST
preserve the complete available alpha payload and its layer, offset, texture, tile, and build
provenance before deriving any summary or mask. Derived views are a fan-out, not a destructive
choice: the same source may be represented as raw occupancy, alpha transitions, atomic connected
components, grouped paste blocks, macro parent regions, cumulative/incremental layer additions,
and cross-tile relationships. A view that is unhelpful for one terrain family must not erase the
other views. Missing or opaque data remains explicitly unavailable; it is never replaced with a
zero mask and never used to claim that the tile contains no authored structure.

## User Scenarios & Testing

### User Story 1 - Build a motif atlas from controlled and real evidence (Priority: P1)

A researcher can inspect a visual atlas of terrain windows showing normalized observation, height, alpha layers, texture-layer identity, auxiliary tileset channels when present, and object-placement evidence side by side.

**Why this priority**: The project needs to verify that the proposed signals actually vary and recur before training another model.

**Independent Test**: Run the atlas builder on the synthetic control corpus and a small real 0.x/1.x slice; confirm that each row has provenance, signal availability, transform metadata, and a visual review panel.

**Acceptance Scenarios**:

1. **Given** a synthetic FBM, lightning, island, sheer-dropoff, or cross-tile control, **when** it is indexed, **then** its source family, variant, pattern ID, and boundary context are visible in the atlas.
2. **Given** a real tile with missing alpha, auxiliary, or object data, **when** it is indexed, **then** the atlas marks that signal unavailable rather than fabricating it.
3. **Given** two windows from the same source pattern, **when** rendered, **then** their transforms and shared-pattern relationship are visible.

### User Story 2 - Detect and retrieve recurring paste families (Priority: P1)

A researcher can search the corpus for repeated terrain fragments that may have been copied, rotated, mirrored, scaled, or placed across tile and chunk boundaries.

**Why this priority**: A retrieved paste scaffold can provide high-value spatial context to the terrain model without requiring the model to rediscover every repeated brush from pixels alone.

**Independent Test**: Plant known transformed motifs across synthetic tile boundaries, search the corpus, and verify that the correct family ranks above unrelated controls with the transform and confidence reported.

**Acceptance Scenarios**:

1. **Given** a motif crossing a tile boundary, **when** searched from either partial tile, **then** the same family is retrieved using the neighboring context.
2. **Given** a rotated or mirrored motif, **when** searched, **then** the transform is reported instead of treating it as a new family.
3. **Given** unrelated fractal controls with similar global complexity, **when** searched, **then** local structure and cross-channel evidence prevent a false match.
4. **Given** a motif seen only once, **when** indexed, **then** it remains an unconfirmed candidate and is not promoted to a reusable paste family.

### User Story 3 - Separate tileset identity from terrain geometry (Priority: P1)

A researcher can identify tileset or biome-family evidence independently from the geometry reconstruction signal, using texture-layer identity, alpha layout, albedo, and auxiliary channels when they exist.

**Why this priority**: Tileset appearance, alpha painting, specular behavior, and terrain shape are related but not interchangeable. Keeping them separate prevents a texture cue from being mistaken for height evidence.

**Independent Test**: Mix controls with the same geometry but different authored tileset profiles and controls with the same tileset profile but different geometry; confirm that the profile and geometry descriptors remain independently queryable.

**Acceptance Scenarios**:

1. **Given** the same height pattern with two tileset profiles, **when** encoded, **then** the geometry descriptor remains stable while the tileset profile changes.
2. **Given** a tileset with an alpha or specular-like auxiliary channel, **when** measured, **then** the channel is recorded as evidence with an availability and correlation score, not as an assumed depth target.
3. **Given** a missing or unsupported auxiliary channel, **when** processed, **then** the profile remains valid with an explicit missing-signal marker.

### User Story 4 - Feed bounded evidence into clean-signal reconstruction (Priority: P2)

A researcher can compare the clean-signal terrain model with and without a retrieved motif or tileset scaffold, using the same held-out families and the same real transfer slice.

**Why this priority**: The purpose of archaeology is to improve reconstruction, but it must earn its place through ablation rather than intuition.

**Independent Test**: Run the Spec 139 model in parity, motif-guided, and tileset-guided modes on held-out synthetic families; report per-signal metrics, seam metrics, and confidence coverage for each mode.

**Acceptance Scenarios**:

1. **Given** a low-confidence or unmatched motif, **when** passed downstream, **then** the terrain model receives an explicit absence/uncertainty state and does not receive a fabricated scaffold.
2. **Given** a validated motif match, **when** passed downstream, **then** its transform, family ID, source provenance, and confidence accompany the scaffold.
3. **Given** a real albedo-normalized minimap with a client-backed height reference, **when** evaluated, **then** the report distinguishes input-only inference from reference-only validation.

### User Story 5 - Keep object placement as an optional later signal (Priority: P3)

A researcher can add normalized object footprints or placement slots to a motif record without making object identity a prerequisite for terrain reconstruction.

**Why this priority**: Objects may reveal pasted workflow patterns, but the earlier dot-like object mask result is not sufficient evidence to make object identity part of the first terrain model.

**Independent Test**: Index controls with none, sparse, dense, overlap, and boundary-crossing placements; verify that object evidence can be enabled or omitted without changing the terrain or motif contracts.

**Acceptance Scenarios**:

1. **Given** a motif with different object assets placed in equivalent slots, **when** compared, **then** the structural slot pattern can match while exact asset identity remains separate.
2. **Given** an object-free tile, **when** processed, **then** the object signal is an explicit empty state and does not affect terrain confidence.

### User Story 6 - Infer paint order and sculpt intent (Priority: P1)

A researcher can inspect whether ordered alpha-layer additions, recurring pastes, and later relief
are consistent with a painted-plan-then-sculpt workflow, including where retexturing broke that
relationship.

**Why this priority**: This is the proposed missing intermediate representation between a minimap
and a heightmap. If it is useful, the geometry model no longer has to rediscover the terrain’s
artist-authored structure from lighting alone.

**Independent Test**: Run the analyzer on synthetic controls with known paint sequences and on a
small real 0.x/1.x slice; verify layer-order evidence, paste membership, height relationship, and
an intact/retextured/resculpted/unknown status are reported separately.

**Acceptance Scenarios**:

1. **Given** synthetic controls where opaque layer 0 is followed by layer-1 rock and later slope additions, **when** analyzed, **then** the ordered additions and their spatial regions are recovered deterministically.
2. **Given** a real tile whose opaque layer 0 has no alpha payload, **when** analyzed, **then** layer 0 is represented as the base tileset and layer 1 is inspected as the first paste/paint candidate without inventing an `alpha_0` array.
3. **Given** a retextured zone with preserved relief but changed alpha relationships, **when** analyzed, **then** the relationship is marked broken/retextured rather than forcing the current alpha to explain the old height.
4. **Given** a validated paint scaffold, **when** passed to Spec 139, **then** it is marked as an inferred intermediate with confidence and provenance, not as ground-truth height.

## Edge Cases

- A pattern begins outside the current tile and only a fragment is visible.
- The same paste is rotated, mirrored, scaled, or shifted by a non-chunk-aligned offset.
- A fractal-looking alpha boundary is unrelated to height and must not become a geometry prior.
- A flat tile contains a strong texturing pattern but no recoverable relief.
- A steep or discontinuous height feature has no matching alpha or tileset change.
- Adjacent tiles use different tileset families and the blend crosses the tile edge.
- Auxiliary texture channels are absent, encoded differently, or visually flat.
- A real tile has objects, water, or other contamination that must be represented as unavailable or separate evidence.
- A repeated motif is caused by a generic fractal statistic rather than a copied spatial arrangement.
- A source group appears in both training and validation through overlapping windows; the split must fail closed.
- Opaque layer 0 is present and layer 1 is missing or has no usable alpha payload.
- Alpha layer order is preserved while terrain was later resculpted or repainted.
- An alpha addition resembles a mountain brush but has no corresponding height change.
- Height relief survives while the original alpha layer is missing or replaced.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST represent terrain observation, geometry, alpha/texturing, tileset identity, motif recurrence, and object placement as separately available signals.
- **FR-002**: The system MUST index windows that cross tile and chunk boundaries and MUST preserve the source coordinates needed to reconstruct their context.
- **FR-003**: The system MUST include deterministic synthetic controls for smooth, flat, mountainous, island, sheer-dropoff, FBM/ridged, lightning/dendritic, burn-like, and cross-tile patterns.
- **FR-004**: The system MUST support deterministic transform variants including arbitrary sub-cell translation, rotation, mirroring, and scale where the source contract permits them.
- **FR-005**: The system MUST produce a visual atlas showing the available signals and their provenance for every reviewed motif candidate.
- **FR-006**: The system MUST distinguish a confirmed recurring paste family from a single occurrence or an unconfirmed similarity.
- **FR-007**: Motif matching MUST use spatial evidence at more than one scale and MUST report the transform, source group, matched signals, and confidence.
- **FR-008**: The system MUST prevent overlapping windows from the same source family or copied pattern from crossing the train/validation boundary.
- **FR-009**: The system MUST record texture-layer IDs, alpha-mask summaries, and tileset-family evidence independently from height targets.
- **FR-010**: Auxiliary channels such as specular-like or depth-related texture evidence MUST be optional, provenance-bearing, and empirically correlated before being used as guidance.
- **FR-011**: The downstream guidance bundle MUST be usable by Spec 139 in parity, motif-guided, and tileset-guided ablations.
- **FR-012**: Missing, flat, or unsupported signals MUST be represented explicitly and MUST NOT be silently replaced with target-derived values.
- **FR-013**: Every guidance bundle MUST include a deterministic content hash, source provenance, transform metadata, and confidence for each included hypothesis.
- **FR-014**: Object evidence MUST remain an optional auxiliary channel and MUST not be required for the first terrain reconstruction model.
- **FR-015**: The pipeline MUST report per-signal metrics and separate motif retrieval quality from terrain reconstruction quality.
- **FR-016**: Real 0.x/1.x data MUST be usable as a small transfer or validation slice without requiring a broad legacy harvest to be treated as training truth.
- **FR-017**: The system MUST preserve logical layer order, MCLY layer identity, and MCAL offset provenance, including the invariant that layer 0 is the opaque base tileset and layer 1 is the first alpha-bearing paste/paint candidate for the targeted client profile.
- **FR-018**: The system MUST derive ordered alpha evidence as cumulative and incremental occupancy hypotheses, without claiming that the derived increments are literal editor operations unless independently proven.
- **FR-019**: The system MUST report the relationship between paint evidence and relief as `intact`, `retextured`, `resculpted`, `unknown`, or `insufficient_data`.
- **FR-020**: The system MUST expose a paint/sculpt-intent scaffold as an intermediate output that can guide Spec 139, while keeping source-side alpha arrays out of the minimap-only deployment input contract.
- **FR-021**: The system MUST measure whether alpha additions, paste families, curvature, and relief improve reconstruction independently; a correlation MUST NOT be treated as causal authoring proof.
- **FR-022**: The system MUST preserve atomic-brush, paste-block, and macro-prefab-context records as separate scales with separate labels, metrics, and confidence.
- **FR-023**: The system MUST link records across scales using spatial overlap, layer order, map/tile provenance, and available height/normal or texture evidence, while allowing a record to have no parent or children.
- **FR-024**: A frozen synthetic reference-model score MAY guide curriculum sampling through reproducible `easy`, `learnable_hard`, and `pathological` bands, but MUST NOT be used as a staleness indicator, pseudo-target, or provenance substitute.
- **FR-025**: Difficulty guidance MUST report per-signal error, seam/boundary error, confidence, and coverage so that one aggregate score cannot hide a dead or pathological signal.
- **FR-026**: The system MUST preserve every available alpha layer payload and its layer, offset, texture, tile, map, build, and decoder provenance before generating derived descriptors.
- **FR-027**: The system MUST expose raw occupancy, transition/stroke, atomic, paste-block, macro-context, ordered-layer, and cross-tile alpha interpretations as independently selectable views.
- **FR-028**: No alpha interpretation may erase another interpretation or silently convert unavailable/opaque data into an empty mask; each view MUST carry its own availability and confidence.

### Key Entities

- **ObservationWindow**: A spatially bounded, possibly cross-tile window with normalized observation, coordinates, source group, and boundary context.
- **SignalBundle**: The available height, albedo-normalized observation, gradients, alpha summaries, texture-layer identity, auxiliary channels, and object evidence for one window.
- **AlphaEvidenceBundle**: The lossless available alpha layers and provenance plus independently derived occupancy, transition, atomic, block, macro, ordered-layer, and cross-tile views.
- **TilesetProfile**: A versioned description of the texture family, layer ordering, alpha statistics, albedo statistics, and optional auxiliary channels.
- **MotifCandidate**: A repeated-pattern hypothesis with descriptors, matched channels, transform, source provenance, and confidence.
- **PasteFamily**: A confirmed or unconfirmed group of motif candidates with recurrence counts and held-out split ownership.
- **BrushScaleRecord**: A scale-specific atomic, paste-block, or macro-prefab-context record with spatial extent, parent/child references, descriptors, provenance, and confidence.
- **GuidanceBundle**: The bounded output consumed by Spec 139, containing optional motif and tileset scaffolds plus uncertainty and provenance.
- **DifficultyGuidance**: A reproducible curriculum-only assessment from a frozen reference model, including per-signal errors, seam/boundary errors, confidence, coverage, and a difficulty band.
- **PipelineRun**: The deterministic run manifest recording corpus hashes, configuration, stage outputs, metrics, and ablation mode.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The synthetic control atlas visibly covers every required terrain regime and every cross-tile family, with no duplicate-position or provenance violations.
- **SC-002**: On a transformed synthetic retrieval benchmark, the intended motif family ranks first for at least 90% of queries and the transform error is reported within the configured tolerance.
- **SC-003**: On a held-out-family benchmark, confirmed motif retrieval beats a complexity-only nearest-neighbor baseline on both family precision and boundary continuity.
- **SC-004**: At least one recurring motif family is identified independently in real 0.x/1.x evidence with matches from multiple source groups, or the report explicitly records that recurrence was not proven.
- **SC-005**: The motif-guided Spec 139 ablation improves held-out-family seam and structural metrics over parity without degrading clean-observation performance beyond the predeclared tolerance.
- **SC-006**: The tileset-guided ablation reports separate tileset classification and terrain reconstruction metrics; neither signal may hide a dead output behind an aggregate score.
- **SC-007**: Every reviewed real sample states whether its height, alpha, auxiliary, and object channels are client-backed, synthesized, unavailable, or used only for validation.
- **SC-008**: Repeating the same run with the same corpus and seed produces identical motif IDs, transforms, hashes, split assignments, and reports.
- **SC-009**: On synthetic controls with known paint order, the analyzer recovers opaque layer 0 as the base and layer 1 plus later layers as ordered additions deterministically.
- **SC-010**: On real 0.x/1.x evidence, the report can distinguish an intact paint/relief relationship from a retextured or resculpted relationship, or explicitly records that the distinction is unproven.
- **SC-011**: A Spec 139 paint-scaffold ablation reports whether the intermediate improves held-out-family structure and seams over parity without using source-side alpha as a deployment input.
- **SC-012**: The visual atlas can show at least one linked atomic brush, paste block, and macro parent across synthetic cross-tile controls, while preserving unlinked and boundary-truncated cases.
- **SC-013**: Re-running curriculum scoring with the same frozen reference checkpoint, corpus, and seed produces identical difficulty bands and per-signal guidance without changing labels, provenance, or validation freshness state.
- **SC-014**: An alpha coverage audit proves that every available source layer is either preserved or explicitly reported unavailable, and that no derived view silently replaces another view or fabricates an empty mask.

## Assumptions

- The synthetic FBM/ridged and dendritic controls are useful hypotheses because they resemble observed terrain structure, but resemblance alone is not treated as proof of a historical brush implementation.
- Existing client readers and v60 control-corpus tooling remain the source of decoded arrays; this feature does not rewrite file parsers or the renderer.
- Albedo stripping is an observation-normalization step. It does not create missing geometry evidence and must carry a confidence map.
- Texture alpha and auxiliary channels may correlate with terrain depth, but their semantics must be measured per client/build and may be absent.
- Exact object identity is deferred; normalized footprint or slot structure is the initial object evidence.
- The first implementation is classical/descriptive retrieval plus small ablations. A large end-to-end multi-model training run is out of scope until the recurrence and visual gates pass.
- A frozen synthetic reference model can provide difficulty guidance for curriculum sampling. Its error is not a dataset-staleness signal and cannot rewrite labels, provenance, or split ownership.
- The 10.2 workflow-map screenshot is corroborating workflow evidence only; the 0.x/1.x client data remains the authority for the initial reconstruction contract.
- “Paint first, sculpt second” is a motivating hypothesis. The system must test opaque layer 0, alpha-bearing layer 1, later layer order, spatial recurrence, and relief correlation rather than treating the developer map as direct 0.x/1.x ground truth.

## Out of Scope

- Reconstructing a complete modern Blizzard terrain editor.
- Treating WDL priors, target-derived height hints, or object masks as deployment inputs for the clean-signal model.
- Making exact tileset names, object identities, or historical authoring intent a prerequisite for height reconstruction.
- Starting a broad multi-client harvest or GPU training run before the control atlas and retrieval gates are accepted.
