# Data Model: Terrain Paste and Fractal Motif Archaeology

## ObservationWindow

One normalized spatial query window. It may cross a tile or chunk boundary.

- `window_id`: deterministic identifier
- `source_kind`: `synthetic_control`, `real_client`, or `derived_review`
- `source_group_id`: split ownership key
- `map_id`, `tile_x`, `tile_y`: source location when available
- `window_origin_uv`: fractional source origin
- `window_size_uv`: normalized extent
- `boundary_context`: neighboring-window and tile/chunk boundary metadata
- `transform`: translation, rotation, mirror, and scale metadata
- `pattern_id`: synthetic known family ID when available
- `availability`: per-signal availability state

## SignalBundle

The signals associated with an observation window.

- `clean_observation_luma_256`: normalized observation luma
- `clean_observation_gradient_256`: deterministic x/y gradients
- `clean_observation_confidence_256`: albedo-normalization confidence
- `height_257`: client-backed or synthetic reference, validation-only for deployment
- `alpha_layers`: ordered alpha summaries or full masks when available
- `texture_layer_ids`: ordered layer and tileset identifiers
- `tileset_auxiliary`: optional normal/specular/depth-like channel references
- `object_slots`: optional normalized footprints or placement slots
- `validity_masks`: per-signal validity masks

## AlphaEvidenceBundle

The preserved alpha substrate and its non-destructive derived views.

- `source_layers`: every available alpha layer with raw payload reference
- `layer_provenance`: MCLY layer ID, MCAL offset, texture ID, tile/map/build, and decoder identity
- `raw_occupancy_view`: thresholded or continuous occupancy with threshold provenance
- `transition_view`: edge, stroke, and alpha-gradient evidence
- `atomic_view`: localized connected components or equivalent fine candidates
- `paste_block_view`: grouped middle-scale regions and member references
- `macro_context_view`: broad full-map parent regions and boundary context
- `ordered_layer_view`: cumulative and incremental layer additions
- `cross_tile_view`: relationships extending beyond the current tile/window
- `availability`: per-view `available`, `unavailable`, or `invalid`
- `confidence`: per-view confidence; unavailable is not an empty mask
- `content_hash`: hash of preserved source references and derivation configuration

## TilesetProfile

- `profile_id`: deterministic profile identifier
- `build_family`: client/build provenance
- `texture_ids`: ordered identifiers
- `alpha_layout_descriptor`: layer count, occupancy, transition, and edge summaries
- `appearance_descriptor`: albedo and optional auxiliary statistics
- `geometry_correlation`: measured relationship to height/fractal descriptors
- `confidence`: profile confidence

## MotifCandidate

- `candidate_id`: deterministic identifier
- `descriptor_version`: descriptor contract version
- `source_window_id`: origin window
- `matched_window_id`: candidate match
- `matched_signals`: height, alpha, texture, observation, auxiliary, or object slots
- `transform`: estimated translation, rotation, mirror, and scale
- `alignment_error`: spatial mismatch score
- `cross_boundary_support`: boundary continuity score
- `confidence`: calibrated match confidence
- `status`: `unconfirmed`, `recurring`, `rejected`

## PasteFamily

- `family_id`: stable family identifier
- `member_candidate_ids`: matching candidate list
- `source_group_ids`: distinct source groups represented
- `recurrence_count`: number of accepted occurrences
- `variant_axes`: transforms and controlled mutations
- `split_assignment`: train/validation/test family ownership
- `evidence_summary`: human-readable provenance

## BrushScaleRecord

A scale-specific view of the same terrain evidence. The three scales are intentionally separate.

- `record_id`: deterministic identifier
- `scale`: `atomic_brush`, `paste_block`, or `macro_prefab_context`
- `source_window_id` or `map_canvas_id`: spatial owner
- `bbox_uv`: normalized extent, including cross-tile extent when available
- `parent_record_ids`: broader records containing this record
- `child_record_ids`: narrower records contained by this record
- `descriptor_refs`: scale-specific descriptor IDs
- `layer_refs`: MCLY/MCAL and ordered-layer provenance
- `height_normal_refs`: optional coupled relief evidence
- `status`: `candidate`, `recurring`, `composite`, `one_off`, `rejected`, or `unavailable`
- `confidence`: calibrated scale-specific confidence
- `provenance`: decoder, build, map, tile, and extraction implementation

## DifficultyGuidance

Curriculum guidance from a frozen synthetic reference model. It is not a truth or freshness label.

- `guidance_id`: deterministic identifier
- `reference_model_id`: checkpoint and architecture identity
- `reference_corpus_hash`: synthetic corpus hash used for the reference model
- `score_config_hash`: scoring and banding configuration
- `per_signal_error`: independent signal metrics
- `seam_boundary_error`: cross-tile and edge metrics
- `confidence_coverage`: confidence and valid-pixel coverage metrics
- `difficulty_band`: `easy`, `learnable_hard`, or `pathological`
- `sampling_weight`: curriculum weight only
- `not_staleness`: literal `true`

## PaintOrderHypothesis

An evidence record describing the likely authored sequence without claiming literal editor history.

- `hypothesis_id`: deterministic identifier
- `base_layer`: opaque MCLY layer 0 and its base texture identity
- `first_paint_layer`: MCLY layer 1, its texture identity, MCAL offset, and availability
- `ordered_layers`: layer 1+ MCLY IDs, texture IDs, MCAL offsets, and availability
- `incremental_occupancy`: derived newly occupied regions per ordered layer
- `cumulative_occupancy`: derived union/coverage evidence
- `paste_family_refs`: recurring motif families associated with each addition
- `relief_features`: curvature, slope, edge, and height evidence
- `relationship_status`: `intact`, `retextured`, `resculpted`, `unknown`, or `insufficient_data`
- `paint_relief_score`: measured association, not causal proof
- `confidence`: calibrated hypothesis confidence
- `provenance`: source/build and decoder identity

## GuidanceBundle

- `bundle_id`: deterministic hash
- `window_id`: consumer window
- `alpha_evidence`: optional preserved source/derived alpha bundle and confidence
- `tileset_profile`: optional profile reference and confidence
- `motif_scaffold`: optional spatial scaffold and transform
- `brush_scale_records`: optional linked atomic, paste-block, and macro-context records
- `paint_scaffold`: optional ordered paint/sculpt-intent scaffold
- `fractal_descriptor`: optional multiscale descriptor
- `object_evidence`: optional slot/footprint evidence
- `difficulty_guidance`: optional curriculum-only score; never a deployment target
- `uncertainty`: per-field confidence and absence markers
- `source_provenance`: all contributing source IDs
- `ablation_mode`: `parity`, `motif_guided`, `tileset_guided`, or `combined`

## PipelineRun

- `run_id`, `seed`, `config_hash`, `corpus_hash`
- `stage_versions`
- `split_manifest_hash`
- `retrieval_metrics`
- `reconstruction_metrics`
- `visual_review_paths`
- `failure_counts`
