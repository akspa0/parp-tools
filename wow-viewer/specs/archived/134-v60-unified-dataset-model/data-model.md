# Data Model: Controlled Synthetic-to-Real Experiment

## ControlSourceManifest

The explicit source/configuration authority for one reproducible run.

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | Manifest schema, currently `v60-control-source-v1` for source-backed runs or the emitted `v60-control-corpus-v1` for procedural controls. |
| `source_era_policy` | string[] | Allowed client eras; initially `0.x`, `1.x`. |
| `seeds` | object[] | Explicit procedural families or client-backed terrain seed identities. |
| `families` | string[] | Complete split groups. |
| `generator_version` | string | Exact synthesis/normalization implementation version. |
| `generation_seed` | integer | Deterministic variant seed. |
| `signal_contract` | string[] | Signals emitted for the run. |

No source manifest means no client-backed real-data operation. A client root is configuration, not a
license to recursively harvest it.

## SyntheticControlRow

One exact input/target pair from the control generator.

| Field | Type | Meaning |
|---|---|---|
| `row_id` | string | Stable row identifier. |
| `source_kind` | string | `procedural_control` or a versioned client-seed kind. |
| `source_seed_id` | string | Terrain seed identity when available. |
| `control_family` | string | Holdout grouping unit. |
| `variant_parameters` | object | Height, lighting, and albedo controls. |
| `input` | string | Initially `terrain_shadow_256`. |
| `target` | string | Initially `height_257`. |
| `split` | enum | `train`, `validation`, or `test`; complete families stay together. |
| `array_hashes` | object | SHA-256 for every emitted array. |
| `npz` | string | Row payload path. |

The current procedural emitter uses `variant`, `input_sha256`, and `target_sha256` as compact
manifest fields; later source-manifest support may add the normalized fields above while preserving
backward-readable v60-control-v1 rows.

## ObjectSieveControlRow

One deterministic terrain control with synthetic object contamination and exact decomposition targets.

| Field | Type | Meaning |
|---|---|---|
| `row_id` | string | Stable row identifier. |
| `terrain_control_row_id` | string | Source control row before object overlay. |
| `object_family` | string | Broad procedural family such as tree, rock, building, bridge, or cluster. |
| `placement_regime` | enum | `none`, `sparse`, `dense`, `overlap`, or `boundary_crossing`. |
| `objectified_terrain_shadow_256` | float32[256,256] | Terrain shadow after controlled object rendering. |
| `terrain_shadow_256` | float32[256,256] | Exact clean terrain-only target. |
| `object_contamination_mask_256` | uint8[256,256] | Pixels requiring object removal/inpainting, including configured object effects. |
| `placement_metadata` | object | Deterministic positions, scale, rotation, and object-family parameters. |
| `split` | enum | `train`, `validation`, or `test`; complete regimes/families stay together. |

`v60-object-library-sieve-v1` is the promoted object corpus. It is separate from the historical
procedural `object-sieve-v1` rows below.

## ObjectLibrarySieveRow

One clean terrain control with one or more real v50 object-library captures composited into it.

| Field | Type | Meaning |
|---|---|---|
| `source_library` | object | Read-only Zarr path, schema, release, eligible count, and parquet hashes. |
| `objectified_terrain_shadow_256` | float32[256,256] | Clean terrain shadow with real library object pixels composited in. |
| `terrain_shadow_256` | float32[256,256] | Unmodified clean terrain control target. |
| `object_contamination_mask_256` | float32[256,256] | Exact union of transformed `capture_mask` silhouettes. |
| `object_instance_id_256` | uint16[256,256] | Per-pixel placed-object identity; zero is background. |
| `object_instances` | object[] | Library ID, asset path/family, source row, scale, rotation, and screen-space centre. |
| `split` | enum | Terrain split plus library-family isolation; no library family crosses train/validation. |

The derived row is immutable and never writes back into the v50 object library. The ground-truth
mask and instance map are loss-side/evaluation targets only; neither is supplied to the sieve model
as an input channel.

The contamination mask is intentionally not the existing `object_geometry_visible_mask_257`.
That 257-grid target describes visible object geometry for numeric terrain supervision; this 256-grid
target describes screen-space minimap contamination that the sieve must remove.

## ObjectMarkerCandidate

One image-plus-footprint candidate used by the separate object identification specialist.

| Field | Type | Meaning |
|---|---|---|
| `row_id` | string | Stable candidate row identifier. |
| `minimap_rgb_256` | float32[3,256,256] | Image presented to the marker model; synthetic controls may use the canonical textureless terrain as the background. |
| `object_candidate_mask_256` | uint8[256,256] | One proposed object footprint; it is an input proposal, not the identity target. |
| `known_object` | uint8 | `1` for a candidate generated from a known library capture, `0` for shifted/empty/unknown controls. |
| `library_id` | string/null | Target identity for positive training/evaluation rows; never supplied to the model. |
| `library_family` | string/null | Family split key; no family crosses train/validation. |
| `placement_metadata` | object | Source row, scale, rotation, centre, and clipping provenance. |
| `split` | enum | `train`, `validation`, or `test`. |

The first corpus derives positive candidates from the exact per-instance masks and real capture RGB
records in `v60-object-library-sieve-v1`. Shifted/empty candidates provide negative knownness
examples without pretending the rejected v50 curriculum dots are precise silhouettes.

## ObjectMarkerMap

The export artifact that marks known objects on one input minimap.

| Field | Type | Meaning |
|---|---|---|
| `known_object_marker_256` | uint16[256,256] | Zero is background/unaccepted; positive values are per-tile marker instance IDs. |
| `identity_rows` | object[] | One row per accepted candidate: marker instance ID, library ID, asset path, bounds/coverage, confidence, retrieval score, and split/provenance. |
| `rejected_candidates` | object[] | Candidate IDs, confidence, nearest library ID/score, and rejection reason. |
| `input_sha256` | string | Hash of the marked minimap input. |
| `checkpoint` | object | Marker checkpoint identity and architecture/config hash. |
| `gallery` | object | Read-only v50 library identity, schema, release, and gallery hash. |
| `skipped_instances` | object[] | Source instances with no visible pixels, including the explicit occlusion/overwrite reason. |

The marker map is deliberately an instance index rather than a library-ID raster. Library IDs are
variable-length strings and cannot safely be represented as pixels; the sidecar table is the
authoritative identity mapping. Marker corpus builders write through an incomplete `.partial`
directory and publish the final directory only after the manifest is complete; a partial directory
is never a trainable corpus.

## HistoricalRealObjectMaskDiagnostic

The immutable v50-backed supervision view used only for object-mask detection.

| Field | Type | Meaning |
|---|---|---|
| `store` | path | Configured canonical v50.1 Zarr store; never copied by the trainer. |
| `release` | string | Expected v50 release, initially `v50.1`. |
| `source_build` | string | Initial build identity, `0_5_3_3368`. |
| `source_filter` | enum | `authored`, `synthetic`, or `all`; initial default is `authored`. |
| `split_policy` | enum | Existing `manifest` split or explicit `map_holdout`. |
| `targets` | string[] | One or both of `object_precise_mask`, `object_mask`. |
| `rows` | object[] | Selected row indices, map/tile identity, source group, and split. |
| `provenance` | object | Store identity, source-group counts, and mask availability audit. |

The v50 `object_geometry_visible_mask_257` is audited separately. If it is empty, that fact is
recorded as unavailable geometry evidence rather than relabeled as object appearance supervision.
The curriculum `object_mask`/`object_precise_mask` projections are tile-level diagnostic labels;
they are not the promoted precision object target and are not interchangeable with
`ObjectLibrarySieveRow`.

## RealSyntheticValidationPair

One same-terrain authored/synthetic minimap pair selected from the v50 mixed curriculum for
validation or explicit input guidance.

| Field | Type | Meaning |
|---|---|---|
| `source_group_id` | string | Pair identity shared by the authored and synthetic rows. |
| `authored_row_index` | integer | Row containing the real/authored minimap RGB and supervision masks. |
| `synthetic_row_index` | integer | Same-tile legacy flat fake-maptexture RGB row; not a terrain-shadow target. |
| `map`, `tile_x`, `tile_y` | string/integer | Identity that must match on both rows. |
| `split` | enum | `train` or `val`; the pair cannot cross the split. |
| `domain_metrics` | object | Authored-vs-flat-synthetic MAE, RMSE, difference fractions, and optional fixed-shadow correlations. |

The pair selector skips incomplete groups only in the reportable count and records that count. It
fails closed for a selected group's mismatched map/tile identity or split assignment. The pair is
not a clean-minimap or terrain-shadow ground-truth claim; it is an absolute-difference diagnostic.
An optional fresh NPZ comparison is valid only when it contains the post-fix C#
`terrain_shadow_256` signal.

## RealObjectMaskExperiment

The report for the user-run RGB-to-real-mask experiment.

| Field | Type | Meaning |
|---|---|---|
| `model_version` | string | Architecture/input variant and target-head set. |
| `input_contract` | string | `minimap_rgb` or explicit `minimap_rgb_edge`; legacy flat synthetic RGB is not a terrain input contract. |
| `target_metrics` | object | Independent thresholded metrics for each requested mask target. |
| `selection_score` | number | Minimum requested-target IoU used for best-checkpoint selection. |
| `split_audit` | object | Map/source-group leakage checks and row counts. |
| `preview_artifacts` | string[] | RGB/truth/prediction/error review images. |
| `geometry_visible_mask_audit` | object | Presence and nonzero coverage of the excluded geometry signal. |
| `real_synthetic_validation` | object|null | Validation-only same-tile flat-maptexture absolute-difference report and optional fixed-shadow comparison. |
| `training_command` | string | Exact PowerShell-ready invocation. |

The experiment has no clean-minimap target. Its output is an object-mask detector, not an inpainting
or height-reconstruction proof.

## ObjectSieveExperiment

The bounded ablation report for object cleaning.

| Field | Type | Meaning |
|---|---|---|
| `variant` | enum | `clean_only`, `auxiliary_mask_loss`, or `predicted_mask_guided`. |
| `clean_metrics` | object | Error against clean `terrain_shadow_256`. |
| `mask_metrics` | object | Pixel metrics against `object_contamination_mask_256`. |
| `metrics_by_regime` | object | Separate results for density, overlap, and boundary cases. |
| `ground_truth_mask_as_input` | boolean | Must be `false`; ground truth is loss-side only. |
| `baseline` | object | Trivial clean-output and mask baselines. |

## VisualCoverageReview

The human-review artifact produced from a validated control corpus.

| Field | Type | Meaning |
|---|---|---|
| `signals_rendered` | string[] | Height, textureless shadow, normals, and derived height edges. |
| `families` | object[] | One summary row per control family with bucket and variation statistics. |
| `missing_expected_families` | string[] | Taxonomy families absent from the reviewed corpus. |
| `complexity_bucket_family_counts` | object | Counts of families in `easy`, `medium`, `hard`, and `pathological`. |
| `cross_tile_coverage` | object[] | Pattern IDs and observed 2x2 tile positions for cross-tile families. |
| `cross_tile_complete` | boolean | True only when each configured cross-tile family has all four positions. |
| `coverage_complete` | boolean | True only when the expected family taxonomy is present. |
| `outputs` | string[] | Family and variant atlas image paths. |

The atlas is an inspection surface, not training evidence. A visually weak or redundant family is a
control-design failure to fix before model training.

## AlbedoOperationRun

The output of processing one texture-bearing minimap into the model's canonical input domain.

| Field | Type | Meaning |
|---|---|---|
| `row_id` | string | Input tile identity. |
| `source_build` | object | Client era/build and configured root fingerprint. |
| `method` | string | Versioned albedo-removal method. |
| `method_version` | string | Reproducible implementation version. |
| `input_artifact` | string | Authored or source minimap artifact. |
| `normalized_artifact` | string | Canonical textureless output. |
| `albedo_artifact` | string | Optional estimated albedo/texture component. |
| `metrics` | object | Texture residual, high-frequency residual, range, and finite-value checks. |
| `status` | enum | `accepted`, `rejected`, or `quarantined`. |
| `failure_reason` | string|null | Required for non-accepted output. |

## TexturelessGateDecision

The fail-closed decision that determines whether an albedo-normalized tile may enter the first
model lane.

| Field | Type | Meaning |
|---|---|---|
| `row_id` | string | Tile identity. |
| `gate_version` | string | Gate implementation and threshold version. |
| `thresholds` | object | Persisted calibrated thresholds. |
| `metrics` | object | Measured residual/texture/quality values. |
| `decision` | enum | `accepted`, `rejected`, or `quarantined`. |
| `reason` | string | Human-readable and stable machine reason. |

Missing normalized artifacts, non-finite pixels, or uncalibrated thresholds are gate failures, not
implicit acceptance.

## DatasetVersionCatalogEntry

The viewer-facing summary for one explicitly discoverable dataset root. This is metadata and
selection state, not a promise that every listed signal has complete row coverage.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable normalized path identity for the current machine. |
| `display_name` | string | Human-readable version/root label, such as `v50.1 / 0.5.3.3368-Kalimdor`. |
| `root` | string | Configured dataset root. |
| `source_kind` | enum | `vlm_project` or `zarr_store`. |
| `map` | string|null | Map inferred from the project or store when available. |
| `tile_count` | integer | Discoverable tile count; zero is valid for a summary-only store. |
| `signals` | string[] | Recognized signal names or project signal groups. |
| `renderable` | boolean | True only when the current viewer can load terrain tiles from this root. |
| `diagnostic` | string|null | Missing contract, unsupported codec, or other fail-closed reason. |

Control NPZ folders and model-run folders are not catalog entries. A Zarr entry may list liquid,
object, tileset, texture, and placement arrays while still reporting `renderable=false` until its
tile decoder and tensor-pack rehydration are proven.

## DatasetVersionSelection

The persisted viewer choice, intentionally separate from client source and secondary map overlay.

| Field | Type | Meaning |
|---|---|---|
| `catalog_root` | string | Folder searched for versioned dataset roots. |
| `selected_root` | string|null | User-selected catalog entry, not an inferred client path. |
| `active_root` | string|null | Last successfully activated renderable project. |
| `client_source_unchanged` | boolean | Selection must not mutate the game-client root/build. |
| `secondary_overlay_unchanged` | boolean | Selection must not silently clear or replace the client map overlay. |

## RealTileObservation

One immutable real image supplied as evidence for reconstruction. The observation may be a full
client-backed tile, an authored minimap, or a low-resolution reference recovered from media.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable source-file identity. |
| `file_path` | string | Original local source path or imported source URI. |
| `kind` | enum | `client_harvest`, `authored_minimap`, `media_reference`, or `unknown`. |
| `map_hint` | string|null | Optional inferred or human-supplied map name; not a proof. |
| `tile_x_hint`, `tile_y_hint` | integer|null | Optional coordinate hints; not a proof of alignment. |
| `source_build_hint` | string|null | Optional client/build or publication context. |
| `source_sha256` | string | Hash of the immutable source bytes. |
| `width`, `height` | integer|null | Native dimensions when decoded; unknown is valid. |
| `available_signals` | string[] | Signals actually present, initially often only `real_rgb`. |
| `unknown_signals` | string[] | Height, normal, liquid, alpha, object, tileset, or texture targets not present/proven. |
| `preprocessing` | object[] | Crop, alignment, de-albedo, upscale, or resampling operations and versions. |
| `usable_as_model_input` | boolean | Whether an explicit input contract admits this observation. |
| `usable_as_target` | boolean | False unless an independent ground-truth artifact is attached. |

The source observation is never overwritten by derived artifacts. A low-resolution media reference
can be a valuable model input while remaining non-renderable and target-free.

## ExperimentRun

One bounded model/evaluation invocation.

| Field | Type | Meaning |
|---|---|---|
| `run_id` | string | Stable run identity. |
| `data_stage` | enum | `control`, `tiny_transfer`, or `expanded_transfer`. |
| `dataset_manifest` | string | Exact corpus or accepted-gate manifest. |
| `model_version` | string | Architecture/checkpoint contract. |
| `training_row_count` | integer | Limited-size experiment input. |
| `baseline` | object | Tile-mean or other trivial baseline metrics. |
| `metrics_by_family` | object | Held-out and transfer metrics. |
| `ambiguity` | object | Weak-signal/flat cases and exclusions. |

## TransferGate

The expansion decision between a tiny real sample and broader processing.

| Field | Type | Meaning |
|---|---|---|
| `control_run_id` | string | Control proof being transferred. |
| `real_gate_report` | string | Albedo/textureless gate report. |
| `real_sample_size` | integer | Accepted 0.x/1.x rows evaluated. |
| `domain_metrics` | object | Input-distribution and failure comparisons. |
| `decision` | enum | `hold`, `expand`, or `diagnose`. |
| `reason` | string | Why expansion is or is not allowed. |
