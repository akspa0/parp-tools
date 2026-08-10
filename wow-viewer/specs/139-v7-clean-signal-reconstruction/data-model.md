# Data Model: V7-Inspired Clean-Signal Terrain Reconstruction

## CleanObservationRow

One row in the new synthetic or accepted-real corpus.

| Field | Type | Contract |
|---|---|---|
| `row_id` | string | Globally unique within the corpus. |
| `source_kind` | enum | `synthetic_control` or `accepted_real`. |
| `source_group_id` | string | Split/leakage identity; variants in one group never cross a family holdout. |
| `clean_observation_luma_256` | float32[256,256] | Finite `[0,1]`, albedo-normalized terrain observation. |
| `clean_observation_gradient_256` | float32[2,256,256] | Finite deterministic x/y gradients of luma. |
| `clean_observation_confidence_256` | float32[256,256] | Finite `[0,1]`; zero is valid only with an explicit absence status. |
| `confidence_status` | enum | `measured`, `absent_explicit`, `rejected`, or `quarantined`. |
| `split` | enum | `train`, `validation`, or `test`. |
| `family` | string | Terrain family and complexity bucket. |
| `observation_provenance` | object | Albedo method/version, renderer/transform version, and parameters. |
| `array_hashes` | object | SHA-256 for every stored array. |

## StructuralTarget

Training/evaluation targets derived from exact `height_257`. These fields are never admitted to an
inference input package.

| Field | Type | Contract |
|---|---|---|
| `height_257` | float32[257,257] | Exact source height grid. |
| `relative_height_257` | float32[257,257] | Versioned per-tile relative-height target. |
| `coarse_relief_257` | float32[257,257] | Fixed low-frequency projection of relative height. |
| `detail_residual_257` | float32[257,257] | `relative_height_257 - coarse_relief_257`, signed. |
| `decomposition_version` | string | Binds kernel/cutoff and range-floor semantics. |

## CleanSignalModelOutput

| Field | Type | Contract |
|---|---|---|
| `coarse_prediction_257` | float32[257,257] | Model coarse branch. |
| `detail_prediction_257` | float32[257,257] | Model signed detail branch. |
| `height_prediction_257` | float32[257,257] | Recomposition and published output. |
| `model_identity` | object | Architecture, profile, parameter count, seed, and config hash. |

## GuidanceRunReport

The report records `loss_profile`, input/target contracts, corpus and split hashes, architecture
identity, best/final epoch, final/coarse/detail MAE, baseline-relative family metrics, and each
enabled structural component. It must also record `forbidden_signals_seen` and keep that list empty
for a valid inference run.

## TransferDecision

| Field | Type | Contract |
|---|---|---|
| `synthetic_report` | path/hash | The accepted control run. |
| `accepted_real_rows` | integer | Count admitted by the albedo gate. |
| `rejected_real_rows` | integer | Count retained but excluded. |
| `domain_metrics` | object | Observation distribution and failure signatures. |
| `decision` | enum | `hold`, `diagnose`, or `expand`. |
| `forbidden_signal_audit` | object | Must report zero target-side reads. |
