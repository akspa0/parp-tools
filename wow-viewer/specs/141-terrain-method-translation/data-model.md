# Data Model: Terrain Method Translation

## ExternalMethodRecord

One researched external method.

- `method_id`: stable local identifier.
- `name`: human-readable method name.
- `source_urls`: repository, paper, model card, or documentation links.
- `source_kind`: `paper`, `github`, `huggingface`, `documentation`, or `dataset`.
- `accessed_at`: UTC review timestamp.
- `input_modalities`: one or more of `rgb`, `dsm`, `point_cloud`, `mask`, `multispectral`, `metadata`.
- `output_signals`: declared outputs such as `ground_mask`, `dtm`, `height_residual`, or `object_mask`.
- `domain`: source domain and expected resolution.
- `license_status`: `verified`, `needs_review`, `unknown`, or `not_applicable`.
- `weights_status`: `not_required`, `available`, `unavailable`, or `not_reviewed`.
- `translation_status`: `reference`, `diagnostic`, `candidate`, `hold`, `rejected`, or `promoted`.
- `rejection_or_hold_reason`: required when not promoted.

## InputContract

The declared input boundary for one evidence run.

- `contract_id`: stable versioned identifier.
- `branch`: `rgb_only`, `height_prior`, `point_cloud`, or `combined`.
- `observable_inputs`: signals available at the claimed runtime boundary.
- `predicted_inputs`: model-produced auxiliaries, such as a predicted object mask.
- `supervision_only_inputs`: training/evaluation arrays that cannot enter inference.
- `forbidden_inputs`: explicit fail-closed names and aliases.
- `runtime_claim`: `none`, `offline_diagnostic`, or `deployment_candidate`.

## MethodEvidenceRun

A deterministic plan or result tied to a method and corpus.

- `run_id`: stable run identity.
- `method_id` and `contract_id`: method and input boundary.
- `corpus_root`, `corpus_manifest_hash`, and `source_row_hashes`.
- `split_identity`: map/family/source-group split description and hash.
- `conditions`: no-mask, predicted-mask, withheld-mask, DSM, or point-cloud condition.
- `baseline_ids`: identity, tile mean, and any signal-specific baseline.
- `metrics`: independent final-height, clean-identity, contaminated-input, cross-tile, family, and mask metrics.
- `forbidden_reads`: exact audit result.
- `decision`: `reference`, `diagnostic`, `candidate`, `hold`, `rejected`, or `promoted`.
- `artifacts`: reports, predictions, atlases, and command record.

## ResearchLead

A novel observation that is not yet a conclusion.

- `lead_id`: stable identifier.
- `observation`: what was actually seen.
- `hypothesis`: proposed explanation, clearly labeled as such.
- `provenance`: client/build/map/tile/window/source-group and signal availability.
- `falsification_test`: smallest test that could disprove the hypothesis.
- `result`: observed outcome, including failure.
- `confidence`: `unconfirmed`, `supported`, `contradicted`, or `insufficient_data`.
- `next_action`: one bounded follow-up.
- `linked_evidence_runs`: evidence required before promotion.

## TranslationDecision

The final classification of a method or experiment.

- `subject_id`: method or run.
- `status`: `reference`, `diagnostic`, `candidate`, `hold`, `rejected`, or `promoted`.
- `reason`: concise evidence-backed explanation.
- `required_next_gate`: next action if not promoted.
- `reviewed_at`: timestamp.
- `reviewer_artifacts`: report and visual evidence references.

## Invariants

1. `deployment_candidate` contracts cannot contain supervision-only or forbidden inputs.
2. A predicted mask must never be labeled as a ground-truth observable.
3. A method with an unresolved license or weight status cannot be a project dependency.
4. A promoted decision requires a linked evidence run and independent baseline-relative metrics.
5. A research lead cannot leave `unconfirmed` without provenance and a falsification result.
