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

## RGBMethodBenchmarkPlan

A deterministic, manifest-only plan for the three RGB/object-mask conditions.

- `schema`: `v60-rgb-method-benchmark-v1`.
- `source_selection`: `authored`, `object_library`, or `both`.
- `source_reports`: source modality, row counts, split identity, model inputs, evaluation-only arrays, and runtime compatibility.
- `conditions`: `no_mask`, `predicted_mask`, and `withheld_mask` with per-source availability.
- `baselines`: tile mean, identity observation, and zero predicted mask declarations.
- `metric_groups`: final height, clean identity, contaminated input, object mask, cross-tile, and family metrics.
- `runtime_eligible_conditions`: conditions backed by a runtime-compatible source.
- `forbidden_reads`: any model-input signal that violates the source contract.
- `next_gate`: the required user review or run before training.

## BenchmarkCondition

One model-input policy within the plan.

- `condition_id`: one of `no_mask`, `predicted_mask`, or `withheld_mask`.
- `model_input_policy`: observation-only or observation plus predicted mask.
- `mask_role`: not provided, predicted only, or evaluation-only withheld from the model.
- `required_predicted_artifact`: whether a separately proven predicted-mask artifact is required.
- `source_reports`: eligible counts and runtime status for each source.

## BaselineDefinition

A named comparison with an explicit metric scope and target-read requirement.

- `baseline_id`: stable identifier.
- `scope`: final height, clean head, or predicted mask.
- `description`: exact comparison behavior.
- `requires_target_for_evaluation`: whether targets are evaluation-only inputs.

## Invariants

1. `deployment_candidate` contracts cannot contain supervision-only or forbidden inputs.
2. A predicted mask must never be labeled as a ground-truth observable.
3. A method with an unresolved license or weight status cannot be a project dependency.
4. A promoted decision requires a linked evidence run and independent baseline-relative metrics.
5. A research lead cannot leave `unconfirmed` without provenance and a falsification result.
6. A synthetic luma/object-control source cannot be labeled runtime-compatible with RGB minimap inference.
7. A target-side mask can support withheld-mask evaluation but cannot appear in `model_input_arrays`.
