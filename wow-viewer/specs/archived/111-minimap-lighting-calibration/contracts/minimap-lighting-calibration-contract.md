# Minimap Lighting Calibration Contract

## Shading-match inference contract

1. Every candidate rendered for scoring MUST come from the production
   `TerrainMinimapCompositor`/`TerrainSolarDirection` path -- the exact code the live viewer and the
   `synthetic-minimap` exporter use. No parallel lighting-direction implementation may be introduced or
   retained anywhere in this feature's code path.
2. The shading-match score MUST be independent of the existing tint-ratio signal: it compares
   directional/gradient structure, not raw color, so it answers a question the existing
   `MinimapLightingProvenance.Infer` tint path cannot.
3. A tile MUST receive exactly one of: a `matched` status with a time-of-day and confidence, an
   explicit `low_confidence_ambiguous`/`low_confidence_flat_terrain` status, or `not_evaluated`. There
   is no fourth, silent outcome.
4. Inference is scoped to 0.5.3.3368 tiles only. A tile from any other build MUST retain
   `ShadingMatchStatus = not_evaluated` and MUST NOT be scored.
5. Every `matched` or low-confidence result MUST record the build fingerprint it was computed against.

## Rebalancing contract

1. Synthetic-lighting-variant sampling weights MUST be derived only from real, computed
   `LightingBucketDistributionReport` data -- never an assumed or hardcoded distribution.
2. A bucket with zero or near-zero real coverage MUST be explicitly flagged (`no_real_baseline`); the
   system MUST NOT infer or fabricate a real-example count for it.
3. Reweighting MUST NOT alter or bypass the existing `source_group_id`/`lighting_variant_id` leak-safety
   tagging. A rebalancing plan that would do so is rejected before it reaches the training pipeline.
4. The resulting training data's model-input contract MUST continue to exclude ground-truth
   time-of-day and lighting-direction fields; the lighting-bucket label is a sampling signal consumed
   only by the data-generation step, never forwarded into the model's input tensor.

## Training/evaluation execution contract

1. No command in this feature's Phase 3 may launch a GPU training run or any cloud compute job (RunPod
   or otherwise) without a separate, explicit user go-ahead given at the point of execution. Completing
   Phases 1-2, or preparing Phase 3's training script/config, does not itself authorize execution.
2. Any executed training run MUST target the existing reconstruction architecture lineage (Spec 108
   `WdlPriorNet` or the currently active, unblocked Spec 102 residual-chain stage). It MUST NOT
   introduce a DepthAnything-family, multi-head, multi-task, or shared-weight model path.
3. A trained candidate checkpoint MUST be compared against the currently deployed checkpoint on the
   existing Spec 108 group-held-out split before any promotion decision.
4. `Outcome = regressed` MUST result in `PromotionDecision = false`. No automatic promotion path may
   bypass this.

## Storage contract

1. All new per-tile and per-build results are additive fields/attrs on the existing per-build Zarr
   store, delivered through the existing C#-to-Python length-prefixed streaming protocol.
2. No new NPZ artifact is introduced as a primary storage format. Derived reports (e.g. the
   lighting-bucket distribution) are computed from the Zarr store's fields, not stored as a second
   independent source of truth.
