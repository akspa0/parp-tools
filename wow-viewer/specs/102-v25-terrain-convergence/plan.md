# Implementation Plan: Spec 102 Numeric-Lattice Recovery

**Status**: BLOCKED — M0 target reharvest and full-source readiness proof are required before any CUDA decision.

**Specification**: [spec.md](spec.md)

## Technical Context

Deployment consumes only RGB minimap pixels. Terrain targets are raw numeric mesh vertices and topology; `height_257` and `normal_xyz_257` are materialized views and validation facts, never a substitute terrain representation. Validation PNGs are post-inference observability only.

M0 remains the one permitted learned stage in this phase: a 3,043,041-parameter RGB-to-one-visible-object-mask model. It has one target, checkpoint, optimizer, loss history, and three-epoch decision gate. It cannot share weights or train jointly with H0/W1/H2. The legacy `object_precise_mask_257` target is contaminated, so there is currently no permitted M0 training target. The old unified V25 trainer stays fail-closed; `ef99e715` is a control-flow reference only and cannot restore its old data/model contract.

## Constitution Check

- New Python remains under `data-harvester/` and uses the existing `uv` environment.
- Dataset construction remains identity-preserving and non-destructive; do not rewrite client readers or create a second parser.
- No DepthAnything, multi-head, multi-task, or shared-weight model path.
- CUDA is explicit and fails closed. Every trainer change has a focused validation path.
- Legacy `object_precise_mask_257` is not a training target. Never substitute it, a reduced mask, a visibility mask, or a fallback mask for the required strict reharvest.
- The replacement target must rasterize transformed object geometry and compare every fragment with raw-MCVT terrain Z at that fragment. Placement/instance/centroid/bounds whole-mask removal is prohibited. The required `strict-geometry-terrain-liquid-fragment-trace-v3` is a variable-length numeric audit sidecar, never a raster-model input: it preserves each fragment and overlap with source identity, transformed coordinates, raw-MCVT interpolation facts, liquid facts, and a verified content hash.
- Liquid coverage/state/height must be resolved at the same fragment before calling terrain or an object visible. Blue/uniform minimaps and unknown water visibility are reject conditions, never empty-mask targets.

## Current Evidence and Decision

- The old numeric-v3 M0 selection is a partial legacy-target transport snapshot: 46 maps / 2,804 rows, selected from V25 curation. It is not coherent target evidence and cannot authorize CUDA.
- The exact staged `3_3_5_12340` probe identifies 125 Map.dbc records, 52 terrain-ready maps, and 5,471 occupied WDT locations.
- Raw V18 contains all 52 terrain-ready map identities / 5,134 valid rows. It also records 367 locations rejected for missing required signals. Separately, eight readable staged maps have height/normals but no canonical minimap RGB; six production maps among them also lack MCLY/MCAL, so there is no deterministic texture/alpha composition fallback for canonical RGB.
- The six map identities missing from numeric-v3 exist in V18. Their legacy classifications are not enough to decide M0 eligibility; they must remain in map/row provenance and be re-evaluated from strict target and liquid evidence.
- The replacement raw-identity store preserves all 5,134 rows / 52 map identities, but its 2,059-row curation and 1,244 / 303 / 512 split inherit the contaminated target. They are historical selection evidence, not a training corpus.
- `coverage_final.json`, the merged seven-signal audit, and their fingerprint prove identity/copy/range facts only. Panels remain inspection-only and do not prove target correctness or water visibility.
- The requested all-3.3.5 readiness is unmet. The eight-map canonical-RGB absence is a frozen-input source gap in the staged client, not a harvester parser bug; reharvesting cannot create inputs that are not there. The affected readable maps include Trial of the Champion, Trial of the Crusader, and Vault of Archavon. The 367 WDT locations also lack required source signals. No hash-bound legacy corpus may authorize M0 until a canonical source is supplied or the user consciously revises the source/input contract, and the strict target contract is resolved.

## Phase 0 — Numeric Contract and Baselines (complete)

1. [x] Extract and preserve raw MCVT vertex Z/topology/world-coordinate truth.
2. [x] Prove dense-view mapping and invalidate `wdl_height_33` as WDL truth.
3. [x] Establish paired `outer_17` / `inner_16` WDL contract and numeric normal audit.
4. [x] Freeze baselines, deployment-input manifest, validation-only rendering, and fixed-light terrain-shadow guidance contract.
5. [x] Record `ef99e715` as a trainer-control-flow reference only.

## Phase 1 — M0 Strict Target Reharvest and Full 3.3.5 Readiness

1. [x] Preserve the legacy 46-map and 52-map coverage/audit artifacts as transport controls only; permanently block them from authorizing CUDA.
2. [ ] Reharvest every 3.3.5 object target from transformed source geometry, retaining per-fragment world position, source identity, raw-MCVT terrain-Z evidence, and classification. The source implementation/serializer contract is now v3-tested; a real staged output remains required before this step is complete.
3. [ ] Add per-fragment liquid coverage/state/height evidence and reject terrain-hidden-by-water or unknown-visibility fragments/tiles instead of inventing empty-object labels. Initial M0 remains dry-only until a per-pixel valid-loss mask exists.
4. [ ] Reconcile the staged-client inventory: include every requested terrain map in provenance and record the eight-map canonical-RGB absence plus 367 missing-required-source WDT locations as frozen source facts. Do not represent them as a reharvest repair. All-map M0 may proceed only if a canonical source is supplied or the user consciously revises the source/input contract.
5. [ ] Build a fresh non-destructive numeric store only from the strict target and retain exact raw-row identity. Legacy object masks may not enter its target path.
6. [ ] Publish the versioned full-3.3.5 machine report: staged inventory -> raw V18 -> numeric row -> strict target -> M0 decision/rejection reasons. Fail on any missing identity, legacy target, unknown water visibility, or unresolved inventory gap. It MUST label the absent canonical RGB/MCLY/MCAL condition as a source-contract failure rather than a parser or reharvest failure.
7. [ ] Freeze a complete-map split only after the report has `training_authorized: true`; all noneligible rows remain reason-coded provenance, not training negatives.
8. [ ] Audit the replacement raw signals/target before coercion and update pre-CUDA validators to require its exact hashes and coverage contract.
9. [ ] Inspect self-describing panels only after the strict target is available; panels remain observability, not training input.
10. [ ] Run exactly one small three-epoch M0 decision only after steps 2-9 pass and a user-approved fresh authorization exists. Otherwise stop; do not start W1.

## Phase 2 — H0 Tile Offset Residual

Opens only after M0 is frozen. Train a separate RGB-to-one-scalar residual over the frozen deployable RGB-flat baseline, then run one three-epoch gate.

## Phase 3 — W1 Paired WDL Lattice Residual

Opens only after M0 and H0 are frozen and real paired WDL arrays are materialized. Predict one 545-sample numeric residual, not a 33x33 image.

## Phase 4 — H2 Mesh-Vertex Residual

Opens only after W1 passes. Predict numeric residual Z at canonical mesh vertices and evaluate primary loss/metrics at real nodes. Render PNG/OBJ/shadow guidance only after numeric proof.

## Deferred Independent Phases

H3 borders, U1 uncertainty, Alpha 0.5.3 target work, WDL export, objects, textures, alpha, liquids, PM4, and writers all require their own separate specification and gate.
