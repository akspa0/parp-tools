# Implementation Plan: Spec 102 Numeric-Lattice Recovery

**Status**: Phase 1 M0 decision complete; W1 blocked on real paired WDL
**Specification**: [spec.md](spec.md)

## Technical Context

The only deployment input is RGB minimap pixels. Existing V18/V25 terrain arrays are labels and evaluation facts, never model inputs. The current unified trainer is permanently fail-closed. Replacement work is a sequence of tiny, independent residual models.

## Constitution Check

- New Python stays under `data-harvester/` and uses the existing environment.
- No parser, client reader, or dataset-builder duplication.
- CUDA is explicit; silent CPU fallback is a failure.
- One model, one residual signal, one checkpoint, and one gate.
- No shared weights or joint training between stages.
- Each slice is independently testable and no phase exceeds ten tasks.

## Phase 0 — Numeric Contract and Baselines

1. Extract raw MCVT vertices/topology from real staged-client data and prove the mapping to the dense export. Any non-vertex array position is invalid for loss.
2. Define the canonical numeric terrain-lattice type: vertex Z, fixed world X/Y, source chunk/local index, and valid-node mask.
3. Prove the paired C# WDL contract from that vertex lattice; retain outer 17x17 and inner 16x16 separately.
4. Deprecate `wdl_height_33`; a 33x33 helper may exist only with explicit lattice-validity metadata.
5. Audit native MCNR normals as numeric vertex geometry and finite-difference evidence; do not train on it yet.
6. Freeze a held-out-map/era split with a per-stage deployment-input manifest and numeric vertex/WDL baselines.
7. Specify the real-client terrain-shadow capture contract: fixed camera/global light; objects, liquids, diffuse textures, alpha composition, and vertex-colour tint disabled; capture manifest records every setting.
8. Generate and reproduce a five-tile staged-client shadow-capture probe. Measure its numeric correlation with raw mesh slope/vertices before building a model around it.
9. Define one-way validation rendering after inference; previews are never deployment inputs. After a vertex model passes numeric proof, its mesh may be deterministically rerendered under the same light as a separately reported guidance metric.
10. Record `ef99e715` as trainer-control-flow reference only; it is non-comparable until rerun on the corrected contract.

## Phase 1 — M0 Object Segmentation and Deterministic Cleaning

1. Train one small RGB-to-object-mask model with its own checkpoint and three-epoch gate.
2. Freeze M0 only if it beats the held-out mask baseline.
3. Produce cleaned minimaps deterministically from the frozen mask; validate raw/clean/mask alignment on real tiles.

**2026-07-12 corrected liquid gate:** the earlier coarse-target run and every
M0 result predating the MH2O repair are invalid. V18 did contain liquid data;
the defect was sparse, half-scale MH2O cell rasterization plus a malformed raw
stream metadata document that prevented a clean repatch. The repaired numeric
v3 store retains the canonical precise mask and real numeric liquid mask,
height, MCNK flags, and source provenance. With an 80% occlusion cutoff, M0
receives 1,901 train / 302 map-validation / 770 era rows; H2 receives 1,880 /
279 / 770. No M0 metric is currently comparable to this cohort. M0 must be
rerun and frozen before W1; W1/H2 remain blocked.

## Phase 2 — H0 Tile Offset Residual

1. Predict one scalar correction residual over the frozen deployable RGB-flat baseline.
2. Train with a dedicated H0 trainer and checkpoint.
3. Run at most three epochs and stop unless held-out offset error beats the RGB-flat baseline.

## Phase 3 — W1 WDL Lattice Residual

This phase opens only after M0 cleaning and H0 are frozen.

1. Freeze H0 and materialize its predictions for the frozen split.
2. Predict one paired `outer_17` + `inner_16` numeric lattice residual from cleaned RGB plus H0 output.
3. Run at most three epochs and stop unless 545-sample L1 beats the H0-plane lattice baseline.

**2026-07-12 correction:** the prior W1 run consumed invalid M0 outputs and is
retracted. The curated store has zero W1-eligible rows because it does not yet
contain real paired `outer_17` / `inner_16` WDL arrays. Derived
`wdl_height_33` is prohibited, so W1 cannot run.

## Phase 4 — H2 Terrain Detail Residual

This phase opens only after W1 passes.

1. Freeze H0/W1 and materialize the deterministic WDL lattice upsample.
2. Predict one mesh-native ADT vertex-Z detail residual from cleaned RGB plus the frozen lattice prediction.
3. Run at most three epochs and stop unless vertex-Z and numeric-normal metrics beat W1 upsampling; render previews only after numeric validation.

## Phase 5 — H3 Border Residual

This phase opens only after H2 passes.

1. Consume adjacent RGB tiles and frozen H2 border predictions.
2. Predict one shared-border correction residual.
3. Validate raw continuity before any deterministic stitching.

## Phase 6 — U1 Uncertainty

This phase opens only after H2 passes and is trained separately from height.

1. Consume RGB and frozen height outputs.
2. Predict one uncertainty map.
3. Validate calibration against held-out H2 error.

## Deferred Independent Phases

WDL export, objects, textures, alpha, liquids, PM4, and binary writers each require separate single-output models or deterministic stages and independent gates.
