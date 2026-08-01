# Tasks: Spec 102 Minimap-Only Reset

## Invalidated Work

The earlier unified V25 architecture, teacher-forced run, predicted-WDL runs, loss-balance smoke, and multi-head quality claims are historical diagnostics only. They violate the one-model/one-residual rule and are not completion proof. The dataset itself may remain useful as a label store after an input-leakage audit.

**2026-07-12 correction:** `wdl_height_33` is a stride-8 raster, not the C# WDL MARE representation. It must not be used as WDL supervision, a validation gate, or an export value. See `docs/architecture/v25-v24-numeric-lattice-recovery-audit-2026-07-12.md`.

## Phase 0 — Contract and Baselines

- [x] **R001 Freeze invalid trainer**: fail before dataset loading or CUDA initialization with the reset reason.
- [x] **R002 Rewrite specification**: make RGB minimap pixels the only deployment input and remove WDL/multi-head prerequisites.
- [x] **R003 Create frozen held-out split manifest**: complete-map holdout plus era holdout; record hashes and row counts.
- [x] **R004 Add deploy-input manifest audit**: H0 forward accepts exactly `minimap_rgb`; runtime writes `input_manifest.json` and fails on signature drift.
- [x] **R005 Register deployable baselines**: zero height, train-global mean, and RGB-derived flat height on the frozen split; per-tile target statistics are forbidden.
- [x] **R006 Audit historical minimap-only checkpoint**: historical `190.31` L1 is marked non-comparable because it used a different split.
- [x] **R007 Checkpoint Phase 0**: no CUDA training; immutable split and baseline report published.

Baseline report: `output/analysis/spec102_minimap_baseline_v1/baseline_report.json`. Frozen counts: 2,381 train, 423 held-out 3.3.5 Northrend, 777 held-out 0.5.3 era. Best deployable L1: 308.889 on Northrend (RGB-flat) and 197.605 on the era holdout (train-global mean).

## Phase 1 — H0 Offset Residual

- [x] **R008 Implement H0 only**: RGB → one scalar correction residual over the frozen RGB-flat baseline; zero initialization starts exactly at that baseline.
- [x] **R009 Implement H0 trainer only**: separate optimizer/checkpoint/history; CUDA-only and three-epoch cap.
- H0 validation gate: beat `289.4451` tile-mean MAE (RGB-flat on frozen Northrend) by 20%, requiring `<=231.5561`.
- H0 v1 failed honestly (`321.3856` validation MAE): it incorrectly relearned RGB-flat from the train-global mean. H0 v2 fixes the residual anchor, normalizes regression scale, and uses batch 32 for more useful steps within the same three epochs.
- [x] **R010 Validate H0**: H0 v2 passed (`178.4316` validation offset MAE, required `<=231.5561`; era MAE `169.1934`; peak VRAM `0.0905 GB`).
- [x] **R011 Stop or freeze H0**: `h0_offset_v2_rgb_residual/checkpoint_best.pt` is the frozen H0 owner; H1 is unblocked.

## Phase 2 — H1 Coarse Relief Residual

- [x] **R012 Materialize frozen H0 outputs** in the H1 startup cache for the immutable split; checkpoint hash is recorded.
- [x] **R013 Implement H1 only**: RGB + frozen H0 → one 33×33 relief residual.
- [x] **R014 Implement and run H1 three-epoch gate** with its own checkpoint/history: attempted five times (v1 defaults, v2 optimization-stability fixes, v3 higher-resolution input, v4 frozen pretrained texture features via `timm`, v5 neighboring-tile context). All five ran their full three-epoch, CUDA-only, frozen-H0-input gate honestly. Best result (v4): `214.6247` validation coarse MAE against a required `<=175.2267`.
  - v5 is a genuine structural fix, not another technique swap: v1-v4 all shared the unexamined assumption that H1 should see only its own isolated tile, which is architecturally suspect for a spatial-relief task (ridgelines/valleys cross tile boundaries) and was caught by user review, not by this process. Fixed via `(build, map, tile_x, tile_y)` adjacency lookup + coarse 4-neighbor context encoder, verified correct against real data (adjacency resolution and flip-mirror mechanics both checked before the GPU run). Result: `215.8985` — did not beat v4, despite the fix being real and correctly implemented. Diagnosed as likely the wrong granularity of context (global-average vector loses directional slope information), not proof neighboring context is irrelevant.
- [ ] **R015 Stop or freeze H1**: no H2 work unless H1 beats the H0 plane. **Not met by any of v1-v5 — H2 remains blocked.** Stopped after five bounded runs (one of them a structural fix, not just hyperparameter search) to report honestly rather than launch a sixth attempt unilaterally. Decision point for the user.

## Phase 3 — H2 Detail Residual

- [ ] **R016 Materialize frozen H1 outputs** and deterministic 257×257 upsampling.
- [ ] **R017 Implement H2 only**: RGB + frozen coarse terrain → one 257×257 detail residual.
- [ ] **R018 Implement and run H2 three-epoch gate** with height, slope, and low-frequency metrics.
- [ ] **R019 Stop or freeze H2** before any border, uncertainty, or non-height work.

## Later Height Models

- [ ] **R020 H3 border residual**: one correction signal, independently trained after H2.
- [ ] **R021 U1 uncertainty**: one uncertainty signal, independently trained after H2.

## Later Phases

WDL export, objects, textures, alpha, liquids, PM4, and writers remain blocked until H2 passes. Every learned addition must remain single-output and independently gated.

## Recovery Phase 0 — Correct Numeric WDL Contract

- [x] **N001 Extract real vertices**: expose raw MCVT vertex Z, fixed X/Y, chunk/local index, and topology from real staged-client tiles without raster interpolation.
- [x] **N002 Prove dense-view mapping**: prove every `height_257` value maps to a raw vertex or mark it invalid; do not call an interpolated cell a vertex.
- [x] **N003 Define terrain lattice**: add the canonical numeric terrain-lattice type and its valid-node mask.
- [x] **N004 Prove paired WDL sampling**: derive outer 17x17 and inner 16x16 from the real lattice against the C# reader/writer.
- [x] **N005 Audit legacy consumers**: enumerate every `wdl_height_33` dataset, trainer, inference, validation, and export use; mark it invalid or migrate it in a later isolated task.
- [x] **N006 Audit normal geometry**: establish native-normal orientation, valid-node mask, scale, and finite-difference agreement with raw vertex Z.
- [x] **N007 Freeze recovery split**: record held-out maps, era holdout, hashes, deployment inputs, and target-only arrays for M0/W1/H2.
- [x] **N008 Publish numeric baselines**: report M0 mask, H0 datum, W1 545-sample lattice, and H2 vertex-Z/normal metrics without target leakage.
- [x] **N009 Define validation-only rendering**: emit PNG/OBJ/mesh previews after numeric validation; prevent them from becoming a dataset input or label.
- [x] **N011 Specify terrain-shadow capture**: add the fixed-camera/global-light capture manifest and disable objects, liquids, diffuse textures, alpha composition, and vertex-colour tint through the canonical viewer capture surface.
- [x] **N012 Prove terrain-shadow determinism**: capture five real staged-client tiles twice and require byte-identical settings plus bounded pixel-difference evidence before using the captures for guidance.
- [x] **N013 Measure shadow-to-mesh signal**: report real terrain-shadow correlation with raw vertex Z/slope as an upper-bound probe; do not train a model in this task.
- [x] **N014 Checkpoint Phase 0**: do not train or alter a trainer until N001-N013 have reproducible real-data evidence.

Phase 0's numeric mesh/lattice gate passed. Numeric report: `output/analysis/spec102_numeric_lattice_v1/baseline_report.json` (split SHA-256 `4ba21d68ca659a091542d00279002b13e5c803005bc2b941503b8b42c1987d5d`). The legacy store was read only at proven real checkerboard nodes; `wdl_height_33` and mixed-parity cells were never consumed. **M0 is now hard-blocked by contaminated object targets and unresolved liquid/all-map source evidence.** W1/H2 remain blocked.

## Recovery Phase 1 — M0 Object Mask

- [x] **N015 Implement M0 control flow only**: RGB -> one object-mask signal, separate checkpoint and bounded CUDA gate. The legacy `object_precise_mask_257` projection path is now a rejected control surface: it may validate fail-closed behavior but may not initialize CUDA or supply a training target.
- [x] **N015A Repair liquid-source coverage only**: strict raw-stream metadata JSON and MH2O 8x8-cell to 16x16-half-step coverage were repaired and eight coastal rows were copied exactly. This does not prove whether terrain/object fragments are visible through water; the legacy liquid-type raster remains excluded.
- [x] **N015B Make M0 validation images self-describing**: replace the unlabeled three-column grid with embedded run/column labels, build/map/tile identity, thresholded per-row IoU/Dice/pixel counts, and a fourth TP/FP/FN agreement column. Add a checkpoint renderer and regenerate both Northrend and Alpha-era panels for the corrected epoch-12 checkpoint.
- [x] **N015C Audit legacy M0 transport signals**: `spec102-dataset-signal-audit-v1` inspects source values before coercion, emits labelled panels, and binds hashes/fingerprints. It proves copy/range integrity only; it cannot certify the contaminated legacy target, terrain visibility, or water visibility and does not authorize CUDA.
- [x] **N015D Freeze the partial-corpus target boundary**: the first signal audit covers 2,804 3.3.5 rows across 46 maps and preserves all 777 0.5.3 rows as explicit exclusions. It is a labelled partial-corpus integrity proof, not the requested all-map M0 corpus.
- [x] **N015E Inventory full 3.3.5 source coverage**: staged discovery records 52 terrain-ready maps/5,471 occupied WDT locations and raw V18 records 52 map identities/5,134 rows. `coverage_final.json` preserves the eight readable maps with height/normals but no canonical minimap RGB, including six production maps that also lack MCLY/MCAL and therefore have no deterministic composition fallback, plus 367 rejected WDT locations. These are staged-source facts, not harvester parser failures or simple-reharvest work. It is inventory evidence, not readiness proof.
- [x] **N015F Preserve the full-map legacy numeric copy**: raw-V18 identity is retained for every row, but its 2,059-row curation and four zero-eligible maps are superseded as M0 target evidence because they inherit the contaminated legacy mask. They must not be called a frozen training corpus.
- [x] **N015G Preserve the legacy split/audit as a fail-closed control**: the 1,244 / 303 / 512 split and fingerprint bind legacy copies only. Their pre-CUDA pass is invalidated as training authorization; 0.5.3 remains separate.
- [ ] **N015H Reharvest strict 3.3.5 visible-object targets**: for every raster fragment, retain transformed geometry identity/world X-Y-Z, raw-MCVT terrain-Z evidence, and below-terrain/visible classification in the verified `strict-geometry-terrain-liquid-fragment-trace-v3` sidecar. It must preserve overlaps, asset-table identity, unresolved-placement facts, and a checked content hash. Never erase a whole above-ground placement or instance from a centroid, bounds, fallback, or missing-asset decision. Source code is unit-tested; no real staged reharvest/probe is yet accepted.
- [ ] **N015I Resolve water visibility and all-map source gaps**: bind liquid coverage/state/height at the same fragment and reject water-hidden or unknown visibility instead of fabricating zeros. Record the eight-map canonical-RGB absence (including Trial of the Champion, Trial of the Crusader, and Vault of Archavon; six production maps also lack MCLY/MCAL) and the 367 missing-required-source WDT locations as frozen staged-source failures; do not claim a reharvest repairs absent inputs. Until a per-pixel valid-loss mask exists, initial M0 is dry-only: any detected liquid rejects that tile rather than creating water-background negatives. All-map M0 remains blocked until a canonical source is supplied or the user consciously revises the source/input contract.
- [ ] **N015J Rebuild coverage, numeric store, split, and audit from the strict target**: report the complete staged inventory -> V18 row -> numeric row -> target -> M0 eligible/rejected chain. Fail closed on any missing identity, legacy target, unknown water state, or unresolved requested-map gap; no CUDA authorization until `training_authorized: true`.
- [ ] **N015K Run one three-epoch build-local M0 decision**: **BLOCKED.** It may run only after N015H-N015J pass and the user explicitly reauthorizes a fresh target contract. Do not extend epochs or materialize a cleaner.
- [ ] **N016 Validate and freeze M0**: the former 3.3.5-to-0.5.3 result is diagnostic only (`0.2764` Northrend / `0.0743` Alpha). Freeze only after N015K's valid strict-target metrics pass; do not start W1 first.

## Recovery Phase 2 — W1 WDL Lattice Residual

- [x] **N017 Implement W1 only**: cleaned RGB + frozen H0 datum -> one 545-sample paired WDL residual. The coordinate-query decoder emits one numeric vector and has no 33x33 target or separate outer/inner heads.
- [ ] **N018 Validate and freeze W1**: **not run on a valid upstream contract**. The prior diagnostic is retracted. The curated store reports zero W1-eligible rows because real paired WDL arrays are absent; `wdl_height_33` is prohibited. W1 remains blocked behind N016 and canonical WDL materialization.

## Recovery Phase 3 — H2 Mesh Vertices

- [ ] **N019 Implement H2 only**: cleaned RGB + frozen deterministic W1 upsample -> one mesh-native ADT vertex-Z residual.
- [ ] **N020 Validate and freeze H2**: beat W1 upsampling on held-out raw vertex-Z and native numeric-normal metrics; after numeric proof, separately report identical-light terrain-shadow agreement and generate previews.
