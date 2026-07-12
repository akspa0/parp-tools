# Tasks: Spec 102 Minimap-Only Reset

## Invalidated Work

The earlier unified V25 architecture, teacher-forced run, predicted-WDL runs, loss-balance smoke, and multi-head quality claims are historical diagnostics only. They violate the one-model/one-residual rule and are not completion proof. The dataset itself may remain useful as a label store after an input-leakage audit.

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
