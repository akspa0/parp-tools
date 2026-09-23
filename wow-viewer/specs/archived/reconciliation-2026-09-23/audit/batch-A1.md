# Batch A1 — ML terrain/minimap/WDL/dataset tooling (specs 108, 109, 111, 112, 114, 115, 117, 118, 123, 124)

Audited against real code under `wow-viewer/data-harvester/` (Python) and `wow-viewer/src/core/` (C#).
All ten specs belong to the "ML & Dataset Tooling (eventually continue)" watch-list epic in
`specs/STATUS.md` (epic 7) — none appear in the current v0.6 weekly implementation order, i.e. this
whole lane is dormant-but-real, not actively worked this week.

---

### 108 Image-Only WDL Prior
- Stated status: Implementing | Tasks: 10/15 checked (T001-T008, T014)
- Scope: RGB-only WDL-lattice predictor (17x17 outer + 16x16 inner) as a coarse prior for V8 terrain
  refinement, plus irregular-motif ("prefab") recovery from real alpha/height cell topology.
- Verified implemented: RGB-only predictor + exact target mapping (`data-harvester/src/harvester/v50/wdl_prior_infer.py`, standalone `--image` path confirmed); trainer/evaluator (`wdl_prior_train.py`,
  `wdl_prior_evaluate.py` via shims `scripts/train_spec103_wdl_prior.py` etc.); `--generated-wdl-priors`
  fail-closed gate + `history.json` binding in `harvester/v50/terrain_refiner_train.py` (T005/T014);
  mixed-curriculum builder `scripts/spec108_build_mixed_curriculum.py` (T009, code present);
  irregular motif recovery `data-harvester/src/harvester/chunk_motifs.py` with contact-sheet rendering
  and D4-transform family matching (T010, code present) + `tests/test_chunk_motifs.py`.
- Partial: T011's tests exist (`tests/test_spec108_mixed_curriculum.py`, `tests/test_chunk_motifs.py`)
  but not under the spec'd `data-harvester/tests/spec108/` path.
- Not implemented: T012/T013/T015 — user-run mixed-corpus training and label-free validation; no
  checkpoint evidence found.
- Checkbox accuracy: 3 unchecked-but-present (T009, T010, T011 partially).
- Operator gates owed: mixed-corpus training run (T012/T013/T015).
- Open residue (spec-stated only): T012/T013/T015 user-run training/validation.
- Superseded by / overlaps: infrastructure (WDL lattice concept) superseded in intent by Spec 117/123
  (real WDL prior over synthetic); this spec's motif/prefab work is independent and not carried
  forward elsewhere.
- Disposition: FOLD
- Proposed epic theme: ML terrain/minimap dataset tooling
- Confidence: high

---

### 109 V50 Clean-Room Dataset and Repository Audit
- Stated status: Active audit | Tasks: ~62/64 checked (9 phases; only T050, T051 unchecked)
- Scope: Fail-closed trust boundary for all legacy datasets, reviewable repo cleanup, and a
  from-scratch verified v50 per-build Zarr dataset family (Kalimdor/Azeroth/PVPZone02/Kalidar) with
  canonical v50 module ownership.
- Verified implemented: extremely large, real `data-harvester/src/harvester/v50/` package (80+
  modules: contracts, identity, inventory, verify_store, verify_v18, migrate, store, cleanup,
  cleanup apply, curriculum, build, client_evidence, path_policy, and dozens more downstream model
  modules); `data-harvester/tests/v50/` (40+ test files); real on-disk stores confirmed at
  `output/datasets/v50/v50.1/0_5_3_3368-{Kalimdor,Azeroth,PVPZone02,Kalidar}.zarr` with real
  `mcnk_flags_16` populated (T005 fix verified in data, not just code); multiple real curriculum
  builds (`curriculum-0_5_3_3368-{strict_v1,dual_v1,dual_v2,dual_v3,obj_v1}.zarr`) and both
  strict/object-inclusive curation manifests per map.
  Every module tasks.md names (`v50_build_dataset.py`, `v50_cleanup_artifacts.py`,
  `v50_audit_artifacts.py`, `v50_pipeline_runner.py`, etc.) exists on disk.
- Partial: T050 (post-cleanup-apply verification) and T051 (memory-bank compression) explicitly
  blocked on the user actually running the real cleanup-apply command — `output/old/` (reports,
  v50 old artifacts) suggests a cleanup pass ran at some point, but T050's own doc-write step is
  unchecked.
- Not implemented: none found beyond T050/T051.
- Checkbox accuracy: accurate (spot-checked; this tasks.md carries its own inline pytest-count/CLI
  receipts throughout, consistent with what's on disk).
- Operator gates owed: T050 (verify+document post-cleanup state), T051 (memory-bank compression).
- Open residue (spec-stated only): T050, T051.
- Superseded by / overlaps: none — this is the foundational data layer every other spec in this
  batch (111,112,114,115,117,118,123) builds on directly.
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: ML terrain/minimap dataset tooling (foundation)
- Confidence: high

---

### 111 Minimap Lighting Calibration and Lighting-Aware Terrain Reconstruction
- Stated status: Draft (tasks.md is far more current than this header) | Tasks: 19/21 checked
- Scope: Shading-match inference to bucket real 0.5.3.3368 tiles by true sun time-of-day (via the
  production `TerrainMinimapCompositor`/`TerrainSolarDirection`), rebalance synthetic lighting
  variants to match the real distribution, then retrain/evaluate the reconstruction model.
- Verified implemented: `src/core/WowViewer.Core.IO/Maps/MinimapShadingMatch.cs` +
  `MinimapLightingProvenance.cs` extension + `tests/WowViewer.Core.Tests/MinimapShadingMatchTests.cs`
  (US1); `data-harvester/src/harvester/spec111/{lighting_buckets,rebalance_lighting_variants,
  checkpoint_comparison}.py` + matching tests in `data-harvester/tests/spec111/` (US2/US3 code);
  `scripts/{report_lighting_buckets,rebalance_lighting_variants,train_spec111_reconstruction}.py`.
- Partial: none beyond the explicitly user-gated items.
- Not implemented: T009 (real bounded 0.5.3.3368 bucketing pass + eyeball check) and T019 (STOP:
  user-authorized training run) — both explicitly user-run and unchecked.
- Checkbox accuracy: accurate.
- Operator gates owed: T009 real bucketing pass; T019 GPU training run with explicit go-ahead.
- Open residue (spec-stated only): T009, T019.
- Superseded by / overlaps: none directly; its rebalanced-sampling output feeds the same model
  lineage Spec 112/114 use.
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: ML terrain/minimap dataset tooling
- Confidence: high

---

### 112 V50-Native Height-First Terrain Model with Dataset Corrections
- Stated status: Draft (stale header; tasks.md shows real progress) | Tasks: 22/23 checked
- Scope: Phase 1 dataset correction (mcnk_flags_16, minimap_rgb_1024 coverage parity, catalog/
  template honesty, authored-minimap capture) + Phase 2 a lean minimap-RGB -> relative-height model,
  Kalimdor/Azeroth only.
- Verified implemented: catalog/reason-vocabulary helpers (`harvester/v50/signal_catalog.py`,
  `contracts.py` additions); real rebuilt Kalimdor/Azeroth v50.1 stores on disk with populated
  `mcnk_flags_16` (confirms the C# `AlphaTensorPackBuilder` fix actually ran against real data, not
  just fixture tests); real dual-source curricula (`curriculum-0_5_3_3368-dual_v{1,2,3}.zarr`);
  `harvester/v50/height_relative_model.py` + `height_relative_train.py` + `v50_train_height_relative.py`
  with the offset-invariant relative-height target contract (v112.1) and Kalimdor/Azeroth-only
  evaluation guard.
- Partial: none.
- Not implemented: T021 — user-run authored-only baseline training; no `height_relative` checkpoint
  found on disk under `data-harvester/checkpoints`, consistent with the box being unchecked.
- Checkbox accuracy: accurate.
- Operator gates owed: T021 training run + SC-004/SC-005 review.
- Open residue (spec-stated only): T021 (training + SC-004/SC-005 verdict).
- Superseded by / overlaps: **T021's actual training happened under Spec 114's T017** instead
  ("authored-only `direct_cnn_v112` baseline... DONE 2026-07-19: FAILED SC-001" — same architecture,
  same curriculum lineage, run under the 114 label). Spec 112's dataset-correction phase (US1/US2) is
  independently complete and load-bearing infrastructure for 114/115/117/118.
- Disposition: ARCHIVE-SUPERSEDED (US3 training subsumed by Spec 114; US1/US2 dataset work stands as
  completed foundation)
- Proposed epic theme: ML terrain/minimap dataset tooling
- Confidence: high

---

### 114 Direct Minimap-to-Terrain Reconstruction
- Stated status: Draft | Tasks: ~45/63 checked (Phases 1-3 done incl. two extra IDs T056-T063; Phases
  4-8, T019-T055, entirely unchecked)
- Scope: Modular reconstruction chain — direct RGB->relative-height (no WDL prior), then separate
  object-cleanup, land-feature classification, and texture-family/alpha reconstruction stages, each
  independently checkpointed.
- Verified implemented (Phase 3 / US1 — direct geometry, real training evidence): `direct_geometry_
  {model,train,infer,materialize}.py`, `geometry_detailer_{model,train,infer}.py`, `spectral_
  guidance.py` all present in `harvester/v50/`; real run receipts inline in tasks.md (`mit_b0-
  authored-v1` best-epoch-92 negative result recorded as evidence per T017; detailer beating
  coarse-only baseline 9.1-11.2% relative per T061/T063/T063-continued).
- Partial: none within Phase 3.
- Not implemented: Phase 4 (US2 object cleanup: `object_visibility_labels.py`, `object_mask_model.py`
  — grepped, absent), Phase 5 (US3 terrain-feature library `terrain_feature_library.py` — absent),
  Phase 6/7 (US4 `texture_family_library.py`, `texture_family_model.py`, `alpha_stack_model.py` —
  all absent), Phase 8 (audit/docs).
- Checkbox accuracy: accurate (unchecked tasks genuinely have no code).
- Operator gates owed: Phase 4-7's user-run training (never reached — code doesn't exist yet).
- Open residue (spec-stated only): Phase 6 (US4A texture-family selection, T036-T043) and Phase 7
  (US4B alpha-stack reconstruction, T044-T051) — the only parts of 114 with no later spec covering
  them.
- Superseded by / overlaps: Phase 4 (US2 object cleanup) is superseded by the more thorough Spec 118
  (occlusion-aware per-object masks, which explicitly supersedes the RGB-difference approach 114
  assumed). Phase 5 (US3 terrain features) is superseded by Spec 115 (a more urgent, narrower cut of
  the same idea, delivered with real training results).
- Disposition: FOLD (texture-family + alpha-stack residue only; geometry/detailer chain is complete
  and load-bearing)
- Proposed epic theme: ML terrain/minimap dataset tooling
- Confidence: high

---

### 115 Terrain Feature Classification for Geometry Deconfounding
- Stated status: Draft | Tasks: no tasks.md (spec.md + plan.md + research.md + data-model.md +
  quickstart.md only)
- Scope: Image-to-feature-class model (real-terrain/road/water/structure/unknown) supervised from
  real per-chunk texture-family ground truth, to deconfound the direct-geometry model's
  color-as-depth-proxy failure (roads decoded as hills).
- Verified implemented: full `harvester/v50/terrain_feature_{labels,label_build,model,train,infer}.py`
  package; label pipeline verified against real ground truth (Kalimdor tile 24,40 MTEX table
  reproduced exactly per research.md); real per-pixel class distribution measured (road 0.26% of
  pixels, terrain 94.71%) driving an inverse-frequency loss and road-IoU promotion gate (never
  accuracy); US2 (retrain geometry with the generated feature map) is ALSO implemented —
  `direct_geometry_model.py` has `in_channels`, `direct_geometry_train.py` has `--feature-store` and
  an explicit `road_region_mae`/`nonroad_region_mae` FR-008 metric — contradicting quickstart.md's
  stale "Phase 5 — NOT YET IMPLEMENTED" note.
- Partial: quickstart.md text is out of date relative to the actual code (US2 reads as unbuilt but
  isn't).
- Not implemented: nothing found missing against spec.md's two user stories.
- Checkbox accuracy: N/A (no tasks.md); quickstart.md's prose status is stale/inaccurate for US2.
- Operator gates owed: real classifier + retrained-geometry training runs — real run artifacts exist
  on disk (`output/old/v50/v50.1/terrain_features/terrain_features-authored-v{1,2}/`, `ood-review-
  v{1,2}/`), so at least one real pass already happened (matches project memory of a measured
  -21.35% road-region height-error result), though the exact receipt JSON wasn't located in this pass.
- Open residue (spec-stated only): none identified beyond the docs being stale.
- Superseded by / overlaps: supersedes Spec 114 US3 (land-feature classification) per its own
  "Relationship to Spec 114" section; consumed as a proven input by Spec 123's design.
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: ML terrain/minimap dataset tooling
- Confidence: medium (no tasks.md/receipts file to cross-check counts against; conclusion rests on
  direct code+on-disk-artifact inspection)

---

### 117 WDL-Lattice Coarse Prior for Terrain Geometry
- Stated status: Draft | Tasks: 21/21 checked
- Scope: Export the settled 545-point WDL-lattice sampling contract as a real v50 signal, prove it's
  learnable standalone from minimap RGB, then bridge a frozen predictor into the existing
  coarse/detailer `--feature-store` contract.
- Verified implemented: `data-harvester/src/harvester/spec117/{lattice_contract,lattice_model,
  lattice_train,lattice_bridge,lattice_evaluate}.py` (lattice_evaluate.py is extra, beyond what
  tasks.md names — later addition); `data-harvester/tests/spec117/` (7 test files, all named tasks
  covered); `scripts/spec117_{train_lattice,lattice_to_feature_map}.py`; `STAGES` widened with
  `"lattice_prior"` in `harvester/v50/model_stage_contract.py`.
- Partial: none — US1/US2/US3(i) fully code-complete per tasks.md's own checkpoint language.
- Not implemented: US3(ii) — the real paired with/without training comparison — is explicitly
  "entirely user-run... nothing further to implement" per the Phase 5 checkpoint note.
- Checkbox accuracy: accurate.
- Operator gates owed: real US2 standalone-predictor training + US3(ii) paired comparison.
- Open residue (spec-stated only): the paired with/without relief-region comparison (US3 acceptance
  2-3, SC-003/SC-004).
- Superseded by / overlaps: **Spec 123 explicitly declares this spec's underlying premise closed** —
  "Supersedes/closes: Spec 117 (RGB->WDL-lattice-from-scratch, plateaued above tile-mean)... decisive
  negative evidence for predicting a coarse prior from RGB alone." The exported-signal
  infrastructure (US1) remains valid; the standalone-predictor modeling direction (US2/US3) is
  superseded by Spec 123's real-WDL-file approach.
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: ML terrain/minimap dataset tooling
- Confidence: high

---

### 118 Per-Object Occlusion-Aware Masks for Object-Deconfounded Terrain Height
- Stated status: Draft | Tasks: 33/33 checked (incl. Phase 5b augmentation, T031-T033)
- Scope: Correct, per-object, occlusion-aware (visible-portion-only) object mask + class + instance
  id as a v50 signal; object-masked terrain-height loss; a from-scratch object segmenter whose
  output bridges into the same `--feature-store` contract Spec 115/117 use, stackable alongside them.
- Verified implemented: C# `TerrainVisibleObjectMaskRasterizer.cs` extended with `visibleInstance`
  paint (confirmed via grep: front-most-fragment instance-id write inside the existing visibility
  guard); `AdtTensorPackBuilder.cs` instance-table wiring; full
  `data-harvester/src/harvester/spec118/` package (`object_contract`, `object_mask_audit`,
  `object_loss`, `object_segment_{model,train,infer}`, `object_feature_bridge`); shared
  `harvester/v50/feature_stores.py` making `--feature-store` repeatable across all three trainers
  (`direct_geometry_train.py`, `geometry_detailer_train.py`, `direct_geometry_materialize.py`) so the
  object prior stacks with, not replaces, Spec 115's terrain-feature prior.
- Partial: none — this is the most completely delivered spec in the batch by task count and depth.
- Not implemented: nothing found missing against spec.md's three user stories.
- Checkbox accuracy: accurate.
- Operator gates owed: real store rebuild carrying the new instance array; real segmenter training +
  paired object-masked-loss comparison (US2/US3's real runs are user-executed per FR-012).
- Open residue (spec-stated only): the real paired training comparisons (US2 acceptance 1, US3
  acceptance 1-2) — code/CLI complete, runs not yet evidenced in this pass.
- Superseded by / overlaps: directly answers and supersedes the informal "object masks" gap tracked
  in user memory (`feedback_object_masks_occlusion_aware_for_loss.md`) — that memory item should be
  considered resolved, not open.
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: ML terrain/minimap dataset tooling
- Confidence: high

---

### 123 Ground-Up v50 Terrain Height Model — Real WDL Prior + Residual Detailer
- Stated status: Draft | Tasks: no tasks.md, no plan.md (spec.md + checklists/requirements.md only)
- Scope: Replace every prior synthetic/predicted coarse-prior attempt (Specs 094/117/121) with the
  real per-map `.wdl` client file (via the existing `WdlSummaryReader`) as the coarse prior, merged
  with synthetic fallback only where uncovered, feeding exactly one residual detailer trained with
  prior-dropout for graceful degradation, using Spec 122's curation manifest for data selection and
  Spec 115's proven semantic classifier as an anti-confound input channel.
- Verified implemented: nothing. `WdlSummaryReader.cs` exists (pre-existing, referenced not built by
  this spec) and `--wdl-prior-dropout` exists in `terrain_refiner_train.py` (pre-existing V7 lineage,
  the precedent this spec says it will reuse) — both are cited prior art, not new work. No
  `spec123` package, no `real_wdl_prior` module, no new v50 catalog signal for a real-WDL-derived
  prior found anywhere in `data-harvester/src/harvester/`.
- Partial: none.
- Not implemented: all four user stories (US1-US4), all 19 FRs, all 8 SCs — this is a pure design
  document with zero delivered code.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: everything — no training, no harvest, no code exists to run yet.
- Open residue (spec-stated only): the entire spec is residue — it is this project's current stated
  design for the next terrain-height model iteration, explicitly built on the verified findings from
  Specs 094/115/116/117/121/122.
- Superseded by / overlaps: declares Specs 117 and 121 closed (see 117's entry above); depends on
  Spec 122 (dataset curation) and Spec 116 (spatially-isolated split), neither in this batch.
- Disposition: FOLD
- Proposed epic theme: ML terrain/minimap dataset tooling (next-model design)
- Confidence: high

---

### 124 Legacy Python Lane Detangling + New C# RunPod Tooling for v50
- Stated status: Draft | Tasks: no tasks.md, no plan.md (spec.md + checklists/requirements.md only)
- Scope: (1) physically relocate confirmed-dead pre-v50 Python lanes (V14/D1, R1, V15, most of V16,
  V16.2, V17, V19, V20, V21, Spec102/V25, an unversioned research island) to an archive location with
  git history intact, leaving load-bearing older-looking files (`v16_curation.py`, `v16_1_dataset.py`,
  V18, V22, V23, V24, Spec103, Spec108, Spec111) explicitly untouched; (2) new C# tools for v50 RunPod
  bundle packaging and pod provisioning, without touching the existing working V23/V24/Spec103
  RunPod infrastructure.
- Verified implemented: nothing. `data-harvester/src/harvester/` still directly contains
  `d1_model.py`, `r1_dataset.py`, `r1_model.py`, `v15_dataset.py`, `v15_model.py`, `v16_dataset.py`,
  `v16_model.py`, `v16_1_dataset.py`, `v16_1_models.py`, `v16_2_dataset.py`, `v16_2_models.py`,
  `v18_dataset.py`, `v19_dataset.py`, `v19_losses.py`, `v19_models.py`, `v20_dataset.py`,
  `v20_models.py`, `v21_scar_*.py`, `pm4_asset_matching/` — none moved to any archive location. No
  `tools/*RunPod*` C# project found anywhere in `wow-viewer/tools/`.
- Partial: none.
- Not implemented: US1 (archival move), US2 (C# packaging tool), US3 (C# provisioning tool), US4
  (load-bearing/uncertain documentation record) — all four user stories, zero delivered.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: none yet reachable — no code exists to run.
- Open residue (spec-stated only): the entire spec is residue — a still-valid, still-unexecuted
  cleanup + tooling plan; its governing audit (which files are safe to move) was performed in-session
  when the spec was written and is recorded in spec.md itself, so re-verification before any future
  move is still required per its own Governing Principle.
- Superseded by / overlaps: none in this batch; overlaps generally with Spec 109's cleanup-manifest
  machinery (a precedent pattern, not the same target) and with Spec 224 governance's monthly cleanup
  cadence.
- Disposition: FOLD
- Proposed epic theme: Infrastructure & Governance / ML dataset tooling cleanup
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 108 | FOLD | 3 (T012/T013/T015 user-run training+validation) | ML terrain/minimap dataset tooling |
| 109 | ARCHIVE-COMPLETE | 2 (T050 post-cleanup verify, T051 memory-bank compression) | ML terrain/minimap dataset tooling (foundation) |
| 111 | ARCHIVE-COMPLETE | 2 (T009 real bucketing pass, T019 gated training run) | ML terrain/minimap dataset tooling |
| 112 | ARCHIVE-SUPERSEDED | 0 (US3 training subsumed by Spec 114 T017) | ML terrain/minimap dataset tooling |
| 114 | FOLD | 2 (Phase 6 texture-family selection, Phase 7 alpha-stack reconstruction) | ML terrain/minimap dataset tooling |
| 115 | ARCHIVE-COMPLETE | 0 | ML terrain/minimap dataset tooling |
| 117 | ARCHIVE-SUPERSEDED | 1 (US3(ii) paired comparison, largely moot per Spec 123) | ML terrain/minimap dataset tooling |
| 118 | ARCHIVE-COMPLETE | 2 (US2/US3 real paired training comparisons) | ML terrain/minimap dataset tooling |
| 123 | FOLD | 1 (entire spec — next-model design, zero code yet) | ML terrain/minimap dataset tooling (next-model design) |
| 124 | FOLD | 1 (entire spec — cleanup+tooling plan, zero code yet) | Infrastructure & Governance / ML dataset tooling cleanup |
