# Tasks: Object-Library Segmentation & Classifier (Spec 119)

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Research**: [research.md](research.md)

All Python under `wow-viewer/data-harvester/` (`src/harvester/spec119/`, `scripts/spec119_*.py`,
`tests/spec119/`). Run via `cd wow-viewer/data-harvester && uv run ...`. Tests are included (TDD)
because the spec mandates testable FRs and the project convention is test-first for library code.

## Phase 1 — Setup

- [ ] T001 Create `wow-viewer/data-harvester/src/harvester/spec119/__init__.py` (empty package marker)
- [ ] T002 Create `wow-viewer/data-harvester/tests/spec119/__init__.py` (empty package marker)

## Phase 2 — Foundational: contract + split (the highest-risk requirement first)

- [ ] T003 Implement `harvester/spec119/object_library_contract.py`: `COARSE_CLASS_INDEX` map (`{"empty":0,"m2":1,"mdx":2,"wmo":3}`), `coarse_label_for_row()`, `derive_asset_family()` (parent dir of normalized path), `derive_fine_family_label()` (grandparent dir token), `segmentation_target()` (mask>0 → binary), `is_blank_capture()` (coverage < threshold), `BLANK_THRESHOLD_DEFAULT=0.01`. Pure functions, no I/O.
- [ ] T004 Widen `harvester/v50/model_stage_contract.py` `STAGES` with `"object_library_classifier"` and `"object_library_segmenter"` (two new entries; schema unchanged)
- [ ] T005 [P] Write `tests/spec119/test_contract.py`: class-index round-trip, `derive_asset_family` on sample paths, `derive_fine_family_label`, blank-threshold relabel (coverage 0.005→empty, 0.02→kept), `segmentation_target` shape/dtype, refusal on bad asset_type
- [ ] T006 Implement `harvester/spec119/split.py`: `build_family_split(rows, held_out_fraction, seed)` → family-isolated split (D-01); `leakage_check(split, rows)` → `verified_violation_count` (numeric-suffix variant pairs must not straddle); deterministic from seed
- [ ] T007 Implement `scripts/spec119_build_split.py`: CLI per [cli-contract.md](contracts/cli-contract.md) §1; dry-run-first (prints counts + violation count, exits without `--write`); refuses to write if `verified_violation_count > 0`
- [ ] T008 [P] Write `tests/spec119/test_split.py`: family isolation (no family in both halves), leakage-check refusal on a synthetic leaky fixture (castle01 in train, castle02 in held-out → violation), determinism (same seed → same split), row-count accounting
- [ ] T009 Run `spec119_build_split.py` dry-run on the real smoke store (`output/object-library-smoke/smoke_wmo.zarr`) to confirm it parses the zarr + parquet and prints counts (read-only, no `--write`)

**Phase 2 exit**: split builds with `verified_violation_count=0`; `tests/spec119/` green; ruff + `py_compile` clean. No model exists yet.

## Phase 3 — User Story 1: Classifier (P1)

- [ ] T010 [US1] Implement `harvester/spec119/classifier_model.py`: `ObjectClassifier` (conv encoder + global pool + linear head, input 128×128×3, output = len(COARSE_CLASS_INDEX) logits), constructable from `base` alone (D-02); `compute_class_weights(labels)` (inverse-frequency, FR-007); `majority_class_baseline(labels)` (FR-005)
- [ ] T011 [P] [US1] Write `tests/spec119/test_classifier_model.py`: forward-pass output shape, param count at `--base 16` (assert < 1M, SC-005), base-only reconstruction round-trip (build from base → load state_dict → forward matches), class-weight computation, majority-class baseline
- [ ] T012 [US1] Implement `harvester/spec119/classifier_train.py`: dataset reader (zarr capture_rgb + parquet labels, blank→empty class per D-04), dry-run-first trainer reusing `harvester.v50.lr_schedule.make_onecycle_scheduler` (warmup-aware stale counter, D-05), class-weighted CE, per-class precision/recall in metrics, majority-class baseline in baselines, `v50-model-stage-run-v1` record (`stage=object_library_classifier`, `promotion_verdict=pending`)
- [ ] T013 [US1] Implement `scripts/spec119_train_classifier.py`: CLI per cli-contract §2; dry-run-first (prints param count, train/held-out counts, majority-class baseline, class weights, exits without `--confirm-run`); `--fine-labels` flag switches to heuristic FineFamilyLabel (run record marks it heuristic)
- [ ] T014 [P] [US1] Write `tests/spec119/test_classifier_train.py`: dry-run plan shape (has param_count, majority_baseline, class_weights, train_count, held_out_count), missing-split refusal, `--fine-labels` marks run heuristic, `--help` argparse verification, missing-store refusal
- [ ] T015 [US1] Run `spec119_train_classifier.py` dry-run on the real smoke store + smoke split to confirm the plan prints (read-only, no `--confirm-run`)

**Phase 3 exit (code-verified)**: dry-run prints param count + majority-class baseline + class weights. **User-run gate (SC-001)**: `--confirm-run` training must beat majority-class baseline by ≥15pp. If it fails, stop and diagnose before Phase 4.

## Phase 4 — User Story 2: Segmenter (P2) — only after Phase 3 passes SC-001

- [ ] T016 [US2] Implement `harvester/spec119/segmenter_model.py`: `ObjectSegmenter` (U-Net-lite: strided double-conv encoder 128→64→32→16 + skip decoder back to 128, single binary foreground channel, D-02), constructable from `base` alone
- [ ] T017 [P] [US2] Write `tests/spec119/test_segmenter_model.py`: forward-pass output shape (1,1,128,128), param count at `--base 16` (< 1M, SC-005), base-only reconstruction round-trip, sigmoid output in [0,1]
- [ ] T018 [US2] Implement `harvester/spec119/segmenter_train.py`: dataset reader (zarr capture_rgb + capture_mask, blank-capture EXCLUSION per D-04), dry-run-first trainer reusing lr_schedule, BCE loss, all-foreground + all-background trivial IoU in baselines (SC-002), per-coverage-bucket IoU in metrics, `stage=object_library_segmenter` run record
- [ ] T019 [US2] Implement `scripts/spec119_train_segmenter.py`: CLI per cli-contract §3; dry-run-first (prints param count, trivial baselines, exclusion count, train/held-out counts)
- [ ] T020 [P] [US2] Write `tests/spec119/test_segmenter_train.py`: dry-run plan shape (has param_count, trivial_baselines, exclusion_count), blank-exclusion count > 0 on a fixture with a blank row, missing-split refusal, `--help` argparse
- [ ] T021 [US2] Run `spec119_train_segmenter.py` dry-run on the real smoke store + smoke split (read-only, no `--confirm-run`)

**Phase 4 exit (code-verified)**: dry-run prints param count + trivial baselines + exclusion count. **User-run gate (SC-002)**: `--confirm-run` training must beat the better trivial baseline by ≥0.20 IoU.

## Phase 5 — User Story 3: Inference + quality lens (P3) — only after Phase 3 is trained

- [ ] T022 [US3] Implement `harvester/spec119/infer.py`: `load_classifier_checkpoint()` / `load_segmenter_checkpoint()` (reconstruct architecture from `base`, refuse if missing — D-02); `infer_classifier(model, image)` → class+confidence+probs; `infer_segmenter(model, image)` → binary mask
- [ ] T023 [US3] Implement `scripts/spec119_infer.py`: CLI per cli-contract §4; loose-PNG input (FR-013); classifier → JSON, segmenter → `<stem>_mask.png`; refuses checkpoint missing `base`
- [ ] T024 [P] [US3] Write `tests/spec119/test_infer.py`: classifier JSON shape (has predicted_class, confidence, per_class_probs), segmenter mask-PNG write (255/0), architecture-reconstruction refusal on a checkpoint missing `base`, runs on a loose PNG with no store
- [ ] T025 [US3] Implement `harvester/spec119/quality_lens.py`: `compute_embeddings(classifier, store)` → penultimate-layer vectors (FR-009: `eval()`/`no_grad()`/no stochastic ops); `find_mislabels(embeddings, predictions, labels)`; `find_near_duplicates(embeddings, threshold, top_k)` (cosine similarity); `flag_low_coverage(rows, threshold)`
- [ ] T026 [US3] Implement `scripts/spec119_quality_lens.py`: CLI per cli-contract §5; dry-run-first (prints summary counts, exits without `--write`); `--write` → `embeddings.parquet` + `quality_report.json`
- [ ] T027 [P] [US3] Write `tests/spec119/test_quality_lens.py`: embedding determinism (recompute → byte-identical for frozen checkpoint), near-duplicate pair detection on a synthetic fixture (two identical vectors → pair found, two orthogonal → no pair), mislabel report sorting (by wrong-class confidence desc), low-coverage flag list
- [ ] T028 [US3] Run `spec119_quality_lens.py` dry-run on the real smoke store + a random-init classifier checkpoint (no CUDA training) to confirm it runs end-to-end and prints summary counts (read-only, no `--write`)

**Phase 5 exit (code-verified)**: quality lens dry-run prints summary counts. **User-run gate (SC-004)**: `--write` then manually inspect top-flagged mislabels (≥50% genuine).

## Phase 6 — Polish & cross-cutting

- [ ] T029 Run full `data-harvester` test suite (`uv run pytest`) + ruff (`uv run ruff check`) + `py_compile` on all touched files; confirm no regressions beyond pre-existing unrelated failures
- [ ] T030 Update `wow-viewer/memory-bank/activeContext.md` + `progress.md` with Spec 119 status (code-verified through Phase 5 dry-runs; user-run training gates remain)

## Dependencies (story completion order)

```
Phase 2 (split) ──► Phase 3 (US1 classifier) ──┬──► Phase 4 (US2 segmenter)
                                                 └──► Phase 5 (US3 quality lens, needs trained US1)
Phase 6 (polish) after all phases
```

- US2 depends on US1 passing SC-001 (Rule 8: don't build the harder model if the easy one failed).
- US3 depends on US1 being *trained* (needs a frozen classifier checkpoint), not just code-complete.
- US2 and US3 are otherwise independent of each other.

## MVP scope

**Phase 2 + Phase 3 only** (the split + classifier). This delivers the cheapest learnability
verdict (SC-001) and is the gate for everything else. If the classifier cannot beat the
majority-class baseline on clean captured images, the segmenter and quality lens are not worth
building — diagnose the data first.
