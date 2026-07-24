# Implementation Plan: Object-Library Segmentation & Classifier (Spec 119)

**Branch**: `119-object-library-classifier` | **Date**: 2026-07-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/119-object-library-classifier/spec.md`

## Summary

The object-library zarr (Spec 118's `capture-objects` pipeline, now WMO-inclusive) is a
self-contained, labeled image dataset — `capture_rgb`/`capture_mask` per asset plus
`asset_type`/path provenance in `assets.parquet` — but nothing yet *learns* from it. This feature
trains two small, from-scratch, independently checkpointed specialists on the library itself: a
**classifier** (US1, captured image → object class) and a **segmenter** (US2, captured image →
per-pixel object mask), then a **quality lens** (US3) that turns the frozen classifier into a
mislabel/near-duplicate auditing tool plus a per-asset embedding.

This is the v50-era successor to the completed Spec 025 object-roof-mask-library model, and is
deliberately distinct from Spec 118 US3's `ObjectSegmentNet` (which segments **minimap tiles**;
Spec 119 segments **captured object crops** — a different input surface with a different supervision
source: the library's own masks/labels, not terrain-tile ground truth).

Delivery order: split (the highest-risk requirement, FR-004) → US1 classifier (the cheap
learnability gate) → US2 segmenter (only if US1 passes SC-001) → US3 quality lens (only if US1 is
trained). No new C# — the library already exists; this is pure Python under
`wow-viewer/data-harvester/`.

## Technical Context

**Language/Version**: Python 3.11+ managed by `uv` under `wow-viewer/data-harvester/`. No C# — the
object-library zarr is already produced by the Spec 118 capture pipeline.

**Primary Dependencies**: PyTorch (CPU dry-run + user-run CUDA), NumPy, Zarr v3 + PyArrow (read the
library; write derived artifacts). No new dependency beyond what Specs 114–118 added.

**Storage**: Reads the object-library zarr read-only (FR-011); writes derived artifacts
(checkpoints, embeddings, quality reports, run records) to a separate `<output-root>/<run-name>/`
dir. Zarr/parquet/JSON only; no NPZ (constitution V).

**Testing**: `pytest` under `data-harvester/tests/` (new `tests/spec119/`); Ruff + `py_compile`;
dry-run-first CLIs refuse to write/train without explicit flags (FR-010).

**Target Platform**: CPU for split-building, dry-runs, unit tests, and the quality lens; CUDA for
user-run training. No client-path assumptions (the library is already harvested; Rule 9 does not
apply to the training step).

**Project Type**: A new Python package `harvester/spec119/` (contract/model/train/infer/quality)
mirroring the Spec 116/117/118 convention + thin `scripts/spec119_*.py` CLIs + a two-entry widening
of `model_stage_contract.STAGES`.

**Performance Goals**: Both models stay in the small class (≤ single-digit-millions of params,
SC-005): classifier ~100–300K at `--base 16`, segmenter ~200–500K at `--base 16`.

**Constraints**: User executes all training (FR-010); dry-run-first CLIs; two independent
checkpoints (Rule 7, no multi-task/shared weights); family-isolated split with a mandatory leakage
check (FR-004); majority-class + trivial baselines always reported (FR-005/SC-002); blank captures
handled per D-04; retrieval integration explicitly deferred (FR-012).

**Scale/Scope**: One object-library zarr (build 0.5.3.368, all M2/MDX/WMO). Two new small models;
one new split; one quality lens. No new harvest, no C#, no trainer changes outside Spec 119.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Repo Independence | PASS | All new code under `wow-viewer/data-harvester/` (`src/harvester/spec119/`, `scripts/`, `tests/spec119/`). No references outside `wow-viewer/`. |
| II. Library-First | PASS | Logic in `src/harvester/spec119/`; `scripts/` stay thin CLIs. Reuses `harvester.object_library`, `harvester.v50.*` — no duplication. |
| III. Real-Data Validation | PASS | Trains/scores against the real object-library zarr (WMO-inclusive, validated this session). Quality lens runs on the full real library. |
| IV. Residual Model Chain | PASS | Two single-output specialists (classifier: image→class; segmenter: image→mask), independently checkpointed, feeding downstream by output only. No shared weights (Rule 7). |
| V. Streaming-First Dataset | PASS | Reads the zarr library; writes parquet/JSON/zarr-derived artifacts. No NPZ. |
| VI. No Game Client Path Assumptions | PASS | The library is already harvested; training takes `--store` as a CLI arg. No client path in source. |
| VII. Small Modular Specialists | PASS | Two tiny nets, each one input/one output, independently trained/replaced (Rule 7). |
| VIII. One Phase at a Time | PASS | Phased delivery below; US2 does not start until US1 passes SC-001; US3 does not start until US1 is trained. |
| IX. User Runs Heavy Work | PASS | All training is `--confirm-run`-gated, dry-run-first (FR-010). The assistant prepares commands only. |
| X. Doc Hygiene | PASS | Spec/plan/research/data-model/contracts/quickstart written before any code (this document set). Memory bank updated after the WMO fix (prior step). |

*Post-Phase-1 re-check*: no changes — the design adds no new dependencies, no C#, no client paths,
and keeps the two-model independence. PASS.

## Phased Delivery

### Phase 0 — Contract + split (the highest-risk requirement first)

1. `harvester/spec119/__init__.py` + `object_library_contract.py`: the `CoarseClassLabel`
   int↔str map, `AssetFamily` derivation, `SegmentationTarget` derivation, blank-threshold logic,
   the two new `STAGES` names. Pure functions, fully unit-testable.
2. `spec119_build_split.py` + `harvester/spec119/split.py`: family-isolated held-out split (D-01)
   + the leakage check (`verified_violation_count`). Dry-run-first; refuses a leaky split.
3. `tests/spec119/test_contract.py` + `test_split.py`: class-map round-trip, family derivation,
   blank-threshold relabel, leakage-check refusal on a synthetic leaky fixture, determinism
   (same seed → same split).

**Phase 0 exit**: split builds on the real library with `verified_violation_count=0`; tests green;
ruff/`py_compile` clean. **No model exists yet.**

### Phase 1 — Classifier (US1, P1)

4. `harvester/spec119/classifier_model.py`: the small conv encoder + linear head, constructable
   from `base` alone (D-02). `compute_class_weights` (FR-007).
5. `spec119_train_classifier.py` + `harvester/spec119/classifier_train.py`: dry-run-first trainer
   reusing `lr_schedule.make_onecycle_scheduler` (warmup-aware, D-05); majority-class baseline in
   `baselines` (FR-005); per-class precision/recall in `metrics`; blank→`empty` class (D-04);
   `v50-model-stage-run-v1` record (`stage=object_library_classifier`).
6. `tests/spec119/test_classifier_model.py` + `test_classifier_train.py`: param count at known
   `base`, base-only reconstruction round-trip, dry-run plan shape, missing-split refusal,
   class-weight computation, `--help` argparse verification.

**Phase 1 exit (code-verified)**: dry-run on the real library prints param count + majority-class
baseline + class weights + train/held-out counts. **User-run gate (SC-001)**: `--confirm-run`
training must beat the majority-class baseline by ≥15pp. If it fails, stop and diagnose before US2.

### Phase 2 — Segmenter (US2, P2) — only after Phase 1 passes SC-001

7. `harvester/spec119/segmenter_model.py`: U-Net-lite (128→64→32→16 + skip decoder), binary
   single-channel, constructable from `base` alone (D-02).
8. `spec119_train_segmenter.py` + `harvester/spec119/segmenter_train.py`: dry-run-first trainer;
   all-foreground/all-background trivial IoU in `baselines` (SC-002); per-coverage-bucket IoU in
   `metrics`; blank-capture exclusion (D-04); `stage=object_library_segmenter` run record.
9. `tests/spec119/test_segmenter_*.py`: param count, base-only reconstruction, dry-run plan,
   blank-exclusion count, trivial-baseline computation.

**Phase 2 exit (code-verified)**: dry-run prints param count + trivial baselines + exclusion count.
**User-run gate (SC-002)**: `--confirm-run` training must beat the better trivial baseline by
≥0.20 IoU.

### Phase 3 — Inference + quality lens (US3, P3) — only after Phase 1 is trained

10. `spec119_infer.py` + `harvester/spec119/infer.py`: loose-PNG inference for both checkpoints
    (FR-013); reconstructs architecture from `base`; refuses a checkpoint missing `base`.
11. `spec119_quality_lens.py` + `harvester/spec119/quality_lens.py`: frozen classifier →
    `embeddings.parquet` (penultimate layer) + `quality_report.json` (mislabels, near-duplicate
    cosine clusters, low-coverage flags). Deterministic (FR-009: `eval()`/`no_grad()`).
12. `tests/spec119/test_infer.py` + `test_quality_lens.py`: loose-PNG classifier JSON shape,
    segmenter mask-PNG write, architecture-reconstruction refusal, embedding determinism
    (recompute → byte-identical), near-duplicate pair detection on a synthetic fixture.

**Phase 3 exit (code-verified)**: quality lens dry-run on the real library prints summary counts.
**User-run gate (SC-004)**: `--write` then manually inspect top-flagged mislabels (≥50% genuine).

## Validation gates (summary)

| Gate | Where | Criterion | Who runs |
|------|-------|-----------|----------|
| Leakage check | Phase 0 | `verified_violation_count=0` | assistant (read-only) |
| SC-001 classifier | Phase 1 | ≥15pp above majority-class baseline | USER (`--confirm-run`) |
| SC-002 segmenter | Phase 2 | ≥0.20 IoU above better trivial baseline | USER (`--confirm-run`) |
| SC-004 quality lens | Phase 3 | ≥50% top mislabels genuine on review | USER (`--write` + manual) |
| SC-005 model size | Phases 1–2 | single-digit-millions params, from-scratch | assistant (dry-run) |
| SC-006 no heavy self-launch | all | all training `--confirm-run`-gated | assistant discipline |

## Out of scope (do not pull in — Rule 8)

- Minimap-crop-to-library-asset retrieval (FR-012 — Spec 118 minimap chain).
- Multi-variant captures (rotated/scaled) — library has one variant per asset.
- Finer-than-coarse taxonomy as the primary metric (SC-001 is coarse; `--fine-labels` is secondary).
- Any C# change — the library already exists and is WMO-inclusive.
- Any change to the existing geometry/terrain trainers — Spec 119 is self-contained.
