# Implementation Plan: Image-Only WDL Prior

**Branch**: `108-image-wdl-prior` | **Date**: 2026-07-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a small, independent RGB-only model that predicts the verified paired WDL lattice: a 17×17 outer grid and a 16×16 inner grid. The active corpus is the existing varied analytic, authored-lighting synthetic store. Train the WDL prior from that store, generate a row-addressed predicted-WDL archive from its RGB rows, and train V8 using that archive rather than a ground-truth WDL channel. Real-client motif archaeology is separate deferred prefab-synthesis research, not a substitute training corpus or a gate on this lane.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch, NumPy, Zarr, PyArrow, uv  
**Storage**: Existing paired Zarr stores; checkpoint `.pt`; generated-prior `.npz` archive  
**Testing**: pytest through `uv run python -m pytest`  
**Target Platform**: Local CUDA for user-run training; CPU for unit and contract tests  
**Project Type**: Data-harvester library plus CLI tools  
**Performance Goals**: Small enough to train on curated representatives; one 256×256 RGB tile produces 545 values in one forward pass  
**Constraints**: Inference reads RGB only; WDL is exactly `::16` outer plus `8::16` inner; no training is started by the agent  
**Scale/Scope**: One prior model, one prediction archive contract, one V8 generated-prior entry point

## Constitution Check

- Repo independence: pass; every file remains below `wow-viewer/`.
- Residual model chain: pass; this is one independent prior model, not a new V8 head.
- Training contract: pass; target, normalization, split, checkpoint, and validation are documented here.
- Synthetic signoff: user-run crater-family holdout, generated-prior archive, and V8 label-free validation. Real-client evaluation is optional and never a corpus substitution.

## Project Structure

```text
data-harvester/
├── src/harvester/spec103/wdl_prior_model.py
├── src/harvester/spec103/wdl_prior_io.py
├── scripts/train_spec103_wdl_prior.py
├── scripts/infer_spec103_wdl_prior.py
├── scripts/infer_spec103_v7.py
└── tests/spec103/test_wdl_prior_sanity.py
```

**Structure Decision**: Keep the learned component, archive contract, and WDL mapping in the harvester library. CLI scripts only load stores and invoke those shared seams.

## Phases

1. Pin the paired WDL and generated-prior archive contracts. **Complete.**
2. Add the RGB-only predictor, trainer, inference writer, visual review artifacts, and CPU tests. **Complete.**
3. Train the WDL prior with a complete synthetic terrain-family holdout and inspect the held-out synthetic visual review. **User-run.**
4. Generate a complete row-addressed predicted-WDL archive from that checkpoint. The archive must bind to the exact synthetic store and include every selected V8 train/validation row. **User-run.**
5. Train V8 against generated outer priors with WDL-derived height hints, then run generated-prior synthetic inference and the label-free harness. **User-run.**
6. Deferred, separate research: run chunk-cell motif archaeology on 0.5.3 only if building a later prefab-derived synthetic corpus. It neither feeds nor blocks phases 3–5.

## Complexity Tracking

No constitution violation or added architecture is required.
