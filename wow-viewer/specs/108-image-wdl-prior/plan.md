# Implementation Plan: Image-Only WDL Prior

**Branch**: `108-image-wdl-prior` | **Date**: 2026-07-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a small, independent RGB-only model that predicts the verified paired WDL lattice: a 17×17 outer grid and a 16×16 inner grid. The prior analytic-only store is a smoke corpus, not a universal-model corpus: lighting variants did not create terrain diversity, and one held-out synthetic tile is not a quality proof. The active next corpus is capped below 256 examples and mixes real and synthetic rows: 144 real 0.5.3 rows quota-balanced across Azeroth, Kalimdor, DeadminesInstance, and PVPZone02, selected by real irregular alpha/relative-height cell motifs; plus 96 diverse synthetic placements. All source groups stay intact across the split. Generate a row-addressed predicted-WDL archive from this mixed corpus, then train V8 using that archive rather than a ground-truth WDL channel.

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
- Mixed-corpus signoff: group-held-out real and synthetic examples, generated-prior archive, visual review across the held-out groups, and V8 label-free validation.

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
3. Build a fast bounded mixed-store selector: read metadata first, then one bounded 16x16-cell alpha/relative-height descriptor pass over 0.5.3 candidates. Retain irregular brush/paste motifs only; never zones or fixed windows. Write a 240-row mixed store with source-group and split evidence. **Implementation + user-run store build.**
4. Train the WDL prior with complete mixed source-group holdout and inspect a multi-row synthetic-and-real visual review. **User-run.**
5. Generate a complete row-addressed predicted-WDL archive from that checkpoint. The archive must bind to the exact mixed store and include every selected V8 train/validation row. **User-run.**
6. Train V8 against generated outer priors with WDL-derived height hints, then run generated-prior mixed inference and the label-free harness. **User-run.**

## Complexity Tracking

No constitution violation or added architecture is required.
