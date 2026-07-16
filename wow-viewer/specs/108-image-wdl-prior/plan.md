# Implementation Plan: Image-Only WDL Prior

**Branch**: `108-image-wdl-prior` | **Date**: 2026-07-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a small, independent RGB-only model that predicts the verified paired WDL lattice: a 17×17 outer grid and a 16×16 inner grid. Generic analytic synthesis established a controlled baseline but transferred poorly (about 50 world-unit outer MAE on the first real tile). The next corpus is therefore grounded in recovered 0.5.3 terrain-art prefab families: analyze real full-map evidence first, then synthesize many placement/lighting variants from those measured families before handing the generated outer grid to V8.

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
- Real-data signoff: pending user-run representative-corpus experiment. Synthetic tests prove plumbing only.

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
3. Run chunk-cell motif archaeology on 0.5.3 Azeroth/Kalimdor: derive alpha and relative-height
   signatures for real terrain cells, grow recurring adjacent signatures into variable-shaped graphs,
   and group transform-equivalent placements. Preserve motifs across chunk/tile borders. Never use
   macro/blocky zones or fixed rectangular windows as the prefab payload. **User-run evidence gate;
   no generic-synthesis retrain first.**
4. Materialize many synthetic placements from those recovered prefab families, with transform,
   amplitude/relief, tileset/layout, and multi-time lighting variants. Hold out whole families.
5. Train the WDL prior on the prefab-derived corpus, inspect synthetic holdout plus 0.5.3 real-tile
   visual reports, then let V8 consume only the generated outer prior.

## Complexity Tracking

No constitution violation or added architecture is required.
