# Implementation Plan: V16.1.3 Height-Channel Normal Model

**Branch**: `016-v16-1-3-height-channel-normal-model` | **Date**: 2026-05-24 | **Spec**: `specs/016-v16-1-3-height-channel-normal-model/spec.md`

## Summary

Add height as an input channel to the V16.1.1 normal model. Single model, clean gradient flow, no separate refiner. 1000-epoch long run with 12GB VRAM autotune.

## Technical Context

**Language/Version**: Python 3.11+, PyTorch 2.x

**Primary Dependencies**: torch, numpy (same as existing V16.1 trainer)

**Storage**: No new Zarr arrays — height_257, height_mean, height_std already in dataset

**Testing**: Bounded smoke (400 train, 48 val), then 1000-epoch long run

**Target Platform**: CUDA GPU, 4070 Ti SUPER (16GB), autotune to 12GB

## Project Structure

```
wow-viewer/specs/016-v16-1-3-height-channel-normal-model/
├── spec.md
├── plan.md
└── tasks.md

wow-viewer/data-harvester/
├── src/harvester/v16_1_models.py          # + V161NormalHeightModel
├── src/harvester/v16_1_dataset.py         # + height_norm input channel
└── scripts/train_v16_1_common.py          # + height-channel wiring
```

## Implementation Phases

### Phase 1: Model + Dataset

1. Add `V161NormalHeightModel` to `v16_1_models.py` — same as `V161NormalModel` but `Conv2d(4, 64)` first layer
2. Add `height_channel` flag to `V161Dataset` — when enabled, append `height_norm` to the input tensor
3. Wire into training script: `--height-channel` flag selects the new model class and dataset mode

### Phase 2: Training Launch

4. Run bounded smoke (400 train, 48 val, 10 epochs) to verify architecture works
5. Launch 1000-epoch run with `--autotune-batch-size --target-vram-gb 12`
