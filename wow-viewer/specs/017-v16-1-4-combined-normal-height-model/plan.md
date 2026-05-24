# Implementation Plan: V16.1.4 Combined Normal + Height Model

**Branch**: `017-v16-1-4-combined-normal-height-model` | **Date**: 2026-05-24 | **Spec**: `specs/017-v16-1-4-combined-normal-height-model/spec.md`

## Summary

Train one model that predicts both normals and height from minimap. Two output heads share a backbone. Single checkpoint, single export, no numerical integration.

## Project Structure

```
wow-viewer/data-harvester/
├── src/harvester/v16_1_models.py          # + V161NormalHeightCombinedModel
├── scripts/train_v16_1_common.py          # + combined loss, CLI flags
├── scripts/train_v16_1_combined.py        # new entrypoint
└── scripts/export_terrain_obj.py          # support combined model loading
```

## Implementation Phases

### Phase 1: Model + Loss

1. Add `V161NormalHeightCombinedModel` to `v16_1_models.py` — shared backbone, two heads
2. Add `_combined_loss` to `train_v16_1_common.py` — normal loss + height loss with weights
3. Register in TASKS dict

### Phase 2: CLI + Launch

4. Add `--normal-weight` and `--height-weight` flags
5. Write `train_v16_1_combined.py` entrypoint
6. Smoke test (400 train, 48 val, 20 epochs)
7. Launch 1000-epoch run
