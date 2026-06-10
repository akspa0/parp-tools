# Tasks: V16.1.4 Combined Normal + Height Model

## Phase 1: Model + Loss

- [x] T001 Add `V161NormalHeightCombinedModel` to `v16_1_models.py`
  - Shared `_UNetBackbone(4)`, `normal_head` (3ch Tanh), `height_head` (1ch)
  - Forward returns `(normals, height)`

- [x] T002 Add `_combined_loss` to `train_v16_1_common.py`
  - Normal loss from `_normal_loss` (reuse existing)
  - Height loss: `_weighted_l1(height_pred, height_norm, weight_257)`
  - Combined: `w_normal * L_normal + w_height * L_height`
  - CLI flags: `--normal-weight` (default 1.0), `--height-weight` (default 1.0)

- [x] T003 Register `combined` task in TASKS dict + write `train_v16_1_combined.py`

## Phase 2: Launch

- [ ] T004 Smoke test: 400 train, 48 val, 20 epochs, batch-size 8, no-compile
- [ ] T005 Launch 1000-epoch run with 12GB autotune
