# Tasks: V16.1.3 Height-Channel Normal Model

## Phase 1: Model + Dataset

- [ ] T001 Add `V161NormalHeightModel` to `v16_1_models.py`
  - Same as `V161NormalModel` but `nn.Conv2d(4, 64, ...)` instead of `Conv2d(3, 64, ...)`
  - Forward: `cat(input, height_norm, dim=1)` → backbone → head → Tanh

- [ ] T002 Add `height_channel` flag to `V161Dataset.__init__`
  - When `height_channel=True`, the input tensor includes `height_norm` as 4th channel
  - `height_norm = (height_raw - h_mean) / h_std` (already computed, just not exposed as input)

- [ ] T003 Add `--height-channel` CLI flag to `train_v16_1_common.py`
  - Pass to `V161Dataset` constructor
  - Select `V161NormalHeightModel` when `task_name == "normal" and args.height_channel`
  - Auto-prefix run dir with `v16_1_3_`

## Phase 2: Launch

- [ ] T004 Smoke test: 400 train, 48 val, 10 epochs, batch-size 8, no-compile
- [ ] T005 Launch 1000-epoch run: `--epochs 1000 --target-vram-gb 12 --autotune-batch-size --height-channel`
