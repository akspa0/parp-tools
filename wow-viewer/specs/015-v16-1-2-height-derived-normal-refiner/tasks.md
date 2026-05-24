# Tasks: V16.1.2 Height-Derived Normal Refiner

**Input**: `specs/015-v16-1-2-height-derived-normal-refiner/spec.md` and `plan.md`

---

## Phase 1: Refiner Model Definition

**Goal**: Add `V161NormalRefiner` to the model definitions.

- [ ] T001 [US1] Add `V161NormalRefiner` class to `wow-viewer/data-harvester/src/harvester/v16_1_models.py`:
  - Input: 4ch (pred_normals 3ch + height_norm 1ch)
  - 3 residual blocks (Conv2d→BN→ReLU→Conv2d→BN, skip connection)
  - Output: 3ch Tanh
  - Export it from the module so `train_v16_1_common.py` can import it

**Checkpoint**: Refiner model exists and can be instantiated with `V161NormalRefiner()(4ch_tensor) → 3ch_tensor`

---

## Phase 2: Refiner Loss and Eval

**Goal**: Add refiner evaluation at best-epoch triggers with before/after loss comparison.

- [ ] T002 [US1] Add `_refiner_refine_and_compare` function in `train_v16_1_common.py`:
  - Takes main_model, refiner, batch, device — runs refiner on `cat(pred.detach(), height_norm)`
  - Computes `L_main(pred, gt)` and `L_main(refined, gt)` using same cosine/vec/nz loss
  - Returns `(refined_normals, refined_loss, raw_loss, improved_bool)`
- [ ] T003 [US1] Wire best-epoch trigger in `run_task`:
  - After the existing best-epoch save, import refiner, instantiate it, run `_refiner_refine_and_compare` on full validation set
  - Log `refiner_improved`, `refiner_loss`, `raw_loss` to the epoch log entry
  - If improved, save `refiner_state_dict` to checkpoint and set `refiner_active = True`

**Checkpoint**: First best-epoch trigger in a bounded smoke run logs `refiner_improved=true/false` with quantitative loss values.

---

## Phase 3: Distillation Loop

**Goal**: When refiner is active, add `L_distill` to the training loss.

- [ ] T004 [US2] Add distillation path in the training loop:
  - After main forward pass, if `refiner_active`: `teacher = refiner(pred.detach(), height_norm)`
  - Compute `L_distill = _masked_mean(cosine(pred, teacher), train_mask)`
  - `L_total = L_main + w_distill * L_distill`
  - Freeze refiner weights (no gradient flow into refiner)
- [ ] T005 [US2] Add CLI flag `--refiner-distill-weight` (default 0.25) to `_parse_args`
- [ ] T006 [US2] Log `train_distill_loss` to epoch metrics when active

**Checkpoint**: A bounded smoke run with refiner active shows `train_distill_loss` in the log, and total loss includes the weighted distill term.

---

## Phase 4: Validation Preview + Wiring

**Goal**: Show refiner output in validation previews. Wire all CLI flags. Use separate runs folder.

- [ ] T007 [US2] Add `refined_gt` panel to `_preview_normal`:
  - After `normal_gt` and `normal_pred`, add a panel showing `_normals_to_rgb(refined[0])`
- [ ] T008 [US3] Add `--refiner-disabled` flag (default False) to bypass refiner entirely
- [ ] T009 [US3] Run directory: auto-create under `normal/runs/v16_1_2_<name>` when refiner is enabled; keep `v16_1_1_<name>` for refiner-disabled mode

**Checkpoint**: A bounded smoke run with `--refiner-disabled` behaves identically to V16.1.1. A run without that flag uses the `v16_1_2_*` runs folder.

---

## Phase 5: Resume from V16.1.1 Checkpoint

**Goal**: Launch V16.1.2 resuming from the current V16.1.1 checkpoint.

- [ ] T010 [US1] Test resume: launch a bounded smoke run with `--resume-checkpoint <v16_1_1_last.pt>` and verify the main model loads, refiner initializes fresh, and first epoch validates correctly.

**Checkpoint**: A command like:
```
uv run python train_v16_1_normal.py \
  --resume-checkpoint ../models/v16_1/normal/runs/v16_1_1_normal_pool800_epoch256_autotune12_compile/checkpoints/v16_1_normal_last.pt \
  --run-name v16_1_2_smoke_refiner \
  --train-max-tiles 400 --train-epoch-tiles 128 --val-max-tiles 48 \
  --autotune-batch-size --target-vram-gb 14
```
...completes epoch 1 and logs the refiner comparison.

---

## Dependencies & Execution Order

- **Phase 1 → Phase 2**: Model must exist before eval can be written
- **Phase 2 → Phase 3**: Eval must prove refiner wins before distillation is worth adding
- **Phase 2 → Phase 4**: Eval outputs must exist before preview panels can show them
- **Phase 5**: Can run as soon as Phase 2 is complete (smoke test against real checkpoint)

### Parallel Opportunities

- T002 and T003 are tightly coupled (single function + wiring) — same-developer
- T004, T005, T006 are tightly coupled — same-developer
- T007, T008, T009 can run in parallel (different sections of the same file)

---

## Implementation Order (Recommended)

1. T001 — Refiner model class
2. T002 + T003 — Eval at best-epoch (smoke to see if refiner improves)
3. **PAUSE: validate with operator** — does the refiner actually improve loss?
4. T004 + T005 + T006 — Distillation loop
5. T007 + T008 + T009 — Previews, flags, runs folders
6. T010 — Resume from V16.1.1 checkpoint