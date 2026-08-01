# Tasks: V24.1 DA-V2 Pretrained Convergence Model (Spec 101)

**Spec**: [spec.md](spec.md)
**Plan**: [plan.md](plan.md)

---

## Slice 1: StageADAV2 model class ✅

- [x] Create `StageADAV2` class in `stage_a.py` wrapping `DepthAnythingV2SmallEncoder`
- [x] DPT-style head: 4 neck levels → fusion → 33×33 quincunx → 17×17 + 16×16
- [x] Support 3ch (minimap-only) and 9ch (guided) input
- [x] Backbone frozen, only LoRA + patch proj + head trainable
- [x] Test: forward shape, param count, backbone frozen, offline load

## Slice 2: SiLogLoss ✅

- [x] Create `SiLogLoss` class with shift parameter for negative heights
- [x] Create `hybrid_loss` = 0.7 SiLogLoss + 0.3 L1
- [x] Test: positive scalar, negative inputs, gradient, perfect prediction, zero-weight

## Slice 3: Scheduler fix ✅

- [x] Fix OneCycleLR `total_steps` to `n_batches * epochs`
- [x] Per-batch stepping for OneCycleLR, per-epoch for CosineAnnealingLR
- [x] Add `--scheduler` flag (onecycle/cosine)

## Slice 4: Trainer integration ✅

- [x] Add `--dav2` flag to `train_v24_stage_a.py`
- [x] Load pretrained DA-V2-Small encoder (local_files_only=True)
- [x] Default lr=1e-4 for LoRA, batch_size=8
- [x] Checkpoint records model_type, loss_type, scheduler_type, dav2, guided
- [x] Add `--loss-type` (l1/silog/hybrid), `--silog-weight`, `--l1-weight`, `--silog-shift`
- [x] Add `--lora-rank`, `--weight-decay`, `--8bit-optimizer`, `--gradient-checkpointing`
- [x] Free V18 preload cache after tensor extraction
- [x] Fix `n` and `params` variable scoping bugs

## Slice 5: build_dav2_input ✅

- [x] Create `build_dav2_input` for 3ch (256×256) and 9ch (256×256 with normal+Sobel)
- [x] Test: 3ch shape, 9ch shape, no-normal fallback

## Slice 6: StageBPromptDA ✅

- [x] Create `StageBPromptDA` in `stage_b.py` (DA-V2 4ch: 3 RGB + 1 depth prompt → 257×257)
- [x] Create `build_promptda_input` helper
- [x] Test: forward shape, param count, backbone frozen, input builder

## Slice 7: PatchGAN discriminator ✅

- [x] Create `WDLDiscriminator` in new `discriminator.py` (~693K params, base=32)
- [x] Create `gan_step` helper (D step + G step with BCEWithLogits + L1)
- [x] Create `_render_quincunx_33` helper
- [x] Add `--gan` flag to trainer with lambda schedule (0→0.1 over epochs 5-30)
- [x] Test: forward shape (33+257), param count, gradient, quincunx render, GAN step

## Test suite ✅

- [x] 15 new tests in `test_stage_a_dav2.py` — all pass
- [x] 10 new tests in `test_dav2_stage_b_discriminator.py` — all pass
- [x] Full v24 suite: 77 passed, 0 failed (was 46 before Spec 101)

## Training validation (in progress)

- [ ] Run 4: 2,011 tiles, LoRA 32, lr=1e-4, 8-bit optimizer, gradient checkpointing
- [ ] Evaluate val_l1 against SC-101-001 target (< 10.0)
- [ ] If not met: try guided (9ch), higher LoRA rank (64), or PatchGAN
- [ ] Train Stage B (PromptDA) with the best Stage A checkpoint
- [ ] Run full pipeline: Stage A → Stage B → stitched OBJ

## Documentation ✅

- [x] Research report at `docs/architecture/v24-convergence-research-2026-07-10.md`
- [x] Spec at `specs/101-v241-dav2-model/spec.md`
- [x] Plan at `specs/101-v241-dav2-model/plan.md`
- [x] Tasks at `specs/101-v241-dav2-model/tasks.md`
- [x] Checklist at `specs/101-v241-dav2-model/checklists/requirements.md`
- [x] Memory bank updated (activeContext.md, progress.md)
- [ ] Architecture doc `docs/architecture/v241-dav2-model-2026-07-10.md` (after training completes)