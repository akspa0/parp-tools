# Implementation Plan: V24.1 DA-V2 Pretrained Convergence Model (Spec 101)

**Created**: 2026-07-10
**Status**: Slices 1-7 implemented, training in progress
**Spec**: [spec.md](spec.md)

---

## Architecture

```
V18 Zarr (per map)
  │
  │  V24.1 Stage A (DA-V2-Small + LoRA + DPT head)
  │  Input: minimap RGB (3ch, 256×256) or guided (9ch, 256×256)
  │  Encoder: DA-V2-Small (24.8M params, frozen) + LoRA (rank 16-32)
  │  Head: DPT-style → 33×33 quincunx → 17×17 outer + 16×16 inner
  │  Loss: hybrid (0.7 SiLogLoss + 0.3 L1)
  ▼
Per-tile WDL prior (17,17) outer + (16,16) inner
  │
  │  V24.1 Stage B (StageBPromptDA — DA-V2 depth completion)
  │  Input: minimap RGB (3ch) + WDL prior (1ch depth prompt) = 4ch, 256×256
  │  Encoder: DA-V2-Small (frozen) + LoRA
  │  Head: DPT-style → 257×257 heightmap
  ▼
257×257 heightmap
  │
  │  Optional: PatchGAN discriminator (auxiliary adversarial loss)
  │  Discriminator: ~693K params, 33×33 patch logits
  │  Lambda: 0 for epochs 1-5, ramp to 0.1 over epochs 5-30, hold at 0.1
  ▼
Stitched OBJ + atlas (Spec 097 algorithm)
```

---

## Slice Status

| Slice | Description | Status | Tests |
|-------|-------------|--------|-------|
| 1 | `StageADAV2` model class | ✅ Done | 5 |
| 2 | `SiLogLoss` + `hybrid_loss` | ✅ Done | 6 |
| 3 | Scheduler fix (OneCycleLR + cosine) | ✅ Done | — |
| 4 | `--dav2` trainer flag + checkpoint config | ✅ Done | — |
| 5 | `build_dav2_input` (3ch/9ch at 256×256) | ✅ Done | 4 |
| 6 | `StageBPromptDA` (depth completion Stage B) | ✅ Done | 4 |
| 7 | `WDLDiscriminator` PatchGAN + `gan_step` | ✅ Done | 6 |
| — | Full v24 test suite | ✅ 77 passed | 0 failed |

---

## Training Results

| Run | Config | Tiles | Epochs | Best val_l1 | Notes |
|-----|--------|-------|--------|-------------|-------|
| Old U-Net | 335K params, L1 | 2,011 | 50 | 190.31 | Baseline (from-scratch) |
| DA-V2 run 1 | LoRA 16, lr=5e-6, hybrid | 500 | 40 | 91.11 | LR too low |
| DA-V2 run 2 | LoRA 16, lr=1e-4, hybrid | 500 | 200 | 48.13 | Overfitting (train 17/val 49) |
| DA-V2 run 3 | LoRA 32, lr=1e-4, wd=1e-3 | 2,011 | 132/200 | — | cuDNN OOM crash |
| DA-V2 run 4 | LoRA 32, 8-bit opt, grad ckpt | 2,011 | in progress | — | VRAM-optimized |

---

## Key Decisions

1. **DA-V2-Small encoder** (not from-scratch U-Net) — the 335K U-Net cannot learn minimap → heightmap from ~2,000 tiles. DA-V2-Small is pretrained on 62M images.
2. **LoRA rank 16-32** (not full fine-tune) — only ~1-2M trainable params, fits on 12 GB GPU.
3. **Hybrid loss** (0.7 SiLogLoss + 0.3 L1) — SiLogLoss is scale-invariant and produces sharper predictions; L1 provides absolute-scale accuracy.
4. **CosineAnnealingLR** (not OneCycleLR) — simpler, per-epoch stepping, no `total_steps` confusion.
5. **lr=1e-4** (not 5e-6) — LoRA needs 20× higher LR than full fine-tuning.
6. **Gradient checkpointing + 8-bit optimizer** — required for 2,011 tiles on 12 GB GPU.

---

## Files Changed

| File | Change |
|------|--------|
| `src/harvester/v24/stage_a.py` | +`StageADAV2`, `SiLogLoss`, `hybrid_loss`, `build_dav2_input` |
| `src/harvester/v24/stage_b.py` | +`StageBPromptDA`, `build_promptda_input` |
| `src/harvester/v24/discriminator.py` | New file: `WDLDiscriminator`, `gan_step`, `_render_quincunx_33` |
| `scripts/train_v24_stage_a.py` | +`--dav2`, `--loss-type`, `--scheduler`, `--gan`, `--8bit-optimizer`, `--gradient-checkpointing`, `--lora-rank`, `--weight-decay`, scheduler fix, V18 cache free |
| `tests/v24/test_stage_a_dav2.py` | New file: 15 tests |
| `tests/v24/test_dav2_stage_b_discriminator.py` | New file: 10 tests |
| `specs/101-v241-dav2-model/spec.md` | New spec |
| `specs/101-v241-dav2-model/checklists/requirements.md` | New checklist |
| `specs/101-v241-dav2-model/plan.md` | This file |
| `docs/architecture/v24-convergence-research-2026-07-10.md` | Research report |
| `memory-bank/activeContext.md` | Updated with V24.1 lane |
| `memory-bank/progress.md` | Updated with Spec 101 entry |