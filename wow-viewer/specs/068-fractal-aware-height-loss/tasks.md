# Tasks: Fractal-Aware Height Loss

**Input**: `specs/068-fractal-aware-height-loss/spec.md`, `specs/068-fractal-aware-height-loss/plan.md`

**Prerequisites**: plan.md (done), spec.md (done)

**Format**: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no code dependency)
- **[Story]**: US1 = Spectral Loss, US2 = Fractal Dim Aux Head, US3 = Cheap Integration

---

## Phase 1 — Spectral Loss (US1)

**Goal**: Add `--spectral-weight` CLI flag and FFT-based radial power spectrum loss to `_v21_height_loss`.

**Independent Test**: Train V21c with `--spectral-weight 0.1`, verify radial power spectrum slope β within 10% of ground truth.

- [x] T001 [US1] Add `--spectral-weight` arg (float, default=0.1) to `_parse_args()` in `scripts/train_v16_1_common.py` after the `--multiscale-weight` arg block
- [x] T002 [US1] Implement `_spectral_loss(pred, target, mask, weight)` function in `scripts/train_v16_1_common.py` before `_v21_height_loss`
- [x] T003 [US1] Wire spectral loss into `_v21_height_loss()` in `scripts/train_v16_1_common.py` — reuse pred/target/train_mask from `_height_loss` return, compute and accumulate spectral_loss

**Checkpoint**: V21c with `--spectral-weight 0.1` trains without error, prints `spectral_loss` in metrics.

---

## Phase 2 — Fractal Dimension Aux Head (US2)

**Goal**: Add 16×16 → 1 scalar-per-patch aux head to V21HeightModel and Hurst-exponent target via variogram.

**Independent Test**: Train V21 with `--fractal-dim-weight 0.05`, verify aux head outputs H ∈ [0.2, 0.8] on validation tiles.

- [x] T004 [US2] Add `_FractalDimHead` module to `src/harvester/v16_1_models.py` — 32→8 conv + BN + ReLU + 8→1 conv + Sigmoid, outputs (B, 1, 16, 16)
- [x] T005 [US2] Modify `V21HeightModel.__init__` to always create `self.fd_head`; `forward` stores `self._pooled16` for loss function access (no output shape change)
- [x] T006 [US2] Add `--fractal-dim-weight` arg (float, default=0.05) to `_parse_args()` in `scripts/train_v16_1_common.py`
- [x] T007 [US2] Implement `_fractal_dim_target(height, mask)` — compute H per 16×16 via Matheron variogram at lags 1 and 2, clamp ∈ [0.1, 0.9], weight patches by ≥50% valid coverage
- [x] T008 [US2] Wire fractal dim loss into `_v21_height_loss()` — compute fd_pred from `model.fd_head(model._pooled16)`, L1 against target, gate with patch_weight

**Checkpoint**: V21 with `--fractal-dim-weight 0.05` trains without error, prints `fd_loss` in metrics.

---

## Phase 3 — Curation Hardening (Data Quality)

**Goal**: Fix mask precedence bug and harden curation defaults so training gets clean data.

- [x] T009 Fix mask precedence in `v16_1_dataset.py:278-283` — swap elif order so `object_precise_mask` wins over `object_filtered_mask`
- [x] T010 Change `--curation-max-liquid-coverage` default from 1.0 → 0.05 (reject tiles >5% water)
- [x] T011 Add general curation enforcement: for height/v21 tasks, auto-set `terrain_validity>=0.20`, `minimap_usefulness>=0.10`, `reject_what_plate=True`, `liquid_coverage<=0.05` when manifest provided
- [x] T012 Add `liquid_coverage` to curation gate logging

---

## Phase 4 — Results: Training Failure

**Run**: `v21c --multiscale-weight 1.0 --gradient-weight 0.05 --normal-consistency-weight 0.05 --spectral-weight 0.1 --fractal-dim-weight 0.05 --epochs 50`

**v18_focus_terrain_v1 manifest (4096 tiles)**

```
Epoch  1/50 | loss=5.4863  val | loss=5.7148  *** best
Epoch  2/50 | loss=5.4817  val | loss=5.6596  *** best
Epoch  3-11 | loss=~5.47   val | loss=~5.73   plateau, never improves
```

**val loss stalled at ~5.66 — FAIL. Target was <0.65.**

**Root cause analysis:**
1. **Spectral loss dominated**: `log-log MSE` between flat-spectrum random init and 1/f terrain produces O(10-50) MSE. At `--spectral-weight 0.1`, this adds ~1-5 to total loss — drowning out L1 signal.
2. **Multi-scale L1 inflates loss**: 5 scales at weight=1.0 sum L1 across 257+128+64+32+16 resolutions → ~5x single-scale L1. Combined with spectral, the total loss magnitude is meaningless for judging actual height error.
3. **No per-component metrics logged**: The `loss` metric is the weighted sum of 7+ terms. Without `spectral_loss`, `fd_loss`, `l1_base` in validation, we can't diagnose which term is broken.
4. **Curation enforcement may have changed data distribution**: Switching from no curation to `terrain_validity≥0.20, liquid≤0.05` reduces tile pool. If manifest was stale, many tiles might be filtered out.

**What to investigate tomorrow:**
1. Add `l1_base`, `spectral_loss`, `fd_loss` to validation metrics printout
2. Run single-scale L1 (no multiscale) with `--spectral-weight 0.01` (10x lower)
3. Run bare V21c with no extra losses as a proper baseline
4. Compare V21c training loss to the 3-channel model's 0.69 result — is the model architecture the bottleneck?

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Spectral Loss)**: No dependencies, can start immediately
- **Phase 2 (Fractal Dim)**: No dependency on Phase 1 — independent files/sections
- **Phase 3 (Integration)**: Depends on Phase 1 AND Phase 2 being complete

### Parallel Opportunities

- T001 (--spectral-weight arg) and T004 (FractalDimHead module) can run in parallel — different files, no shared state
- T006 (--fractal-dim-weight arg) and T002 (spectral loss function) can run in parallel — different sections of same file but no code dependency
- T003 (wire spectral loss) and T008 (wire fd loss) are sequential within their own phases but T003 from Phase 1 has no dependency on Phase 2 tasks

### Within Each Phase

- T001 → T002 → T003 (sequential: arg → function → wiring)
- T004 → T005 (sequential: head module → model integration)
- T006 → T007 → T008 (sequential: arg → target function → wiring)
- T009 → T010 (sequential: baseline first, then comparison)

---

## Session Summary (2026-06-20)

### What was implemented
- **Spectral loss**: FFT radial power spectrum MSE (`_spectral_loss`)
- **Fractal dim aux head**: 16×16 Hurst regression head + Matheron variogram target (`_FractalDimHead`, `_fractal_dim_target`)
- **Curation hardening**: mask precedence fix (`object_precise_mask` wins), `--curation-max-liquid-coverage` default 0.05, auto-enforcement for height/v21 tasks
- All code imports and forward/backward smoke-tested successfully

### What was discovered
- **V21c val loss 5.66 — model not converging.** The 3-channel baseline achieved 0.69.
- Likely causes: (1) spectral weight too high (0.1 → log-MSE dominates), (2) multi-scale L1 at weight=1.0 inflates loss ~5x, (3) no per-component metrics in val output to diagnose
- **Next session should start with debugging the loss landscape**, not adding more features

### Uncommitted changes
- `data-harvester/scripts/train_v16_1_common.py` — spectral loss, fractal dim loss, curation defaults
- `data-harvester/src/harvester/v16_1_models.py` — `_FractalDimHead`, `V21HeightModel.fd_head`
- `data-harvester/src/harvester/v16_1_dataset.py` — mask precedence fix
