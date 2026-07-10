# Implementation Plan: 096-v24-minimap-deploy

**Status**: Draft
**Created**: 2026-07-09
**Parent spec**: [`spec.md`](spec.md)
**Parent lane**: Spec 094 (V24 WDL prior + lattice detailer)

---

## Goal

Ship the **minimap-to-prior deployment path** that Spec 094 promised in FR-013 and User Story 3 scenario 5, but did not wire. Two artifacts:

1. A **trained `StageAMinimapOnly` checkpoint** on the curated open-world V24 corpus (`3_3_5_12340`, 2,011 kept tiles).
2. A standalone **`infer_v24_stage_a_png.py`** CLI that loads a PNG, runs the model, and writes a WDL prior NPZ + 4-up preview PNG. No V24 store, no V18 store, no staged client.

Plus the honest measurement: did the minimap-only regime actually beat the `block_reduce` baseline on the held-out validation set?

---

## Architecture Sketch

This slice reuses everything from Spec 094 that is already correct. Nothing new is invented.

```
PNG (any source)
  │
  │  PIL.Image.open().convert("RGB").resize(256, 256, BILINEAR) / 255.0
  ▼
cleaned_minimap: (256, 256, 3) float32
  │
  │  stage_a.build_minimap_only_input(cleaned_minimap)
  ▼
input: (3, 64, 64) float32  ← mean-pooled 4× from 256
  │
  │  StageAMinimapOnly().load_state_dict(ckpt)  ← trained slice 1
  ▼
field: (1, 64, 64) float32
  │
  │  F.interpolate(field, (33, 33), bilinear, align_corners=True)
  ▼
quincunx: (33, 33) float32
  │
  │  outer = quincunx[::2, ::2]  (17, 17)
  │  inner = quincunx[1::2, 1::2]  (16, 16)
  ▼
prior_outer: (17, 17) float32 (normalized)   × HEIGHT_SCALE=100  → world units
prior_inner: (16, 16) float32 (normalized)   × HEIGHT_SCALE=100  → world units
  │
  ▼
np.savez(output, outer=..., inner=..., prior_unavailable=...)
```

No V18 reads. No V24 store reads. No C# subprocess. No `clean_minimap` per default. Single-file. Deterministic.

For the training side: the existing `train_v24_stage_a.py --minimap-only` is the trainer. It already exists, already works in tests, just has not been run on real data with a real `--epochs 50` flag and a real corpus.

---

## Slice Breakdown (RULE 8 — one phase at a time, validated before next)

### Slice 1 — Train the minimap-only Stage A checkpoint

**Touches**:
- `wow-viewer/data-harvester/src/harvester/v24/stage_a.py` — **no change** (model class already exists).
- `wow-viewer/data-harvester/scripts/train_v24_stage_a.py` — **no change** (`--minimap-only` flag already exists).
- `wow-viewer/data-harvester/tests/v24/test_stage_a.py` — **add** `test_stage_a_minimap_only_forward_shape_and_params` and `test_stage_a_minimap_only_pre_train_is_constant`.
- New artifacts under `wow-viewer/output/v24_validation/v24_minimap_only_3_3_5_12340_<YYYYMMDD>/`:
  - `stage_a.pt` (checkpoint, the only artifact that matters for Slice 2)
  - `stage_a_metrics.json`
  - `loss_history.jsonl`
  - `peak_vram.json`
  - `batch_autotune.json`

**Run command** (one line, the only thing to execute):
```bash
cd wow-viewer/data-harvester && \
  uv run python scripts/train_v24_stage_a.py \
    --v24-store output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
    --v18-store output/datasets/v18/3_3_5_12340.zarr \
    --output output/v24_validation/v24_minimap_only_3_3_5_12340_20260709 \
    --minimap-only --epochs 50 --seed 94 \
    --autotune-batch-size --log-interval 1
```

**Validation gate**:
- `tests/v24/test_stage_a.py -m v24 -q` passes (≥ 33 tests).
- `stage_a.pt` exists; loads as `StageAMinimapOnly(base=28, in_channels=3)`; config dict contains `"minimap_only": true`; params ≤ 1M.
- `stage_a_metrics.json` reports `train_tiles + val_tiles = 2011`, `epochs_run ≥ 1`, `peak_vram_gb < 2.0`.
- `loss_history.jsonl` has 50 lines (no early stopping).
- The pre-train head is a constant (asserted in the new test, also observable in epoch 0 metrics).

**Honest failure mode**: if `best_val_l1 > block_reduce_baseline_l1` for the minimap-only regime, the slice still ships — the trained model is real, the metrics are real, and Risk 1 is documented in the spec. The model is a real, deployable thing even if it does not beat the trivial baseline. We do **not** tune hyperparameters to make the number look good.

---

### Slice 2 — `infer_v24_stage_a_png.py` standalone inference script

**Touches**:
- New file: `wow-viewer/data-harvester/scripts/infer_v24_stage_a_png.py`. ~80 lines. CLI: `--checkpoint --image --output [--preview] [--seed 94] [--device auto] [--strict-checkpoint / --lenient-checkpoint]`. Loads PNG, builds minimap-only input, runs model, denormalizes, writes NPZ + 4-up preview PNG.
- New tests: `wow-viewer/data-harvester/tests/v24/test_infer_stage_a_png.py`. ~40 lines. Three tests:
  1. `test_infer_png_runs_end_to_end` — write a 256×256 RGB PNG, run the script via subprocess, load the NPZ, assert shape and `prior_unavailable=False`.
  2. `test_infer_png_deterministic_across_seeds` — run twice with different `--seed`, assert `np.array_equal` on `outer` and `inner`.
  3. `test_infer_png_refuses_cheat_checkpoint` — write a 13-channel-in-channels Stage A checkpoint, run the script, assert exit code ≠ 0 and a clear error.

**Validation gate**:
- `tests/v24/test_infer_stage_a_png.py -m v24 -q` passes (3 tests).
- The full suite is still green: `uv run python -m pytest tests/v24 -m v24 -q` passes with ≥ 36 tests (33 from Slice 1 + 3 new).
- Manual sanity check: run the script on a real V18-derived minimap PNG; the NPZ outer/inner values are in a sensible world-unit range (-500 to 4000).

**Implementation notes** (for the implementer — this is the design, not code yet):
- Use PIL's `Image.open(...).convert("RGB")`. Resize to 256 with `BILINEAR`. Float32 in [0, 1].
- Reuse `stage_a.build_minimap_only_input` (already exists in [`stage_a.py:147`](wow-viewer/data-harvester/src/harvester/v24/stage_a.py:147)) to mean-pool to 64×64.
- Load checkpoint with `torch.load(..., map_location=device, weights_only=True)`.
- Assert `checkpoint["config"]["in_channels"] == 3` and `checkpoint["config"].get("minimap_only", False) is True` when `--strict-checkpoint` is on (the default).
- Instantiate `stage_a.StageAMinimapOnly(base=checkpoint["config"]["base"])`. Load state.
- `model.eval()`. `torch.use_deterministic_algorithms(True)`.
- Run on the 4D input (1, 3, 64, 64). Multiply output by `HEIGHT_SCALE = 100.0` to denormalize.
- Save NPZ with `outer` (17,17) float32, `inner` (16,16) float32, `prior_unavailable=False`, plus a metadata dict as a `.npz` extra field.
- For `--preview`, build a 4-up PNG with `PIL.Image.new("RGB", (1024, 256), (0,0,0))` and paste: input minimap, outer (17×17 → 256×256 nearest), inner (16×16 → 256×256 nearest), quincunx (33×33 → 256×256 nearest). No matplotlib.

---

### Slice 3 — Validation comparison (minimap-only vs cheat)

**Touches**:
- `wow-viewer/data-harvester/scripts/validate_v24.py` — add `--minimap-only-checkpoint <ckpt>` arg. When given, evaluate the minimap-only Stage A on the same held-out rows and add a `stage_a_minimap_only` block to the report.
- New artifacts under `wow-viewer/output/v24_validation/v24_minimap_only_3_3_5_12340_<YYYYMMDD>_validation/`:
  - `report.json` (full validation report, including the new block)
  - `preview.png` (existing side-by-side, still from the cheat regime)

**Validation gate**:
- `validate_v24.py --minimap-only-checkpoint <ckpt>` exits 0.
- `report.json` contains `stage_a_minimap_only_l1 < block_reduce_baseline_l1` (the SC-002-MINIMAP gate from the spec).
- The gap `cheat_l1 - minimap_only_l1` is recorded as a number, not hidden.
- SC-004 determinism still holds for the minimap-only pipeline.

**Honest failure mode**: if `minimap_only_l1 ≥ block_reduce_l1`, the slice still ships, the spec is honest about the failure, and Spec 095 (learned minimap cleaner) and Spec 097 (alt deployment) get real motivation. The metric is recorded as fact, not sanded down.

---

### Slice 4 — Memory bank + progress sync + summary doc

**Touches**:
- `wow-viewer/memory-bank/activeContext.md` — add to the "WDL prior + lattice detailer lane (V24)" section: minimap-only training run, deployment script, SC-002-MINIMAP pass/fail, the link to the new summary doc.
- `wow-viewer/memory-bank/progress.md` — add a 2026-07-09 entry. Also fix the orphaned `>>>>> REPLACE` marker that was left dangling from the prior session.
- New file: `wow-viewer/docs/architecture/v24-minimap-deploy-2026-07-09.md`. Architecture doc for the minimap-only path: training, script, metrics, determinism, hardware envelope. Same shape as the existing `v24-validation-2026-07-06.md`.

**Validation gate**:
- Memory bank is updated and the `>>>>> REPLACE` marker is gone.
- Architecture doc is committed.
- No spec doc or memory bank file is older than the work in this slice.

---

## Risk Mitigations (Spec 096 risks re-stated with action)

| Risk | Severity | Mitigation in this plan |
| --- | --- | --- |
| Minimap-only regime does not beat `block_reduce` baseline | High | Honest recording in Slice 3. Spec 095/097 fallback noted. Do not tune hyperparameters to make the number look good. |
| 50 epochs insufficient | Medium | Loss curve visible in `loss_history.jsonl`. If not plateaued by epoch 40, log a warning. Do not auto-extend; user decides. |
| PNG decoding edge cases | Low | Standard `PIL.Image.open().convert("RGB")` path. Non-RGB PNGs convert cleanly. Refusal on missing file / non-image. |
| CPU fallback | Low | `pick_device("auto")` already in [`train_common.py:35`](wow-viewer/data-harvester/src/harvester/v24/train_common.py:35). Same pattern in the inference script. Determinism holds on CPU. |
| Multi-file batch | Low | Out of scope. Single-file mode only. Multi-file is a separate spec. |

---

## Constitution Re-Check (post-plan, AGENTS.md rules)

- **RULE 1 (no edits to `gillijimproject_refactor`)**: ✓ this slice does not touch it.
- **RULE 2 (all new code in `wow-viewer`)**: ✓ all four slices land under `wow-viewer/data-harvester/`.
- **RULE 3 (no rewrite of game client reading tooling)**: ✓ no new WDL reader, no new terrain parser. The model and trainer already exist.
- **RULE 4 (`wow-viewer` repo-independent)**: ✓ no cross-repo imports.
- **RULE 5 (one Python environment)**: ✓ all new code under `wow-viewer/data-harvester/`.
- **RULE 6 (no mutation of training scripts without a plan)**: ✓ this plan IS the plan; the trainer is reused as-is.
- **RULE 7 (small modular residual-predicting models)**: ✓ `StageAMinimapOnly` is a 3-channel-in, 1-out module, ≤ 1M params, no shared weights, no multi-task head.
- **RULE 9 (no `H:\CLIENTS`)**: ✓ no client paths. The training corpus is the V24 store on disk.
- **RULE 10 (`AlphaWdtWriter` frozen)**: ✓ not touched.
- **RULE 11 (doc hygiene, plans bite-sized)**: ✓ 4 slices, each independently validatable, ≤ 1 day each.
- **RULE 8 (one phase at a time)**: ✓ each slice ends with a validation gate before the next starts.

---

## Out of Scope (from the spec, re-asserted)

- Stage B PNG deployment (Stage B needs more than a PNG; the existing `infer_v24_stage_b.py` covers the V18-store-row deployment story).
- Learned minimap cleaner (Spec 095).
- New WDL or minimap readers.
- C# changes.
- RunPod / Pod packaging.
- Dataset schema changes.

---

## End of Plan

Ready to execute. Each slice is small, the validation gates are concrete, and the honest failure modes are recorded. The user's actual question — "can I drop a PNG into a CLI and get a WDL prior?" — is answered by Slice 2 with a real, working script. The honest answer to "is the minimap-only regime any good?" is answered by Slice 3 with a real number, not a hand-wave.
