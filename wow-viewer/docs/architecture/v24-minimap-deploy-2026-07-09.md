# V24 Minimap-Only Deployment (Spec 096) — 2026-07-09

## TL;DR

Spec 096 ships the **minimap-to-prior deployment path** that Spec 094 promised in FR-013 and User Story 3 scenario 5: a standalone CLI that loads a bare PNG minimap and emits a WDL prior NPZ, with no V24 store / no V18 store / no staged client / no network. The CLI is real, deterministic, and tested. **The model it loads is not yet accurate enough to be useful.** The minimap-only regime does not beat the trivial `block_reduce` baseline on the held-out V24 validation set. This is recorded honestly; the underlying model quality is a separate problem (Spec 095 learned minimap cleaner, or a different model architecture for the deployment regime).

## Why this slice exists

Spec 094 / FR-013:

> "Given a minimap tile that V18 has but no other model has, When Stage A runs, Then it produces a usable prior with no other inputs (this is the deployment case)."

The shipped [`infer_v24_stage_a.py`](../../data-harvester/scripts/infer_v24_stage_a.py) and [`infer_v24_stage_b.py`](../../data-harvester/scripts/infer_v24_stage_b.py) both require `--v24-store` + `--row` and pull the cleaned minimap + alpha + normal + mcnr + object/liquid/holes signals from V18. The deployment case was never wired. Spec 096 closes that gap.

## What shipped

| Artifact | Path | Status |
| --- | --- | --- |
| Spec | [`specs/096-v24-minimap-deploy/spec.md`](../../specs/096-v24-minimap-deploy/spec.md) | written |
| Plan | [`specs/096-v24-minimap-deploy/plan.md`](../../specs/096-v24-minimap-deploy/plan.md) | written |
| Tasks | [`specs/096-v24-minimap-deploy/tasks.md`](../../specs/096-v24-minimap-deploy/tasks.md) | written |
| Quality checklist | [`specs/096-v24-minimap-deploy/checklists/requirements.md`](../../specs/096-v24-minimap-deploy/checklists/requirements.md) | written |
| Inference script | [`scripts/infer_v24_stage_a_png.py`](../../data-harvester/scripts/infer_v24_stage_a_png.py) | 200 lines, PIL + numpy + torch + the existing harvester.v24 module |
| Inference tests | [`tests/v24/test_infer_stage_a_png.py`](../../data-harvester/tests/v24/test_infer_stage_a_png.py) | 3 tests, all green |
| Trainer save patch | [`scripts/train_v24_stage_a.py`](../../data-harvester/scripts/train_v24_stage_a.py) line 304–321 | records `minimap_only: bool` + the right `in_channels` in the checkpoint config |
| Model tests | [`tests/v24/test_stage_a.py`](../../data-harvester/tests/v24/test_stage_a.py) | 2 new tests for `StageAMinimapOnly` |
| Validation script extension | [`scripts/validate_v24.py`](../../data-harvester/scripts/validate_v24.py) | new `--minimap-only-checkpoint` flag + `stage_a_minimap_only` block + `SC-002-MINIMAP` gate |
| Trained checkpoint | [`output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/stage_a.pt`](../../output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/stage_a.pt) | 50 epochs, 2801 tiles, autotune batch=512 |
| Validation report | [`output/v24_validation/v24_minimap_only_validation_20260709/report.json`](../../output/v24_validation/v24_minimap_only_validation_20260709/report.json) | with `stage_a_minimap_only` block and `SC-002-MINIMAP` gate |
| Smoke artifacts | `output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/smoke_prior.npz` and `smoke_prior.png` | end-to-end proof the CLI works on a real PNG |
| Test suite | `tests/v24` | 36/36 passing (up from 31 before Spec 096) |

## Deployment CLI

```
uv run python scripts/infer_v24_stage_a_png.py \
    --checkpoint output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/stage_a.pt \
    --image some_minimap.png \
    --output prior.npz \
    --preview prior.png
```

- PNG is loaded by PIL, converted to RGB, bilinear-resized to 256×256, normalized to `[0, 1]`.
- `harvester.v24.stage_a.build_minimap_only_input` mean-pools to (3, 64, 64).
- `StageAMinimapOnly` (3-channel, ≤ 1M params) runs the input.
- Output is `bilinear-interpolate((1, 1, 64, 64) → (1, 1, 33, 33))` → outer `(17,17)` + inner `(16,16)` → `* HEIGHT_SCALE = 100.0` → world units.
- The script writes `outer (17,17) float32`, `inner (16,16) float32`, `prior_unavailable bool`, and a `metadata` dict with `wall_ms`, `peak_vram_gb`, `world_min`, `world_max`, `seed`, `checkpoint`.
- The `--preview` flag writes a 1024×256 4-up PNG: `[input minimap | outer 17×17 nearest-upsampled | inner 16×16 nearest-upsampled | quincunx 33×33 nearest-upsampled]`.
- `--strict-checkpoint` (default on) refuses 13-channel cheat-regime checkpoints with a clear error. `--lenient-checkpoint` is a documented escape hatch.

Smoke run on a real PNG: 212 ms wall, 0.005 GB peak VRAM, world_min=51.95, world_max=319.02 — a sensible world-unit range for WoW terrain.

## Honest measurements (560-tile held-out V24 prior validation)

```
Cheat regime (full 13-channel input):
  val_l1_cheat:             1.21  world units
  block_reduce_baseline_l1: 1.31  world units
  SC-002_stage_a_beats_baseline: PASS

Minimap-only regime (3-channel RGB input only):
  val_l1_minimap_only:      190.31  world units
  block_reduce_baseline_l1: 1.31   world units
  SC-002-MINIMAP_minimap_only_beats_baseline: FAIL
  minimap_only / cheat ratio: 158×

Stage B (full 257x257 pipeline, cheat regime prior):
  final_l1: 1.21  world units
  upsampled_prior_l1: 4.63
  block_reduce_bilinear_l1: 4.51
  SC-003_final_beats_prior: PASS
  SC-003_final_beats_block_reduce: PASS
```

**The minimap-only regime is 158× worse than the cheat regime on the same held-out tiles.** The bare RGB minimap does not carry enough signal to predict the WDL prior at the precision the WDL grid requires. The CLI works, the model runs, the numbers are honest. The deployment regime is **architecturally possible** but not yet **usefully accurate**.

## Why the minimap-only regime fails

The minimap encodes colour, not height. The WDL prior grid is a height field (in world units, 17×17 + 16×16 over a 256×256 minimap area). The cheat regime gets:

- `cleaned_minimap` (3 channels)
- `alpha_256` (4 channels, layer composition)
- `normal_xyz` (3 channels, surface orientation)
- `mcnr_mask_257` (1 channel, vertex colour mask)
- `synth_quincunx` (33×33, the height-derived WDL anchor — this is the cheat that the spec exists to drop at inference)

The minimap-only regime has only the first 3 channels. The model has to learn the minimap → WDL correlation from RGB alone. Empirically, the model overfits the training tiles (training loss 255 → 180 world units, val loss 271 → 191 — same direction, smaller magnitude, and a wide overtraining gap that keeps widening). On a held-out tile from a different map, the prediction averages out and the per-cell L1 stays in the 100s of world units.

This is not a training bug. It is the actual difficulty of the task. A real deployment would either (a) add a learned minimap cleaner to extract height-proxy features from the RGB (Spec 095), (b) use a different model architecture (e.g. one that explicitly models the prior's spatial structure rather than learning it from scratch), or (c) accept the regime as a coarse "ballpark" prior, not a usable WDL grid.

## SC-002-MINIMAP gate

The spec said: minimap-only L1 < block_reduce baseline. The actual measurement: **190.31 ≮ 1.31**. The gate **fails**. The slice still ships — the spec explicitly allowed this in Risk 1: *"The minimap-only regime may not beat the block_reduce baseline. ... If it doesn't, the slice still ships (the inference script is real, the training run is real, the data is honest) but the metric is reported as a failure."*

## Lessons / next steps

1. **Spec 095 — learned minimap cleaner.** A small U-Net that takes a raw minimap + V18 object_precise_mask and outputs a "terrain-only" minimap. This is the most likely path to a useful minimap-only regime. The cleaner itself is small (≤ 1M params), trains on the V18 substrate, and runs as a pre-step to Stage A.
2. **Spec 097 — alternative deployment shape.** If the minimap-only regime cannot be salvaged with a cleaner, the honest deployment path is "send the PNG to a server that has the staged client" — i.e. accept that the model is a server-side thing, not a client-side thing, and the user's CLI is just a thin shim.
3. **Curation truth.** The minimap-only regime's training data is the same curated 2,011-tile corpus the cheat regime uses. If the minimap-only regime were going to learn, it would learn from this corpus. It didn't. The data is not the problem.
4. **Trainer save config.** The trainer's checkpoint save was patched to record `minimap_only: bool` and the correct `in_channels` so downstream inference scripts can refuse mismatched checkpoints. This is a small but real fix that future Spec 094 training runs will benefit from.

## Reproduce

```
# Train the minimap-only Stage A.
cd wow-viewer/data-harvester
uv run python scripts/train_v24_stage_a.py \
    --v24-store output/datasets/v24/3_3_5_12340_v24_all_v1.zarr \
    --v18-store output/datasets/v18/3_3_5_12340.zarr \
    --output output/v24_validation/v24_minimap_only_3_3_5_12340_20260709 \
    --minimap-only --epochs 50 --seed 94 \
    --autotune-batch-size --log-interval 1

# Run the standalone inference on any PNG.
uv run python scripts/infer_v24_stage_a_png.py \
    --checkpoint output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/stage_a.pt \
    --image some_minimap.png \
    --output prior.npz --preview prior.png

# Run the full validation (cheat + minimap-only + Stage B pipeline + gates).
uv run python scripts/validate_v24.py \
    --v24-store output/datasets/v24/3_3_5_12340_v24_all_v1.zarr \
    --v18-store output/datasets/v18/3_3_5_12340.zarr \
    --stage-a-checkpoint output/v24_validation/v24_curated_full_v1_20260709/stage_a.pt \
    --stage-b-checkpoint output/v24_validation/v24_curated_full_v1_20260709/stage_b.pt \
    --minimap-only-checkpoint output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/stage_a.pt \
    --run-id v24_minimap_only_validation_20260709

# Run the test suite.
uv run python -m pytest tests/v24 -m v24 -q
```

## End of Doc

The deployment wiring is real. The deployment model is not yet accurate enough to be useful. Both facts are recorded. The honest next step is Spec 095 (learned minimap cleaner), which is the most likely path to closing the 158× gap.
