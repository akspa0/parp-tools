# Quickstart: 089 — DA-V2-Small LoRA Height Predictor with Cross-Tile Consistency

**Phase 1 quickstart. Companion to `plan.md`, `research.md`, and `data-model.md`.**

This is the operator/developer on-ramp for Spec 089. It covers the routing fix, Phase 0 validation gate, V22 dataset prerequisite, and the planned local/RunPod command surface.

---

## 1. Routing and Scope

- Active spec directory: `wow-viewer/specs/089-dav2-height-predictor/`
- Active implementation surface: `wow-viewer/data-harvester/src/harvester/v23/`
- Dataset prerequisite: Spec 088 V22 stores under `wow-viewer/output/datasets/v22/`
- RunPod packaging pattern owner: Spec 079
- Spec Kit PowerShell helpers must be run from `wow-viewer/` (or below), not the monorepo root, so `.specify/` is discoverable.

Local source work through Phase 7 is now complete. Treat the remaining work as proof/evidence work: cached/pretrained local quality, Pod smoke, full-corpus training, CAI seam review, and determinism evidence.

---

## 2. Prerequisites

- Python 3.11+ managed by `uv`
- `wow-viewer/data-harvester/pyproject.toml` synced locally
- At least one V22 store, ideally `wow-viewer/output/datasets/v22/3_3_5_12340.zarr/`
- Trusted staged clients only under `output/tmp/wowarchive-clients/` if a fresh V22 store must be built

`H:\CLIENTS` is forbidden.

---

## 3. Phase 0 Validation Gate

Run from `wow-viewer/data-harvester/`.

```powershell
cd wow-viewer/data-harvester
uv sync
uv run python -c "import harvester.v23"
uv run pytest tests/v23 -m v23 -q
```

Expected result:

- `uv sync` resolves the added V23 dependencies
- `import harvester.v23` succeeds
- pytest collects only the gated V23 surface and exits cleanly

2026-07-03 local proof note:

- The stale uv-managed 3.11 environment was replaced with a fresh `.venv` on `C:\Python314\python.exe`.
- `pyproject.toml` now includes the missing `src/` packaging metadata, so plain `uv run` can import `harvester`.
- `uv run python -c "import harvester.v23"` now prints `import-ok`.
- `uv run pytest tests/v23/test_dataset.py tests/v23/test_channels.py -m v23 -q` passed with `10 passed`.

If these commands fail on another machine, stop and repair the environment before treating Phase 1 as complete.

---

## 4. V22 Dataset Prerequisite

If `wow-viewer/output/datasets/v22/3_3_5_12340.zarr/` already exists, inspect it first:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/inspect_v22_dataset.py summary `
    --store ../output/datasets/v22/3_3_5_12340.zarr
```

If it does not exist, follow Spec 088's two-step build:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py enrich `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --client-root ../output/tmp/wowarchive-clients/3_3_5_12340 `
    --enrichment-output ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --build-key 3_3_5_12340 `
    --limit 1

uv run python scripts/build_v22_dataset.py build `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --enrichment ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --output ../output/datasets/v22/3_3_5_12340.zarr
```

The bounded proof script for Spec 088 remains the preferred validation route when rebuilding from scratch.

---

## 5. Planned Phase 1 Dataset Commands

These commands are the current Phase 1 validation path now that `channels.py`, `dataset.py`, and the prune-table script exist in source.

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_tileset_prune_table.py `
    --dataset-dir ../output/datasets/v22 `
    --builds 0_5_3_3368 3_3_5_12340 4_0_0_11927 `
    --top-k 256 `
    --output ../output/datasets/v22/tileset_prune_v23_union.json

uv run python -c "from harvester.v23 import V23HeightDataset; ds = V23HeightDataset('../output/datasets/v22/3_3_5_12340.zarr', build='3_3_5_12340'); print(ds[0]['input'].shape, ds[0]['target_height'].shape)"
```

Expected contract:

- input tensor shape `[15, 256, 256]` in `full` mode
- target tensor shape `[1, 257, 257]`

---

## 6. Local Training Smoke

V23 trains from V22 stores. V22 is the V18 substrate plus enrichment, so the trainer does not read V18 directly, but the V18 root arrays are expected to be present in V22. For real local training, also pass the V18 curation manifest. The trainer uses it to keep the same curated tile pool and to select validation from mismatch-rich rows instead of arbitrary first-N rows.

The current curated Northrend local CUDA envelope proof used this command shape:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/train_v23_height.py `
    --dataset-dir ../output/datasets/v22 `
    --builds 3_3_5_12340 `
    --maps Northrend `
    --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet `
    --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json `
    --epochs 1 `
    --train-max-tiles 16 `
    --val-max-tiles 4 `
    --device cuda `
    --target-vram-gb 12 `
    --memory-profile 12gb `
    --batch-size 1 `
    --grad-accum-steps 4 `
    --gpct-K 2 `
    --gpct-weight 0.1 `
    --sdc-weight 0.1 `
    --bias-free-mask-ratio 0.15 `
    --log-interval 1 `
    --deterministic `
    --seed 42 `
    --run-name v23_curated_northrend_labeled_smoke_20260705
```

Expected artifacts:

- live console lines for run setup, train/val batch loss components, epoch summaries, checkpoints, metrics, and CUDA peak VRAM
- `../models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/loss_history.jsonl`
- `../models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/peak_vram.json`
- `../models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/metrics.json`
- `../models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/checkpoints/`
- validation preview PNGs
- metrics/config metadata with commit SHA and data hashes

The operator-facing loss contract is:

```text
[v23] epoch=1/1 phase=train status=start step=1 batch=1/16
[v23] epoch=1/1 phase=train status=done step=1 batch=1/16 samples=1/16 pct=6.2 elapsed=12.4s eta=3m06s optimizer_step=no loss=... affine_loss=... gradient_loss=... sdc_loss=... gpct_loss=... lr=... gpu_alloc_gb=... gpu_reserved_gb=...
[v23] epoch=1/1 phase=val status=done step=1 batch=1/4 samples=1/4 pct=25.0 elapsed=2.0s eta=6.0s optimizer_step=no loss=... affine_loss=... gradient_loss=... sdc_loss=... gpct_loss=...
[v23] epoch=1/1 summary train_loss=... val_loss=... best_val_loss=...
```

Use `loss` / `train_loss` / `val_loss` as the quality trend. Lower is better. `affine_loss`, `gradient_loss`, `sdc_loss`, and `gpct_loss` are the component breakdown that explains which term is dominating. `peak_vram.json` is only capacity evidence; it does not say whether the model is learning.

How to read the status fields:

- `phase=train` means weights can change. `phase=val` means forward-only measurement; it does not train.
- `status=start` prints before the batch work starts. `status=done` prints after that batch finishes and includes loss.
- `step` is the user-visible step counter inside the phase. In training, this advances with train batches.
- `batch=A/B` is current batch out of total batches for that phase.
- `samples=A/B` is current tiles seen out of total tiles in that phase.
- `pct`, `elapsed`, and `eta` are phase progress, elapsed phase time, and estimated remaining phase time.
- `optimizer_step=yes` means this batch triggered an optimizer update. With `--grad-accum-steps 4`, only every fourth train batch updates weights; the other train batches accumulate gradients.
- `loss` is the training objective being optimized or measured. The component losses explain what is contributing to it.
- `gpu_alloc_gb` and `gpu_reserved_gb` are memory status. They are utilization/capacity signals, not learning quality.

`loss_history.jsonl` is the machine-readable training trace. It contains one JSON line per train/val batch plus one epoch summary line with `train_loss`, `val_loss`, and `best_val_loss`.

For a larger local curated run, use `--train-max-tiles 2000` plus startup batch autotune. The autotune path probes CUDA batch candidates before epoch 1, selects the largest candidate that fits under `target_vram_gb * autotune_safety_factor`, writes `batch_autotune.json`, and then rebuilds the real train/val loaders using that selected batch size.

Recommended current local corpus command for the important maps available in local V22 stores:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/train_v23_height.py `
    --dataset-dir ../output/datasets/v22 `
    --builds 0_5_3_3368 3_3_5_12340 `
    --maps Azeroth Kalimdor Kalidar PVPZone01 PVPZone02 Northrend Expansion01 `
    --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet `
    --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json `
    --epochs 2 `
    --train-max-tiles 2000 `
    --val-max-tiles 256 `
    --val-interval 2 `
    --val-preview-interval 2 `
    --device cuda `
    --target-vram-gb 12 `
    --memory-profile 12gb `
    --batch-size 1 `
    --grad-accum-steps 4 `
    --autotune-batch-size `
    --autotune-batch-candidates 1 2 4 8 12 16 24 32 40 48 `
    --autotune-safety-factor 0.85 `
    --gpct-K 2 `
    --gpct-weight 0.1 `
    --sdc-weight 0.1 `
    --bias-free-mask-ratio 0.15 `
    --log-interval 1 `
    --deterministic `
    --seed 42 `
    --run-name v23_curated_2k_keymaps_autotune_20260705
```

Current local V22 map coverage note: `0_5_3_3368` has `Azeroth`, `Kalimdor`, and `PVPZone02`; `3_3_5_12340` has `Azeroth`, `Kalimdor`, `PVPZone01`, `PVPZone02`, `Northrend`, and `Expansion01`; `Kalidar` is present in the curation manifest but not in the currently inspected V22 stores. The map filter selects the intersection of `--builds` and `--maps`.

Validation is forward-only measurement, not training. It is still real GPU time. For the 2-epoch local command above, `--val-interval 2` skips validation after epoch 1 and validates on the final epoch only. For longer runs, use `--val-interval 5` or higher if you want fewer measurement passes.

If `batch_autotune.json` shows the selected batch is the last candidate and reserved VRAM is still well below `effective_target_vram_gb`, extend `--autotune-batch-candidates`. The `v23_curated_2k_keymaps_autotune_20260705` run selected batch `24` with only about `6.21 GB` peak reserved against a `10.2 GB` effective target, so later local runs should include larger candidates like `32`, `40`, and `48`.

If `sdc_loss` stays exactly `0.0` across all batches while `--sdc-weight` is nonzero, treat that as a loss-mask bug rather than healthy training. The SDC path now uses fractional valid-mask weights so sparse terrain-valid tiles do not zero the whole SDC term.

When running pytest commands from this quickstart, do not paste summary arrows like `->` from chat output. They are not part of the command.

2026-07-05 proof note: `3_3_5_12340` V18 and V22 both contain `Northrend` with 1,131 tiles. The curated `--maps Northrend` run completed on an RTX 4070 Ti SUPER with zero CUDA OOM, peak allocated VRAM `0.3959296 GB`, checkpoint config `maps = ["Northrend"]`, and checkpoint config `curation_manifest = "../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet"`. The preview PNGs now include panel labels plus map/tile and curation bucket/mismatch metadata. The trainer now prints live setup, batch loss values, component breakdowns, epoch summaries, checkpoint paths, metrics paths, and peak CUDA VRAM; use `--log-interval N` to reduce batch log volume. The local HF cache did not contain `depth-anything/Depth-Anything-V2-Small-hf`, so this is an envelope/curation-path proof, not pretrained-quality evidence.

---

## 7. Planned Deterministic Inference / CAI Proof

```powershell
cd wow-viewer/data-harvester
uv run python scripts/infer_v23_height.py `
    --checkpoint models/v23/height/runs/smoke_v23/checkpoints/v23_height_best.pt `
    --v22-store ../output/datasets/v22/3_3_5_12340.zarr `
    --build 3_3_5_12340 `
    --tiles 30,48 `
    --cai-r 16 `
    --deterministic `
    --seed 42 `
    --save-preview `
    --output-dir models/v23/height/runs/smoke_v23/inference_seed42
```

Repeat with `--seed 12345` and compare the outputs bit-for-bit.

---

## 8. Planned RunPod Bundle Flow

V23 does not invent a second Pod bootstrap route. It follows Spec 079's pattern after the V23 packager exists:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/package_v23_runpod.py `
    --bundle-name v23_smoke_bundle `
    --dataset-dir ../output/datasets/v22 `
    --builds 0_5_3_3368 3_3_5_12340 `
    --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json `
    --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet `
    --include-v22-subset-tiles 2000 `
    --output-tar runpod/v23/dist/v23_smoke_bundle.tar
```

Pod-side validation sequence:

```bash
bash runpod/v23/install_deps.sh
bash runpod/v23/verify_bundle.sh
bash runpod/v23/smoke.sh
bash runpod/v23/train.sh
```

With no positional arguments, `runpod/v23/train.sh` now runs the current curated key-map proof shape:

```bash
DATASET_DIR=data/v22 \
RUN_NAME=v23_curated_2k_keymaps_runpod \
bash runpod/v23/train.sh
```

Default full-train wrapper settings:

- builds: `0_5_3_3368 3_3_5_12340`
- maps: `Azeroth Kalimdor Kalidar PVPZone01 PVPZone02 Northrend Expansion01`
- curation: `config/curation_manifest.parquet`
- prune table: `config/tileset_prune_table.json`
- train/val size: `TRAIN_MAX_TILES=2000`, `VAL_MAX_TILES=256`
- validation cadence: `VAL_INTERVAL=2`, `VAL_PREVIEW_INTERVAL=2`
- optimizer envelope: `MEMORY_PROFILE=24gb`, `TARGET_VRAM_GB=22`, `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=4`
- startup autotune candidates: `1 2 4 8 12 16 24 32 40 48 64 80 96`
- loss knobs: `GPCT_K=2`, `GPCT_WEIGHT=0.1`, `SDC_WEIGHT=0.1`, `BIAS_FREE_MASK_RATIO=0.15`
- observability: `LOG_INTERVAL=1`

Override with environment variables for later full-corpus training:

```bash
TRAIN_MAX_TILES=25000 \
EPOCHS=4 \
VAL_INTERVAL=1 \
VAL_PREVIEW_INTERVAL=1 \
RUN_NAME=v23_height_full_corpus_v1 \
bash runpod/v23/train.sh
```

If `config/curation_manifest.parquet` or `config/tileset_prune_table.json` is missing, full training fails early. That is intentional; this path must not silently fall back to uncurated first-N tiles.

---

## 9. Per-Phase Checklist

Before moving from one implementation phase to the next:

1. The current phase's pytest target passes.
2. Any required real-data proof for that phase is recorded.
3. `wow-viewer/specs/089-dav2-height-predictor/{plan,tasks}.md` still matches the current work.
4. `wow-viewer/memory-bank/activeContext.md` and `progress.md` are updated if task status changed materially.

---

## 10. Related Reading

- Spec: `specs/089-dav2-height-predictor/spec.md`
- Plan: `specs/089-dav2-height-predictor/plan.md`
- Research: `specs/089-dav2-height-predictor/research.md`
- Data model: `specs/089-dav2-height-predictor/data-model.md`
- V22 schema: `docs/architecture/v22-dataset-signals-2026-06-30.md`
- Spec 088 quickstart: `specs/088-v22-enrichment-from-v18/quickstart.md`
- Spec 079 spec: `specs/079-runpod-integration-guide/spec.md`
- Workspace continuity: `wow-viewer/memory-bank/activeContext.md`
