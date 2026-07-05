# V23 Height Predictor

Date: 2026-07-03

## Scope

Spec 089 implements a single-signal terrain-height model on top of the V22 Zarr contract. The active local stack is:

- `DepthAnything-V2-Small` encoder with a replaced patch-embed conv for the V23 channel contract
- frozen base backbone plus LoRA adapters on the actual attention projection layers
- compact DPT-style height head plus affine anchor
- V22-backed dataset adapter, deterministic trainer, deterministic inference CLI, and CAI-style stitched inference helper

This model predicts only `height_257`. It does not joint-train normals, liquids, objects, or WDL-derived priors.

## Input And Target Contract

- Input: default 15-channel tensor built from `minimap_rgb`, `alpha_256`, dominant `mcly_tileset_ids`, `normal_xyz`, and `terrain_valid_mask`
- Target: `height_257`, with liquid pixels overridden from resampled `liquid_height`
- Provenance: canonical V22 stores remain `paths_only`; model/tileset identity lives in inventories and audit sidecars, not embedded M2/WMO/BLP payload blobs
- Dataset relationship: V23 trains from V22. V22 is built from the V18 substrate plus enrichment, so V23 does not read V18 directly but still depends on V18-derived root arrays being present in the V22 store. V23 local training must also consume the V18 curation manifest so training/validation uses the curated tile pool and mismatch-rich validation rows.

## Local Implementation Status

Local implementation is complete through the package boundary:

- Phase 0: package wiring, dependency surface, import smoke
- Phase 1: V22-to-V23 dataset adapter and channel builder
- Phase 2: DA-V2-Small encoder wrapper and LoRA surface
- Phase 3: head/model composition
- Phase 4: affine, gradient, SDC, GPCT, and bias-free masking losses
- Phase 5: deterministic trainer and checkpoint format
- Phase 6: deterministic inference CLI and CAI stitch helper
- Phase 7: RunPod helper scripts plus `package_v23_runpod.py`

The remaining incomplete boundary is hardware validation:

- local CUDA peak-VRAM proof exists for the 12 GB real-GPU envelope, with the caveat that HF pretrained DA-V2-Small weights were unavailable locally and the encoder used random DepthAnything initialization
- no real RunPod pod smoke or full-corpus training proof
- no cross-pod determinism evidence yet

## Local Proof

Focused V23 pytest coverage exists for:

- dataset/channel contract
- encoder/head/model contract
- loss functions
- deterministic two-run smoke training
- deterministic inference
- CAI stitching
- RunPod bundle packaging

The current local gate is the focused V23 suite:

```text
uv run python -m pytest tests/v23 -m v23 -q
```

At the end of the current local implementation pass, the suite is expected to stay green before reopening external Pod validation.

T035 local CUDA smoke was run on 2026-07-04:

```text
uv run python scripts/train_v23_height.py --dataset-dir ../output/datasets/v22 --builds 3_3_5_12340 --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json --epochs 2 --train-max-tiles 16 --val-max-tiles 4 --device cuda --target-vram-gb 12 --memory-profile 12gb --batch-size 1 --grad-accum-steps 4 --gpct-K 2 --gpct-weight 0.1 --sdc-weight 0.1 --bias-free-mask-ratio 0.15 --deterministic --seed 42 --run-name t035_local_12gb_20260704
```

Artifacts:

- `wow-viewer/models/v23/height/runs/t035_local_12gb_20260704/peak_vram.json`
- `wow-viewer/models/v23/height/runs/t035_local_12gb_20260704/metrics.json`
- `wow-viewer/models/v23/height/runs/t035_local_12gb_20260704/checkpoints/v23_height_best.pt`
- `wow-viewer/models/v23/height/runs/t035_local_12gb_20260704/val_preview_2/tile_0.png`

Peak allocated VRAM was `0.408541184 GB`; peak reserved VRAM was `0.457179136 GB`; no CUDA OOM occurred. The run warned that `depth-anything/Depth-Anything-V2-Small-hf` was not cached locally and could not be downloaded, so this is a real CUDA trainer/envelope proof, not a pretrained quality proof.

V23 local training now supports map targeting through `--maps`. `3_3_5_12340` V18 and V22 both contain `Northrend` with 1,131 tiles. A Northrend-specific local CUDA smoke was run on 2026-07-05:

```text
uv run python scripts/train_v23_height.py --dataset-dir ../output/datasets/v22 --builds 3_3_5_12340 --maps Northrend --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json --epochs 2 --train-max-tiles 16 --val-max-tiles 4 --device cuda --target-vram-gb 12 --memory-profile 12gb --batch-size 1 --grad-accum-steps 4 --gpct-K 2 --gpct-weight 0.1 --sdc-weight 0.1 --bias-free-mask-ratio 0.15 --deterministic --seed 42 --run-name t035_northrend_local_12gb_20260705
```

Artifacts:

- `wow-viewer/models/v23/height/runs/t035_northrend_local_12gb_20260705/peak_vram.json`
- `wow-viewer/models/v23/height/runs/t035_northrend_local_12gb_20260705/metrics.json`
- `wow-viewer/models/v23/height/runs/t035_northrend_local_12gb_20260705/checkpoints/v23_height_best.pt`
- `wow-viewer/models/v23/height/runs/t035_northrend_local_12gb_20260705/val_preview_2/tile_0.png`

The checkpoint config records `maps = ["Northrend"]`; peak allocated VRAM was `0.408541184 GB`; no CUDA OOM occurred. The same HF-cache caveat applies.

The earlier Northrend smoke still used only map filtering. The corrected local proof consumes the V18 curation manifest and labels previews:

```text
uv run python scripts/train_v23_height.py --dataset-dir ../output/datasets/v22 --builds 3_3_5_12340 --maps Northrend --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet --tileset-prune-table ../output/datasets/v22/tileset_prune_v23_union.json --epochs 1 --train-max-tiles 16 --val-max-tiles 4 --device cuda --target-vram-gb 12 --memory-profile 12gb --batch-size 1 --grad-accum-steps 4 --gpct-K 2 --gpct-weight 0.1 --sdc-weight 0.1 --bias-free-mask-ratio 0.15 --log-interval 1 --deterministic --seed 42 --run-name v23_curated_northrend_labeled_smoke_20260705
```

Artifacts:

- `wow-viewer/models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/peak_vram.json`
- `wow-viewer/models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/loss_history.jsonl`
- `wow-viewer/models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/metrics.json`
- `wow-viewer/models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/checkpoints/v23_height_best.pt`
- `wow-viewer/models/v23/height/runs/v23_curated_northrend_labeled_smoke_20260705/val_preview_1/tile_0.png`

The checkpoint config records the curation manifest and thresholds. The validation preview is labeled (`minimap`, `target_height`, `pred_height`, `abs_error`) and includes the map/tile plus curation bucket/mismatch score. Peak allocated VRAM was `0.3959296 GB`; no CUDA OOM occurred. The same HF-cache caveat applies.

The operator-facing learning signal is the trainer loss, not `peak_vram.json`. Console output must include train/val batch `loss=...` values, epoch `train_loss=...`, `val_loss=...`, and `best_val_loss=...`, plus component breakdowns named `affine_loss`, `gradient_loss`, `sdc_loss`, and `gpct_loss`. `loss_history.jsonl` is the persistent batch/epoch trace for plotting and post-run comparison. `peak_vram.json` is only memory-capacity proof.

The user-visible training heartbeat is the batch status line. With `--log-interval 1`, the trainer prints `status=start` before a train/val batch begins and `status=done` after it finishes. `step`, `batch`, `samples`, `pct`, `elapsed`, and `eta` show progress through the current phase. `optimizer_step=yes` marks batches that actually update weights after gradient accumulation; `optimizer_step=no` means the batch only accumulated gradients or was validation. CUDA memory fields are capacity signals only.

V23 also has startup batch autotune for local CUDA utilization. `--autotune-batch-size` probes the candidate ladder from `--autotune-batch-candidates`, compares PyTorch peak reserved memory against `--target-vram-gb * --autotune-safety-factor`, writes `batch_autotune.json`, and then uses the selected batch size for the real train/val loaders. The intended 12 GB local command uses `--batch-size 1` as the safe floor, `--grad-accum-steps 4`, `--gpct-K 2`, and candidate ladder `1 2 4 8 12 16 24`.

Validation is forward-only measurement work. It does not update model weights, but it still costs GPU time. `--val-interval` is honored: skipped epochs save `v23_height_last.pt` and write `validation_skipped=true` to `loss_history.jsonl`; validation runs only on scheduled epochs and on the final epoch when `--val-interval > 0`. The local 2-epoch key-map command uses `--val-interval 2 --val-preview-interval 2` so epoch 1 stays training-only.

For the current local 2K key-map run, use builds `0_5_3_3368 3_3_5_12340` and maps `Azeroth Kalimdor Kalidar PVPZone01 PVPZone02 Northrend Expansion01`. The currently inspected V22 stores cover all of those except `Kalidar`; the curation manifest contains `Kalidar`, but no current local V22 store for a Kalidar-bearing build is present in `../output/datasets/v22`.

## RunPod Bundle Surface

Repo-owned bundle inputs now live under `wow-viewer/data-harvester/runpod/v23/`:

- `install_deps.sh`
- `verify_bundle.sh`
- `smoke.sh`
- `train.sh`

`scripts/package_v23_runpod.py` builds a BYOD tar containing:

- the V23 Python modules
- the V22 read dependency surface actually imported by the trainer
- `train_v23_height.py` and `infer_v23_height.py`
- optional `tests/v23/`
- a bounded V22 subset store under `data/v22/`
- a `manifest.json` with `contains_game_client_files = false`

## Next Proof Order

The next proof order is:

- cache or download the HF DA-V2-Small checkpoint and rerun through the curated manifest path when quality or pretrained behavior is being evaluated
- keep local stability proof curated and map-targeted first, especially `--maps Northrend` for Wrath terrain knowledge
- run the T046 24 GB Pod smoke with the same artifact capture discipline
- only then consider full-corpus remote training

Remote proof is still required eventually for:

- 24 GB Pod smoke without CUDA OOM
- peak VRAM capture
- full-corpus training run
- CAI seam review on real validation tiles
- same-commit cross-pod determinism

The earlier remote attempt is not a trustworthy proof owner. The active next gate is the local 12 GB path.

If the remote path is reopened later, the V23 RunPod selector should prefer consumer cards first: `3090`, then `4090`, then `5090`, ahead of workstation fallbacks. The earlier cheap-first ordering selected `A4500` before those consumer targets, which is no longer the intended route.

Those are proof tasks, not missing local source files.
