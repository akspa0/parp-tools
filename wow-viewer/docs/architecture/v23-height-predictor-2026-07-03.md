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

- no local CUDA peak-VRAM proof for the 12 GB real-GPU envelope
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

The next proof order is local first:

- local 12 GB CUDA smoke with `target-vram-gb = 12`
- capture `peak_vram.json`
- only then consider any remote retry

Remote proof is still required eventually for:

- 24 GB Pod smoke without CUDA OOM
- peak VRAM capture
- full-corpus training run
- CAI seam review on real validation tiles
- same-commit cross-pod determinism

The earlier remote attempt is not a trustworthy proof owner. The active next gate is the local 12 GB path.

If the remote path is reopened later, the V23 RunPod selector should prefer consumer cards first: `3090`, then `4090`, then `5090`, ahead of workstation fallbacks. The earlier cheap-first ordering selected `A4500` before those consumer targets, which is no longer the intended route.

Those are proof tasks, not missing local source files.
