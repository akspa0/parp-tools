# Tasks: DA-V2-Small LoRA Height Predictor with Cross-Tile Consistency

**Input**: Design documents from `/specs/089-dav2-height-predictor/`

**Prerequisites**: `plan.md` (required), `spec.md` (required for user stories), `research.md`, `data-model.md`, `quickstart.md`, and `contracts/` (generated during the completed `speckit-plan` pass for this spec).

**Tests**: Tests are included. Each US1–US6 user story has its own test file under `tests/v23/`. Tests are written before implementation per the spec template's recommendation where it improves bug surfaces; for purely additive modules (channels.py, losses.py), tests and implementation can land together.

**Organization**: Tasks grouped by phase, then by user story inside the phase. Format: `[ID] [P?] [Story] Description`.

## Path Conventions

Per the plan's Project Structure, all paths are under `wow-viewer/data-harvester/` for Python sources/scripts, and under `wow-viewer/specs/089-dav2-height-predictor/` and `wow-viewer/docs/architecture/` for documentation. Paths below assume the data-harvester root unless prefixed with `wow-viewer/`.

---

## Phase 0: Repo Setup & Dependency Verification

**Purpose**: V23 package skeleton importable. CI baseline green.

- [ ] T001 [P] Add `transformers>=4.40`, `peft>=0.10`, `bitsandbytes>=0.43` to `wow-viewer/data-harvester/pyproject.toml` under `[project.dependencies]`. Run `uv sync` and confirm lockfile resolves.
- [x] T002 [P] Create `wow-viewer/data-harvester/src/harvester/v23/__init__.py` exposing public surface (`V23HeightDataset`, `V23HeightPredictor`, `V23Checkpoint`, `run_cai_inference`, `build_channel_tensor`) with `NotImplementedError` stubs.
- [x] T003 [P] Create empty module files: `channels.py`, `dataset.py`, `encoder.py`, `head.py`, `model.py`, `losses.py`, `inference.py`, `checkpoint.py`. Each file contains only module docstring + `raise NotImplementedError` for exported classes/functions referenced by `__init__.py`.
- [x] T004 [P] Create `wow-viewer/data-harvester/tests/v23/__init__.py` (empty). Create `tests/v23/conftest.py` that skips all tests under `tests/v23/` when not running pytest with `-m v23` (avoid noise on heritage test runs).
- [ ] T005 Run `uv sync && uv run python -c "import harvester.v23"` and `uv run pytest -q`. Confirm exit 0 with 0 collected tests under the v23 marker. No heritage test under `wow-viewer/data-harvester/tests/` regressed.

2026-07-03 status note: dependency declarations and the V23 package/test skeleton are source-applied. T001 and T005 remain open because local shell execution fails before command launch with `The "path" argument must be of type string. Received undefined`, so `uv sync`, import smoke, and pytest have not been executed.

**Checkpoint**: Pending. V23 package wiring is source-applied, but dependency install, import smoke, and CI baseline validation still need to run.

---

## Phase 1: V23 Dataset Adapter (User Story 1)

**Purpose**: `V23HeightDataset` reads V22 Zarr and produces the documented 15-channel Input Channel Contract + `height_257` target.

**Independent Test**: `pytest tests/v23/test_dataset.py tests/v23/test_channels.py -m v23` green; one real V22 tile loads with documented shapes.

### Tests for User Story 1

- [ ] T006 [P] [US1] Write `wow-viewer/data-harvester/tests/v23/test_channels.py` covering: (a) `CHANNEL_ORDER` constant matches the documented Spec Input Channel Contract indices 0–14 in order; (b) `build_channel_tensor(zarr_tile)` returns tensor shape `[15, 256, 256]` for synthetic input; (c) degrade mode `minimap_only` produces a 3-channel tensor with indices `[0,1,2]`; (d) degrade mode `minimap_alpha` produces a 7-channel tensor with indices `[0,1,2,3,4,5,6]`. (Tests must FAIL before implementation.)
- [ ] T007 [P] [US1] Write `wow-viewer/data-harvester/tests/v23/test_dataset.py` covering: (a) `V23HeightDataset(store_path, build="3_3_5_12340").__getitem__(0)` returns dict with `input` shape `[15, 256, 256]`, `target` shape `[1, 257, 257]`, `valid_mask` shape `[1, 257, 257]`; (b) for a synthetic tile with `liquid_mask > 0`, target equals `liquid_height_257` at those pixels; (c) for a synthetic tile missing `normal_xyz` array when `--input-mode minimap_alpha`, input has 7 channels and `valid_mask` reflects the absent channels as False; (d) docstring on `V23HeightDataset` lists every channel index, source, dtype, normalization, fill policy.

### Implementation for User Story 1

- [ ] T008 [US1] Implement `wow-viewer/data-harvester/src/harvester/v23/channels.py`:
  - `CHANNEL_ORDER` constant (list of 15 channel-name strings in documented order).
  - `InputMode` enum (`FULL`, `MINIMAP_ONLY`, `MINIMAP_ALPHA`, `MINIMAP_ALPHA_NORMAL`).
  - `CHANNEL_INDICES` mapping from `InputMode` → list of int indices active.
  - `build_channel_tensor(zarr_tile, mode=InputMode.FULL, tileset_prune_table=None)` → torch tensor `[len(CHANNEL_INDICES[mode]), 256, 256]`.
  - Normalisation rules for `minimap_rgb` (uint8→float32 / 255.0, ImageNet mean/std) and `alpha_256` (already float32, [0,1]).
  - Tileset one-hot encoding against the prune table (top-K=256 by default).
- [ ] T009 [US1] Implement `wow-viewer/data-harvester/scripts/build_tileset_prune_table.py`:
  - Reads a V22 store's `tilesets/tileset_paths` + all `mcly_tileset_ids` arrays.
  - Counts frequency of each unique tileset id.
  - Emits JSON `{original_id: pruned_index, ...}` sorted by frequency such that pruned indices 0..K-1 cover the top-K tilesets; everything outside maps to `K` (OOV bucket).
  - Default `--top-k 256`. Default `--builds` to all V22 builds under `--dataset-dir`.
  - Default output path `../output/datasets/v22/tileset_prune_<run>.json`.
- [ ] T010 [US1] Implement `wow-viewer/data-harvester/src/harvester/v23/dataset.py`:
  - `V23HeightDataset(torch.utils.data.Dataset)` reads V22 Zarr via `harvester.v22_zarr_io`.
  - Bypass liquid-override via resampling `liquid_height_256` to `liquid_height_257` using bicubic + masking by `liquid_mask_256` resampled to 257.
  - `terrain_valid_mask_257` derived from `mcnr_mask_257 & ~liquid_mask_257 & ~object_mask_257_binarized`.
  - Degrade-mode zero-fill + per-channel `valid_mask` tracking.
  - Docstring lists every channel contract detail.
- [ ] T011 [US1] Run `pytest tests/v23/test_dataset.py tests/v23/test_channels.py -m v23` and confirm green.
- [ ] T012 [US1] Real-data proof: locate or produce a V22 store under `wow-viewer/output/datasets/v22/3_3_5_12340.zarr/`. If absent, run `uv run python scripts/build_v18_dataset.py build --build 3_3_5_12340 --allow-zarr-write` then `uv run python scripts/build_v22_dataset.py enrich ... && uv run python scripts/build_v22_dataset.py build ...` per Spec 088 quickstart. Then run `uv run python -c "from harvester.v23 import V23HeightDataset; ds = V23HeightDataset('../output/datasets/v22/3_3_5_12340.zarr', build='3_3_5_12340'); print({k: v.shape for k, v in ds[0].items()})"` and record the printed dict in a Phase 1 validation note. Confirm shapes match documented contract.

**Checkpoint**: US1 acceptance scenarios 1–4 pass; documented Input Channel Contract is verified against a real V22 tile.

---

## Phase 2: DA-V2-Small Encoder + LoRA + Patch-Embed Swap (User Story 2)

**Purpose**: Frozen DA-V2-Small + LoRA-r16 adapters + first-conv swap; stock DA-V2-Small forward bit-identical when LoRA disabled.

**Independent Test**: `pytest tests/v23/test_encoder.py -m v23` green; three random-input stock-DA-V2-Small-vs-LoRA-disabled forward passes are bit-identical.

### Tests for User Story 2

- [ ] T013 [P] [US2] Write `wow-viewer/data-harvester/tests/v23/test_encoder.py` covering: (a) instantiate `DepthAnythingV2SmallEncoder(in_channels=15)`, count params: non-LoRA base frozen (all `requires_grad=False`), LoRA adapter params `< 2_000_000`, first patch-embed conv params trainable (all `requires_grad=True`); (b) forward pass on `[2, 15, 518, 518]` produces the documented DPT feature pyramid (output is a dict/list of intermediate features at DA-V2-Small's documented intermediate shapes); (c) under `disable_lora()` context, forward pass is bit-identical to a fresh `transformers.DepthAnythingV2ForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')` forward pass on a `[2, 3, 518, 518]` (only 3ch — minimap-only equivalent) input via `torch.allclose(atol=0, rtol=0)` on three random seeds.

### Implementation for User Story 2

- [ ] T014 [US2] Implement `wow-viewer/data-harvester/src/harvester/v23/encoder.py`:
  - `DepthAnythingV2SmallEncoder(nn.Module)`. Constructor takes `in_channels`, LoRA config flags (`lora_rank=16`, `lora_alpha=32`, `lora_dropout=0.05`), and HF weights source `model_id='depth-anything/Depth-Anything-V2-Small-hf'`.
  - Load via `transformers.AutoModelForDepthEstimation.from_pretrained(model_id)`. Extract `pretrained.model` (DPT encoder) into self.pretrained_encoder. Freeze all `pretrained_encoder.parameters()`.
  - Apply LoRA: `peft.LoraConfig(target_modules=['q_proj','k_proj','v_proj','out_proj'], r=16, lora_alpha=32, lora_dropout=0.05, bias='none')` then `peft.get_peft_model(self.pretrained_encoder, lora_config)`.
  - Replace the first patch-embed conv: preserve the original input projection conv. Construct a fresh `nn.Conv2d(in_channels=in_channels, out_channels=<hidden_dim>, kernel_size=original.kernel_size, stride=original.stride, padding=original.padding)`. Initialise via `nn.init.normal_(mean=0.0, std=0.02)`. Wire in via attribute assignment. Mark this conv `requires_grad=True`.
  - Implement `forward(x) -> FeatureDict` exposing the intermediate features consumed by the V23HeightHead (Phase 3). The shape contract is documented in the module docstring with explicit tensor shape annotations.
  - Implement `disable_lora()` context manager that calls `peft.disable_adapter_layers()` on enter + `peft.enable_adapter_layers()` on exit.
  - Docstring cites DA-V2 paper + peft docs + HF model card.
- [ ] T015 [US2] Implement `disable_adapter_layers` test in `test_encoder.py`: under `disable_lora()` on a 3-channel input, the encoder forward should match a fresh `transformers.AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')` forward on the same input bit-for-bit across three random seeds (using `torch.manual_seed(N)` before each noise draw, then `torch.allclose(atol=0, rtol=0)`).
- [ ] T016 [US2] Run `pytest tests/v23/test_encoder.py -m v23` and confirm green.

**Checkpoint**: US2 acceptance scenarios 1–3 pass; LoRA config + first-conv swap matches the FR-003 / FR-004 spec contracts.

---

## Phase 3: V23 Height Decoder Head + Combined Model (User Story 3)

**Purpose**: DPT-style head < 5M params producing `[B, 1, 257, 257]` disparity + `[B, 2]` affine anchor; combined `V23-HeightPredictor` forward.

**Independent Test**: `pytest tests/v23/test_head.py tests/v23/test_model.py -m v23` green; head param count under 5M.

### Tests for User Story 3

- [ ] T017 [P] [US3] Write `wow-viewer/data-harvester/tests/v23/test_head.py` covering: (a) instantiate `V23HeightHead(encrypted=feature_dict_schema)`, fed synthetic features matching the schema, output disparity shape `[B, 1, 257, 257]` and affine anchor shape `[B, 2]`; (b) `sum(p.numel() for p in head.parameters() if p.requires_grad) < 5_000_000`; (c) disparity returned is float32 in `[0, 1]` (sigmoided); (d) affine anchor composition `disparity * scale + shift` produces realistic metric height range (sanity: not all zero, not all NaN).
- [ ] T018 [P] [US3] Write `wow-viewer/data-harvester/tests/v23/test_model.py` covering: (a) `V23-HeightPredictor(in_channels=15)` forward pass on `[2, 15, 518, 518]` returns a `V23ModelOutput` dataclass with `.disparity`, `.affine_anchor`, `.metric_height`; (b) `.metric_height` shape `[2, 1, 257, 257]`, dtype float32; (c) total trainable param count `sum(p.numel() for p in model.parameters() if p.requires_grad) < 8_000_000`.

### Implementation for User Story 3

- [ ] T019 [US3] Implement `wow-viewer/data-harvester/src/harvester/v23/head.py`:
  - `V23HeightHead(nn.Module)`. Constructor requires a `feature_dict_schema` argument (dict of feature names → shape templates) so the reassembly blocks can be sized against the encoder's actual output.
  - Reassembly blocks reduced in width from the canonical 256-DA-V2-Large widths down to 128 (encoder feature dim) at the finest scale to keep param count under 5M.
  - Residual fuse + final sigmoid conv → disparity `[B, 1, H, W]` then `F.interpolate(mode="bicubic", align_corners=False, size=(257, 257))`.
  - Affine anchor head: pool encoder features via `mean over spatial dim`, then `MLP(features_dim → 64 → 2)` → `(scale, shift)`. Sigmoid scale to keep in `[0, 1]`; tanh shift scaled to `[-1, 1]`.
- [ ] T020 [US3] Implement `wow-viewer/data-harvester/src/harvester/v23/model.py`:
  - `V23HeightPredictor(nn.Module)` wraps `DepthAnythingV2SmallEncoder` + `V23HeightHead`.
  - Forward returns `V23ModelOutput` (dataclass) with `.disparity`, `.affine_anchor`, `.metric_height = disparity * scale + shift` (broadcasting scale/shift over H×W).
  - Under `model.eval()` + `torch.no_grad()`, the forward is deterministic given the same input — but this property belongs to Phase 6 to formally verify across seeds.
- [ ] T021 [US3] Implement the `feature_dict_schema` generator: small helper `infer_encoder_feature_schema(encoder)` runs a forward pass with a single synthetic input and captures intermediate shapes. Cache via `functools.lru_cache(maxsize=1)`. Wired into the head constructor in `model.py`.
- [ ] T022 [US3] Run `pytest tests/v23/test_head.py tests/v23/test_model.py -m v23` and confirm green.

**Checkpoint**: US3 acceptance scenarios 1–3 pass; total trainable param count under 8M per SC-008.

---

## Phase 4: Loss Stack — Lssi, Lgm, SDC, GPCT, Bias-Free Masking (User Story 4 part 1)

**Purpose**: Five loss components implemented as pure-PyTorch functions; combiner under weight flags.

**Independent Test**: `pytest tests/v23/test_losses.py -m v23` green; zero-weighted components are bypassed cleanly.

### Tests for User Story 4 (Part 1)

- [ ] T023 [P] [US4] Write `wow-viewer/data-harvester/tests/v23/test_losses.py` covering each component:
  - Lssi: input `pred=[2,1,257,257]`, `target=[2,1,257,257]`, both random; output scalar >= 0; gradient flows through both; least-squares alignment per-sample is internally applied before computing MSE.
  - Lgm: Sobel-style gradient of `pred` and `target`; L1 between gradients; output scalar >= 0; gradient flows.
  - SDC: spatial-distance-constraint matrix at patch size 16×16 (configurable); output scalar >= 0; gradient flows.
  - GPCT: input is a list of K=4 sub-tile predictions + a list of overlap-coordinate tuples; L2 between features at overlap regions; output scalar; gradient distributed over all 4 sub-tile pixels in overlap.
  - BiasFreeMasking helper: input `[B, C, H, W]`, ratio 0.15, ancestor-shuffled RNG; output tensor with same shape, masked patch indices recoverable via the returned mask.
  - `compute_v23_loss(...)`: total = weights × components; if `weights['gpct']=0.0`, GPCT is bypassed (no forward through the GPCT path).
  - All gradients:
    - zero_grad the model, compute a single loss component, backward, assert each param has nonzero gradient where applicable.
    - Zero weight for a component, expected contribution == 0 in total.

### Implementation for User Story 4 (Part 1)

- [ ] T024 [US4] Implement `affine_invariant_lssi(pred, target, mask=None)` in `wow-viewer/data-harvester/src/harvester/v23/losses.py`. Port from DepthAnything-V2 training code at `https://github.com/DepthAnything/Depth-Anything-V2/blob/main/loss.py`. Citation in docstring. Returns scalar loss.
- [ ] T025 [US4] Implement `gradient_matching_lgm(pred, target, mask=None)` in `losses.py`. Port from DA-V2 source. Returns scalar.
- [ ] T026 [US4] Implement `spatial_distance_constraint(features_pred, features_target, patch_size=16)` in `losses.py`. Per DepthAnything-AC paper (arXiv 2507.01634) §3.2. Returns scalar.
- [ ] T027 [US4] Implement `gpct_overlap_consistency(sub_tile_preds, sub_tile_features, overlap_coords, feature_loss=True)` in `losses.py`. Per PRO paper (arXiv 2503.22351) §3.1. Returns scalar.
- [ ] T028 [US4] Implement `apply_bias_free_masking(input_tensor, ratio=0.15, generator=None)` in `losses.py`. Returns `(masked_tensor, mask_indices)`. Masked patches are filled with channel-mean RGB (not zero, not noise).
- [ ] T029 [US4] Implement `compute_v23_loss(outputs, target, weights, valid_mask=None)` in `losses.py`. Returns `(total_loss, components_dict)`. Each component gated by its weight being > 0.0.
- [ ] T030 [US4] Run `pytest tests/v23/test_losses.py -m v23` and confirm green. Cross-check gradient flow on a tiny synthetic model containing one Linear + Conv as a sanity probe.

**Checkpoint**: US4 acceptance scenario 1 (zero-weight bypass) verified; components are isolated and testable independently.

---

## Phase 5: Training Script `train_v23_height.py` (User Story 4 part 2)

**Purpose**: Working trainer with deterministic mode, GPCT batch packing, bias-free masking integration, checkpoint metadata, validation previews.

**Independent Test**: A 2-epoch 4-tile smoke run completes; checkpoint + preview PNG saved.

### Tests for User Story 4 (Part 2)

- [ ] T031 [P] [US4] Write `wow-viewer/data-harvester/tests/v23/test_train_smoke.py` covering: (a) call `train_v23_height.main` with synthetic-data arguments `--epochs 2 --train-max-tiles 4 --val-max-tiles 2 --device cpu --deterministic --seed 42` against a 4-tile synthetic V22-shaped store fixture; assert checkpoint file `models/v23/height/runs/smoke_<run>/checkpoints/v23_height_last.pt` exists; assert preview PNG `models/v23/height/runs/smoke_<run>/val_preview_2/tile_0.png` exists; assert checkpoint metadata includes `seed=42`, `commit_sha`, `input_mode=full`, `gpct_weight`, `bias_free_mask_ratio`; (b) re-run with `--seed 12345` and same args; load both checkpoints' `model_state`; assert `torch.allclose(state_dict_42, state_dict_12345, atol=0, rtol=0)`.

### Implementation for User Story 4 (Part 2)

- [ ] T032 [US4] Implement `wow-viewer/data-harvester/src/harvester/v23/checkpoint.py`:
  - `V23Checkpoint` dataclass: `config: dict`, `model_state: Dict[str, torch.Tensor]`, `optimizer_state: Dict[str, Any]`, `epoch: int`.
  - `save_checkpoint(path, ckpt)` writes torch.save with the full config dict including `seed`, `commit_sha` (via `subprocess.check_output(['git','rev-parse','HEAD']).strip().decode()`), `input_mode`, all CLI flag values, `v22_store_path_hash`, `tileset_prune_table_hash`.
- [ ] T033 [US4] Implement `wow-viewer/data-harvester/scripts/train_v23_height.py`:
  - argparse with all documented flags: `--dataset-dir`, `--builds`, `--input-mode`, `--tileset-prune-table`, `--epochs`, `--lr`, `--batch-size`, `--grad-accum-steps`, `--gpct-K`, `--gpct-weight`, `--gpct-feature-loss`, `--sdc-weight`, `--spectral-weight`, `--bias-free-mask-ratio`, `--val-max-tiles`, `--val-interval`, `--val-preview-interval`, `--target-vram-gb`, `--device`, `--deterministic`, `--seed`, `--run-name`, `--output-dir`, `--resume-checkpoint`.
  - Dataset construction: `V23HeightDataset` with the configured `--input-mode` and `--tileset-prune-table`.
  - Optimizer: `bitsandbytes.optim.PagedAdamW8bit(model.parameters(), lr=args.lr)`. Resume from checkpoint state if `--resume-checkpoint` set.
  - Mixed precision: `torch.cuda.amp.autocast(dtype=torch.bfloat16)` around forward + loss; `GradScaler` is no-op for bf16 (used for API symmetry).
  - Gradient checkpointing: `model.encoder.gradient_checkpointing_enable()` when device is cuda.
  - Deterministic mode under `--deterministic`: set `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `torch.utils.data.DataLoader(generator=torch.Generator().manual_seed(seed))`.
  - Single-tile baseline path under `--gpct-weight 0`: process batch of tiles -> forward -> loss -> backward.
  - GPCT path under `--gpct-weight > 0`: split each tile in the batch into K=4 overlapping sub-tiles (offsets `[0,0],[step,0],[0,step],[step,step]` where `step = (256 - 64)/(K**(1/2)-1)`). Forward all K sub-tiles through the encoder (one batched forward), compute disparity at 256×256, upsample to 257×257, return patch predictions + features + overlap coordinates. Feed to `gpct_overlap_consistency`. Add to total loss.
  - Bias-free masking: under `--bias-free-mask-ratio > 0`, apply `apply_bias_free_masking` to the minimap channels (indices 0-2) before forward.
  - OOM catch: `try/except torch.cuda.OutOfMemoryError`; on first OOM, halve `--batch-size`, retry once. Persistent OOM after retry becomes fatal.
  - Validation loop every `--val-interval` epochs: per-tile L1 of anchored metric height vs target `height_257`, save preview PNG every `--val-preview-interval`.
  - Checkpoint save every epoch + best.
- [ ] T034 [US4] Run `pytest tests/v23/test_train_smoke.py -m v23` and confirm green.
- [ ] T035 [US4] Run on real GPU (if available locally) for 2 epochs on a 16-tile real V22 subset. Confirm zero CUDA OOM at `--device cuda --batch-size 4 --gpct-K 4 --target-vram-gb 22 --bias-free-mask-ratio 0.15` on an RTX-class card. Capture peak VRAM via `torch.cuda.max_memory_allocated() / 1e9` and record in the run folder's `peak_vram.json`. If no local CUDA available, defer this validation to Phase 8 RunPod smoke.

**Checkpoint**: US4 acceptance scenarios 1–4 pass; checkpoint format full per FR-019.

---

## Phase 6: Deterministic Inference + CAI Stitching (User Story 5)

**Purpose**: `infer_v23_height.py` is byte-reproducible; CAI-R=16 produces no seam.

**Independent Test**: Two-seed determinism test bit-identical. CAI seam cross-tile L1 drops >= 50% vs no-CAI on a 3×3 grid.

### Tests for User Story 5

- [ ] T036 [P] [US5] Write `wow-viewer/data-harvester/tests/v23/test_inference_determinism.py` covering: (a) load a trained smoke checkpoint; (b) run inference on a single tile twice with `--seed 42` and `--seed 12345` under `--deterministic`; assert `torch.allclose(atol=0, rtol=0)`.
- [ ] T037 [P] [US5] Write `wow-viewer/data-harvester/tests/v23/test_cai_stitch.py` covering: (a) load 3×3 tile grid from V22 store fixture; (b) run CAI-R=16 inference and CAI-R=1 inference; (c) cross-tile L1 along every shared edge is < 50% with CAI vs without; (d) saved preview PNG exists.

### Implementation for User Story 5

- [ ] T038 [US5] Implement `wow-viewer/data-harvester/src/harvester/v23/inference.py`:
  - `run_cai_inference(model, store, tile_xy_grid, cai_r=16)`: build R overlapping sub-tile crops with overlap stride `(256 - overlap)//(R-1)` along both axes. Run model on each crop. For each output pixel, accumulate predictions and a coverage count into a running-mean buffer over a `(py*256, px*256)` output array. Skip outside-grid positions. Return the stitched tensor.
  - When `cai_r=1`, the function is equivalent to single-pass tile inference on each grid cell without overlap.
  - Hard guarantee: under `model.eval() + torch.no_grad()` (set by the caller), the running mean is the only averaging step.
- [ ] T039 [US5] Implement `wow-viewer/data-harvester/scripts/infer_v23_height.py`:
  - argparse: `--checkpoint`, `--v22-store`, `--build`, `--tiles <list>` (or `--tile-grid NxM`), `--output-dir`, `--cai-r`, `--seed`, `--deterministic`, `--save-preview`, `--fp16`.
  - Load checkpoint, restore config (input_mode, tileset_prune_table path, gpct flags).
  - For single-tile inference (`--tiles` lists one): forward + save disparity + metric height NPZ + preview PNG.
  - For multi-tile inference: call `run_cai_inference` with the requested `--cai-r`. Save stitched NPZ + preview PNG.
  - Set deterministic flags under `--deterministic`.
- [ ] T040 [US5] Run `pytest tests/v23/test_inference_determinism.py tests/v23/test_cai_stitch.py -m v23` and confirm green.

**Checkpoint**: US5 acceptance scenarios 1–3 pass; SC-002 + SC-004 + SC-005 verifiable.

---

## Phase 7: RunPod Bundle (User Story 6)

**Purpose**: Spec-079-compliant tar ships V23 to a 24 GB RunPod Pod; one-command bootstrap.

**Independent Test**: Pod install completes; `smoke.sh` runs without CUDA OOM at the documented 24 GB config.

### Implementation for User Story 6

- [ ] T041 [P] [US6] Implement `wow-viewer/data-harvester/runpod/v23/install_deps.sh`:
  - `set -euo pipefail`; `cd /workspace/v23_bundle`.
  - `uv sync --frozen` (uses packaged `pyproject.toml` + `uv.lock`).
  - Install extra: `uv pip install transformers peft bitsandbytes`.
  - Validate download: `python -c "from transformers import AutoModelForDepthEstimation; AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')"` — first-download caches to `/runpod-volume/hf_cache/` via `HF_HOME` env.
  - Log to `/workspace/bootstrap.log`.
- [ ] T042 [P] [US6] Implement `runpod/v23/verify_bundle.sh`: `set -euo pipefail`; check `manifest.json` exists; check `contains_game_client_files == "false"`; check `src/harvester/v23/__init__.py` exists; check `scripts/train_v23_height.py` and `scripts/infer_v23_height.py` exist; `python -c "from harvester.v23 import V23HeightPredictor"` succeeds (no NotImplementedError for the public surface).
- [ ] T043 [P] [US6] Implement `runpod/v23/smoke.sh`:
  - `set -euo pipefail`.
  - `uv run python scripts/train_v23_height.py --epochs 2 --train-max-tiles 4 --val-max-tiles 2 --device cuda --target-vram-gb 22 --batch-size 4 --gpct-K 4 --gpct-weight 0.1 --bias-free-mask-ratio 0.15 --deterministic --seed 42 --run-name smoke_v23`.
  - Asserts the checkpoint + preview exist; exits non-zero on failure.
- [ ] T044 [P] [US6] Implement `runpod/v23/train.sh` — minimal wrapper: `exec uv run python scripts/train_v23_height.py "$@"`.
- [ ] T045 [US6] Implement `wow-viewer/data-harvester/scripts/package_v23_runpod.py`:
  - argparse: `--bundle-name`, `--v22-store-subset-path`, `--tileset-prune-table`, `--output-tar`, `--include-v22-subset-tiles <int>`.
  - Copies `src/harvester/v23/`, `scripts/train_v23_height.py`, `scripts/infer_v23_height.py`, `pyproject.toml` (trimmed to v23 deps only), `uv.lock`, `runpod/v23/{install_deps,verify_bundle,smoke,train}.sh`, `tests/v23/`.
  - Generates or copies the V22 Zarr subset (using a small `zarr` slice of the first N tiles of each build root).
  - Writes `manifest.json` with `contains_game_client_files: false`, the tree hash audit, and a `v23_bundle_version` field.
- [ ] T046 [US6] Manual Pod validation: launch a 24 GB RunPod RTX 4090 Pod (per Spec 079 — `--manual-pod` fallback if REST API fails). Upload the tar via `scp -P <port> <tar> root@<ip>:/workspace/`. `tar -xf v23_bundle.tar -C /workspace/v23_bundle`. Run `bash runpod/v23/install_deps.sh && bash runpod/v23/verify_bundle.sh && bash runpod/v23/smoke.sh`. Confirm smoke passes without CUDA OOM; capture `nvidia-smi` peak during smoke into `/workspace/v23_bundle/peak_vram.json`.

**Checkpoint**: US6 acceptance scenarios 1–3 pass; SC-001 partially verified (smoke); SC-007 verified (manifest audit).

---

## Phase 8: Real-Data Validation against a Staged V22 Store

**Purpose**: End-to-end real-data proof. SC-001 (no OOM), SC-003 (L1 vs V21 baseline), SC-004 (no seam), SC-005 (cross-tile L1 with CAI), SC-006 (determinism across Pods), SC-007 (bundle manifest clean).

**Independent Test**: All six SC items evidenced in the run folder.

### Implementation for User Story (Real-Data Validation)

- [ ] T047 Real-data: confirm or build a V22 Zarr store at `wow-viewer/output/datasets/v22/3_3_5_12340.zarr/` per Spec 088 quickstart. If building fresh, record hashes of the staged client root + V18 store + V22 store into a `data_provenance.json` for reproducibility.
- [ ] T048 Ship the V23 bundle to a 24 GB RunPod Pod per Phase 7. Run `bash runpod/v23/train.sh --dataset-dir /runpod-volume/v22 --builds 0_5_3_3368 3_3_5_12340 4_0_0_11927 --epochs 4 --train-max-tiles 25000 --val-max-tiles 256 --val-interval 1 --val-preview-interval 1 --device cuda --target-vram-gb 22 --batch-size 4 --gpct-K 4 --gpct-weight 0.1 --sdc-weight 0.1 --bias-free-mask-ratio 0.15 --deterministic --seed 42 --run-name v23_height_full_corpus_v1` (this is the canonical training command for the spec). Capture peak VRAM via `nvidia-smi dmon`sampling or by reading `torch.cuda.max_memory_allocated()` from the trainer's log.
- [ ] T049 Capture V21 baseline L1 for the same validation set (saved into `models/v23/height/runs/v23_height_full_corpus_v1/baselines/v21_val_l1.json` from re-running `train_v18.py height` reference eval or by referencing a previously-recorded V21 baseline number). Compute V23 per-tile val L1 from saved predictions. Compute diff. Record into `models/v23/height/runs/v23_height_full_corpus_v1/sc_003_v21_comparison.json`.
- [ ] T050 Run CAI-R=16 inference on a 3×3 validation tile grid using `infer_v23_height.py --tiles <3x3 list> --cai-r 16 --deterministic --seed 42 --save-preview --output-dir models/v23/height/runs/v23_height_full_corpus_v1/cai_3x3/`. Also run CAI-R=1 variant for the same grid. Save both preview PNGs.
- [ ] T051 Run cross-tile L1: read both CAI-R=1 and CAI-R=16 stitched outputs, compute L1 along every shared tile edge, average over edges per output. Record `--cai-r=1_avg_edge_l1` and `--cai-r=16_avg_edge_l1` into `sc_005_cai_comparison.json`. Confirm CAI-R=16 is at least 50% lower.
- [ ] T052 Determinism proof on Pod: run `infer_v23_height.py --tiles <single tile> --seed 42 --deterministic --output-dir /tmp/det_proof_42/`. Run again with `--seed 12345 --output-dir /tmp/det_proof_12345/`. Diff the two NPZ files; assert bit-identical. Record into `sc_006_determinism_proof.json`.
- [ ] T053 Manual visual review: open the CAI-R=16 preview PNG. Confirm no visible seam at tile boundaries. Note the observation in `sc_004_visual_review.md` inside the run folder (text file body: "Reviewed by <operator>; preview at <relative path>; seam observed: no / yes; observation: <free text>").
- [ ] T054 Update `wow-viewer/memory-bank/activeContext.md` to record V23 status. Compress aggressively: ~10 lines describing V23 completion, Phase 1–8 status, and what the next active work is. Update `wow-viewer/memory-bank/progress.md` with a one-line V23 entry dated 2026-07-03.
- [ ] T055 Write `wow-viewer/docs/architecture/v23-height-predictor-2026-07-03.md` covering: architecture overview (DA-V2-Small + LoRA + DPT head + affine anchor), loss stack (Lssi + Lgm + SDC + GPCT + BiasFree), training envelope (24 GB RunPod), inference envelope (6 GB, fp16), determinism strategy (3-layer), validation results (SC-001 through SC-007 with file references), open questions (e.g. cross-build tileset id contract: union-top-K vs per-build choose), and downstream V24+ integration plan (height output feeds into normal/liquid/etc models as frozen input).

**Checkpoint**: All SC-001 through SC-007 verified and recorded. The spec is "implementation complete" only when these files exist with the expected contents.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 0**: No dependencies. Start immediately.
- **Phase 1**: Must complete after Phase 0 (deps).
- **Phase 2**: Depends on Phase 1's `channels.py` for channel count constant.
- **Phase 3**: Depends on Phase 2's encoder forward shapes.
- **Phase 4**: Depends on Phase 3's output shape contract.
- **Phase 5**: Depends on Phases 1–4 (dataset + encoder + head + losses).
- **Phase 6**: Depends on Phase 5 (loads checkpoint format produced by trainer).
- **Phase 7**: Depends on Phases 5–6 (wraps active scripts).
- **Phase 8**: Depends on Phase 7 (uses RunPod bundle).

### Within-Phase Dependencies

- Each phase has tests at the top (where TDD improves bug surface) followed by implementation tasks. Tests can be marked [P] to allow parallel writing with reference implementation tasks.
- Within a phase, tasks marked [P] can run in parallel (different files, no dependencies).
- The final task in each phase is always the run-validation task — it must execute after all other tasks in the phase.

### Parallel Opportunities

- Phase 0: T001, T002, T003, T004 all touch different files; parallel.
- Phase 1: T006 + T007 (tests) and T008 + T009 (channels.py + prune script) can run in parallel; T010 depends on T008.
- Phase 2: T013 (test) + T014 (encoder.py) parallel; T015 depends on T014.
- Phase 3: T017, T018 (tests) parallel; T019, T020, T021 implementation tasks sequence: T021 first (feature schema helper), then T019 (head), then T020 (model).
- Phase 4: All tests T023 in parallel with implementation tasks T024–T028; T029 (combiner) depends on all component-loss functions.
- Phase 7: T041, T042, T043, T044 (install/verify/smoke/train shell scripts) parallel; T045 (packager) depends on all of them; T046 (Pod validation) depends on T045 + RunPod provisioning.

---

## Implementation Strategy

### MVP First — Phase 1 + Phase 2 + Phase 3 + Phase 5 minimal

A real-evidence MVPkra commit-able intermediate is the Phase-1-done state: dataset adapter shape verified against real V22 tile. Even without LoRA, an unfrozen DA-V2-Small + fresh head trained on `minimap_rgb` only would be a meaningful baseline. Phases 2–8 incrementally specialize.

### Incremental Delivery

Each phase's checkpoint produces a committable diff on its own. Stop-and-validate at any checkpoint per the spec's "One Phase at a Time" rule (RULE 8).

### Anti-Risks

- Phase 0 task list explicitly avoids feature work; stubs return `NotImplementedError`. This keeps Phase 1 unblocked by file-existence concerns.
- Phase 1's prune-table script (T009) is independent of `dataset.py`, so the prune table can be built standalone before the dataset adapter is complete. The dataset adapter uses it via `--tileset-prune-table`.
- Phase 2's LoRA-on-`out_proj` may not exist on all DA-V2-Small sub-blocks (some blocks use `proj` instead of `out_proj`). The target_modules list uses both. If peft complains about a missing `out_proj`, the implementation falls back to `[proj]` and raises a warning. Captured in T014.
- Phase 5's GPCT-batched forward may OOM at K=4 on 24 GB if the model also has #channels=15. If running K=4 batched at `--batch-size 4 --gpct-K 4` (= 16 effective forward) does OOM, fall back to GPCT-K=4 with sequential sub-tile forwards inside the training step (slower but fits). Documented in `train_v23_height.py` argument help.

---

## Notes

- `[P]` tasks = different files, no dependencies.
- Each task maps to a single user story via `[US1]`...`[US6]` for traceability back to the spec.
- Each phase ends with a "Run `pytest ...`" or "Run real-data proof ..." task as the validation gate.
- Commits are one-logical-change-per-commit. The checkpoint task T005, T011, T022, T030, T034, T040, T046 each tag a checkpoint commit.
- Phase 8 records evidence files into the run folder; the run folder is the spec's "implementation complete" evidence ledger.
