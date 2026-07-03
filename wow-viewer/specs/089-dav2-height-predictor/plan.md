# Implementation Plan: DA-V2-Small LoRA Height Predictor with Cross-Tile Consistency

**Branch**: `089-dav2-height-predictor` | **Date**: 2026-07-03 | **Spec**: [`spec.md`](./spec.md)

**Input**: Feature specification at `wow-viewer/specs/089-dav2-height-predictor/spec.md`

## Summary

**V23-HeightPredictor** is one single-signal model: a DepthAnything-V2-Small backbone, LoRA-r16 fine-tuned on the encoder's Q/K/V/O projections, with a fresh DPT-style decoder head plus a 2-parameter affine anchor, predicting only `height_257` from the documented V22 Input Channel Contract (15 channels: minimap RGB, alpha, one-hot tileset ids, normals, terrain-valid mask). No WDL priors. Cross-tile grayscale drift and detail hallucination are eliminated structurally via affine-invariant loss Lssi + gradient-matching Lgm + Spatial Distance Constraint (DepthAnything-AC) + Grouped Patch Consistency Training (PRO 2025) + Bias-Free Masking + Consistency-Aware Inference (PatchFusion CVPR 2024). Target training envelope: single RunPod 24 GB GPU (RTX 4090 / A10 / A16 / L4 class). Target inference envelope: 6 GB consumer GPU at fp16, < 4 GB peak VRAM, < 3 s wall-time per tile single-pass, < 15 s with CAI-R=16. Bit-reproducible output regardless of seed: there is no noise to seed in deterministic single-forward-pass networks.

## Technical Context

**Language/Version**: Python 3.11+ (per `wow-viewer/data-harvester/pyproject.toml`)

**Primary Dependencies**:
- `torch` (>=2.1, current uv-locked)
- `transformers` (DA-V2-Small HF loader)
- `peft` (LoRA adapter injection)
- `bitsandbytes` (`PagedAdamW8bit` to fit 24 GB envelope)
- `zarr` v3 (V22 dataset reads; already in lockfile)
- `numpy`, `pillow` (already in lockfile)
- `tqdm`, `tensorboard` (already in lockfile)
- Optional: `pyyaml` (checkpoint metadata serialisation, may already be available)

**Storage**: V22 Zarr stores under `wow-viewer/output/datasets/v22/<build>.zarr/` (per Spec 088). Trained checkpoints under `wow-viewer/data-harvester/models/v23/height/runs/<run-name>/checkpoints/`. RunPod bundle tar under `wow-viewer/data-harvester/runpod/v23/dist/`.

**Testing**: pytest under `wow-viewer/data-harvester/tests/v23/`. RunPod smoke (`smoke.sh`) is the cloud-environment integration test.

**Target Platform**: Windows 11 local development, RunPod Ubuntu 22.04 (image `runpod/pytorch:2.3.1-cuda12.1-cudnn8-devel`) for production training. Inference: any 6 GB+ GPU consumer card.

**Project Type**: Python library + CLI scripts; data-harvester package surface.

**Performance Goals**: < 22 GB peak VRAM at batch=4×4 (K=4 GPCT), bf16, gradient checkpointing. < 4 GB peak VRAM at inference, fp16, batch=1. Inference wall-time per tile < 3 s single-pass.

**Constraints**: Deterministic forward+backward under `torch.manual_seed(N) + model.eval() + torch.use_deterministic_algorithms(True)`. Pipeline isolation: zero `gillijimproject_refactor` imports anywhere. All client data via `output/tmp/wowarchive-clients/`; never `H:\CLIENTS`.

**Scale/Scope**: ~1 build × ~10K tiles for smoke proof; full training corpus = 3 builds (`0_5_3_3368`, `3_3_5_12340`, `4_0_0_11927`) × ~10K tiles = ~30K tiles. 4–6 epochs to converge. Total trainable params < 8M (patch-embed + LoRA adapters + head + affine head).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | ✅ Pass | All work under `wow-viewer/data-harvester/`; no path exits `wow-viewer/` |
| II. Library-First | ✅ Pass | Model code lives in `wow-viewer/data-harvester/src/harvester/v23/`; scripts are thin wrappers |
| III. Real-Data Validation | ✅ Pass | Phase 8 is dedicated real-data proof against a staged V22 store |
| IV. Residual Model Chain | ✅ Pass | V23 predicts ONE signal (`height_257`) only. No joint heads. No shared weights with other models. V23 is the *top-of-chain* height model; downstream normal/liquid/etc. models of later specs (V24+) will consume V23's output as a frozen input — the "chain outputs become downstream inputs" rule is honored in spirit by V23's role, not by internal multi-task heads |
| V. Streaming-First Dataset Pipeline | ✅ Pass | V23 reads V22 Zarr stores directly (Zarr is the on-disk artifact; no intermediate NPZ tiles) |
| VI. No Game Client Path Assumptions | ✅ Pass | All client references via `output/tmp/wowarchive-clients/`; no `H:\CLIENTS` |
| Safety: Read-Only Reference Codebase | ✅ Pass | Zero writes to `gillijimproject_refactor` |
| Safety: Format Reader/Writer Ownership | ✅ Pass | V23 does not introduce new format readers; only consumes V22 Zarr + the DepthAnything-V2-HF model loader |
| Safety: Terrain Alpha Risk Area | ✅ Pass | No MCAL/MCLY/edge-fix changes; V23 only *reads* `alpha_256` as an input feature |
| Safety: AlphaWdtWriter Frozen | ✅ Pass | Untouched |
| Workflow: One Phase at a Time | ✅ Pass | Each phase has a validation gate; no phase begins before the prior phase passes |
| Workflow: Spec Docs are Source of Truth | ✅ Pass | `docs/architecture/v23-height-predictor-2026-07-03.md` will be created and kept in sync |
| Workflow: Training Script Changes Require Validation | ✅ Pass | Phase 8 records the validation run + results before training is "done" |
| Workflow: Memory Bank Discipline | ✅ Pass | Phase 8 final task updates `activeContext.md` + `progress.md` |
| Workflow: Bite-Sized Plans | ✅ Pass | Each phase has <= 8 tasks; one concern per task |

No constitution violations. No Complexity Tracking table required.

## Project Structure

### Documentation (this feature)

```text
specs/089-dav2-height-predictor/
├── spec.md              # feature spec (already written)
├── plan.md              # this file
├── research.md          # Phase 0 research decisions
├── data-model.md        # Phase 1 entities and state transitions
├── quickstart.md        # Phase 1 operator guide
├── contracts/           # Phase 1 contract surfaces
└── tasks.md             # concrete task breakdown (/speckit-tasks output)
```

Architecture doc lives at repository level:

```text
wow-viewer/docs/architecture/
└── v23-height-predictor-2026-07-03.md   # Phase 8 final write-up
```

### Source Code (repository root: `wow-viewer/data-harvester/`)

```text
wow-viewer/data-harvester/
├── pyproject.toml                          # Phase 0: add peft + bitsandbytes + transformers (if missing)
├── src/
│   └── harvester/
│       └── v23/                            # NEW package
│           ├── __init__.py                 # public surface
│           ├── channels.py                 # Input Channel Contract constants + builder (Phase 1)
│           ├── dataset.py                  # V23HeightDataset — V22→tensor adapter (Phase 1)
│           ├── encoder.py                  # DA-V2-Small backbone w/ LoRA + patch-embed swap (Phase 2)
│           ├── head.py                     # V23HeightHead DPT decoder + affine anchor head (Phase 3)
│           ├── model.py                    # V23-HeightPredictor = encoder + head combined (Phase 3)
│           ├── losses.py                   # Lssi, Lgm, SDC, GPCT, BiasFreeMasking (Phase 4)
│           ├── inference.py                # CAI running-mean (Phase 6)
│           └── checkpoint.py              # V23Checkpoint format + determinism metadata (Phase 5)
├── scripts/
│   ├── build_tileset_prune_table.py        # NEW: derive top-K=256 tileset prune table per build (Phase 1)
│   ├── train_v23_height.py                 # NEW: training entrypoint (Phase 5)
│   └── infer_v23_height.py                 # NEW: inference entrypoint w/ CAI (Phase 6)
├── tests/
│   └── v23/                                # NEW tests subpackage
│       ├── __init__.py
│       ├── test_channels.py                # Phase 1
│       ├── test_dataset.py                 # Phase 1
│       ├── test_encoder.py                  # Phase 2
│       ├── test_head.py                     # Phase 3
│       ├── test_model.py                    # Phase 3
│       ├── test_losses.py                   # Phase 4
│       ├── test_train_smoke.py              # Phase 5
│       ├── test_inference_determinism.py    # Phase 6
│       └── test_cai_stitch.py               # Phase 6
└── runpod/
    └── v23/                                 # NEW RunPod bundle surface
        ├── install_deps.sh                  # Phase 7
        ├── verify_bundle.sh                 # Phase 7
        ├── smoke.sh                         # Phase 7
        ├── train.sh                          # Phase 7
        └── dist/                             # tars go here (gitignored)
```

**Structure Decision**: A new `v23/` *package* (not a flat `v23_*.py` set) under `src/harvester/`. The repo convention is flat `v18_*.py`/`v19_*.py`/`v20_*.py`, but V23 has ≥7 closely-related modules (encoder/head/model/losses/dataset/inference/checkpoint). A package keeps the import surface tidy. The public API surface in `v23/__init__.py` exposes `V23HeightDataset`, `V23HeightPredictor`, `V23Checkpoint`, `run_cai_inference`, `build_channel_tensor`. Internal modules are not exported.

## Implementation Phases

### Phase 0 — Repo Setup & Dependency Verification

**Goal**: V23 package is importable; deps install; CI baseline green.

**Approach**: Add `transformers`, `peft`, `bitsandbytes` to `pyproject.toml` if not already present. Create the `src/harvester/v23/` package skeleton with empty modules and an `__init__.py` exposing the public surface (with `NotImplementedError` stubs). Add a `tests/v23/` skeleton. Run `uv sync && uv run pytest tests/v23 -q` and confirm green (0 collected, 0 errors). This phase is a no-functional-code phase; it just guarantees the wiring exists before any logic lands.

**Validation**: `uv sync` succeeds. `uv run python -c "import harvester.v23"` succeeds. All existing tests still pass. `dotnet build WowViewer.slnx` is unaffected (no C# touched).

**Max Steps**: 5.

---

### Phase 1 — V23 Dataset Adapter (User Story 1)

**Goal**: `V23HeightDataset` loads a V22 Zarr store and emits the documented Input Channel Contract + `height_257` target.

**Approach**: Implement `channels.py` first (channel index constants, normalization rules, fill policies; pure-data module, no I/O). Then implement `build_tileset_prune_table.py` to derive the top-K=256 tileset id mapping per build (read V22 `tilesets/tileset_paths` + `mcly_tileset_ids`, compute frequency, write a JSON `{id → pruned_index}` map). Then implement `V23HeightDataset` in `dataset.py` reading from a V22 Zarr store via the existing `v22_zarr_io` module; assemble the 15-channel input tensor + 1-channel target with the liquid_height override. Degrade modes controlled by `--input-mode {full,minimap_only,minimap_alpha,minimap_alpha_normal}`. Tests cover tile shape, liquid override, degrade mode zero-fill+valid_mask, and channel contract documentation.

**Validation**: `pytest tests/v23/test_dataset.py tests/v23/test_channels.py` green. Run on a real staged V22 store (Spec 088 produce flow run first if no V22 store exists in `output/datasets/v22/`) and confirm one-tile load produces documented shapes. Spec.US1 acceptance scenarios 1, 2, 3, 4 pass.

**Max Steps**: 8.

---

### Phase 2 — DA-V2-Small Backbone with LoRA + Patch-Embed Swap (User Story 2)

**Goal**: `DepthAnythingV2SmallEncoder` loads the HF DA-V2-Small weights, applies LoRA-r16 adapters to every transformer block's Q/K/V/O, replaces the first patch-embed conv for configurable `in_channels`, and exposes a "LoRA disabled" forward pass identical to stock DA-V2-Small.

**Approach**: Use `transformers.models.depth_anything_v2.DepthAnythingV2ForDepthEstimation` as the loader; pull the encoder sub-module. Apply `peft.LoraConfig(target_modules=["q_proj","k_proj","v_proj","out_proj"], r=16, lora_alpha=32, lora_dropout=0.05, bias="none")` via `peft.get_peft_model`. Replace `model.preprocessor.config.patch_size` conv layer's input channel count by constructing a fresh `nn.Conv2d` from `in_channels` to `hidden_dim`, initialised from zero-mean normal std=0.02, then wiring it in via attribute assignment. Preserve frozen-state on all original DA-V2 params. Expose `disable_lora()` context manager that zeroes adapter contribution (using `peft`'s `disable_adapter_layers`). Forward returns the multi-stage feature pyramid consumed by the DPT head (Phase 3).

**Validation**: `pytest tests/v23/test_encoder.py` green. All three US2 acceptance scenarios pass (frozen base + <2M LoRA + trainable patch-embed; forward shape matches DA-V2-Small intermediate shapes; LoRA-disabled forward is bit-identical to stock DA-V2-Small via `torch.allclose(atol=0, rtol=0)` on three random inputs).

**Max Steps**: 7.

---

### Phase 3 — V23 Height Head + Affine Anchor + Combined Model (User Story 3)

**Goal**: `V23HeightHead` consumes the encoder feature pyramid and outputs `[B, 1, 257, 257]` disparity + `[B, 2]` affine anchor. `V23-HeightPredictor` combines encoder + head. Total head params < 5M.

**Approach**: Implement a small DPT-style decoder in `head.py` — reassembly blocks + fused feature head per Ranftl et al. 2021, but with reduced channel widths fit V23's small-model budget (≤ 256 channels at finest scale instead of 256-relative-to-large-DA-V2's 768). Add a parallel small MLP head (256→1→1) on pooled/FFN features for the affine anchor (predicts scale + shift). Compose encoder + head inside `V23HeightPredictor` in `model.py`; the forward returns a `V23ModelOutput` dataclass with `.disparity`, `.affine_anchor`, `.metric_height` (= `disparity * scale + shift`, computed for loss/eval convenience but not for training target chain). All output resolutions enforced via `F.interpolate(mode="bicubic", align_corners=False)` to 257×257.

**Validation**: `pytest tests/v23/test_head.py tests/v23/test_model.py` green. US3 acceptance scenarios 1, 2, 3 pass — for shape, anchor composition formula correctness, and parameter count measured via `sum(p.numel() for p in head.parameters() if p.requires_grad) < 5_000_000`.

**Max Steps**: 6.

---

### Phase 4 — Loss Stack: Lssi, Lgm, SDC, GPCT, Bias-Free Masking (User Story 4 part 1)

**Goal**: All five loss components implemented as functions in `losses.py` that combine into the full V23 loss. Each component independently tested.

**Approach**: Implement each loss as a pure-PyTorch function callable from `train_v23_height.py`. Port Lssi and Lgm directly from the DepthAnything-V2 training repo (citation in docstring, link to source file). Implement SDC per DepthAnything-AC paper §3.2 (intra-patch pairwise distance constraint in feature space). Implement GPCT per PRO paper §3.1 — accept K sub-tile predictions + their overlap-coordinates pairs, compute L2 between feature tensors at overlap positions; default feature-level constraint is on via flag `--gpct-feature-loss`. Implement Bias-Free Masking as a per-step input transform in `losses.py` (mask K ratio of patches in the minimap channel only; replace with channel-mean RGB on the masked patch). Implement `compute_v23_loss(outputs, target, *, weights)` combiner that returns the weighted sum + a dict of component values for logging.

**Validation**: Unit tests in `tests/v23/test_losses.py` for each component's mathematical contract (gradient flows, output shape, masked-pixel behavior). Combiner test: zeroed weight for a component disables it cleanly.

**Max Steps**: 8.

---

### Phase 5 — Training Script `train_v23_height.py` (User Story 4 part 2)

**Goal**: A working trainer that runs all US4 acceptance scenarios (gpct-weight 0 baseline, GPCT K=4 with `--gpct-weight 0.1`, `--bias-free-mask-ratio 0.15`, workspace isolation, checkpoint metadata).

**Approach**: Implement `train_v23_height.py` with argparse including all FR-009, FR-010, FR-011, FR-012, FR-014 flags plus dataset selection (`--dataset-dir`, `--builds`, `--input-mode`, `--tileset-prune-table`), training loop control (`--epochs`, `--lr`, `--grad-accum-steps`), validation control (`--val-max-tiles`, `--val-interval`, `--val-preview-interval`), and run organisation. Optimizer = `bitsandbytes.optim.PagedAdamW8bit`. Mixed precision via `torch.cuda.amp.autocast(dtype=torch.bfloat16)` + `GradScaler` (no-op for bf16). Gradient checkpointing on the encoder via `model.encoder.gradient_checkpointing_enable()`. Deterministic mode: if `--deterministic`, set `torch.manual_seed(N)`, `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`. Checkpoint format in `checkpoint.py` writes config-dict + state-dict + commit SHA (`git rev-parse HEAD`) + V22 store path hash. Validation loop saves a per-tile preview PNG (input minimap + ground-truth height + predicted disparity + predicted metric height + per-tile abs error) to `models/v23/height/runs/<run>/val_preview_<epoch>/tile_<N>.png`. Implement OOM-catcher that halves effective batch size and retries once; persistent OOM is fatal with clear log message.

**Validation**: `pytest tests/v23/test_train_smoke.py` runs a 2-epoch synthetic-data or micro-V22-subset run and confirms checkpoint + preview exist. US4 acceptance scenarios 1, 2, 3, 4 pass.

**Max Steps**: 9.

---

### Phase 6 — Deterministic Inference + CAI Stitching (User Story 5)

**Goal**: `infer_v23_height.py` produces byte-reproducible output across seeds; CAI-R=16 produces no visible seam on a 3×3 tile grid.

**Approach**: Implement `inference.py` with `run_cai_inference(model, store, cai_r=16)` that loads R overlapping sub-tile shifts (offset by stride `(tile_w - overlap)//(R-1)` along each axis), runs the model on each shift, and averages predictions on covered pixels via a running-mean accumulator. When `cai_r=1`, falls back to single-pass tile inference. Implement `infer_v23_height.py` CLI with flags `--checkpoint`, `--v22-store`, `--build`, `--tiles <list>`, `--output-dir`, `--cai-r`, `--seed`, `--deterministic`, `--save-preview`. Implement determinism test script `tests/v23/test_inference_determinism.py` that runs the same input twice with different seeds under `--deterministic` and asserts `torch.allclose(atol=0, rtol=0)`. Implement CAI seam test `tests/v23/test_cai_stitch.py` that loads a 3×3 grid from a real (or synthetic-positioned) store, runs CAI-R=16 vs R=1, and asserts the cross-tile L1 along boundaries drops by >= 50% with CAI.

**Validation**: US5 acceptance scenarios 1, 2, 3 pass. Manual visual review of saved preview PNG.

**Max Steps**: 7.

---

### Phase 7 — RunPod Bundle (User Story 6)

**Goal**: A Spec-079-compliant tar that ships V23 + a V22 Zarr subset to a 24 GB RunPod Pod, with one-command bootstrap.

**Approach**: Write `runpod/v23/install_deps.sh` (set -euo pipefail; `uv sync` against packaged `pyproject.toml` + `install --no-deps transformers peft bitsandbytes zarr`), `verify_bundle.sh` (md5 list check; `python -c "from harvester.v23 import V23HeightPredictor"` import smoke; `bash smoke.sh --dry-run` to confirm wiring), `smoke.sh` (`uv run python scripts/train_v23_height.py --epochs 2 --train-max-tiles 4 --val-max-tiles 2 --device cuda --target-vram-gb 22 --batch-size 4 --gpct-K 4 --gpct-weight 0.1 --bias-free-mask-ratio 0.15` — bounded proof), `train.sh` (full training command as a wrapper around `train_v23_height.py`). Write a packager script `wow-viewer/data-harvester/scripts/package_v23_runpod.py` that builds the tar: includes `src/harvester/v23/`, `scripts/train_v23_height.py`, `scripts/infer_v23_height.py`, `pyproject.toml` (trimmed), `runpod/v23/*.sh`, a V22 Zarr subset (configurable subset size), and `manifest.json` with `contains_game_client_files: false` + tree hash audit. Validate the bundle's `verify_bundle.sh` on a fresh 24 GB Pod (RTX 4090 preferred). If RunPod API integration is needed, follow Spec 079's `--manual-pod` fallback path; do not re-derive RunPod integration.

**Validation**: Bundle exists. `verify_bundle.sh` reports OK on Pod. `smoke.sh` runs to completion without CUDA OOM at the documented config. US6 acceptance scenarios 1, 2, 3 pass.

**Max Steps**: 6.

---

### Phase 8 — Real-Data Validation against a Staged V22 Store

**Goal**: End-to-end real-data training proof against a populated V22 Zarr store, satisfying SC-001, SC-003, SC-004, SC-005, SC-006, SC-007 simultaneously.

**Approach**: Pre-flight: confirm a V22 Zarr store exists under `output/datasets/v22/3_3_5_12340.zarr`; if absent, run Spec-088 production flow (`build_v18_dataset.py build` then `build_v22_dataset.py enrich` then `build_v22_dataset.py build`). Run V23 training on RunPod for 4 epochs over the full 3-build corpus via `runpod/v23/train.sh`, capturing peak VRAM via `nvidia-smi` polling, all loss-component metrics via the trainer's logged JSON, and per-epoch validation L1. Save baseline V21 per-tile L1 metric into the run folder for direct comparison. After training, export validation previews via `infer_v23_height.py --tiles <3x3 validation grid> --cai-r 16`. Run determinism proof: `infer_v23_height.py --seed 42 ...` and `infer_v23_height.py --seed 12345 ...`; assert bit-identical. Update `memory-bank/activeContext.md` and `memory-bank/progress.md` with V23 status + research notes. Write `docs/architecture/v23-height-predictor-2026-07-03.md` summarising the stack (architecture, loss components, RunPod envelope, determinism contract, validation results, downstream integration plan for V24+ normal/liquid models).

**Validation**: SC-001 pass (no VM OOM at 24 GB), SC-003 pass (per-tile L1 drops >= 25% vs V21 baseline, recorded), SC-004 pass (CAI-R=16 preview has no visible seam — manual review noted in the architecture doc), SC-005 pass (cross-tile L1 with CAI < 50% of without-CAI, metric saved as JSON), SC-006 pass (two Pod runs at same config + commit produce bit-identical final weights via determinism test), SC-007 pass (manifest audit clean). Spec is "implementation complete" only when all five SC items are evidenced.

**Max Steps**: 8.

---

## Complexity Tracking

> **Not needed** — Constitution Check passes on every principle. No violations to justify.

## Notes on Phase Boundaries

- **Phase 1 → Phase 2 dependency**: Phase 2 needs the channel count from Phase 1's `channels.py`. Implement Phase 1 first.
- **Phase 2 → Phase 3 dependency**: Phase 3 reads Phase 2's encoder output shapes. Implement Phase 2 first.
- **Phase 3 → Phase 4 dependency**: Phase 4's loss components expect `(B, 1, 257, 257)` disparity output shape from Phase 3. Implement Phase 3 first.
- **Phase 4 → Phase 5 dependency**: Phase 5 imports Phase 4's `compute_v23_loss`. Implement Phase 4 first.
- **Phase 5 → Phase 6 dependency**: Phase 6 loads checkpoints produced by Phase 5. Implement Phase 5 first.
- **Phase 6 → Phase 7 dependency**: Phase 7's `smoke.sh` and `train.sh` call scripts produced in Phase 6 with a fallback for install-time. Implement Phase 6 first.
- **Phase 7 → Phase 8 dependency**: Phase 8 uses the bundle from Phase 7 to launch the RunPod training run. Implement Phase 7 first.

No phase can ship before its validation gate passes. Spec's "One Phase At A Time" rule (AGENTS.md RULE 8) is enforced: validation evidence is recorded in the per-phase commit message before the next phase begins.

## Determinism Strategy (Cross-Phase Reference)

Three layers of determinism, applied at three different layers:

1. **Per-call determinism** (Phase 5 training, Phase 6 inference): `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. Plus `model.eval()` + `torch.no_grad()` at inference. This is the seed-stable path the user demanded.
2. **Per-architecture determinism** (Phase 4 losses): affine-invariant Lssi removes *global-scale hallucination by math*. Spatial Distance Constraint removes *texture-template hallucination by patch-level geometry*. Bias-Free Masking prevents *dataset-bias overfit by input dropout*. None of these depend on the seed.
3. **Per-inference-stack determinism** (Phase 6 CAI): a running mean over R=16 overlapping tile shifts averages out residual disagreement. Combined with seed-stable per-call determinism and seed-independent loss training, this gives the cross-tile-consistent, byte-reproducible output the user demanded.

These three layers correspond directly to the user's three pain points: "same seed → different colored outputs" (Layer 1), "hallucinated details" (Layer 2), "cross-tile grayscale drift" (Layer 3).

## Open Question — Tracking top-K Tileset Identity Across Builds

Spec FR **FR-007** says the documented 15-channel input contract uses one-hot top-K=256 tileset ids. Two builds have disjoint tileset id spaces (`tileset_paths` per-build). Training across multiple builds requires either (a) per-build prune tables with per-build head-side calibration (more code, simpler math), or (b) union-top-K across all V22 builds used for training (simpler code, requires a global prune table). Option (b) is recommended and assumed by the plan; the per-build prune table (`build_tileset_prune_table.py`) emits a unified table at `output/datasets/v22/<run>/tileset_prune_<run>.json` covering all `--builds` flag values. If the user prefers per-build isolation, Phase 1 task "build_tileset_prune_table.py" changes scope — flag this in Phase 1 review.

## Relationship to Existing Code

| Existing Surface | Reused / Untouched | Notes |
|---|---|---|
| `src/harvester/v22_zarr_io.py` | Reused | Read V22 stores |
| `src/harvester/v22_patched_signals.py` | Reused | Liquid/normal-masking derivation |
| `scripts/build_v22_dataset.py` | Untouched | Phase 8 may run it to produce a fresh store |
| `scripts/train_v18.py` framework | Pattern reference only | Different task; not forked |
| `src/harvester/v19_losses.py` | Referenced only if V19 had affine-invariant precedent | Confirmed it does not — port from DA-V2 directly |
| `runpod/` directory (Spec 079 shell) | Pattern reference only | Spec 079's bundle template is a per-project copy |
| `tests/` existing pytest harness | Reused | Existing conftest patterns followed |

No writes to `gillijimproject_refactor`. No writes outside `wow-viewer/data-harvester/` (except the spec/architecture files under `wow-viewer/specs/` and `wow-viewer/docs/architecture/`). No C# touched.
