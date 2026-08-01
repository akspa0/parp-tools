# Implementation Plan: 094-wdl-prior-v24

**Feature Branch**: `094-wdl-prior-v24`
**Created**: 2026-07-06
**Owner**: wow-viewer
**Source**: [`./spec.md`](./spec.md), [`./research.md`](./research.md), [`./data-model.md`](./data-model.md), [`./checklists/requirements.md`](./checklists/requirements.md)

> **Amended 2026-07-06**: see `spec.md` "Implementation Amendments" — verified C# reader shape (17×17 outer + 16×16 inner int16, MAHO not read), WDLs resolved from MPQs via `NativeMpqService` (batch-first shim modes), V18 actual schema (`minimap_rgb` 256², `object_precise_mask` float32, `liquid_mask` name, `holes_16` available, `no_object_minimap` on 0_5_3), paired-array V24 store schema (A5), exact quincunx 33→257 upsample (A6), Stage A interpolation heads (A7), and the added C#-grounded V22 dataset audit lane (A8). The phases below are executed with those amendments in force.

## Technical Context

**Goal**: Build a v7-style two-stage height predictor (low-res WDL prior + small lattice detailer) on V18 substrate, using real WDL coverage where it exists and synthetic WDL coverage everywhere else.

**Tech stack**:
- Python 3.10+ under `wow-viewer/data-harvester/`, `uv`-managed per RULE 5.
- PyTorch 2.x for both models (no DA-V2, no DPT, no LoRA, no GPCT, no CAI).
- Zarr v2 for dataset storage.
- NumPy for the minimap cleaner and the synthetic-WDL shim wrapper.
- A small C# CLI shim (`WowViewer.Tool.WdlRead`) that wraps the existing C# WDL reader and the existing C# terrain→WDL path in `WowViewer.Core.IO`. C# is not modified.

**Hardware envelope**: 6 GB consumer GPU (e.g. GTX 1660 Super, RTX 3050). fp16 mixed precision. No RunPod, no Pod packaging.

**Total model size**: ≤ 3M trainable params (Stage A ≤ 1M + Stage B ≤ 2M). Inference VRAM < 4 GB. Inference wall-time < 3 s/tile.

**Substrate**: V18 Zarr store (Spec 001) at `wow-viewer/output/datasets/v18/<build>.zarr/`. V18 arrays are read but not modified.

**Real WDL source**: Staged-client `.wdl` files at `output/tmp/wowarchive-clients/<build>/World of Warcraft/<map>.wdl`. Read via the C# shim. C# is not modified.

## Architecture Constraints

- **RULE 1**: `gillijimproject_refactor` is read-only. V24 does not touch it.
- **RULE 2**: All new code in `wow-viewer`. V24 lives under `wow-viewer/data-harvester/`.
- **RULE 3**: No rewrite of game client reading tooling. V24 wraps the existing C# WDL reader and the existing C# terrain→WDL path. No new WDL parser in Python.
- **RULE 4**: `wow-viewer` is repo-independent. The C# shim is in-repo (a new project under `wow-viewer/tools/`).
- **RULE 5**: One Python environment. All V24 Python under `wow-viewer/data-harvester/`.
- **RULE 6**: Each phase is a separate, testable change. Phase 0/1 changes don't bundle with Phase 2 changes.
- **RULE 7**: Small, modular, residual-predicting models. Stage A predicts WDL prior directly. Stage B predicts a residual.
- **RULE 9**: No `H:\CLIENTS`. Real WDLs are read from `output/tmp/wowarchive-clients/`.
- **RULE 8**: One phase at a time. Each phase ends with a validation gate before the next phase starts.
- **RULE 11**: Doc hygiene. Memory bank updated at session end.

## Phased Implementation

The spec is broken into 7 phases, each with a concrete validation gate. A phase is not done when its code is written; it is done when its gate passes.

### Phase 0: C# WDL Reader Audit (Foundation)

**Goal**: Confirm the C# WDL reader's actual output shape and document it.

**Tasks**:
1. Locate the existing C# WDL reader in `WowViewer.Core.IO`. Confirm it can be invoked from .NET code (already known; this is a sanity check).
2. Write a one-off audit script that reads 5 MARE entries from a real staged-client `.wdl` (e.g. Azeroth, one per quadrant + center) for each target build (`0_5_3_3368`, `3_3_5_12340`).
3. Record the per-MARE output shape and dtype for each build. Confirm the MARE outer+inner layout matches the wowdev wiki (17×17+16×16 on 3.3.5a) or document the difference.
4. Document the audit at `wow-viewer/docs/architecture/wdl-reader-shape-audit-2026-07-XX.md`.

**Files touched**:
- `wow-viewer/docs/architecture/wdl-reader-shape-audit-2026-07-XX.md` (new)

**Validation gate**:
- The audit document exists and records the per-MARE output shape for both target builds.
- The shape is recorded in the spec's `data-model.md` (E7) as the authoritative shape for V24.

**Out of scope**: Implementing the C# reader. The C# reader is the source of truth; we audit it, we don't modify it.

### Phase 1: C# WDL Reader CLI Shim (Foundation)

**Goal**: Build a tiny CLI shim that lets Python call the existing C# WDL reader and the existing C# terrain→WDL path.

**Tasks**:
1. Create `wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead/` (a new .NET project).
2. Implement the shim: takes `--wdl <path> --tile-x <int> --tile-y <int> --output <npz>` and emits the MARE grid as NPZ. Re-uses the existing C# WDL reader in `WowViewer.Core.IO`.
3. Extend the shim to support the synthetic-WDL mode: `--height-257 <npz> --liquid-mask <npz> --output <npz>`. Re-uses the existing C# terrain→WDL path.
4. Add a `--help` flag. Add a `--version` flag.
5. Add the shim to `WowViewer.slnx`.
6. Verify `dotnet build` succeeds.
7. Verify the shim runs on one real staged-client `.wdl` file and emits a non-zero NPZ with the expected shape (per the Phase 0 audit).
8. Verify the shim runs on one synthetic input and emits a non-zero NPZ with the expected shape.

**Files touched**:
- `wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead/WowViewer.Tool.WdlRead.csproj` (new)
- `wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead/Program.cs` (new)
- `wow-viewer/WowViewer.slnx` (modified: add the new project)

**Validation gate**:
- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeds with 0 errors.
- `dotnet run --project wow-viewer/tools/wdl-read -- --help` prints usage.
- `dotnet run --project wow-viewer/tools/wdl-read -- --wdl <real-path> --tile-x 32 --tile-y 32 --output /tmp/mare.npz` produces a non-zero NPZ with the C# reader's actual shape.
- `dotnet run --project wow-viewer/tools/wdl-read -- --height-257 /tmp/height.npz --output /tmp/synth.npz` produces a non-zero NPZ.

**Out of scope**: Modifying the C# WDL reader. Modifying the C# terrain→WDL path. The shim is a thin wrapper, not a re-implementation.

### Phase 2: Synthetic WDL Builder (Python Wrapper)

**Goal**: Wrap the C# terrain→WDL shim in a Python module that V24 can call.

**Tasks**:
1. Create `wow-viewer/data-harvester/src/harvester/v24/synth_wdl.py` with `def build_synth_wdl(height_257, liquid_mask_256=None) -> np.ndarray`.
2. The function writes `height_257` and `liquid_mask_256` to a temp NPZ, invokes the C# shim via subprocess, reads the output NPZ, and returns the array.
3. Add `wow-viewer/data-harvester/src/harvester/v24/__init__.py` (empty, marks the package).
4. Add a `pyproject.toml` check: `import harvester.v24.synth_wdl` works.
5. Add a `pytest` test in `tests/v24/test_synth_wdl.py` that calls `build_synth_wdl` on a synthetic 257×257 height array and asserts the output shape matches the C# reader's shape (per the Phase 0 audit).
6. Verify `uv run python -m pytest tests/v24/test_synth_wdl.py -q` passes.

**Files touched**:
- `wow-viewer/data-harvester/src/harvester/v24/__init__.py` (new)
- `wow-viewer/data-harvester/src/harvester/v24/synth_wdl.py` (new)
- `wow-viewer/data-harvester/tests/v24/__init__.py` (new)
- `wow-viewer/data-harvester/tests/v24/test_synth_wdl.py` (new)

**Validation gate**:
- `uv run python -m pytest tests/v24/test_synth_wdl.py -q` passes.
- `build_synth_wdl(height_257=...)` on a synthetic 257×257 array returns an array with the C# reader's shape (per the audit).

**Out of scope**: Implementing a new WDL synthetic algorithm. The function is a thin wrapper around the existing C# path.

### Phase 3: Real WDL Reader (Python Wrapper)

**Goal**: Wrap the C# WDL reader shim in a Python module that V24 can call.

**Tasks**:
1. Create `wow-viewer/data-harvester/src/harvester/v24/wdl_reader.py` with `def read_wdl_mare(wdl_path, tile_x, tile_y) -> tuple[np.ndarray, np.ndarray | None] | None`.
2. The function invokes the C# shim via subprocess, reads the output NPZ, and returns the (MARE grid, MAHO bitmask or None) tuple.
3. Add a `pytest` test in `tests/v24/test_wdl_reader.py` that calls `read_wdl_mare` on a real staged-client `.wdl` file (requires the staged client to be staged; use the test data path conventions from `data-paths.md`) and asserts the output shape matches the C# reader's shape.
4. Verify `uv run python -m pytest tests/v24/test_wdl_reader.py -q` passes.

**Files touched**:
- `wow-viewer/data-harvester/src/harvester/v24/wdl_reader.py` (new)
- `wow-viewer/data-harvester/tests/v24/test_wdl_reader.py` (new)

**Validation gate**:
- `uv run python -m pytest tests/v24/test_wdl_reader.py -q` passes on a real staged-client `.wdl` file.
- `read_wdl_mare` returns the C# reader's actual shape.

**Out of scope**: Re-implementing the WDL reader. The function is a thin wrapper.

### Phase 4: Merged WDL Prior Builder (Stage 0)

**Goal**: Build the merged-coverage WDL prior for every V18 tile. Emit the V24 Zarr store.

**Tasks**:
1. Create `wow-viewer/data-harvester/src/harvester/v24/merged_wdl_prior.py` with `def build_merged_wdl_prior(...)` per FR-005 in the spec.
2. Create `wow-viewer/data-harvester/scripts/build_wdl_prior.py` with `build` and `infer` subcommands per FR-006 in the spec.
3. Add a `pytest` test in `tests/v24/test_build_wdl_prior.py` that runs `build_wdl_prior.py build` on a 5-tile V18 subset (uses a synthetic V18 store with 5 tiles; no real staged client required for the test) and asserts the V24 store has `wdl_prior`, `wdl_prior_source`, `wdl_prior_confidence` arrays populated with the correct shapes and value ranges.
4. Add a `pytest` test in `tests/v24/test_merged_wdl_prior.py` that unit-tests the merge rules (real-agreeing, real-disagreeing, missing-real, audit-empty cases) on synthetic inputs.
5. Verify `uv run python -m pytest tests/v24/ -q` passes (all v24 tests so far).
6. Run a bounded real-data build: `uv run python scripts/build_wdl_prior.py build --v18-store <real V18 path> --staged-client <real staged-client path> --output ../output/datasets/v24/3_3_5_12340_smoke.zarr --limit 5` and verify the V24 store is created and has the right shape and coverage stats.

**Files touched**:
- `wow-viewer/data-harvester/src/harvester/v24/merged_wdl_prior.py` (new)
- `wow-viewer/data-harvester/scripts/build_wdl_prior.py` (new)
- `wow-viewer/data-harvester/tests/v24/test_merged_wdl_prior.py` (new)
- `wow-viewer/data-harvester/tests/v24/test_build_wdl_prior.py` (new)
- `wow-viewer/output/datasets/v24/3_3_5_12340_smoke.zarr/` (build output, gitignored)

**Validation gate**:
- `uv run python -m pytest tests/v24/ -q` passes.
- The bounded 5-tile real-data build produces a V24 store with `wdl_prior.shape` matching the C# reader's shape (per the Phase 0 audit), `wdl_prior_source` in {0, 1, 2}, `wdl_prior_confidence` in [0, 1], and combined real+synthetic coverage ≥ 95%.
- `inspect_v24_dataset.py summary --store <v24.zarr>` reports the coverage stats correctly.

**Out of scope**: Stage A. Stage B. The minimap cleaner. Those are separate phases.

### Phase 5: Minimap Cleaner (User Story 2)

**Goal**: Build the pure-NumPy minimap cleaner that uses V18's tile-level `object_precise_mask` to remove object roofs.

**Tasks**:
1. Create `wow-viewer/data-harvester/src/harvester/v24/clean_minimap.py` with `def clean_minimap(minimap_rgb, object_precise_mask) -> np.ndarray` per FR-010 in the spec.
2. Create `wow-viewer/data-harvester/scripts/clean_minimap.py` with the CLI per FR-009 in the spec.
3. Add a `pytest` test in `tests/v24/test_clean_minimap.py` that unit-tests the cleaning on synthetic inputs (all-object, no-object, partial-object, audit-empty cases).
4. Verify `uv run python -m pytest tests/v24/test_clean_minimap.py -q` passes.

**Files touched**:
- `wow-viewer/data-harvester/src/harvester/v24/clean_minimap.py` (new)
- `wow-viewer/data-harvester/scripts/clean_minimap.py` (new)
- `wow-viewer/data-harvester/tests/v24/test_clean_minimap.py` (new)

**Validation gate**:
- `uv run python -m pytest tests/v24/test_clean_minimap.py -q` passes.
- `clean_minimap.py --help` prints usage.
- A bounded run on 5 V18 tiles produces 5 cleaned minimap NPZs that show object pixels replaced.

**Out of scope**: A model-based cleaner. The cleaner is NumPy-only per the spec.

### Phase 6: Stage A (Minimap → WDL Prior Correlation)

**Goal**: Train a small U-Net (≤ 1M params) that maps cleaned minimap + alpha + normal + mcnr + synthetic WDL to the WDL prior.

**Tasks**:
1. Create `wow-viewer/data-harvester/src/harvester/v24/stage_a.py` with the Stage A model and dataset adapter. Model is a small U-Net with ≤ 1M trainable params. Inputs: `[cleaned_minimap (down-sampled), alpha (down-sampled), normal (down-sampled), mcnr (down-sampled), synthetic_wdl (down-sampled)]`. Output: WDL prior (same shape as the C# reader's per-MARE output).
2. Create `wow-viewer/data-harvester/scripts/train_v24_stage_a.py` with the training script per FR-012 in the spec. Loss: L1 with `wdl_prior_confidence` as sample weight and `wdl_prior_source != 2` as sample selection. Optimizer: AdamW, lr=1e-3, cosine annealing, fp16 autocast, gradient checkpointing on encoder.
3. Create `wow-viewer/data-harvester/scripts/infer_v24_stage_a.py` with the inference script per FR-013 in the spec.
4. Add a `pytest` test in `tests/v24/test_stage_a.py` that builds a synthetic Stage A model, runs a forward pass on a synthetic input, and asserts the output shape matches the C# reader's shape. Also asserts the param count is ≤ 1M.
5. Add a `pytest` test that runs `train_v24_stage_a.py` for 2 epochs on a 5-tile V24 subset and asserts the loss decreases.
6. Verify `uv run python -m pytest tests/v24/test_stage_a.py -q` passes.
7. Run a bounded real-data training: `uv run python scripts/train_v24_stage_a.py --v24-store <real V24 path> --output <run_id> --epochs 10 --limit 50` and verify the loss decreases.

**Files touched**:
- `wow-viewer/data-harvester/src/harvester/v24/stage_a.py` (new)
- `wow-viewer/data-harvester/scripts/train_v24_stage_a.py` (new)
- `wow-viewer/data-harvester/scripts/infer_v24_stage_a.py` (new)
- `wow-viewer/data-harvester/tests/v24/test_stage_a.py` (new)
- `wow-viewer/output/v24_validation/<run_id>/stage_a.pt` (training output, gitignored)
- `wow-viewer/output/v24_validation/<run_id>/loss_history.jsonl` (training output, gitignored)

**Validation gate**:
- `uv run python -m pytest tests/v24/test_stage_a.py -q` passes.
- Stage A model has ≤ 1M trainable params (asserted in test).
- 2-epoch training on a 5-tile V24 subset shows loss decreasing.
- 10-epoch training on a 50-tile V24 real-data subset shows loss decreasing.
- `infer_v24_stage_a.py` runs on a single V24 tile in < 1 second on a 6 GB consumer GPU.

**Out of scope**: Stage B. Validation report. Comparison.

### Phase 7: Stage B (Lattice Detailer)

**Goal**: Train a small conv-deconv (≤ 2M params) that maps upsampled prior + cleaned minimap + V18 channels to a 257×257 residual.

**Tasks**:
1. Create `wow-viewer/data-harvester/src/harvester/v24/stage_b.py` with the Stage B model and dataset adapter. Model is a small conv-deconv with ≤ 2M trainable params. Inputs: `[bilinear_upsample(stage_a_prior, 257), cleaned_minimap, alpha, normal, mcnr, object_mask]`. Output: 257×257 residual.
2. Create `wow-viewer/data-harvester/scripts/train_v24_stage_b.py` with the training script per FR-017 in the spec. Loss: L1 gated to non-liquid, non-object, non-MAHO-hole pixels. Optimizer: AdamW, lr=1e-3, cosine annealing, fp16 autocast, gradient checkpointing.
3. Create `wow-viewer/data-harvester/scripts/infer_v24_stage_b.py` with the full pipeline (Stage A + Stage B) per FR-018 in the spec.
4. Add a `pytest` test in `tests/v24/test_stage_b.py` that builds a synthetic Stage B model, runs a forward pass on a synthetic input, and asserts the output shape is (B, 1, 257, 257). Also asserts the param count is ≤ 2M.
5. Add a `pytest` test that runs `train_v24_stage_b.py` for 2 epochs on a 5-tile V24 subset and asserts the loss decreases.
6. Add a determinism test: two `infer_v24_stage_b.py` runs with different seeds produce bit-identical output.
7. Verify `uv run python -m pytest tests/v24/test_stage_b.py -q` passes.
8. Run a bounded real-data training: `uv run python scripts/train_v24_stage_b.py --v24-store <real V24 path> --stage-a-checkpoint <path> --output <run_id> --epochs 10 --limit 50` and verify the loss decreases.

**Files touched**:
- `wow-viewer/data-harvester/src/harvester/v24/stage_b.py` (new)
- `wow-viewer/data-harvester/scripts/train_v24_stage_b.py` (new)
- `wow-viewer/data-harvester/scripts/infer_v24_stage_b.py` (new)
- `wow-viewer/data-harvester/tests/v24/test_stage_b.py` (new)
- `wow-viewer/output/v24_validation/<run_id>/stage_b.pt` (training output, gitignored)

**Validation gate**:
- `uv run python -m pytest tests/v24/test_stage_b.py -q` passes.
- Stage B model has ≤ 2M trainable params (asserted in test).
- 2-epoch training on a 5-tile V24 subset shows loss decreasing.
- 10-epoch training on a 50-tile V24 real-data subset shows loss decreasing.
- Full pipeline (Stage A + Stage B) inference on 1 tile in < 3 seconds on a 6 GB consumer GPU.
- Two `infer_v24_stage_b.py` runs with different seeds produce bit-identical output.

**Out of scope**: Validation report. Comparison. Memory bank updates.

### Phase 8: Validation Report + Final Artifacts

**Goal**: Produce the final validation report, update the memory bank, and write the summary doc.

**Tasks**:
1. Create `wow-viewer/data-harvester/scripts/inspect_v24_dataset.py` with `summary` and `tile` subcommands.
2. Create `wow-viewer/data-harvester/scripts/validate_v24.py` that runs the validation report per FR-025 in the spec.
3. Run the full pipeline on a 50-tile V24 real-data validation set: build the V24 store, train Stage A, train Stage B, run the validation report.
4. Verify SC-001 through SC-008 pass per the spec.
5. Update `wow-viewer/memory-bank/activeContext.md` to add Spec 094 to the "Recent background still live" section.
6. Update `wow-viewer/memory-bank/progress.md` with a 2026-07-06 entry summarizing the spec.
7. Write `wow-viewer/docs/architecture/v24-validation-2026-07-XX.md` with the validation report summary.

**Files touched**:
- `wow-viewer/data-harvester/scripts/inspect_v24_dataset.py` (new)
- `wow-viewer/data-harvester/scripts/validate_v24.py` (new)
- `wow-viewer/memory-bank/activeContext.md` (modified: add Spec 094)
- `wow-viewer/memory-bank/progress.md` (modified: add 2026-07-06 entry)
- `wow-viewer/docs/architecture/v24-validation-2026-07-XX.md` (new)

**Validation gate**:
- `validate_v24.py` runs on a 50-tile V24 real-data validation set and emits a JSON report.
- SC-001 through SC-008 all pass per the report.
- The memory bank is updated.
- The summary doc exists at `wow-viewer/docs/architecture/v24-validation-2026-07-XX.md`.

**Out of scope**: Replacing V23. Promoting V24 to production. V24 is a research slice; the spec is bounded.

## Phase Dependencies

```
Phase 0 (audit) → Phase 1 (C# shim) → Phase 2 (synth WDL wrapper)
                                     → Phase 3 (real WDL wrapper)
                                     → Phase 4 (merged prior + V24 store)
                                                       ↓
                                          Phase 5 (minimap cleaner) → Phase 6 (Stage A) → Phase 7 (Stage B) → Phase 8 (validation)
```

Phases 2 and 3 depend on Phase 1. Phase 4 depends on Phases 2 and 3. Phase 5 is independent. Phase 6 depends on Phases 4 and 5. Phase 7 depends on Phase 6. Phase 8 depends on Phase 7.

## Open Items for `speckit-tasks` (Next Skill)

The next skill (`speckit-tasks`) will break each phase into bite-sized tasks. Each task should be:
- A single concern (per RULE 11: "one concern per step, independently validatable, max 10 steps per phase").
- Independently testable (a single pytest or a single CLI run).
- Documented in `tasks.md` with a clear pass/fail criterion.

## Notes

- The spec intentionally does not use V23's DA-V2 / LoRA / GPCT / CAI infrastructure. V24 is small and runs locally.
- The C# shim is the only C# code added. It is a thin wrapper around existing C# paths in `WowViewer.Core.IO`.
- V24 consumes V18 directly. V22 is out of scope.
- The `block_reduce(height_257)` baseline is the trivial "no learning" answer. Stage A and Stage B both need to beat it to pass the success criteria.
- The user's "no over-engineering" directive is enforced by the spec's hard caps (≤ 3M params, no RunPod, no DA-V2, L1 loss only) and by the small, sequential phase structure.
