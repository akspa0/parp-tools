# Research: 094-wdl-prior-v24

**Purpose**: Resolve the open questions in [`./spec.md`](./spec.md) and document the technical decisions that need to be made before implementation.
**Created**: 2026-07-06

## Q1: C# Python shim form

**Question**: How does V24's Python code call the existing C# WDL reader in `WowViewer.Core.IO` and the existing C# terrain→WDL path used by `WoWViewer`'s "click on map to spawn" visualization?

**Decision**: Build a tiny CLI shim under `wow-viewer/tools/wdl-read/` (or a similar location) that takes a `.wdl` path + tile_x + tile_y as args and emits the MARE grid + MAHO bitmask as NPZ. The shim is built once and is the canonical Python-callable entry point to both C# paths (WDL reader and terrain→WDL path). V24's Python code calls the shim via subprocess.

**Rationale**:
- The C# side is not modified (per RULE 1 and the user's hard rule "C# whenever possible for I/O").
- A CLI shim is the lowest-friction bridge: it works on any Python + .NET install, requires no in-process bridge configuration, and is easy to test in isolation.
- The shim can be reused by V24, future specs, and tooling. It's a one-time cost.
- Alternative considered: pythonnet in-process bridge. Rejected because pythonnet adds a non-trivial Python dependency and complicates the venv (per RULE 5, the data-harvester env is `uv`-managed and adding pythonnet is a meaningful change).
- Alternative considered: HTTP service. Rejected because it adds a runtime process to manage and is overkill for batch dataset building.

**Decision details**:
- Shim name: `WowViewer.Tool.WdlRead` (under `wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead/`).
- CLI signature: `dotnet run --project wow-viewer/tools/wdl-read -- --wdl <path> --tile-x <int> --tile-y <int> --output <npz-path>`.
- Output: NPZ with keys `outer`, `inner` (whatever the C# reader actually returns per MARE — the shape is the C# reader's output, period), and `holes` (optional, present only if the C# reader exposes MAHO).
- The same shim is used for the terrain→WDL path: `dotnet run --project wow-viewer/tools/wdl-read -- --height-257 <npz> --liquid-mask <npz> --output <npz-path>`. The shim detects which operation to perform based on the input flags.
- The shim re-uses the existing C# WDL reader code in `WowViewer.Core.IO` and the existing C# terrain→WDL path. It is a thin wrapper, not a re-implementation.

**Status**: Recommended default. Confirmed by user.

## Q2: Minimap cleaning quality

**Question**: Is the NumPy-based 8-connected median cleaner (using V18's tile-level `object_precise_mask`) sufficient for V24, or should V24 train a model-based cleaner?

**Decision**: NumPy-based cleaner for this spec. Model-based cleaner is a future lane.

**Rationale**:
- V24 is research-shaped. The cleaner is a means to an end (a clean minimap input to Stage A), not the focus of the spec.
- The V18 `object_precise_mask` is tile-level, not per-object. A NumPy-based 8-connected median fill is the right complexity match for a tile-level mask.
- A model-based cleaner would require a separate spec (training data: V18 minimap + tile-level mask → cleaned minimap; model: small U-Net). This is out of scope for V24.
- V22 per-object mask data is suspect (per the user). A model-based cleaner that depended on V22 per-object data would inherit the suspect data.

**Decision details**:
- The cleaner is a pure NumPy function in `wow-viewer/data-harvester/src/harvester/v24/clean_minimap.py`.
- Algorithm: for each object pixel, replace with the median of the surrounding non-object 8-connected neighbourhood. If no non-object neighbour exists, replace with the global mean colour.
- Loss gate: `object_precise_mask` is used to skip object pixels in Stage A and Stage B's loss.
- Future lane: a model-based cleaner trained on V18's tile-level mask is a separate spec, blocked on V22 per-object data verification.

**Status**: Recommended default. Confirmed by user.

## Q3: WDL grid shape

**Question**: What is the WDL grid shape the C# WDL reader actually returns per MARE? The wowdev wiki says 17×17 + 16×16 = 545 per MARE for 3.3.5a, but the user has flagged that the wiki may not be 100% accurate and that the C# reader (built from decompiled clients) is the de-facto ground truth.

**Decision**: The spec accommodates whatever the C# reader returns. The V24 Zarr schema uses generic array names (`wdl_prior`, `wdl_prior_source`, `wdl_prior_confidence`, `wdl_prior_holes`) with shapes determined by the C# reader's actual output. The synthetic-WDL builder, the merged-coverage builder, and Stage A all consume the C# reader's shape as a runtime parameter, not as a hard-coded constant.

**Rationale**:
- The user has explicitly stated the C# reader is the source of truth. Hard-coding 17×17+16×16 in the Python code would be a guess.
- Alpha-era WDLs (0.5.3) may have a different layout than LK-era WDLs (3.3.5a). The C# reader handles this; V24 reads whatever the C# reader returns per build.
- The Zarr store is keyed by tile, not by MARE. The V24 schema is per-tile, with the WDL prior grid shape recorded as a build-time metadata field.

**Decision details**:
- The `wdl_prior` array is per-tile float32, shape = whatever the C# reader returned for that build. The shape is recorded in `wdl_prior.attrs['shape_per_tile']` and `wdl_prior.attrs['shape_source'] = 'csharp_wdl_reader'`.
- The synthetic-WDL builder returns the same shape (it wraps the C# terrain→WDL path, which produces the same per-MARE shape).
- Stage A's output is the same shape.
- The synthetic-WDL builder, merged-coverage builder, and Stage A all read the shape from the V24 store's metadata on load. No hard-coded shape constants.
- An audit of the C# WDL reader's actual output for 0_5_3_3368 and 3_3_5_12340 is a Phase 0 task. The audit is documented in `wow-viewer/docs/architecture/wdl-reader-shape-audit-2026-07-XX.md` once it runs.

**Status**: Recommended default. To be confirmed by the C# WDL reader audit (Phase 0 task).

## Additional Technical Decisions

### D1: Loss functions for Stage A and Stage B

**Decision**: Both stages use simple L1 loss (mean absolute error). No fancy multi-component loss stacks.

**Rationale**:
- V24 is small and simple. The user has explicitly pushed back on over-engineering (no DA-V2, no LoRA, no GPCT, no CAI, no PagedAdamW8bit).
- L1 is robust to outliers and easy to debug.
- L1 is the right loss for "predict the WDL prior" (regression on float32 heights).
- L1 is the right loss for "predict the residual" (regression on float32 height differences).
- Per-cell sample weights (`wdl_prior_confidence`) and per-cell sample selection (stratify by `wdl_prior_source`) handle the real-vs-synthetic imbalance without needing a custom loss.

**Decision details**:
- Stage A loss: `L = sum(confidence * |predicted - target|) / sum(confidence)` over all cells where `source != 2` (learned-fill excluded).
- Stage B loss: `L = mean(|predicted_residual - target_residual|)` over non-liquid, non-object, non-MAHO-hole pixels.
- No gradient matching, no SDC, no bias-free masking. The user explicitly said no over-engineering.

### D2: Optimizer and precision

**Decision**: AdamW (not 8-bit), fp16 mixed precision (not bf16).

**Rationale**:
- The models are small (≤ 1M + ≤ 2M params). 8-bit AdamW is unnecessary and would add a dependency on bitsandbytes.
- fp16 is the standard for consumer GPU inference; bf16 is the standard for Hopper / Ada datacenter. 6 GB consumer GPU = fp16.
- AdamW is the standard optimizer for small vision models. No reason to use anything else.

**Decision details**:
- Optimizer: AdamW with default betas (0.9, 0.999) and weight_decay=1e-4.
- LR: 1e-3 with cosine annealing to 0 over the training run. No warmup (the models are small enough that warmup is unnecessary).
- Mixed precision: fp16 autocast for forward, fp32 master weights, fp32 gradients.
- Determinism: `torch.manual_seed`, `model.eval`, `torch.use_deterministic_algorithms(True)` everywhere.

### D3: Hardware envelope

**Decision**: Local training and inference on a 6 GB consumer GPU. No RunPod.

**Rationale**:
- The user has explicitly pushed back on RunPod / V23-style infrastructure. V24 is small and runs locally.
- The total trainable params are ≤ 3M. fp16 activations at 256×256 are < 100 MB. The full inference VRAM is < 4 GB, well within a 6 GB consumer GPU.
- Training memory is bounded by the optimizer state (AdamW: 2× model size for moments + 1× for master weights = ~36 MB for 3M params) + activations. With gradient checkpointing, training fits in < 8 GB.

**Decision details**:
- Validation hardware: any 6 GB+ consumer GPU (GTX 1660 Super, RTX 3050, etc.).
- Peak inference VRAM: < 4 GB at fp16.
- Peak training VRAM: < 8 GB at fp16 with gradient checkpointing.
- Inference wall-time: < 3 s/tile on a 6 GB consumer GPU.
- No RunPod, no Pod packaging, no Spec 079 bundle.

### D4: Zarr store layout

**Decision**: V24 Zarr store is a V18 Zarr store extended with 3 new arrays (`wdl_prior`, `wdl_prior_source`, `wdl_prior_confidence`) and 1 optional array (`wdl_prior_holes`, present only if the C# reader exposes MAHO). All V18 arrays are present and unchanged.

**Rationale**:
- V18 is the substrate (per the spec). V24 extends V18 additively.
- The 3+1 array schema matches the C# reader's output shape (whatever that is). No hard-coded 17×17 or 16×16.
- The `wdl_prior` array is float32, the source is uint8 (3 values: 0/1/2), the confidence is float32 in [0, 1], and the holes (optional) is bool.

**Decision details**:
- V24 store path: `wow-viewer/output/datasets/v24/<build>.zarr/`.
- V18 arrays (unchanged): `minimap_rgb`, `height_257`, `alpha_256`, `mcnr_mask_257`, `normal_xyz`, `object_precise_mask`, `liquid_mask_256`, etc.
- New V24 arrays:
  - `wdl_prior`: per-tile float32, shape = C# reader's per-MARE output.
  - `wdl_prior_source`: per-tile uint8, same shape, values 0/1/2.
  - `wdl_prior_confidence`: per-tile float32, same shape, values in [0, 1].
  - `wdl_prior_holes` (optional): per-tile bool, same shape (or a coarser shape if MAHO is 16-uint16 vs 17×17).
- Build-time metadata in `wdl_prior.attrs`:
  - `shape_per_tile`: the shape of the WDL prior grid.
  - `shape_source`: the string 'csharp_wdl_reader'.
  - `build_id`: which WoW client build this store was built from.
  - `coverage_real_ratio`, `coverage_synthetic_ratio`, `coverage_learned_fill_ratio`: per-store coverage stats.

### D5: Validation gates

**Decision**: Each phase ends with a concrete validation check before the next phase starts. The gates are:

- **Phase 1 gate**: `WowViewer.Tool.WdlRead` shim works on at least one real staged-client `.wdl` file. The shim emits a non-zero NPZ with the C# reader's actual shape. `py_compile` + CLI `--help` pass.
- **Phase 2 gate**: `build_synth_wdl.py --help` passes. A bounded run on a 5-tile V18 subset produces a synthetic WDL NPZ that matches the C# reader's shape (per the audit in Phase 0).
- **Phase 3 gate**: `build_wdl_prior.py build --v18-store <path> --staged-client <path> --output <v24.zarr>` runs on 5 tiles and produces a V24 store with `wdl_prior`, `wdl_prior_source`, `wdl_prior_confidence` populated. Coverage stats pass: real+synthetic ≥ 95%.
- **Phase 4 gate**: `clean_minimap.py --help` passes. A bounded run on 5 V18 tiles produces cleaned minimap NPZs that, when compared to the raw minimap, show object pixels replaced.
- **Phase 5 gate**: `train_v24_stage_a.py --help` passes. A 5-epoch training run on 5 V18 tiles converges (loss decreases). Stage A is ≤ 1M params. Inference on 1 tile is < 1 s.
- **Phase 6 gate**: `train_v24_stage_b.py --help` passes. A 5-epoch training run on 5 V18 tiles converges. Stage B is ≤ 2M params. Full pipeline (Stage A + Stage B) inference on 1 tile is < 3 s.
- **Phase 7 gate**: `validate_v24.py` runs on a 50-tile V24 validation set and emits a JSON report at `output/v24_validation/<run_id>/report.json` with SC-001 / SC-002 / SC-003 / SC-004 / SC-005 all pass.

**Rationale**:
- The spec is structured as 7 sequential phases. Each phase ends with a measurable gate (RULE 8: "you cannot work on Phase N+1 until Phase N is done. Done means validated, not coded.").
- A failed gate blocks the next phase. The user has explicitly called out the false-positive risk; concrete validation gates are the structural fix.

## Open Questions Resolved

All 3 open questions in the spec are resolved with the recommended defaults above. The user confirmed all three.

## Out of Research Scope (Deferred to Plan / Tasks)

- Specific U-Net architecture for Stage A (depth, width, skip connections). Deferred to plan.md and tasks.md as bite-sized implementation choices. The spec caps total params at ≤ 1M for Stage A and ≤ 2M for Stage B; the exact architecture is an implementation detail.
- Specific conv-deconv architecture for Stage B. Same.
- Exact training hyperparameters (LR, batch size, epochs). Deferred to tasks.md; the spec gives the framework (AdamW, fp16, ≤ 50 epochs, determinism required).
- Specific Zarr compression settings. Deferred to tasks.md; Zarr's default compression is fine for this spec.
