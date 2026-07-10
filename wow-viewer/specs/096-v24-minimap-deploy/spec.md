# Feature Specification: V24 Minimap-Only Deployment Wiring (Spec 096)

**Feature Branch**: `096-v24-minimap-deploy`
**Created**: 2026-07-09
**Status**: Draft
**Owner**: wow-viewer
**Parent**: Spec 094 `094-wdl-prior-v24` (the WDL prior + lattice detailer)

---

## User Description (Verbatim)

The V24 model is supposed to take a minimap image and produce a WDL prior. The current inference scripts both require a V24 Zarr store + row index; that is not the deployment story. I want to be able to drop a bare PNG minimap into a CLI and get a WDL prior NPZ out, the way Spec 094's FR-013 and User Story 3 scenario 5 say it should work. The current code does not do that. The current `StageAMinimapOnly` model class exists but was never trained and has no inference entry point. Make this work.

---

## Problem Statement

Spec 094 `094-wdl-prior-v24` shipped a full implementation of the WDL prior + lattice detailer. The reported metrics (Stage A real-cell L1 = 0.397, Stage B final L1 = 0.857 world units, both better than the `block_reduce` baseline by 5–6.5×) are real but were earned in the **cheat regime** — the input includes V18 `height_257`-derived synthetic WDL, alpha, normal, mcnr, and object/liquid/holes masks. The "cheat" channels exist to be **dropped at inference** for the deployment case. The deployment path was never wired:

- [`StageAMinimapOnly`](wow-viewer/data-harvester/src/harvester/v24/stage_a.py:101) is a separate `nn.Module` (3-channel input, no residual) but **no checkpoint was trained for it**. The trainer's `--minimap-only` flag exists ([`train_v24_stage_a.py:137`](wow-viewer/data-harvester/scripts/train_v24_stage_a.py:137)) but has never been exercised on real data.
- [`infer_v24_stage_a.py`](wow-viewer/data-harvester/scripts/infer_v24_stage_a.py:1) and [`infer_v24_stage_b.py`](wow-viewer/data-harvester/scripts/infer_v24_stage_b.py:1) both require `--v24-store` + `--row` (or `--map --tile-x --tile-y`). There is no `--image` / `--png` entry point.
- The only minimap-only training infrastructure that was tested is the `StageAMinimapOnly` definition in [`stage_a.py:101`](wow-viewer/data-harvester/src/harvester/v24/stage_a.py:101) and a single test that the model is bit-identical under a fixed seed ([`tests/v24/test_stage_a.py:36`](wow-viewer/data-harvester/tests/v24/test_stage_a.py:36)). No real-data minimap-only training has been run.
- The "deployment" metric (Stage A's L1 in the **minimap-only regime** vs the `block_reduce` baseline) has never been measured. The memory bank reports `val_l1_minimap_only` as part of validation, but on a model that was not trained in that regime.

The user has been waiting on a usable inference path. The shipped metrics in the cheat regime do not answer the question the user actually asked. This spec closes that gap.

---

## What This Spec Does

1. **Train** the [`StageAMinimapOnly`](wow-viewer/data-harvester/src/harvester/v24/stage_a.py:101) checkpoint on the curated open-world V24 corpus (`3_3_5_12340`, 2,011 kept tiles, hard+pathological difficulty buckets). One training run, 50 epochs, gpu-resident data, batch-autotune, the standard V24 trainer log surface. The checkpoint lands at a known path under the run output dir.
2. **Add** a new inference script `infer_v24_stage_a_png.py` that loads a PNG minimap, runs the minimap-only Stage A checkpoint, and emits the `(17,17)` outer + `(16,16)` inner WDL prior NPZ plus a 4-up preview PNG. No V24 store required, no V18 store required, no staged client required. Pure minimap-to-prior.
3. **Measure** the minimap-only Stage A on a held-out curated V24 validation set. Report `val_l1_minimap_only` against the `block_reduce` baseline and the full-input Stage A (cheat regime) as an upper bound. Honest comparison.
4. **Update** the memory bank and `docs/architecture/v24-validation-2026-07-XX.md` so the deployment story is recorded as fact, not as a future intent.

---

## What This Spec Does NOT Do (Explicit Out of Scope)

- **No new minimap-only Stage B.** Stage B consumes `upsampled_prior + cleaned_minimap + alpha + normal + mcnr + object_mask` — five channels a bare PNG does not carry. The honest deployment path for Stage B is to drive it from a V18 store row (V18 has all of alpha, normal, mcnr, object_mask, liquid_mask, holes). The existing [`infer_v24_stage_b.py`](wow-viewer/data-harvester/scripts/infer_v24_stage_b.py:1) does exactly that (joined via `v18_row` from the V24 store). A PNG-only Stage B would require synthesizing fake alpha/normal/mcnr/object and would be a lie. Spec 097, if the user wants it, handles Stage B as a separate deployment shape.
- **No rewriting of the cheat-regime Stage A** or its trainer. The `--minimap-only` flag is the only new code path on the trainer. The cheat-regime trainer and its validation numbers are the upstream paper trail and stay as-is.
- **No RunPod work.** Per Spec 094 the minimap-only trainer is small (≤ 1M params). It runs locally on the 12 GB envelope the way V23 does. If a cloud pass is needed later, it is a separate spec.
- **No C# changes.** V24 wraps the existing C# WDL reader and the existing C# terrain→WDL path. None of that is touched.
- **No V18 build changes.** V18 is the substrate; we read from it, we do not modify it.
- **No dataset schema changes.** The V18 substrate, the V24 store, and the curated corpus are the inputs. Nothing is reshaped.
- **No "learned" minimap cleaner.** The NumPy 8-connected median cleaner from Spec 094 ([`clean_minimap.py`](wow-viewer/data-harvester/src/harvester/v24/clean_minimap.py:1)) is reused. A learned cleaner (Spec 095) is a separate spec.

---

## User Scenarios & Testing

### User Story 1 — Train the minimap-only Stage A (Priority: P1)

As a V24 owner, I can run one shell command against the curated V24 corpus and produce a minimap-only Stage A checkpoint, training summary, and a per-epoch loss history.

**Why this priority**: Without a trained checkpoint, the deployment script has nothing to load. The user has explicitly asked for a working minimap-to-prior path.

**Acceptance Scenarios**:

1. **Given** the existing curated open-world V24 store at `output/datasets/v24/3_3_5_12340_openworld_curated.zarr` and its underlying V18 store, **When** `train_v24_stage_a.py --minimap-only --v24-store <...> --v18-store <...> --output <run_dir> --epochs 50 --seed 94 --autotune-batch-size` runs to completion, **Then** the run directory contains `stage_a.pt`, `stage_a_metrics.json`, `loss_history.jsonl`, `peak_vram.json`, and `batch_autotune.json` (when autotune is on).
2. **Given** the training run completed, **When** `stage_a_metrics.json` is read, **Then** `params` is the Stage A minimap-only parameter count, `best_val_l1` is the best per-tile weighted L1 in world units across the run, `best_epoch` is the epoch that produced it, `epochs_run` is ≥ 1 and ≤ 50, `peak_vram_gb` is < 2.0 (12 GB envelope), and `train_tiles` + `val_tiles` = 2011.
3. **Given** the training run completed, **When** `loss_history.jsonl` is read, **Then** it contains one line per epoch with `epoch`, `train_loss`, `val_l1`, `val_l1_real_cells`, `val_l1_synth_cells`, and `lr`. `val_l1_minimap_only` is NOT in this file (it is a separate eval pass, see Slice 3).
4. **Given** the minimap-only Stage A model is zero-initialized at the output head (per [RULE 7](AGENTS.md:142) residual contract), **When** the model is evaluated on a held-out tile **before any training** (sanity check), **Then** the prediction equals the zero-initialized head output — i.e. a constant `(17,17)` and `(16,16)` field close to the per-dataset mean (the deployment regime is "predict the full prior," not "residual over a baseline"). This is the expected pre-train behavior; the test that asserts it lives in `tests/v24/test_stage_a.py`.
5. **Given** the same random seed (`--seed 94`) and the same data, **When** the run is repeated, **Then** the best `val_l1` matches the prior run to within `1e-4` (training determinism for this run shape; matches the existing trainer contract for the cheat regime).

### User Story 2 — Drop a PNG into a CLI and get a WDL prior (Priority: P1)

As a V24 owner, I can point a CLI at a single minimap PNG and get back a NPZ containing the `(17,17)` outer + `(16,16)` inner WDL prior plus a 4-up preview PNG. No V24 store, no V18 store, no staged client, no network.

**Why this priority**: This is the deployment story the spec promised. It is the deliverable the user actually asked for.

**Acceptance Scenarios**:

1. **Given** a trained minimap-only Stage A checkpoint and a 256×256 PNG minimap, **When** `infer_v24_stage_a_png.py --checkpoint <ckpt> --image <png> --output <npz> [--preview <png>]` runs, **Then** it writes `<npz>` with keys `outer` (17,17 float32), `inner` (16,16 float32), `prior_unavailable` (bool scalar, `False` on success), and the four-up preview PNG with the panels `[input minimap | prior outer (as 17×17 upsampled) | prior inner (as 16×16 upsampled) | prior outer/inner as a 33×33 quincunx]`. The four-up preview is the "did this work?" check.
2. **Given** a PNG that is not 256×256 (e.g. 512×512 or 128×128), **When** the script runs, **Then** it resizes the PNG to 256×256 with bilinear filtering before model input. The user is not expected to know the model's input size. The script reports the resize in the console log.
3. **Given** a PNG with a transparency channel, **When** the script runs, **Then** the alpha channel is dropped (RGB only). A user-supplied `--alpha-mask <npz>` is the right way to provide object masking; the PNG's alpha is not used as a mask.
4. **Given** the script runs successfully, **When** the output NPZ is loaded and `outer`/`inner` are denormalized to world units (multiply by 100.0, the V24 `HEIGHT_SCALE`), **Then** the values are within a reasonable world-unit range (e.g. roughly [-500, 4000] for a typical WoW terrain tile; the script emits a `world_min`/`world_max` summary in the console log so a sanity-check is built in).
5. **Given** the same PNG and the same checkpoint, **When** the script is run twice with different `--seed` values, **Then** the output NPZ is bit-identical (`np.array_equal`). The minimap-only regime is fully deterministic under `torch.manual_seed` + `eval()` + `use_deterministic_algorithms(True)`.
6. **Given** a 1×1 PNG, an empty PNG, or a corrupt PNG, **When** the script runs, **Then** it exits non-zero with a clear error message and writes nothing to `--output`. No silent failures.
7. **Given** a checkpoint that was trained in the cheat regime (not minimap-only), **When** the script tries to load it, **Then** it exits non-zero with a clear error message: "checkpoint was trained with IN_CHANNELS=13, this script requires the minimap-only 3-channel model." The `--strict-checkpoint` flag is on by default; `--lenient-checkpoint` is a documented escape hatch for debugging.

### User Story 3 — Measure the minimap-only Stage A on the held-out V24 set (Priority: P1)

As a V24 owner, I can run a single command that loads the trained minimap-only checkpoint, evaluates it on the same held-out V24 validation rows used for the cheat-regime validation, and produces a JSON report comparing `val_l1_minimap_only` against the `block_reduce` baseline and the full-input cheat-regime Stage A.

**Why this priority**: Without this comparison, we do not know whether the minimap-only regime is real or theatre. The user has been told the model works; this is the proof that the deployment regime also works.

**Acceptance Scenarios**:

1. **Given** the trained minimap-only Stage A checkpoint and the same `3_3_5_12340` curated V24 store used for training, **When** `validate_v24.py` (extended) runs with the minimap-only checkpoint, **Then** the report contains `stage_a_minimap_only_l1` (per-tile weighted L1 in world units, on the same held-out rows as the cheat regime).
2. **Given** the validation report, **When** reviewed, **Then** `stage_a_minimap_only_l1` < `block_reduce_baseline_l1` on the held-out rows. This is the new SC-002 gate: the minimap-only regime must beat the no-learning baseline. (The cheat regime passes this trivially; the question is whether dropping to 3 channels also passes.)
3. **Given** the validation report, **When** reviewed, **Then** `stage_a_minimap_only_l1` is reported alongside the cheat regime's `val_l1_cheat` and the gap between them is recorded. The honest comparison is the headline.
4. **Given** the validation report, **When** reviewed, **Then** the bit-identical-different-seeds check still passes for the minimap-only pipeline (SC-004 contract).

### User Story 4 — Standalone CLI works against arbitrary minimap sources (Priority: P2)

As a V24 owner, I can run the minimap-only script against (a) a PNG that has nothing to do with V18, (b) a PNG that came from a V18 store's `minimap_rgb` array, (c) a PNG that came from a real game client capture, and (d) a PNG that came from another terrain generator. In all four cases the script emits a WDL prior NPZ. The model's behaviour is not contingent on the PNG's provenance.

**Why this priority**: This is the user-facing meaning of "drop a PNG in." If the script only works on V18-derived PNGs, the deployment story is a half-truth.

**Acceptance Scenarios**:

1. **Given** a PNG with arbitrary content, **When** the script runs, **Then** it returns a valid WDL prior NPZ (any 17×17 and 16×16 float32 arrays; values are not guaranteed to be useful, but the script does not crash or refuse).
2. **Given** a real V18-derived minimap PNG (e.g. a 256×256 export of `v18["minimap_rgb"][r]`), **When** the script runs, **Then** the resulting WDL prior is in the same family as the V24 store's `wdl_prior_outer` / `wdl_prior_inner` for the same tile (within the per-tile L1 the validation report records).

### Edge Cases

- A minimap PNG with non-RGB channels (e.g. RGBA, L, P, CMYK). Script converts to RGB; transparency is dropped.
- A minimap PNG smaller than 32×32 or larger than 4096×4096. Script refuses (out of reasonable minimap-size range) with a clear error. The model is trained on 256×256 minimaps; anything outside the 64–1024 range is likely a user error.
- A checkpoint whose `in_channels` does not match the 3-channel minimap-only model. Script refuses (see User Story 2 acceptance scenario 7).
- A path that points to a directory, not a file. Script refuses with a clear error.
- A machine without a CUDA device. Script falls back to CPU; the user is told wall-time will be longer. Determinism still holds on CPU.
- Two PNGs in one script invocation (e.g. via shell glob). Not supported in this slice — single-file mode only. Multi-file is a separate spec.

---

## Functional Requirements

### Slice 1: Train the minimap-only Stage A checkpoint

- **FR-101**: A new run, `output/v24_validation/v24_minimap_only_3_3_5_12340_<YYYYMMDD>/`, contains a trained `stage_a.pt` for the minimap-only model and the standard V24 trainer artifacts (`stage_a_metrics.json`, `loss_history.jsonl`, `peak_vram.json`, `batch_autotune.json`). The trainer is the existing `train_v24_stage_a.py` with `--minimap-only`. No new trainer script.
- **FR-102**: The training run uses the curated open-world V24 store (`3_3_5_12340`, 2,011 kept tiles) joined against the V18 store. The training corpus is the same one used for the cheat-regime validation in 2026-07-07 / 2026-07-09.
- **FR-103**: Training runs for 50 epochs, seed 94, with `--autotune-batch-size`. Early stopping is disabled (`--patience 0`) so the full 50 epochs run.
- **FR-104**: The trained model is a `StageAMinimapOnly` instance. The checkpoint config records `{"base": 28, "in_channels": 3, "minimap_only": true}` so downstream inference scripts can refuse mismatched checkpoints (see FR-107).
- **FR-105**: `tests/v24/test_stage_a.py` adds two tests:
  - `test_stage_a_minimap_only_forward_shape_and_params` — assert `StageAMinimapOnly` forward returns `(B,17,17)` and `(B,16,16)`, parameter count ≤ 1M.
  - `test_stage_a_minimap_only_pre_train_is_constant` — assert the zero-init head output on a random input is constant across spatial positions (the minimap-only model has no synth baseline, so pre-train output is the constant head bias).

### Slice 2: `infer_v24_stage_a_png.py` standalone inference script

- **FR-106**: A new script `wow-viewer/data-harvester/scripts/infer_v24_stage_a_png.py` exists. CLI: `--checkpoint <ckpt> --image <png> --output <npz> [--preview <png>] [--seed 94] [--device cpu|cuda] [--strict-checkpoint / --lenient-checkpoint]`. Defaults: `--seed 94`, `--device auto`, `--strict-checkpoint` on.
- **FR-107**: The script loads the checkpoint, asserts the checkpoint config is minimap-only (`config["in_channels"] == 3` and `config["minimap_only"] is True`), asserts the checkpoint model state matches the `StageAMinimapOnly` architecture, and refuses with a clear error otherwise. `--lenient-checkpoint` skips these asserts.
- **FR-108**: The script loads the PNG via PIL, converts to RGB, resizes to 256×256 (bilinear), normalizes to float32 in [0, 1], and feeds it through `stage_a.build_minimap_only_input`. No V18 store is opened. No minimap cleaner is run by default (PNG minimaps are assumed clean — the user can preprocess with `clean_minimap.py` if they need it). An optional `--alpha-mask <npz>` flag enables object masking if the user has a per-tile mask.
- **FR-109**: The script runs the model in `eval()` mode under `torch.use_deterministic_algorithms(True)`, denormalizes the output (`* HEIGHT_SCALE = 100.0`), and writes the NPZ with keys `outer`, `inner`, `prior_unavailable`, plus a small metadata dict (`world_min`, `world_max`, `wall_ms`, `peak_vram_gb` if CUDA).
- **FR-110**: When `--preview <png>` is given, the script writes a 4-up PNG: input minimap (256×256) | outer 17×17 upsampled to 256×256 (nearest) | inner 16×16 upsampled to 256×256 (nearest) | outer/inner as a 33×33 quincunx upsampled to 256×256 (nearest). The preview is for visual sanity-check only.
- **FR-111**: Determinism (FR-014 / SC-004) holds: two runs with different `--seed` produce bit-identical output. The seed argument exists only to prove this.

### Slice 3: Validation comparison (minimap-only vs cheat)

- **FR-112**: `validate_v24.py` accepts an optional `--minimap-only-checkpoint <ckpt>` flag. When given, the report adds a `stage_a_minimap_only` block with `params`, `val_l1_minimap_only`, and `block_reduce_baseline_l1` (the same baseline used for the cheat regime). When absent, the block is omitted.
- **FR-113**: The new SC-002-MINIMAP gate is recorded in the report: `stage_a_minimap_only_l1 < block_reduce_baseline_l1` (same shape as the existing SC-002).
- **FR-114**: The full validation report at `output/v24_validation/<run_id>/report.json` is committed and the summary lands at `docs/architecture/v24-minimap-deploy-validation-2026-07-XX.md`.

### Slice 4: Memory bank + progress sync

- **FR-115**: `wow-viewer/memory-bank/activeContext.md` "WDL prior + lattice detailer lane (V24)" section is updated with the minimap-only deployment training result, the inference script entry, and the SC-002-MINIMAP gate pass/fail.
- **FR-116**: `wow-viewer/memory-bank/progress.md` gets a new 2026-07-09 entry summarising the slice, the metric, and the open question (is minimap-only as good as cheat? — recorded honestly even if the answer is "no, but it beats the baseline").

---

## Success Criteria

- **SC-096-001**: `train_v24_stage_a.py --minimap-only` produces a checkpoint on the curated `3_3_5_12340` corpus that loads, runs forward, and saves a `stage_a.pt` that the new inference script can load.
- **SC-096-002**: `infer_v24_stage_a_png.py` runs end-to-end on a real PNG (export of `v18["minimap_rgb"][r]`) and produces a WDL prior NPZ + 4-up preview PNG in well under 1 second on a 6 GB consumer GPU (target: < 100 ms on the actual hardware).
- **SC-096-003**: The minimap-only Stage A on a held-out curated V24 validation set achieves `val_l1_minimap_only < block_reduce_baseline_l1` in world units. This is the new SC-002-MINIMAP gate. If it fails, the spec is not done.
- **SC-096-004**: Bit-identical-different-seeds: two `infer_v24_stage_a_png.py` runs on the same PNG with `--seed 11` and `--seed 22` produce `np.array_equal` outputs (SC-004 contract).
- **SC-096-005**: Existing test suite remains green: `uv run python -m pytest tests/v24 -m v24 -q` passes with the two new tests added, total ≥ 33 passing.
- **SC-096-006**: Memory bank + progress.md are updated at slice completion. No "I forgot to update the docs" session.
- **SC-096-007**: The minimap-only regime is reported with a clear, honest comparison: `minimap-only L1 / cheat L1 / block_reduce L1`, in the same world units, on the same held-out rows. The gap is recorded as fact.

---

## Key Entities

- **Minimap-only Stage A checkpoint**: `output/v24_validation/v24_minimap_only_3_3_5_12340_<YYYYMMDD>/stage_a.pt`. A `StageAMinimapOnly` instance with base=28, in_channels=3, zero-initialized 1×1 head. Config dict carries `minimap_only: true`.
- **WDL prior NPZ**: 3 arrays — `outer (17,17) float32`, `inner (16,16) float32`, `prior_unavailable bool` — plus metadata. Same shape as the V24 store's `wdl_prior_outer` / `wdl_prior_inner` arrays. Compatible with downstream consumers.
- **4-up preview PNG**: a 1024×256 (4 × 256-wide panels) PNG showing the input minimap and three prior visualizations. Diagnostic only; not consumed downstream.
- **Validation report** (extended): `output/v24_validation/v24_minimap_only_3_3_5_12340_<YYYYMMDD>/report.json`. Same shape as the existing report, with the new `stage_a_minimap_only` block.

---

## Risks

- **Risk 1 (high)**: The minimap-only regime may not beat the `block_reduce` baseline. WoW minimaps carry a lot of object / shadow / compositing noise that the cheat regime's alpha/normal/mcnr/object channels help disambiguate. If the minimap-only regime fails SC-002-MINIMAP, the slice still ships (the inference script is real, the training run is real, the data is honest) but the metric is reported as a failure. Spec 097 then has a real motivation: "minimap-only underperforms; we need either a learned minimap cleaner (Spec 095) or to scope minimap-only to a specific tile-quality tier."
- **Risk 2 (medium)**: 50 epochs of minimap-only training may be insufficient. If val_l1 has not plateaued by epoch 50, the run logs a warning and the user can re-run with `--epochs 100`. We do not auto-extend; that is a separate decision.
- **Risk 3 (low)**: PNG decoding edge cases (truncated, non-standard color profiles, 16-bit). The script uses PIL's standard `Image.open().convert("RGB")` path; documented behavior. A user-side 16-bit PNG is the user's bug; the script logs and exits non-zero.
- **Risk 4 (low)**: The minimap-only model size (≤ 1M params) is small enough to run on CPU. The script does not require CUDA. The `peak_vram` and `wall_ms` numbers differ between CPU and CUDA; the metadata records both.
- **Risk 5 (low)**: The script does not need to be a long-running service. Single-file mode only. Multi-file batch mode is a separate spec.

---

## Assumptions

- The curated open-world V24 store from the 2026-07-09 training run is still on disk at `output/datasets/v24/3_3_5_12340_openworld_curated.zarr` (or an equivalent path; the run command is parameterized). If it is not, the training corpus is rebuilt with the existing `build_wdl_prior.py build` command first.
- The trained minimap-only checkpoint is small enough to commit to git (≤ 5 MB). If it is larger than that, `.gitignore` rules are updated and the checkpoint is documented as a download artifact.
- The user accepts that the minimap-only regime may be lower-quality than the cheat regime. The honest answer to "is this as good?" is recorded in the validation report.
- Stage B is out of scope for this spec. The user's question was about the minimap-to-prior path; Stage B's PNG deployment is a separate question because Stage B needs more than a PNG. This is explicit in the Out of Scope section.
- The `clean_minimap.py` module is good enough for the optional `--alpha-mask` path. A learned cleaner is a separate spec.

---

## Open Questions (For User Review Before Plan)

1. **Minimap-only training corpus scope**: should this spec train on the **full 2,011 curated tiles** (matches the cheat regime validation, gives the most reliable minimap-only number), or a **smaller held-out subset** (e.g. 200 tiles) for a faster turnaround? Recommended: full corpus, 50 epochs, ~30 min on the 12 GB envelope. The full corpus is the only one that gives a minimap-only number comparable to the cheat regime.
2. **Optional minimap cleaning on PNG input**: should the inference script apply `clean_minimap.py` to the PNG by default (assuming a transparent-channel-as-mask heuristic), or only when an explicit `--alpha-mask <npz>` is given? Recommended: by default, no cleaning (the user is responsible for clean input; this is the simplest contract). The optional flag is a documented escape hatch. A learned cleaner is Spec 095, not this spec.
3. **Where the inference script lives**: `wow-viewer/data-harvester/scripts/infer_v24_stage_a_png.py` (matches the existing `infer_v24_stage_a.py` naming) or a new top-level CLI like `wow-viewer/tools/v24-prior/`? Recommended: same scripts directory, same naming pattern. Consistent with the rest of the v24 scripts.

---

## End of Spec

This spec is bounded, honest, and ships the actual deployment story. The hard preconditions (the trainer flag exists, the model class exists, the V24 + V18 stores exist, the curated corpus exists) are explicit. The success criteria are measurable against the same `block_reduce` baseline the cheat regime was measured against. The risk that minimap-only underperforms is recorded as a real possibility with a real fallback spec (095/097), not hand-waved.

---

## Implementation Amendments (2026-07-09 — user feedback after smoke)

After shipping the four slices, the user ran the deployment wrapper against a real PNG minimap from the pre-alpha reconstruction work (`G:\WoW\rebuilding-prealpha\aligned-grid-3_tile2-1\tile_31_27.png`) and found the resulting OBJ was the closest thing to the prior needed to build the proper full v24 terrain reconstruction model. This is a real shift in the spec's status: the deployment wiring is no longer "real but model quality is bad" — it is "real and the mesh is a useful pre-alpha prior for downstream full-resolution reconstruction work." The honest 190.31-world-unit L1 is the same number; the framing changed because the user is now using the output to drive reconstruction work, not as a final answer.

### A1: New CLI surface

Two new scripts, both production-ready, added after the four shipped slices:

- `wow-viewer/data-harvester/scripts/v24_prior_to_obj.py` — standalone CLI that takes a prior NPZ (or a directory of priors) and writes a single-tile OBJ+MTL+texture or a grid-stitched OBJ+atlas. The X-axis flip is fixed (image-X is flipped to world-X at load time via `np.fliplr`, so the mesh opens with the source minimap oriented the same way it appears in any 2D viewer; the OBJ writer's existing V-flip handles the Y axis). The grid-stitch mode takes `--grid-from-priors <dir>`, auto-discovers `<stem>.prior.npz` + `<stem>.png` pairs, and writes one `terrain.obj` with a `atlas.png` texture covering all tiles.

- `wow-viewer/data-harvester/scripts/v24_run_on_png.py` — one-shot wrapper that auto-discovers the latest minimap-only checkpoint, runs the inference, calls the OBJ exporter, and writes the prior NPZ + preview PNG + mesh. Three modes:
  1. Single PNG: `uv run python scripts/v24_run_on_png.py some.png` → NPZ + preview + mesh.
  2. Export a random V18 minimap: `--export-v18-minimap` (optional seed) → exports + runs the wrapper.
  3. Batch a folder: `--batch-dir <dir>` → every PNG gets its own prior NPZ + mesh, then all are stitched into `<batch-dir>/stitched_mesh/terrain.obj` + `<batch-dir>/stitched_mesh/atlas.png`.

### A2: Spec 095 scope clarification

With the mesh output wired, the minimap-only regime is the right size for "pre-alpha prior for reconstruction" work. The 158× gap to the cheat regime is real but not blocking for downstream use: the user explicitly noted this is "the closest we've been to having the prior we need." Spec 095 (learned minimap cleaner) is still the right next step for closing the L1 gap, but its motivation is now "tighter pre-alpha prior" rather than "make minimap-only useful at all." The cleaner also feeds the full V24 reconstruction model, not just the deployment path.
