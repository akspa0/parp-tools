# Implementation Plan: V7-Inspired Clean-Signal Terrain Reconstruction

**Branch**: `139-v7-clean-signal-reconstruction` | **Date**: 2026-08-10

**Spec**: [spec.md](spec.md)

## Summary

Recover v7's useful multi-scale coarse/detail and structural-loss bias inside a new image-only
contract. The model will consume only a versioned albedo-normalized observation package, train on
exact synthetic height supervision, compare the existing pyramid CNN/SegFormer/U-Net candidates,
and transfer only after the real albedo gate accepts a tiny 0.x/1.x sample.

## Implementation checkpoint — 2026-08-10

Phase 2 foundational contracts, Phase 3 model contracts, and the synthetic packaging core of Phase
4 are implemented. The clean model slice
uses local random-initialized encoders for `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2`, one
shared feature adapter, independent coarse/detail heads, and a clamped recomposed height output.
The model identity records its input/output schemas, architecture, profile, parameter count, and
configuration hash; reconstruction tests load the saved state into a rebuilt identity. The corpus
builder consumes only validated control NPZs, derives the declared synthetic observation,
materializes structural targets, and publishes a hash-bound manifest atomically. The visual-review
slice is implemented as a family/variant/cross-tile atlas with provenance JSON. The loss slice now
provides versioned parity and v7 structural profiles, differentiable point/gradient, frequency,
curvature, edge, transition, border, and LF/HF terms, and independent component tensors. The shared
trainer slice now fixes deterministic within-family/complete-family split identities, four-channel
NPZ loading, per-signal/per-family/per-bucket evaluation, and best/last checkpoint binding. The
PowerShell-ready dry-run/user-run CLI is implemented with fresh-output refusal.

## User-run checkpoint — 2026-08-10

The user completed the six-cell CUDA within-family matrix at `train-size=32` and `epochs=80`.
`pyramid_cnn/v7_structural_v1` is the best cell at validation MAE `0.145868`, versus the same
architecture/parity cell at `0.150999` and the within-family tile-mean baseline at `0.181995`.
Structural guidance improved every architecture, but the same-architecture lift was 3.40% for
pyramid, 4.23% for SegFormer, and 7.72% for U-Net, below the 10% success threshold. This justifies
a larger-capacity confirmation run while keeping promotion gated on the complete-family holdout.

That confirmation is now complete. It used one fresh full-profile `pyramid_cnn` structural cell over
all 76 complete-family training rows and 32 held-out rows. The plan reported 1,579,586 parameters
and no forbidden signals before the user-owned CUDA run.

## User-run checkpoint — 2026-08-10 (complete-family result)

The completed run is recorded under
`output/datasets/v60/v7-clean-signal-runs/pyramid-full-structural-complete-v1`. At best epoch 37,
`pyramid_cnn/v7_structural_v1` reached final-height MAE `0.173904` against the complete-family
tile-mean baseline `0.191047`, an `8.97%` aggregate improvement. The run used CUDA, 76 train rows,
and 32 held-out rows; best and last checkpoints plus the JSON report exist on disk.

The promotion gate is held: `cross_tile_burn` regressed `15.52%`, `cross_tile_lightning` regressed
`229.79%`, and the pathological bucket regressed `2.81%`. The aggregate SC-004 improvement clears
the 5% floor, but the cross-tile family regressions trigger the explicit no-promotion acceptance
scenario. Real transfer remains blocked; this checkpoint is diagnostic evidence for the next
bounded failure-analysis slice.

## Next bounded slice — checkpoint failure diagnosis

The checkpoint is useful as a frozen probe: load it against the exact held-out rows, export each
prediction and absolute-error field, and inspect the cross-tile atlas. This distinguishes a seam or
context failure from a signal that is not recoverable from one 256×256 clean observation. The new
`v60_diagnose_clean_signal_checkpoint.py` command performs prediction only; it does not train or
admit real transfer. The user-run command is recorded in the quickstart.

## Implementation slice — constant-field stability

The diagnostic atlas showed a concrete model failure rather than an unexplained cross-tile gap:
`flat-v00` and `cross_tile_lightning-v01` have nearly identical four-channel inputs, but the
zero-padding checkpoint emits the same large spatial ramp while both targets are effectively flat
zero. The model now defaults to `reflect-3x3-v1` padding for every spatial 3×3 convolution. A
constant-input regression test covers all three architectures, and identity reconstruction retains
legacy zero-padding support for existing checkpoints. The targeted user-run retraining command is
recorded in the quickstart under a fresh `v2-reflect-padding` output root.

## User-run checkpoint — 2026-08-10 (reflect-padding result)

The user completed the fresh `v2-reflect-padding` full-profile CUDA run. The best checkpoint was
epoch 80 with final-height MAE `0.137891` versus the same `0.191047` tile-mean baseline, a `27.82%`
aggregate improvement compared with `8.97%` for the zero-padding checkpoint. The CPU diagnostic
confirmed the invented flat-input ramp is gone: `cross_tile_lightning-v01` now predicts a nearly
constant field, but still has MAE `0.229543` against a near-zero target. `cross_tile_lightning`
remains `61.17%` worse than its baseline and `cross_tile_burn` remains `30.15%` worse, so the
cross-family promotion hold remains active.

## Next bounded slice — within-family cross-tile learnability

The next run is not another complete-family rerun. It uses all 81 within-family training rows
(three variants from each family) and 27 one-variant validation rows to determine whether the
cross-tile patterns are learnable once the model sees examples from those families. If they pass
within-family but fail complete-family, this is a family-coverage/generalization limit. If they
still fail, the clean observation does not expose enough information for those targets.

## Implementation slice — real-terrain synthetic bridge

The existing v50/v60 real/synthetic pair store is not a clean v60 training corpus: its synthetic
side is a legacy flat fake maptexture, and its authored side has not passed albedo normalization.
The bounded bridge therefore consumes only the harvested `terrain_shadow_256` plus independently
extracted `height_257` from real-client terrain NPZs and labels the rows
`real_terrain_synthetic`. `v60_build_real_terrain_synthetic_corpus.py` publishes a normal
four-channel corpus without mutating source NPZs; `v60_evaluate_clean_signal_checkpoint.py` runs
image-only checkpoint evaluation with an explicit forbidden-read audit.

The first existing NPZ sample contains only 16 Alpha/Azeroth rows (15 train, 1 validation). That
directory came from an older quickstart diagnostic, not from the complete v50.1 store. The current
reflect-padding checkpoint scored MAE `0.323879` versus a `0.157124` tile-mean baseline on all 16
rows (`-106.13%` relative improvement), so this is diagnostic evidence of a real-domain failure,
not a promotion result. The complete v50.1 mixed curriculum store contains 1,330 synthetic rows:
688 Kalimdor and 642 Azeroth, but this particular pre-Spec-133 store has raw `shadow_mask` and no
`terrain_shadow_256`. The Zarr-backed builder now reads that source without mutating it, preserves
original `index.parquet` row indices in provenance, and creates a complete-family map-held-out
split (Kalimdor train, Azeroth validation). Its explicit `shadow_mask` mode is a geometry-only raw
MCSH diagnostic; it is not the deployment-clean signal and must not be silently relabeled as one.
Authored RGB remains blocked on the missing albedo-normalization gate.

## User-run checkpoint — 2026-08-10 (real-terrain bridge probe)

The user trained the full `pyramid_cnn`/`v7_structural_v1` model on 15 Alpha/Azeroth bridge rows
with one validation row. The best checkpoint was epoch 4 at MAE `0.313952` versus the one-row
`0.109902` tile-mean baseline (`-185.66%`); the pasted epoch-24 snapshot was worse at `0.380639`.
Evaluating that best checkpoint across all 16 bridge rows produced MAE `0.293371` versus the
`0.157124` all-row baseline (`-86.71%`). The coarse error dominates while detail error is small,
so this is not merely a late-epoch fluctuation.

## Implementation slice — complete v50.1 bridge source

`v60_build_real_terrain_synthetic_zarr.py` is dry-run by default and publishes the full synthetic
side only after `--confirm-build`. The 2026-08-10 dry run against
`v50/v50.1/curriculum-0_5_3_3368-obj_v1.zarr` reported 1,330 source rows, 688 train rows from
Kalimdor, and 642 validation rows from Azeroth. This fixes the previous scope error: the 16-row
NPZ directory remains a small failure diagnostic, while the Zarr bridge is the usable multi-map
diagnostic corpus. Because this store lacks `terrain_shadow_256`, use the explicit
`--input-signal shadow_mask` mode for a raw-MCSH geometry diagnostic. A deployment-clean bridge
still requires a post-Spec-133 store containing `terrain_shadow_256`; authored minimap RGB remains
excluded.

## Next bounded slice — bridge source integrity and multi-map expansion

Do not retrain the same 16 rows again. The current bridge contains two effectively flat targets,
large height-range variation, and non-uniform shadow/mask coverage. Audit and preserve those source
quality bands, then add approved rows from additional maps/builds before another real-bridge run.
Authored RGB normalization remains a separate implementation gate.

## Technical Context

**Language/Version**: Python 3.11; existing C# harvest/compositor remains the signal authority.

**Primary Dependencies**: PyTorch, existing `timm`, existing Hugging Face Transformers, NumPy,
Pillow, Zarr/NPZ corpus utilities already present in `data-harvester`.

**Storage**: Versioned NPZ shards for the first control corpus; existing v50/v60 Zarr stores are
read-only sources or lineage references. No copyrighted dataset is packaged.

**Testing**: Focused pytest contract tests, `ruff`, `py_compile`, deterministic hash comparison,
dry-run CLI audits, and user-run GPU training/visual review.

**Target Platform**: Windows PowerShell development and user-owned CUDA training; CPU tests must
remain available.

**Project Type**: Python data/model library plus PowerShell-ready CLI scripts.

**Performance Goals**: The first candidates remain in the existing small-model range and must run
the 256×256 smoke contract on CPU. Training performance is reported by the user-run CLI rather than
claimed by tests.

**Constraints**: No WDL, height, normals, liquid, object, alpha, or target-derived arrays at
inference. No external weights in the first bakeoff. Heavy generation/training is user-launched.

**Scale/Scope**: A deterministic 32–128-row synthetic control corpus first, followed by a tiny
accepted 0.x/1.x transfer sample. Later client eras and object prediction remain out of scope.

## Constitution Check

| Gate | Status | Evidence |
|---|---|---|
| New code remains under `wow-viewer` | PASS | All model, corpus, trainer, and tests are under `wow-viewer/data-harvester`. |
| Existing readers/compositor remain authorities | PASS | The plan consumes existing C# outputs and does not rewrite file readers or lighting. |
| User owns heavy work | PASS | Generation, training, and transfer CLIs are dry-run by default and require explicit confirmation. |
| No target-derived deployment input | PASS | Four-channel clean observation package is the only inference contract. |
| Per-signal evidence | PASS | Coarse, detail, final, frequency, edge, and family metrics are reported independently. |
| Phase boundaries | PASS | Synthetic contract and control gate precede real normalization/transfer. |

## Project Structure

```text
wow-viewer/data-harvester/
├── src/harvester/v60/
│   ├── clean_signal_corpus.py       # observation/target manifest and validation
│   ├── clean_signal_inputs.py       # four-channel deployment-safe input assembly
│   ├── clean_signal_targets.py      # coarse/detail target decomposition
│   ├── clean_signal_losses.py       # parity and v7 guidance loss components
│   ├── clean_signal_diagnostics.py   # checkpoint predictions and failure atlases
│   ├── clean_signal_model.py        # architecture adapters and v7-style heads
│   ├── clean_signal_train.py        # shared trainer/evaluator/report writer
│   ├── clean_signal_transfer.py     # prepared-corpus transfer audit
│   └── real_terrain_synthetic.py    # real-client terrain synthetic bridge corpus
├── scripts/
│   ├── v60_build_clean_signal_corpus.py
│   ├── v60_validate_clean_signal_corpus.py
│   ├── v60_train_clean_signal.py
│   ├── v60_build_real_terrain_synthetic_corpus.py
│   ├── v60_evaluate_clean_signal_checkpoint.py
│   ├── v60_visualize_clean_signal.py
│   └── v60_transfer_clean_signal.py
└── tests/v60/
    ├── test_clean_signal_inputs.py
    ├── test_clean_signal_targets.py
    ├── test_clean_signal_losses.py
    ├── test_clean_signal_model.py
    └── test_clean_signal_contract.py
```

## Architecture and signal decisions

1. The input adapter accepts exactly four channels: normalized luma, x-gradient, y-gradient, and
   albedo confidence. It refuses target-side arrays in inference mode.
2. The target adapter creates a versioned low-frequency coarse field and signed detail residual from
   `height_257`; the target decomposition is training-only and never serialized as an input.
3. Each encoder candidate exposes a shared feature contract. A v7-style two-head decoder emits
   coarse and detail fields, and the evaluator recomposes and scores final height.
4. `pyramid_cnn` is the first implementation candidate, `segformer_b0` is the transformer
   comparison, and `unet_lite_v2` is the control. No DPT or external checkpoint is needed here.
5. The loss registry keeps point/gradient parity separate from the v7 structural stack. The first
   structural run excludes adversarial and recovery/object terms, and every enabled or disabled term
   remains available as an independent metric.
6. Spatially constant clean observations must remain spatially constant through the model; the
   default `reflect-3x3-v1` padding policy prevents zero-padding boundary information from becoming
   a learned position prior. Legacy zero-padding identities remain loadable for diagnosis only.

## Phase 0 — Contract and evidence

1. Freeze the input, target decomposition, forbidden-signal, and provenance contracts.
2. Add fixtures showing an accepted clean observation, missing confidence, textured rejection, and
   forbidden target arrays.
3. Define both within-family learnability and complete-family transfer splits.

## Phase 1 — Synthetic corpus and visualization

1. Derive the four-channel clean observation package from the authoritative synthetic observation.
2. Emit exact target and training-only coarse/detail arrays with hashes and synthesis metadata.
3. Add deterministic validators and visual panels for observation, confidence, target, coarse field,
   detail residual, and recomposed height.
4. Validate all required terrain complexity and cross-tile families before training.

## Phase 2 — Model and loss bakeoff

1. Adapt `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2` to the shared v7-style two-head output.
2. Implement parity and structural loss configurations with independent component metrics.
3. Add dry-run architecture/loss matrix reporting and fail-closed output directories.
4. User runs the within-family learnability matrix first, then the complete-family gate for the
   strongest configurations.

## Phase 3 — Real transfer gate

1. Consume only accepted albedo-normalized 0.x/1.x observations.
2. Run image-only inference with forbidden-signal audit and fixed visual artifacts.
3. Write hold/diagnose/expand transfer decision; no broad harvest or later-era processing follows
   without `expand`.

## Constitution Re-check

The design remains within the constitution: it removes the old WDL dependency, keeps all signals
visible and ablatable, uses project-owned data, and leaves heavy execution to the user. The
four-channel artifact and target decomposition are validated, and the user-owned complete-family
run is recorded. The remaining gate is a promotion decision after cross-tile failure analysis;
real transfer is not authorized by the aggregate improvement alone.
