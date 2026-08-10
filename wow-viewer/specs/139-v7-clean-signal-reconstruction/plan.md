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
materializes structural targets, and publishes a hash-bound manifest atomically. The next slice is
visual review, not a user-run training command.

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
│   ├── clean_signal_model.py        # architecture adapters and v7-style heads
│   ├── clean_signal_train.py        # shared trainer/evaluator/report writer
│   └── clean_signal_transfer.py     # accepted-real transfer audit
├── scripts/
│   ├── v60_build_clean_signal_corpus.py
│   ├── v60_validate_clean_signal_corpus.py
│   ├── v60_train_clean_signal.py
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
   structural run excludes adversarial and recovery/object terms.

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
visible and ablatable, uses project-owned data, and leaves heavy execution to the user. The only
remaining gate is implementation validation of the exact four-channel artifact and target
decomposition before any training command is authorized.
