# Phase 0 Research: V7-Inspired Clean-Signal Reconstruction

**Date**: 2026-08-10

## Evidence reviewed

- `data-harvester/src/harvester/spec103/v7_model.py` — the historical 117M multi-scale U-Net,
  global/local output heads, optional WDL trestle, and bounds head.
- `data-harvester/src/harvester/spec103/v7_inputs.py` — the pinned 13-channel input assembly and
  its target-derived WDL, height-hint, normal, liquid, object, and brush channels.
- `data-harvester/src/harvester/spec103/v7_losses.py` — full-spectrum, gradient, Sobel, Laplacian,
  transition, tile-border, SSIM, and recovery-focus terms.
- `specs/archived/103-image-only-reconstruction/research-v7-contract.md` — the authoritative
  historical contract and its explicit caveats.
- `specs/134-v60-unified-dataset-model/` — the current clean control corpus, albedo gate plan, and
  four-model bakeoff.
- The user-run `v60-architecture-bakeoff-report-v1` — `pyramid_cnn` was best at `0.236665`, but the
  tile-mean baseline was `0.191047`; all candidates failed overall, while cross-tile lightning and
  burn were the dominant failures.

## Decision 1 — transfer v7's structural bias, not its input contract

The useful v7 idea is the explicit separation of broad relief from local detail plus loss-side
structural guidance. The WDL trestle and the remaining 13-channel inputs are rejected because they
are either unavailable at inference or derived from the target. The new lane therefore predicts a
coarse relative field and a signed detail residual, then recomposes one `height_257` output.

Alternatives considered:

- Revive the original 117M v7 model: rejected because its capacity, WDL trestle, and input leakage
  would obscure the clean-signal question.
- Keep the current one-channel model unchanged: rejected because the bakeoff showed smooth models
  lose cross-tile and high-frequency structure even on valid controls.
- Add a WDL prior predicted by a separate model: rejected for this lane; it recreates the prior
  dependency and makes the first question two coupled models instead of one image-only contract.

## Decision 2 — make albedo-normalized observation the only deployment authority

The deployment input is a four-channel package: normalized textureless luminance, its x/y image
gradients, and the albedo operation's confidence map. Each channel is computable from an arbitrary
minimap before model inference. Confidence is provenance from the albedo operation, not a target
mask. The synthetic corpus must render or derive the same package and preserve its operation
metadata.

Alternatives considered:

- Feed authored RGB directly: rejected because texture and material color are known shortcuts and
  are precisely what the albedo operation is intended to remove.
- Feed native normals, liquid, object, or alpha signals: rejected because they are not guaranteed
  to exist for an arbitrary minimap and several are answer-side or geometry-side signals.
- Feed only the one-channel `terrain_shadow_256` control input: retained as a diagnostic baseline,
  but not as the full new contract because it omits the real albedo-operation confidence and makes
  the transfer boundary ambiguous.

## Decision 3 — use the existing architecture registry, with pyramid as the first candidate

The first candidates are `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2` control. They share one
clean-input/coarse-detail-output contract and one evaluator. `pyramid_cnn` is first because it was
the strongest current bakeoff candidate; SegFormer tests a different multi-scale inductive bias;
U-Net establishes the low-capacity control. The generic DPT candidate is deferred because the prior
run was flat from epoch one and did not justify another expensive lane.

No external weights are part of the initial experiment. Architecture identity, seed, and parameter
count are recorded for every run.

## Decision 4 — make v7 losses independently ablatable

The first loss matrix has a point/gradient parity cell and a structural-guidance cell. Structural
guidance includes full 2D log-spectrum, Laplacian curvature, Sobel edges, target-gradient
transition focus, tile-border emphasis, and explicit low/high-frequency target bands. SSIM,
adversarial loss, and object/recovery weighting are excluded from the first clean lane; they can
hide whether the core terrain relation is learnable and have no necessary deployment signal.

Every component receives its own validation metric or ablation record. A lower aggregate loss is not
enough to promote a model if a complexity bucket or structural component regresses.

## Decision 5 — separate learnability from family transfer

The current family-held-out split is valuable for transfer but too coarse to answer whether the new
signal can be learned at all. The new experiment reports both:

1. a within-family variant split for learnability and loss ablation; and
2. a complete-family holdout for generalization, including cross-tile and pathological families.

The second gate remains the promotion gate. The first gate prevents an OOD split from causing another
architecture-only detour.

## Rejected shortcuts

- Ground-truth coarse fields, bounds, height ranges, normals, WDL, or masks as inference channels.
- Reusing the old v50 synthetic RGB as if it were a terrain-shadow target.
- Using a GAN or external image corpus to make outputs look plausible.
- Calling synthetic success a real-data transfer result without the albedo gate.
