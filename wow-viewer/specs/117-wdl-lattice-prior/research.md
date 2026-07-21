# Research: WDL-Lattice Coarse Prior for Terrain Geometry

**Feature**: 117-wdl-lattice-prior | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This document resolves the open design questions from the plan's Technical Context before any code
is written. Each decision records what was chosen, why, and what was rejected. Decisions are
referenced by id (D-0n) from `data-model.md`, `contracts/`, and `tasks.md`.

---

## D-01 — How a scalar lattice satisfies the existing class-probability `--feature-store` contract

**Decision**: The generated lattice is upsampled from its native 545 sparse points (17×17 outer +
16×16 inner) to a dense per-pixel 256×256 field via bilinear interpolation over the sparse grid,
then written as a `feature_map` array of shape `(N, 1, 256, 256)` in a store carrying
`schema="v115-feature-map-v1"` and `class_count=1`. **No changes to `direct_geometry_train.py` or
`geometry_detailer_train.py` are required.** Both trainers already validate a `--feature-store`
purely structurally — `schema == "v115-feature-map-v1"`, `class_count >= 1`, a `feature_map` array
present, row coverage of every selected row via `source_row_index` — and concatenate exactly
`class_count` channels onto RGB before the first convolution. Neither trainer inspects the
channel's *semantics* (they never assume the concatenated channels sum to 1 or represent a
probability simplex); `class_count` is read purely as "how many extra channels", never used in any
softmax- or probability-specific code path on the consuming side. A `class_count=1` scalar height
channel satisfies that contract exactly as written, with no special-casing.

**Rationale**: Both trainers were extended and proven this session specifically to be
generic over *what* generated signal they consume, not just the Spec 115/116 terrain-feature
classes — the contract is already schema/shape-based, not classifier-specific. Reusing it exactly
means zero risk to two trainers that are the only real, honestly-validated result in this project's
history (56.1% relief-region improvement). Defining a second, lattice-specific ingestion path would
duplicate validated code for no benefit and would need its own independent proof before being
trusted.

**Alternatives rejected**:
- *A new `v117-wdl-lattice-map-v1` schema with its own trainer code path*: doubles the surface area
  that could silently regress the proven trainers, for a distinction (probability vs scalar) that
  the trainers never actually enforce or rely on. Rejected as unjustified complexity.
- *Feed the lattice as 545 raw sparse values (no upsampling) via a separate small side-network*:
  would require new architecture on the coarse/detailer side (a second input path, a fusion layer),
  directly contradicting "no changes to the already-validated trainers." The dense-upsample-then-
  concatenate approach reuses 100% of the existing input path.
- *Nearest-neighbour upsampling (matching how Spec 116's per-chunk structure classes were
  upsampled)*: appropriate for categorical per-chunk labels (a chunk has exactly one class), wrong
  for a continuous height field, where nearest-neighbour would introduce artificial 16px-aligned
  step discontinuities the real height surface does not have. Bilinear is the correct choice for a
  scalar field, exactly as the coarse/detailer stages themselves already use bilinear when resizing
  their own coarse relief field internally.

---

## D-02 — Trivial baseline for the standalone predictor (US2 learnability gate)

**Decision**: The trivial baseline for the standalone lattice predictor is the **per-tile mean of
its own 545 real lattice samples**, predicted uniformly for all 545 points — the exact same
"tile-mean" baseline concept already established for the coarse and detailer stages (`tile_mean` in
`height_relative_train.py`), applied at lattice resolution instead of full 257×257 resolution. US2's
learnable/not-learnable verdict (FR-004/SC-002) is: does the trained predictor's held-out
lattice-point MAE beat this baseline.

**Rationale**: Consistency. Every other stage in this project's residual chain is judged against a
tile-mean baseline computed at its own native resolution; inventing a different baseline concept for
the lattice (e.g. a global corpus mean, or a map-mean) would make US2's result incomparable to every
other honesty check this session already ran, and would reopen exactly the "is this baseline fair"
question Spec 116 US4 was built to close once.

**Alternatives rejected**:
- *Flat baseline (predict the middle of the normalized range everywhere)*: already tried and already
  known to be a much weaker reference across every prior stage (it never wins); redundant to
  recompute here.
- *A map-level or corpus-level mean*: would conflate the per-tile-scarcity failure mode this
  project already corrected for (see Spec 116's "no model beats tile-mean" reversal) — the tile-mean
  baseline is deliberately per-tile precisely because it is the hardest honest reference to beat.

---

## D-03 — Held-out split reuse, not a new split

**Decision**: The standalone predictor (US2) and the chain-integration comparison (US3) both use
the existing Spec 116 spatially-isolated held-out split (`spec116-held-out-0_5_3_3368-dual_v2` or
its successor built against whatever store the lattice signal export lands in) verbatim. This
feature does not construct a new split.

**Rationale**: A second independently-built split would make every result in this feature
incomparable to the just-established real baseline (the structure-augmented detailer's 56.1% figure)
that US3's whole comparison depends on. Reusing the identical split is what makes "compare against
the already-established baseline" (spec US3 acceptance 2) meaningful at all.

**Alternatives rejected**:
- *Build a fresh held-out split scoped to this feature*: technically straightforward (the tooling
  already exists and is proven), but would trigger the project-wide rule that a new split
  invalidates absolute comparison with all prior results — exactly the cost this decision avoids
  paying for no benefit, since nothing about this feature requires a different split shape.

---

## Summary of decisions carried into Phase 1

| ID | Decision | Consumed by |
|----|----------|--------------|
| D-01 | Lattice → dense `(N,1,256,256)` `v115-feature-map-v1`-shaped bridge; zero trainer changes | data-model.md (Generated Lattice Store), contracts/, tasks.md |
| D-02 | Trivial baseline = per-tile mean of the tile's own 545 real lattice samples | data-model.md, contracts/, tasks.md |
| D-03 | Reuse the existing Spec 116 held-out split verbatim; no new split | quickstart.md, tasks.md |
