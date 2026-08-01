# Research: Relational Terrain Layer Reconstruction

**Feature**: 116-relational-terrain-layers | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This document resolves every NEEDS CLARIFICATION from the plan's Technical Context and the one
Constitution Principle IV tension, before any code is written. Each decision records what was
chosen, why, and what was rejected. Decisions are referenced by id (D-0n) from `data-model.md`,
`contracts/`, and `tasks.md`.

---

## D-01 — Where the relational layer-entry data lives (no new harvest)

**Decision**: The "layer entries as rows" are extracted from signals already present in the v50
Zarr curriculum store — **no new harvest pass is required** (spec Assumption). Concretely, for each
curriculum row (one tile) the store carries:

- `mcly_texture_ids` — `(16, 16, 4)` int32: per chunk `(cy, cx)`, per slot `s ∈ {0..3}`, the
  **local MTEX index** (foreign key into that tile's own texture-name table).
- `mcly_layer_mask` — `(16, 16, 4)` float32: per chunk/slot coverage (0..1). Slot 0 (base) is
  always opaque and carries **no authored alpha** (spec Motivation point 3 / FR-008).
- `mcly_tileset_ids` — `(16, 16, 4)` int32: the **global** tileset index (not persisted to a
  name list — see Spec 115 `terrain_feature_labels.py` for why it is unreliable).
- `alpha_256` — `(256, 256, 4)` uint8: MCAL alpha for the detail layers.
- `height_257`, `minimap_rgb`, `mcnk_flags_16` — geometry, appearance, flags.

A **layer entry row** is the tuple `(tile_row, chunk_y, chunk_x, slot, local_texture_id,
coverage)`. The texture reference is resolved to a **surface family** by joining
`local_texture_id` against the per-tile texture-name dump (the same join Spec 115's
`terrain_feature_labels.derive_row_labels` already performs) and classifying the leaf name with
the frozen `v115.1` taxonomy.

**Rationale**: The spec's Motivation point 2 states the foreign-key join is already performed by
the existing feature-label derivation. Reusing it avoids a second texture-name resolution path and
keeps the family vocabulary identical to the deployed Spec 115 classifier, so US1's
family→slot measurement and US3's predictions speak the same ordinals.

**Alternatives rejected**:
- *Re-derive from `mcly_tileset_ids` (global index)*: Spec 115 falsified the global index→name
  mapping against the real client (it maps Kalimdor 24,40 to Alterac textures when the true table
  is four Darkshore textures). The local index + name dump is the verified path.
- *A new harvest pass*: the spec explicitly assumes none is needed; the signals already exist.

---

## D-02 — US1 family→slot consistency metric and decision threshold

**Decision**: US1 measures, for every surface family `f`, the distribution of the slot ordinal `s`
it occupies across all chunk/slot rows in the corpus, then computes a single summary **consistency
score** = the mean over families of the max slot probability `max_s P(s | f)` (a family is
"consistent" if it almost always lands in one slot). The **decision threshold** is a configurable
constant (default 0.70): if the summary score is at/above it, slot-keyed prediction is viable; if
below, heads MUST key on family and slot becomes a training-time grouping only (spec US1 acceptance
2). The threshold and the resulting recommendation are written to a **durable decision artifact**
(JSON, hash-bound to the store + taxonomy) consumed verbatim by US3.

**Rationale**: The spec requires a go/no-go architecture decision with no model trained. A
max-slot-probability summary is directly interpretable ("a given family lands in one slot X% of the
time") and is the quantity US3's head design depends on. The threshold is a constant, not a fitted
parameter, so it cannot be gamed.

**Alternatives rejected**:
- *Entropy of the slot distribution*: less interpretable to a non-technical reader and not a clean
  go/no-go cutoff.
- *Per-slot family purity (inverse direction)*: answers "is a slot always one family" rather than
  "does a family always take one slot", which is the question US3 needs (a minimap shows the
  family, not the slot).

---

## D-03 — US2 non-linear fit, explained variance, and bimodality test

**Decision**: US2 fits, **per tile and per detail layer**, a non-linear mapping from surface
properties `{elevation, slope}` (derived from `height_257` downsampled to the 16×16 chunk grid) to
that layer's coverage (`mcly_layer_mask[..., s]`). The fitter is a `scikit-learn`
`GradientBoostingRegressor` (small, non-linear, no GPU). Explained variance
`1 - SS_res/SS_tot` is reported per (tile, layer). Bimodality is tested two ways and both reported:
(1) Hartigan's dip test on the explained-variance distribution, and (2) a two-component Gaussian
mixture BIC comparison vs one component. The report explicitly states whether a distinct
high-coupling population exists and its tile share (spec US2 acceptance 2), and — if the
non-linear result disagrees with the prior weak linear analysis — records that the linear test was
**underpowered for threshold relationships**, not noise (spec US2 acceptance 3).

**Rationale**: The working hypothesis (masks distilled from higher-res source with hand fix-ups)
predicts a bimodal coupling. A non-linear fitter can detect threshold relationships a linear fit
cannot, which is exactly the underpowered-linear-test failure mode the spec calls out.

**Alternatives rejected**:
- *A single global fit*: would average away the bimodality the spec is looking for; per-tile fits
  are required to expose a high-coupling subpopulation.
- *Deep model fit*: overkill, GPU-bound, and not interpretable as explained variance; the user
  runs heavy work and this is a cheap CPU measurement.

---

## D-04 — Resolving Constitution Principle IV (the central design tension)

**Decision**: US3 "predict layer structure" is **decomposed into independent single-output models**,
one per detail slot, each with its own checkpoint, its own training script, and its own promotion
gate — exactly as FR-014 and constitution IV require. Each model predicts **one** signal: the
surface family occupying that slot, as a per-chunk classification over the 5 families. Coverage is
**not** a separate trained regressor by default: if US2 finds strong shape→coverage coupling, the
slot's coverage is *derived* from predicted geometry (D-03); if US2 finds weak coupling, a separate
independent coverage regressor per slot is added as its own checkpoint. The base layer (slot 0) is
never predicted — it is always opaque (FR-008) — so there are at most **three** structure models
(slots 1–3), each independently replaceable.

**Rationale**: The constitution forbids multi-task heads and shared weights, and requires each model
to predict one residual signal and be independently replaceable. A single "row predictor" with
multiple heads would violate all three. Decomposing by slot also matches the data: successive
detail layers are a *gradient* of fineness, not interchangeable, so each slot is a distinct
prediction problem. Independent checkpoints mean improving slot-3 never forces retraining slot-1.

**Alternatives rejected**:
- *One multi-task head sharing a trunk*: violates constitution IV + FR-014; prevents independent
  per-slot replacement; the exact monolithic pattern the project retired.
- *Predict the full texture reference (local MTEX id) directly*: a minimap shows what a texture
  *looks like* (family), not which local index it has — US1 exists precisely to confirm this. The
  local id is recovered post-hoc by legality repair (D-05), not predicted.
- *Predict coverage with a shared trunk across slots*: shared weights, forbidden.

---

## D-05 — FR-007 legality guarantee without ground truth at prediction time

**Decision**: A predicted family is **not** a texture reference until it is resolved against a
tile's own texture table. The model emits family probabilities per chunk/slot (no table consumed).
Legality is enforced in two regimes:

1. **Reconstruction of a real client tile** (the deployment contract): the tile's MTEX table *is*
   available — it is part of the client file being reconstructed, not model input. A post-hoc
   **legality resolver** picks, for each predicted family, a legal local texture id whose
   classified family matches the prediction (preferring the tile's existing entries; falling back
   to any same-family entry). Predictions with no legal same-family entry are **rejected** and the
   chunk is marked low-confidence (SC-004 = 100% of emitted references are legal).
2. **Out-of-distribution hand-painted image** (no tile, no table): there is no table to resolve
   against, so the system emits the family probabilities plus an explicit **"no legal table
   available"** audit record and never fabricates a texture reference (spec Edge Case + US3
   acceptance 3). This is the honest low-confidence expression the spec requires.

**Rationale**: FR-005/FR-006 forbid consuming ground-truth tables as model input; FR-007 requires
emitted references to be legal. These are reconciled by making the table a *post-hoc resolution
target*, never an input feature. The model never sees the table; the resolver does, only to
constrain output to legal entries.

**Alternatives rejected**:
- *Predict the local id directly and clamp to the table size*: clamping does not guarantee the id
  is a *valid, same-family* entry; it can emit a legal-but-wrong-family id and still pass a naive
  range check.
- *Refuse to emit any reference without a table*: would make the OOD case (US3 acceptance 3)
  impossible; the spec requires it to complete and emit an auditable record.

---

## D-06 — US4 spatially-isolated held-out set construction

**Decision**: A new held-out split is built over the tile grid (tile coordinates from
`index.parquet`). It guarantees **zero** held-out tiles share an edge **or corner** (8-neighbour)
with any training tile (FR-010 / SC-005). Construction is a deterministic graph problem: tiles are
nodes, 8-neighbour adjacency is edges; a held-out region is grown by selecting a seed and
flood-filling a buffer of width ≥1 ring of training tiles around every held-out tile, so no
held-out tile is ever 8-adjacent to another held-out tile that touches training. The builder reports
the verified violation count (must be 0) and the train/held-out counts. The split is written as a
Parquet manifest + JSON identity (store hash, build id, taxonomy revision, adjacency rule) so every
later result can name which held-out set produced it (FR-016 / SC-010). Rebuilding the split
**invalidates absolute comparison** with all prior results; the report states this and names the
baseline requiring re-run (FR-017).

**Rationale**: The spec's Motivation point 4 shows 99.6% of held-out tiles currently have a
training tile as an immediate edge-neighbour, so current scoring measures interpolation between
memorised boundaries. An 8-neighbour isolation buffer is the minimum that removes edge-vertex
sharing (adjacent tiles share edge vertices exactly).

**Alternatives rejected**:
- *4-neighbour (edge-only) isolation*: corner-touching tiles still share a vertex; the spec says
  edge **or corner**.
- *Random tile holdout*: reproduces the exact adjacency leakage the spec measured.
- *Reusing the Spec 114 grouped split*: it has the 99.6% leakage the spec calls out as the defect.

---

## D-07 — US4 relief stratification and the trivial baseline

**Decision**: Every evaluation partitions locations into **flat** and **relief-bearing** strata by
the local height variation in `height_257` (e.g. std over a chunk above a small threshold; the
threshold is a reported constant, not tuned per model). Error is reported **separately** per stratum
(FR-011 / SC-006), and the **trivial baseline** (tile-mean predictor) error is reported alongside
each stratum. Promotion is **refused** for any model that does not beat the trivial baseline on
**relief-bearing** regions (FR-012 / SC-007) — the honest bar, since no model in this project has
ever cleared it. Reused-piece overlap between train and held-out is measured and reported (FR-013 /
SC-008) using the existing cross-map dihedral block-matching machinery from Spec 113
(`minimap_alignment.py`), restricted to cross-set pairs.

**Rationale**: 39% of height patches are effectively flat and 51 of 120 sampled tiles are >90%
base-only, so aggregate error is dominated by terrain a constant predictor already solves.
Stratification is the only way to see whether a model actually learns relief.

**Alternatives rejected**:
- *Aggregate MAE as the gate*: the spec explicitly forbids it (FR-009 for structure; the same logic
  applies to height); it hides the failure behind flat terrain.
- *A fixed relief threshold baked into code*: the threshold is a reported constant but is not
  tuned per model, so it cannot be gamed to pass a gate.

---

## D-08 — US3 promotion metric (per-class, never aggregate accuracy)

**Decision**: US3's promotion gate is **per-class recall and IoU for every structural class**, with
the rarest class (~2% of locations) deciding promotion by its own recall/IoU (FR-009 / SC-003).
Aggregate accuracy is computed for reporting only and is explicitly **not** a gate. The in-run
baseline is the majority-class predictor (predict the most common family everywhere), computed on
the held-out set, making the degenerate "always terrain" solution explicit — mirroring the
established Spec 115 `terrain_feature_train.py` pattern. Class weights are capped inverse-frequency
(Spec 115 measured that uncapped weights drove road to ~8× over-prediction).

**Rationale**: The spec's SC-003 sets explicit per-class IoU floors (≥0.60 for classes ≥5%,
≥0.40 for the ~2% rarest class). A model that scores high aggregate accuracy while missing the
rarest class is the exact failure mode the spec forbids as a gate.

**Alternatives rejected**:
- *Macro/micro-averaged IoU as the gate*: hides the rarest class, which the spec makes the
  deciding class.
- *Accuracy gate*: explicitly forbidden by FR-009.

---

## D-09 — US5 feeding predicted (never ground-truth) structure into geometry

**Decision**: US5 materializes a **frozen** US3 checkpoint's *generated* structure output into a
derived store (bound to the checkpoint hash, source stores immutable — the exact pattern of Spec 115
`feature_map_materialize.py` and Spec 114 `direct_geometry_materialize.py`). Height reconstruction
(Spec 114's geometry chain) is then trained **with** this predicted-structure channel concatenated
to its RGB input and **without** it, on the **same** Phase C held-out set, and relief-region error
is compared (US5 acceptance 2). The geometry model consumes the *predicted* structure only; ground
truth never reaches it (US5 acceptance 1 / FR-006).

**Rationale**: This is the payoff story but depends on every story above. Materializing the frozen
output (not wiring the classifier live) keeps the geometry trainer independent of the classifier's
weights and makes the structure channel independently replaceable — the residual-chain discipline.

**Alternatives rejected**:
- *Live classifier in the geometry training loop*: couples two models' weights, violating
  constitution IV and preventing independent replacement.
- *Feeding ground-truth structure to geometry*: violates FR-006 and would measure an upper bound,
  not the deployable system.

---

## D-10 — Execution ownership and dry-run-first discipline

**Decision**: Every training and heavy rebuild is **user-run** from a documented CLI with time and
memory estimates (FR-018). Every training script validates and prints its full plan and exits
**without training** by default, requiring an explicit `--confirm-run` to consume compute (FR-015),
exactly as `v50_train_direct_geometry.py` / `train_spec111_reconstruction.py` already do. Every run
records an **identity binding** (sha256 of inputs, config, held-out set, taxonomy) in a
schema-validated run record (FR-016), reusing the `model_stage_contract.py` validator pattern.

**Rationale**: Project Rule 0 (the user runs training) and FR-015/FR-016/FR-018 are non-negotiable
and carried from prior specs.

**Alternatives rejected**: none — this is a constraint, not a choice.

---

## Open questions for the user (none blocking Phase 1 design)

None. Every NEEDS CLARIFICATION is resolved above. The two configurable constants (D-02
consistency threshold default 0.70; D-07 relief-stratum threshold) are documented defaults the user
can override at run time; they are not blocking.