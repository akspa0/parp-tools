# Feature Specification: V50-Native Height-First Terrain Model with Dataset Corrections

**Feature Branch**: `112-v50-height-model`

**Created**: 2026-07-18

**Status**: Draft

**Input**: User description: "v50-native height-first terrain model with dataset corrections.
Replace the rejected legacy spec103/spec108 model lane (WdlPriorNet absolute-elevation RGB-to-WDL +
V8 refiner) with a new, lean, v50-native model designed against the actual v50.1 signal corpus.
Phase 1 is dataset correction driven by the 2026-07-18 per-signal coverage audit; Phase 2 is the
model: minimap RGB in, per-tile RELATIVE height out. Corpus scope: Kalimdor and Azeroth ONLY.
Standing constraints: per-build Zarr only, ground-truth lighting/time never a model input, no
DepthAnything-family architectures, all training user-executed CUDA with explicit per-run
go-ahead, curriculum stores carry the full frozen signal catalog."

## Context and prior rulings this spec encodes

- **The legacy model lane is rejected, not inherited.** The spec103/spec108 `WdlPriorNet` predicts
  absolute global elevation from minimap RGB. A real training run on the v50.1 corpus (2026-07-18)
  proved the failure structurally: its best validation loss occurred at epoch 1 and degraded
  monotonically afterward against a held-out map whose mean altitude (+381 world units) lies far
  from the training maps' (−150 to +32). Absolute altitude is not inferable from minimap pixels;
  any target that embeds it inherits that flaw. User ruling: "We needed a sane dataset to move past
  the insane model that doesn't really work right."
- **Small maps are not evaluation material and are excluded entirely.** PVPZone02 (~60 usable
  tiles) and Kalidar (~24) are too small to gauge anything. This spec's corpus is Kalimdor and
  Azeroth only; validation is a within-map stratified holdout drawn from both.
- **The dataset audit found real gaps** (2026-07-18, measured on all four v50.1 stores): the
  manifest template declares four signals the frozen Spec 109 catalog explicitly dropped
  (`mddf_mask`, `modf_mask`, `object_filtered_mask`, `model_focus_mask` — all 0% populated, dead
  zero arrays); `mccv_rgb` is declared but 0% everywhere because MCCV does not exist in 0.5.3-era
  client data; `mcnk_flags_16` is promised by the catalog but 0% everywhere (never emitted by the
  extraction stream); `minimap_rgb_1024` coverage is 40–92% versus `minimap_rgb`'s ~100%
  (suspected file-contention decode failures between the two parallel synthesis processes reading
  the same archives).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Corrected, Honest v50.1 Corpus (Priority: P1)

The dataset operator rebuilds the Kalimdor and Azeroth stores so that every signal the store
declares is either genuinely populated or explicitly recorded as unavailable-with-reason — no
dead zero arrays, no catalog/template contradictions, and high-resolution minimap coverage equal
to standard-resolution coverage.

**Why this priority**: Every downstream decision (model targets, curriculum contents, evaluation)
reads the store's declarations. A store that declares dead signals or silently under-covers a
signal poisons those decisions; this was the user's core complaint ("we don't have all the signals
we need").

**Independent Test**: Run a per-signal coverage audit over the rebuilt stores. Every declared
signal shows either substantive coverage or an explicit unavailability record; the four
catalog-dropped signals are absent; `mcnk_flags_16` has real per-tile data; `minimap_rgb_1024`
row coverage equals `minimap_rgb` row coverage exactly.

**Acceptance Scenarios**:

1. **Given** the frozen Spec 109 signal catalog, **When** the manifest template is regenerated,
   **Then** the template's signal set matches the catalog exactly — signals the catalog dropped do
   not appear, and a build-era-impossible signal (`mccv_rgb` on 0.5.3) is recorded as
   unavailable-by-era rather than declared and zero-filled.
2. **Given** a rebuilt Kalimdor or Azeroth store, **When** `mcnk_flags_16` is sampled on tiles
   with real terrain, **Then** it contains genuine per-chunk flag data, not zeros.
3. **Given** a rebuilt store, **When** the rows carrying a 256px synthesized minimap are compared
   with rows carrying a 1024px synthesized minimap, **Then** the two sets are identical — any tile
   whose 256px synthesis succeeded also has its 1024px counterpart.
4. **Given** the rebuilt stores, **When** finalize and verify run, **Then** every populated
   signal's declared content identity matches its observed identity, and every gap is named with
   a reason (era-unavailable, tile-has-no-source-data), never silently zero-filled.

---

### User Story 2 - Full-Catalog Training Curriculum, Big Maps Only (Priority: P2)

The dataset operator builds a trainer-facing curriculum store that carries every signal in the
frozen catalog (not a legacy field subset), draws rows from Kalimdor and Azeroth only, and
assigns a deterministic within-map stratified holdout from both maps.

**Why this priority**: The first curriculum silently carried 7 of ~18 signals because a legacy
field list was reused, and its first split drew validation mass from maps too small to measure.
The curriculum is the model's entire world; it must present the full corrected corpus.

**Independent Test**: Build the curriculum from the rebuilt stores; verify its signal list equals
the frozen catalog's populated set, its rows come only from Kalimdor and Azeroth, both maps
contribute validation rows, and rebuilding it yields an identical split.

**Acceptance Scenarios**:

1. **Given** rebuilt Kalimdor and Azeroth stores plus their reviewed curation manifests, **When**
   the curriculum is built, **Then** it contains only reviewed keep-rows from those two maps,
   copied bit-for-bit, with full source lineage.
2. **Given** the built curriculum, **When** its signals are listed, **Then** every populated
   catalog signal is present — nothing narrowed to a historical trainer's field list.
3. **Given** two builds of the same curriculum from the same inputs, **When** their splits are
   compared, **Then** they are identical (deterministic), and each of the two maps contributes
   validation rows proportional to its kept-row count.
4. **Given** any curriculum build request including PVPZone02 or Kalidar rows, **When** the build
   runs, **Then** those maps are refused with an explicit message, not silently included.

---

### User Story 3 - Height-First Model with a Relative Target (Priority: P3)

The model operator trains a small, lean model that takes minimap RGB as input and produces
per-tile relative (tile-normalized) height — a target that structurally cannot fail on cross-map
absolute-altitude offsets — and evaluates it against a within-map holdout on the two big maps.

**Why this priority**: This is the goal the dataset exists for, but it is only trustworthy after
US1/US2 make the data honest. The relative-height contract is the explicit correction of the
legacy model's demonstrated structural flaw.

**Independent Test**: Train on the US2 curriculum (user-executed run); validation error improves
over epochs rather than peaking at epoch 1; reconstructed relief on held-out Kalimdor and Azeroth
tiles is visually and numerically superior to a predict-the-tile-mean baseline.

**Acceptance Scenarios**:

1. **Given** the model's target definition, **When** the same terrain shape is presented at two
   different absolute altitudes, **Then** the target values are identical — altitude offset is
   not part of what the model is asked to learn.
2. **Given** a training run on the curriculum, **When** validation metrics are tracked per epoch,
   **Then** the best epoch is not the first epoch, and final validation error beats the
   tile-mean-prediction baseline by a margin recorded in the run summary.
3. **Given** a trained checkpoint, **When** evaluation is requested on any map outside Kalimdor
   and Azeroth, **Then** the tooling refuses or clearly labels the output as out-of-scope —
   small maps never silently enter an evaluation.
4. **Given** the run's summary, **When** it is reviewed, **Then** it records the curriculum
   identity, split mode, target contract version, and baseline comparison — enough to reproduce
   or reject the run without re-deriving context.

---

### Edge Cases

- A tile whose 256px minimap synthesis legitimately failed (no decodable texture) must be absent
  from both resolutions and recorded with a reason — coverage equality (US1) is between the two
  resolutions, not a demand that synthesis never skip.
- A tile with genuinely flat terrain has near-zero height variance; the relative target must
  remain well-defined (no divide-by-near-zero normalization blowups), and such tiles must not
  dominate the validation metric.
- If the extraction stream cannot be made to emit `mcnk_flags_16` for 0.5.3 (an era/format
  limitation discovered during implementation rather than a wiring bug), the catalog — not the
  template — is corrected, with the same era-unavailability treatment as `mccv_rgb`.
- Rebuilding stores must not destroy the existing v50.1 stores until the rebuilt replacements
  pass finalize/verify (the Spec 109 Phase 8 staging discipline applies).
- The legacy spec103/spec108 trainers remain on disk for other lanes; nothing in this spec may
  route new work through them, but nothing deletes them either.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The v50 manifest template MUST be derivable from and consistent with the frozen
  Spec 109 signal catalog; a signal the catalog drops MUST NOT appear in the template, and
  template generation MUST fail loudly on any divergence rather than shipping a contradiction.
- **FR-002**: A signal that cannot exist for a build's era (e.g. `mccv_rgb` before MCCV was
  introduced) MUST be recorded as unavailable-by-era with that reason, and MUST NOT be declared
  as a zero-filled array.
- **FR-003**: `mcnk_flags_16` MUST carry real extracted per-chunk flag data in rebuilt 0.5.3
  stores, or — if extraction proves era-impossible — be reclassified per FR-002 with the finding
  documented; silent zero-fill is not an outcome.
- **FR-004**: For every rebuilt store, the set of rows with a 1024px synthesized minimap MUST
  equal the set of rows with a 256px synthesized minimap; the synthesis pipeline MUST NOT lose
  tiles to concurrent-access failures.
- **FR-005**: Rebuilt stores MUST pass the existing v50 finalize/verify gates (content identity,
  row lineage, fail-closed on mismatch) before replacing prior stores; prior stores MUST remain
  intact until then.
- **FR-006**: The curriculum builder MUST carry every populated signal from the frozen catalog,
  MUST accept only Kalimdor and Azeroth as source maps for this lane (refusing others
  explicitly), and MUST produce deterministic within-map stratified splits with both maps
  represented in validation.
- **FR-007**: The model's height target MUST be invariant to per-tile absolute altitude: adding a
  constant offset to a tile's heights MUST NOT change the target. The normalization MUST remain
  numerically stable on near-flat tiles.
- **FR-008**: The model MUST consume only signals available at inference time from a minimap
  image; ground-truth lighting/time, absolute coordinates, and map identity MUST NOT be inputs.
- **FR-009**: Training MUST be user-executed with an explicit per-run go-ahead; tooling prepares
  and prints exact commands but never launches training itself.
- **FR-010**: Every training run MUST record: curriculum content identity, split mode, target
  contract version, per-epoch validation metrics, and a tile-mean-baseline comparison, in a
  machine-readable summary.
- **FR-011**: Evaluation tooling MUST restrict validation/holdout and reported metrics to
  Kalimdor and Azeroth; any request touching other maps fails closed or is explicitly labeled
  out-of-scope.
- **FR-012**: All stores remain per-build Zarr with the established identity/lineage discipline;
  no NPZ side-channels.

### Key Entities

- **Frozen Signal Catalog**: The Spec 109 authority listing every v50 signal, its policy, and its
  era availability; the single source the manifest template and curriculum contents derive from.
- **Rebuilt Per-Map Store**: A v50.1 complete store for Kalimdor or Azeroth whose declared
  signals are all either populated or explicitly unavailable-with-reason.
- **Full-Catalog Curriculum**: The trainer-facing store: reviewed keep-rows from the two big
  maps, all populated catalog signals, deterministic within-map split, full lineage.
- **Relative-Height Target Contract**: The versioned definition of the tile-normalized height
  representation (and its inverse for reconstruction) that the model trains against.
- **Training Run Summary**: The machine-readable record binding a checkpoint to its curriculum
  identity, split, target contract, metrics, and baseline comparison.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A coverage audit of the rebuilt Kalimdor and Azeroth stores shows zero declared
  signals with 0% population; every declared signal is populated or carries an explicit
  unavailability reason.
- **SC-002**: `minimap_rgb_1024` row coverage equals `minimap_rgb` row coverage on both rebuilt
  stores (100% agreement between the two sets, not merely similar percentages).
- **SC-003**: The curriculum contains only Kalimdor and Azeroth rows, carries every populated
  catalog signal, and two independent rebuilds produce byte-identical splits.
- **SC-004**: A training run on the corrected curriculum reaches its best validation error after
  epoch 5 or later (no epoch-1 peak), and beats the tile-mean-prediction baseline on held-out
  tiles from both maps.
- **SC-005**: Reconstructed held-out tiles from both Kalimdor and Azeroth are judged by the user
  to show genuine relief structure (not washed-out means), using the same side-by-side review
  discipline as prior minimap fidelity gates.
- **SC-006**: No training or heavy extraction run in this spec is executed by the assistant; every
  such run's command appears in the documentation with a time estimate and is user-launched.

## Assumptions

- The 2026-07-18 audit's coverage numbers are representative (50-row samples per store); the
  Phase 1 rebuild re-measures on full stores.
- `mcnk_flags_16` is presumed extractable for 0.5.3 (MCNK chunks carry flags in this era) and its
  0% coverage is presumed a wiring gap, not an era limitation; the edge case above covers the
  alternative.
- The 1024px coverage deficit is presumed to be concurrent-archive-access contention; if root
  cause differs, FR-004's outcome (set equality) still governs.
- Object-inclusive versus strict curation manifests both remain available; this lane's curriculum
  uses the strict (object-free) profile because the target is ground height, which objects
  occlude. Object-aware modeling remains a separate future lane.
- The Spec 102 constitution constraints (no DepthAnything-family, no multi-head/shared-weight
  commitments) apply to Phase 2 architecture selection; growth beyond height is a follow-on
  decision, not designed in now.
- PVPZone02 and Kalidar stores remain on disk untouched; exclusion is a curriculum policy, not a
  deletion.
