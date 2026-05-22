# Implementation Plan: V16.2 Patched Signal Expansion

**Spec**: `011-v16-2-patched-signal-expansion/spec.md`
**Created**: 2026-05-22

## Phase 1: Contract And Signal Inventory

**Goal**: Freeze the `V16.2` dataset contract before touching stores so the
patch path has one authoritative signal list and metadata vocabulary.

### Step 1.1 — Promote the dataset contract from V16 to V16.2
Update the active dataset/model docs so `V16.2` becomes the named contract for
the richer compact corpus rather than leaving the new signals as side artifacts.
**Validation**: the active dataset spec and `V16.2` spec agree on the upgraded
contract and the new signal families.

### Step 1.2 — Enumerate the upgraded signal set
Write the exact `V16.2` signal inventory, including the current precise-mask
family, renderer-truth visibility surfaces, and terrain-only guidance surfaces
that will be patchable into sidecar stores.
**Validation**: the signal list is explicit enough that loader and patch work
do not need to guess names or semantics.

### Step 1.3 — Define compatibility rules with older object-mask surfaces
Specify which object-related arrays remain coarse compatibility signals, which
arrays are the richer precise surfaces, and how trainers should interpret them
without deleting older useful signals.
**Validation**: the contract distinguishes legacy/coarse versus upgraded/precise
surfaces unambiguously.

### Step 1.4 — Freeze the current proof boundary
Record that real renderer-truth validation currently exists only for
`0_5_3_3368` and `3_3_5_12340` on the bounded Azeroth tile proof roots.
**Validation**: continuity/docs no longer imply six-build closure for the new
signal lane.

## Phase 2: Sidecar Store Schema

**Goal**: Make the existing compact stores structurally ready for `V16.2`
signals without requiring full corpus regeneration or immediate mutation of the
finalized V16 base stores.

### Step 2.1 — Define the sidecar store layout for new arrays
Add the new `V16.2` arrays to a compact sidecar-store contract with exact
shapes, dtypes, and optional-coverage semantics.
**Validation**: the schema can describe a base-plus-sidecar build even before
every build has complete upgraded coverage.

### Step 2.2 — Extend metadata and presence flags
Define the `index.parquet` and validation metadata additions needed to report
per-tile or per-build presence for the new signals.
**Validation**: the metadata contract exposes new-signal coverage without an
external manual manifest.

### Step 2.3 — Define resumable sidecar patch state rules
Specify how interrupted patch runs report partial progress so an incomplete
upgrade does not falsely look complete.
**Validation**: the contract includes restart-safe or resumable semantics for
the patch path.

## Phase 3: Sidecar Patch-And-Reindex Workflow

**Goal**: Upgrade existing Zarr stores by adding only the new signal surfaces
in sidecars and refreshing metadata rather than rebuilding already-correct data
or disturbing the finalized V16 base corpus.

### Step 3.1 — Add bounded sidecar signal patch commands
Extend the dataset tooling with a patch surface that can write the new precise
mask and terrain-guidance arrays into additive sidecar stores.
**Validation**: one build store can be patched without rewriting unrelated base
arrays.

### Step 3.2 — Reindex metadata after patching
Refresh `index.parquet`, presence flags, and validation summaries so loaders and
audits immediately see the upgraded contract.
**Validation**: the reindexed base-plus-sidecar build reports the new signals
directly after the patch run.

### Step 3.3 — Preserve compactness expectations
Measure the storage delta of the upgraded arrays and confirm the sidecar path
keeps the compact-corpus win instead of silently exploding disk usage.
**Validation**: a before/after store-size comparison is recorded for at least
one representative build.

## Phase 4: Cross-Build Validation Matrix

**Goal**: Prove the renderer-truth signal lane on the intended client matrix
before any later merge of sidecar signals back into the base V16 stores.

### Step 4.1 — Run the remaining real capture proofs
Extend the same bounded validation capture path across `0_5_5_3494`,
`0_7_0_3694`, `3_0_1_8303`, and `4_0_0_11927`.
**Validation**: a six-build coverage matrix exists with concrete artifact roots.

### Step 4.2 — Gate merge-on-success semantics
Define that any future merge of sidecar signals into the canonical base stores
depends on the build matrix being complete enough for the targeted model lane.
**Validation**: the plan distinguishes sidecar proof from later base-store
promotion.

## Phase 5: Signal Generation And Ingestion

**Goal**: Wire the current new-signal sources into the patch path so the stores
can actually receive the upgraded data.

### Step 5.1 — Ingest precise-mask sources
Map the current precise object-mask sources into the `V16.2` patch workflow so
the upgraded sidecars capture the richer object separation data.
**Validation**: a patched tile exposes the new precise-mask arrays and their
coverage metadata.

### Step 5.2 — Ingest renderer-derived terrain guidance
Map renderer-truth artifacts such as `no_object_minimap` into the `V16.2`
sidecar stores in a form that is aligned and usable by training.
**Validation**: a patched tile exposes aligned terrain-guidance arrays plus the
corresponding presence metadata.

### Step 5.3 — Define fallback behavior for mixed coverage
Keep the corpus usable when some builds or tiles lack the new artifacts.
**Validation**: patched and unpatched tiles can coexist in the same base-plus-
sidecar build without breaking loaders or validators.

## Phase 6: Loader And Trainer Upgrade

**Goal**: Let training consume the richer `V16.2` corpus directly from the same
compact base-plus-sidecar stores.

### Step 6.1 — Upgrade dataset loaders to read V16.2 signals
Extend the dataset layer so training code can load the new precise-mask and
terrain-guidance arrays from patched base-plus-sidecar stores.
**Validation**: a dataset smoke read prints or records the new signal presence
for patched tiles.

### Step 6.2 — Upgrade the normal-lane contract first
Use the normal lane as the first bounded consumer of the `V16.2` guidance and
precise-mask surfaces without redefining raw terrain tensors as rendered truth.
**Validation**: a bounded smoke training run consumes the upgraded signals and
writes review artifacts that make the new inputs visible.

### Step 6.3 — Keep compatibility fallbacks explicit
Preserve a clear fallback path so training remains runnable on partial-coverage
stores and older tiles during the transition.
**Validation**: one smoke run succeeds on a mixed-coverage sample set.

## Phase 7: Corpus Validation On Existing Stores

**Goal**: Prove the patch-and-reindex workflow works on real existing stores,
not just on synthetic or empty test cases.

### Step 7.1 — Patch one representative build sidecar
Run the full `V16.2` upgrade path on one existing build as the first real proof
surface.
**Validation**: the chosen build gains the new arrays and updated metadata
without a full rebuild or base-store rewrite.

### Step 7.2 — Validate spatial alignment and review artifacts
Inspect patched outputs so the new precise masks and terrain-guidance surfaces
are visibly aligned with the existing terrain data.
**Validation**: review artifacts exist and show patched-signal alignment on real
tiles.

### Step 7.3 — Expand to the remaining stores
After the first build proof, apply the same patch-and-reindex path across the
rest of the retained corpus.
**Validation**: all target stores report `V16.2` signal coverage through their
updated metadata.

## Phase 8: Operator Workflow And Handoff

**Goal**: Leave the upgrade lane with exact commands, proof gates, and routing
so later chats continue from `V16.2` instead of reopening the contract.

### Step 8.1 — Publish operator patch commands
Write the hand-runnable commands for locating new signal sources, patching an
existing store, reindexing it, and validating the result.
**Validation**: the commands are documented and map to real workflow seams.

### Step 8.2 — Publish training guidance for upgraded stores
Document how a trainer should consume the `V16.2` signal set and what fallback
behavior applies when some new signals are missing. Also document the first
orchestration slice as a manifest-driven build and signal selector instead of a
hard-coded one-model pipeline.
**Validation**: operator docs clearly distinguish required versus optional
upgraded signals and the orchestration seam is explicit enough for a fresh
chat.

### Step 8.3 — Sync continuity and stop
Update continuity so fresh chats treat `V16.2` as the active corpus-upgrade
lane for precise masks, terrain-only guidance, and patch-only store expansion.
**Validation**: continuity surfaces point future work at the `V16.2` spec and
plan instead of the older narrower drafts.