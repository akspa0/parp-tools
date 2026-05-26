# Implementation Plan: Object Roof Mask Library and Minimap Sieve

**Feature**: `025-object-roof-mask-library-and-minimap-sieve`

## Constitution Check

- Repo independence: pass, all work stays under `wow-viewer/`
- Dataset-first: pass, the first slice builds a curation library before any new model work
- Validation: required on staged client data and existing corpus stores
- One phase at a time: enforced below

## Phase 1 — Roof Library Curation

Goal: turn existing object placement metadata into a reusable roof exemplar library.

1. Define the roof exemplar schema and provenance fields.
2. Extend MdxViewer capture to render a known-used object asset one at a time with explicit pose/transform metadata.
3. Store the per-asset captures in a separate object-visual Zarr datastore.
4. Deduplicate repeated object families into canonical exemplars plus variants.
5. Emit atlas, catalog, and evidence artifacts for manual QA.

Validation:

- Bounded corpus run emits roof exemplars with stable IDs and asset-path metadata.
- Object-visual Zarr outputs exist for the same sample set and can be queried independently of the terrain stores.
- Catalog includes at least one building-heavy family with reviewable provenance.

## Phase 2 — Object-Roof Mask Generation

Goal: generate object coverage masks for minimap inputs using the curated roof library.

1. Define the object-mask label contract for minimap tiles.
2. Build a mask generator that uses placement metadata when available, with SAM2 as the first promptable host.
3. Train a separate transformer-based object-identification model that learns pose-aware roof / object signals from the curated library.
4. Implement the model in the Python `uv` workflow using the Hugging Face transformers stack as the first host, and keep SAM3 as a gated upgrade path if the Hugging Face token unlocks it.
5. Add a learned fallback path for tiles missing direct placement metadata.
6. Emit review artifacts showing minimap, object mask, provenance, and object-family outputs side by side.

Validation:

- Bounded mask run produces non-empty masks on object-rich tiles.
- Mixed-coverage tiles still process without dropping samples.
- The separate object-identification model outputs stable object-family / pose-aware signals for at least one proof tile.

## Phase 3 — Training Integration

Goal: feed the object-roof signal into training as an auxiliary sieve / ignore mask.

1. Wire the object-roof signal into the normal-lane data pipeline.
2. Preserve raw terrain tensors as the authoritative target.
3. Use the object signal to downweight or sieve object pixels before terrain prediction.
4. Add evidence outputs that prove the object signal was consumed.
5. Feed object-identification outputs into the main V18 model as auxiliary inputs or side channels where useful.

Validation:

- Smoke training run completes with the auxiliary signal enabled.
- Preview artifacts show the model saw the object mask and still trained on terrain truth.

## Phase 4 — Operational Proof

Goal: prove the lane on staged real-data anchors.

1. Run the roof library on at least one building-heavy build/map pair.
2. Run the object-mask generator on a known object-rich tile.
3. Run a bounded training comparison with and without the auxiliary signal.
4. Record the outcome and update continuity docs.

Validation:

- Evidence package shows stable catalog, useful masks, and a bounded training comparison.
