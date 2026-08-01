# Data Model: Canonical Dataset Curation and Signal-Mismatch Bucketing

**Feature**: 122-dataset-curation | **Date**: 2026-07-30 | **Spec**: [spec.md](spec.md)

This document defines the entities and on-disk schemas for Spec 122. It references decisions from
[research.md](research.md) by id (D-0n). New artifacts are written alongside an existing v50
store's own `index.parquet`; the store's signal arrays (Zarr) and existing `index.parquet` are
read-only inputs (FR-014) — curation never writes into them.

---

## Entities

### Tile Curation Record

The per-tile classification result. One row in `curation_manifest.parquet` per tile that exists in
the source store's `index.parquet` — full coverage is mandatory (FR-008, SC-006): a tile with zero
findings and a "clean" bucket is still a row, not an absence.

| Field | Type | Notes |
|---|---|---|
| `build` | string | matches the source store's row identity |
| `map` | string | matches the source store's row identity |
| `tile_x`, `tile_y` | int32 | matches the source store's row identity |
| `source_row_index` | int64 | index into the source store, for O(1) join back to signal arrays |
| `difficulty_bucket` | string enum | `easy` \| `medium` \| `hard` \| `pathological` (ports `v16_curation.DIFFICULTY_BUCKETS`) |
| `coverage_bucket` | string enum | `well_covered` \| `low_coverage` \| `blank` (ports `is_blank_what_plate` + `mcly_painted_coverage`) |
| `lighting_bucket` | string enum | `matched` \| `low_confidence_ambiguous` \| `low_confidence_flat_terrain` \| `not_evaluated` (ports `MinimapShadingMatch`/`spec111/lighting_buckets.py` verbatim status vocabulary — D-04 keeps the exact existing strings so no downstream consumer's string comparison silently breaks) |
| `synthetic_fidelity_status` | string enum | `evaluated` \| `not_evaluable` — whether a fidelity finding exists for this tile (US3) |
| `synthetic_fidelity_score` | float32, nullable | the best-candidate correlation `MinimapShadingMatch` already computes (0.0-1.0), null when `not_evaluable` |
| `finding_count` | int32 | denormalized count of rows this tile has in `curation_findings.parquet`, so "does this tile have any findings" doesn't require a join |
| `curation_run_id` | string | binds this row to the `CurationRunRecord` that produced it (FR-012 reproducibility, FR-013 auditability) |

**Validation rules**:

- Every `source_row_index` present in the source store's `index.parquet` MUST have exactly one
  corresponding row here (FR-008; enforced as a hard gate in the C# writer, not just tested —
  SC-006).
- `difficulty_bucket` and `coverage_bucket` are independent axes on the same tile (a tile can be
  simultaneously `easy` and `low_coverage`) — they are separate columns, never collapsed into one
  combined label (spec Edge Cases, FR-010).
- A bucket value is never inferred when its backing check could not run; `lighting_bucket` uses
  `not_evaluated` (not a guessed value) exactly as `MinimapShadingMatch` already does for
  out-of-build-scope tiles (FR-011).

### Mismatch Finding

A specific detected problem on a tile. One row in `curation_findings.parquet` per (tile, finding) —
a tile with zero findings has zero rows here (not a null row), a tile with three findings has three
rows (FR-010).

| Field | Type | Notes |
|---|---|---|
| `build`, `map`, `tile_x`, `tile_y`, `source_row_index` | as above | joins back to `curation_manifest.parquet` and the source store |
| `category` | string enum | `height_normal_mismatch` \| `non_finite_value` \| `has_flag_mismatch` \| `synthetic_fidelity_gap` (ports `mismatch_detector.py`'s reason vocabulary plus the new US3 category) |
| `severity` | string enum | `none` \| `low` \| `medium` \| `high` (ports `mismatch_detector._severity` exactly — D-04 keeps the existing 4-level vocabulary) |
| `reason` | string | human-readable, e.g. `"height_flat_vs_normal_varied"`, `"insufficient_normal_coverage"` — ports `mismatch_detector.py`'s exact reason strings where the check is equivalent, so the SC-003 comparison is a direct string/set diff, not a fuzzy remap |
| `evaluability` | string enum | `evaluated` \| `not_evaluable` — distinct from severity=`none`; `not_evaluable` means the check could not run (missing dependent signal, out-of-scope build), `severity=none` means it ran and found nothing (FR-011, spec Edge Cases) |
| `signal` | string, nullable | which source signal(s) the finding concerns, e.g. `"normal_xyz"`, `"minimap_rgb_authored"` — free text, not an enum, since new checks may reference new signal names without a schema migration |
| `curation_run_id` | string | as above |

**Validation rules**:

- A finding with `evaluability=not_evaluable` MUST NOT carry a `severity` other than a sentinel
  that a consumer cannot mistake for "checked and clean" — the writer sets `severity=none` only
  when `evaluability=evaluated`; for `not_evaluable` rows `severity` is written as the literal
  string `not_evaluable` too, so no downstream filter can accidentally conflate "not checked" with
  "checked, no problem" by filtering on `severity` alone (this directly closes the ambiguity the
  spec's Edge Cases section calls out for missing-signal checks).

### Curation Manifest

The pair of Parquet tables above, considered together as one versioned artifact per store
(D-02). Not a single file — the two tables plus the run record together are "the manifest."

| Artifact | Path convention |
|---|---|
| `curation_manifest.parquet` | `<store root>/curation/<curation_run_id>/curation_manifest.parquet` |
| `curation_findings.parquet` | `<store root>/curation/<curation_run_id>/curation_findings.parquet` |
| `curation_run.json` | `<store root>/curation/<curation_run_id>/curation_run.json` |
| `<store root>/curation/latest` | a plain-text pointer file containing the most recent `curation_run_id`, so downstream tooling that wants "the current manifest" doesn't need to enumerate run directories — re-running `curate` never overwrites a prior run in place (FR-012 reproducibility stays inspectable across reruns) |

### Curation Run Record

Small JSON provenance record, schema `v50-curation-run-v1` — mirrors the existing
`v50-model-stage-run-v1` convention (D-03) but describes a classification pass, not a training run.

| Field | Type | Notes |
|---|---|---|
| `schema` | const | `"v50-curation-run-v1"` |
| `curation_run_id` | string | unique id for this invocation; referenced by every row's `curation_run_id` |
| `store_path` | string | the v50 store this run classified |
| `build_fingerprint` | string | the client build identity, matching this repo's existing fingerprint convention |
| `checks_run` | list[string] | which bucket dimensions and finding categories actually executed (a check can be skipped entirely if its store lacks the backing signal, distinct from running-and-finding-nothing) |
| `tile_count` | int | total tiles classified — MUST equal the source store's row count (SC-006 gate, checked by the writer before it will report success) |
| `bucket_counts` | map[string, map[string, int]] | per-bucket-dimension, per-value tile counts, e.g. `{"difficulty_bucket": {"easy": 400, "hard": 12, ...}}` |
| `finding_counts` | map[string, int] | per-category finding counts across the whole run |
| `tool_version` | string | the C# curation library's version/build identity, for reproducibility (FR-012) |
| `created_at` | timestamp | |

### Selection Record

Not a new stored artifact — an existing-pattern requirement (FR-013) that any downstream consumer
selecting a subset of buckets records that selection in **its own** run record, alongside a
reference to the `curation_run_id` it read. This feature does not define a new schema for this;
future trainer specs are expected to add a `curation_selection` block (bucket filter used, tile
counts selected/excluded, `curation_run_id` referenced) to whatever run-record schema they already
use (`v50-model-stage-run-v1` or successor) — the requirement is that the reference exists and is
reconstructable, not that this feature owns a new artifact type for it.

---

## Query Contract (FR-009: every bucket equally accessible)

No new query engine is introduced. "Query by bucket or finding" means: read
`curation_manifest.parquet` (or `curation_findings.parquet`) with `pyarrow`/`pandas` and filter by
column — exactly as `mismatch_detector.py`'s own `MismatchReport.to_parquet()` output and
`load_curation_keys()` in `v16_curation.py` are already read today. The design commitment (US2) is
that this same, trivial column-filter operation works identically regardless of which bucket value
is requested — there is no separate "recovery" code path for non-clean buckets, because there is
only ever one code path: read the table, filter a column.
