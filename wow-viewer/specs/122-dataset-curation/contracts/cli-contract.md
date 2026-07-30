# CLI Contract: Canonical Dataset Curation and Signal-Mismatch Bucketing

**Feature**: 122-dataset-curation | **Date**: 2026-07-30

One new subcommand on the existing `WowViewer.Tool.Harvest` (D-01). Dry-run/print-plan by default
(D-03); requires an explicit `--write` to persist anything. No Python CLI is added by this
feature — the Python-side changes are library-level thin readers (`harvester/curation_store.py`)
consumed directly, not new user-facing commands.

---

## `curate` (C#, `WowViewer.Tool.Harvest curate`)

```text
WowViewer.Tool.Harvest curate
  --clients-root <path>            # configured client library root (constitution VI — never hardcoded)
  --build <build id>                # e.g. 0_5_3_3368
  --store <path to v50 store>       # the store whose tiles get classified
  [--checks <comma-separated list>] # subset of check categories to run; default = all available for this store's signals
  [--map <map name>]                # optional: scope to one map (still writes full-coverage output for that scope)
  [--write]                         # persist curation_manifest.parquet / curation_findings.parquet / curation_run.json
```

- **Default (no `--write`)**: reads the store's `index.parquet` and signal arrays read-only, prints
  the planned tile count, which checks will run (and which are skipped because their backing
  signal is absent from this store — printed explicitly, not silently dropped), and the output
  paths that would be written. Exits without writing anything (matches every dry-run-first CLI in
  this repository).
- **With `--write`**: runs every planned check over every tile, writes
  `curation_manifest.parquet`, `curation_findings.parquet`, and `curation_run.json` under
  `<store>/curation/<curation_run_id>/`, and updates the `<store>/curation/latest` pointer.
  Verifies `tile_count` in the run record equals the source store's row count before reporting
  success (SC-006) — a partial write is a failure, not a partial success.
- A tile whose backing signal for a given check is absent (e.g. no `normal_xyz` for the
  height-normal-mismatch check, no `minimap_rgb_authored` for the synthetic-fidelity check) still
  gets a row: the check's finding is written with `evaluability=not_evaluable` (FR-011), never
  silently omitted from `curation_findings.parquet` and never guessed.
- Re-running `curate --write` against an unchanged store with the same tool version produces
  bit-identical `bucket_counts`/`finding_counts` in a new `curation_run_id` (FR-012); it never
  overwrites a prior run's directory.
- `--checks` exists so a future check addition (or a re-run limited to just the newly-added check,
  e.g. after a bug fix in one detector) does not require re-running every other check — the run
  record's `checks_run` field always states exactly what ran, whether or not `--checks` was passed.

---

## Python-side read access (no new CLI, library-only)

```python
from harvester.curation_store import load_curation_manifest, load_curation_findings

manifest = load_curation_manifest(store_path)          # -> pandas/pyarrow table, one row per tile
findings = load_curation_findings(store_path)           # -> table, one row per finding

clean_tiles = manifest[manifest.coverage_bucket == "well_covered"]
mismatched_findings = findings[findings.category == "height_normal_mismatch"]
```

- Both loaders resolve `<store>/curation/latest` by default; an explicit `curation_run_id` may be
  passed to pin a specific run instead of "most recent."
- Neither loader filters, drops, or reorders rows relative to what `curate --write` produced — this
  is the FR-009/US2 guarantee expressed as code: querying `pathological` tiles is the identical
  operation (a column filter on the same table) as querying `clean` tiles.
- These are thin readers only (D-04); `v16_curation.py` and `mismatch_detector.py` are updated to
  call through this module rather than recomputing bucket/mismatch logic themselves, once the
  SC-003 comparison (below) passes.

---

## Legacy comparison command (one-time, SC-003 gate, not a permanent CLI)

Before `mismatch_detector.py` is converted to a thin reader (D-04), run its existing detection logic
and the new C# `HeightNormalMismatchDetector` against the same real store and diff the flagged sets.
This is a one-time validation script (`data-harvester/scripts/spec122_compare_legacy_mismatch.py`
or equivalent), not a permanent part of the curation contract — it exists to produce the SC-003
comparison report, then its job is done.

```text
uv run python scripts/spec122_compare_legacy_mismatch.py \
  --store <v50 store already classified by `curate --write`> \
  [--report <output path>]
```

- Prints/writes a diff: tiles flagged by the legacy detector but not the new one, and vice versa,
  with the reason each disagreed (if any). A clean match (or a documented, justified improvement —
  e.g. the new detector catching a case the old one missed) is required before D-04's migration
  proceeds for that script.
