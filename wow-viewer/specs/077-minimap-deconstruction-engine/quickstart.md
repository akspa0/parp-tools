# Spec 077 Quickstart — Per-Object Capture Library

This quickstart walks through the first end-to-end run of the spec 077
per-object capture library. It is intentionally Python-first: the
enumerator, builder, and reviewer all run from the data-harvester and
need nothing more than a V18 Zarr store (or a mock capture directory for
the CI integration test).

## Prerequisites

- `uv` is installed and the harvester environment is bootstrapped:

  ```bash
  cd wow-viewer/data-harvester
  uv sync
  ```

- A V18 Zarr store exists under `wow-viewer/output/datasets/v18/`. If
  you do not have one, the integration test
  `tests/test_object_library_e2e.py` proves the full pipeline with
  synthetic placements and capture artifacts.

## End-to-end run on real data

1. **Enumerate capture jobs** from a V18 build:

   ```bash
   cd wow-viewer/data-harvester
   uv run python scripts/enumerate_object_capture_jobs.py \
       --build 3_3_5_12340 \
       --include-modf \
       --output ../../output/datasets/object-library/jobs_3_3_5_12340.jsonl
   ```

   This reads `output/datasets/v18/3_3_5_12340.zarr/placements.parquet`
   and `index.parquet`, collapses to one job per (instance_type,
   normalized asset path), and writes a JSONL. Clutter/tree assets are
   skipped by default; pass `--keep-clutter` to keep them.

2. **Capture each asset** (currently a manual step). The C# capture
   tool will eventually emit `<variant_id>_image.png`,
   `<variant_id>_mask.png`, and `<variant_id>_pose.json` into a flat
   directory. The `variant_id` is the SHA1-16 hex prefixed with
   `objvar_`; the builder computes it from the library id + build + pose
   using the C#/Python contract in `data-harvester/src/harvester/object_library.py`.

3. **Build the library**:

   ```bash
   cd wow-viewer/data-harvester
   uv run python scripts/build_object_library.py \
       --jobs ../../output/datasets/object-library/jobs_3_3_5_12340.jsonl \
       --captures-dir /path/to/captures \
       --output-root ../../output/datasets/object-library \
       --run-name smoke_3_3_5_12340 \
       --target-size 128
   ```

   This writes
   `wow-viewer/output/datasets/object-library/smoke_3_3_5_12340.zarr/`
   with `capture_rgb/`, `capture_mask/`, `assets.parquet`,
   `index.parquet`, and group-level `metadata.json`. Jobs without
   matching capture artifacts become `not_attempted` entries — they are
   never silently dropped.

4. **Render review artifacts**:

   ```bash
   cd wow-viewer/data-harvester
   uv run python scripts/review_object_library.py \
       --library ../../output/datasets/object-library/smoke_3_3_5_12340.zarr \
       --output-dir ../../output/analysis/object-library/smoke_3_3_5_12340
   ```

   Open `wow-viewer/output/analysis/object-library/smoke_3_3_5_12340/index.html`
   to see per-family contact sheets and the top-level library summary.

## CI integration test (no client data required)

```bash
cd wow-viewer/data-harvester
uv run pytest tests/test_object_library.py tests/test_object_library_e2e.py -q
```

`test_object_library_e2e.py` stages three synthetic capture jobs (two
captured, one with no artifacts), runs the builder and reviewer, and
verifies that:

- The Zarr store has the expected shape and the two captured entries
  have non-zero masks.
- `assets.parquet` shows two `captured` and one `not_attempted` entry.
- The reviewer writes one contact sheet per family.

## Where things live

| Concern | Path |
|---------|------|
| C# data contracts | `wow-viewer/src/core/WowViewer.Core/Maps/ObjectLibraryEntry.cs`, `ObjectCaptureVariant.cs` |
| C# tests | `wow-viewer/tests/WowViewer.Core.Tests/ObjectLibraryContractsTests.cs` |
| Python contract | `wow-viewer/data-harvester/src/harvester/object_library.py` |
| Python tests | `wow-viewer/data-harvester/tests/test_object_library.py`, `test_object_library_e2e.py` |
| Enumerator | `wow-viewer/data-harvester/scripts/enumerate_object_capture_jobs.py` |
| Builder | `wow-viewer/data-harvester/scripts/build_object_library.py` |
| Reviewer | `wow-viewer/data-harvester/scripts/review_object_library.py` |

## Open Phase 2 work

- **T007** xUnit tests for a future C# writer (T009 C# side).
- **T009 C# side** object-library Zarr/Parquet writer. The current
  Python path is sufficient for the first proof; add a C# writer when
  the C# capture lane (T010) needs to publish directly.
- **T010** one-object-at-a-time capture extension in
  `WowViewer.Tool.ValidationCapture`. Until that lands, the builder
  consumes whatever the operator stages in the captures directory.
