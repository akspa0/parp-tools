# Minimap Lighting Calibration Quickstart

## Phase 1 automated proof (shading-match inference)

From `I:\parp\parp-tools`:

```powershell
dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MinimapShadingMatchTests|FullyQualifiedName~MinimapLightingProvenanceTests|FullyQualifiedName~TerrainSolarDirectionTests"
dotnet build wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug
```

## Dataset-wide bucketing pass (0.5.3.3368 only)

User-run against a configured client root; reads every dataset tile with both an authored minimap and
ground-truth terrain for the selected build, streams shading-match results into the existing per-build
Zarr store, and writes a distribution report. This is a bulk pass over the whole 0.5.3.3368 corpus and
can take a while -- start bounded before running the whole build.

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -- \
  minimap-lighting-calibrate --client-root "H:\CLIENTS\<0.5.3.3368-build>" \
  --zarr-store "<existing 0.5.3.3368 store path>" --limit 5
```

For the first bounded run, inspect a handful of `matched` tiles by eye: render the winning
`ShadingMatchedTimeOfDayHours` candidate next to the real authored minimap and confirm the shadow
direction actually lines up, the same way today's real-vs-synthesized side-by-side caught the original
sun-direction bug. Only then remove `--limit` for a full-build pass.

```powershell
uv run --directory wow-viewer/data-harvester python scripts/report_lighting_buckets.py \
  --zarr-store "<0.5.3.3368 store path>" --map "<optional single map, else whole build>"
```

Confirm `sum(BucketCounts) + NotEvaluatedCount + LowConfidenceCount == TotalEligibleTiles` in the
report before trusting it for Phase 2.

## Phase 2 rebalancing check

```powershell
uv run --directory wow-viewer/data-harvester python -m pytest tests/spec111/test_lighting_bucket_rebalancing.py
uv run --directory wow-viewer/data-harvester python scripts/rebalance_lighting_variants.py \
  --distribution-report "<Phase 1 report>" --dry-run
```

`--dry-run` prints the resulting per-bucket sampling weights and flags any `no_real_baseline` buckets
without touching the actual training-variant generation; inspect this before wiring it into a real
training config.

## Phase 3 retrain and evaluate -- requires explicit go-ahead

Do not run `train_spec111_reconstruction.py` without first confirming with the user that this is the
moment to spend the GPU/cloud time. This is the same discipline already used for Spec 108's training
steps. When authorized:

```powershell
uv run --directory wow-viewer/data-harvester python scripts/train_spec111_reconstruction.py \
  --rebalanced-config "<Phase 2 output>"
```

After training, the checkpoint comparison against the currently deployed model must show
`Outcome != regressed` before any promotion is considered.
