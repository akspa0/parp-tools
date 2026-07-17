# Minimap Lighting Calibration Quickstart

## Phase 1 automated proof (shading-match inference)

From `I:\parp\parp-tools`:

```powershell
dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MinimapShadingMatchTests|FullyQualifiedName~MinimapLightingProvenanceTests|FullyQualifiedName~TerrainMinimapCompositorTests|FullyQualifiedName~TerrainSolarDirectionTests"
dotnet build wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug
```

Current proof: 42/42 focused C# tests; Debug Harvest build 0 errors.

## Dataset-wide bucketing pass (0.5.3.3368 only) — v50 lane

The active dataset lane is **v50** (Spec 109). The shading-match inference is not a separate
command: the C# harvester computes it at extraction time, chained onto the tint-based analysis in
`AnalyzeAuthoredMinimapLighting` (`WowViewer.Tool.Harvest`), gated internally on the exact
0.5.3.3368 build fingerprint. Any other build's tiles pass through untouched with
`shading_match_status = not_evaluated`.

**Spec 109 dependency, stated plainly**: the clean-room V50 per-build store builder does not exist
yet (`v50_build_dataset.py` fails closed by design), so there is no V50 store to bucket today. Two
integration requirements land on Spec 109's builder when it is implemented: (1) its
fresh-extraction path must use a stream profile that performs the full texture decode, or the
shading-match analysis is skipped (`analysis_requires_full_texture_decode`); (2) it must carry
`minimap_lighting` — including the shading-match fields — as one of its DatasetSignals.

Until then, a bounded harvester run (its full-texture-decode profile is the transport detail, not
a dataset-lane choice) exercises the C# inference end-to-end against the real client:

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -- \
  harvest-stream --client-root "H:\CLIENTS\<0.5.3.3368-build>" --map "<MapName>" \
  --stream-profile v22 | <existing Python Zarr writer entry point>
```

For the first bounded run, inspect a handful of `matched` tiles by eye: render the winning
`shading_matched_time_of_day_hours` candidate with `synthetic-minimap --time-hours <that hour>`
next to the real authored minimap and confirm the shadow direction actually lines up -- the same
side-by-side method that caught both original sun-direction bugs. Only then run the whole build.

```powershell
uv run --directory wow-viewer/data-harvester python scripts/report_lighting_buckets.py \
  --store-path "<0.5.3.3368 store path>" --map "<optional single map, else whole build>"
```

The report generator enforces `sum(bucket_counts) + not_evaluated_count + low_confidence_count ==
total_eligible_tiles` and fails loudly if any tile falls out of reconciliation. Tiles from stores
written before this feature carry no shading-match field at all and are surfaced separately as
`tiles_without_shading_match_field`, never silently folded into "not evaluated".

## Phase 2 rebalancing check

```powershell
uv run --directory wow-viewer/data-harvester python -m pytest tests/spec111/
uv run --directory wow-viewer/data-harvester python scripts/rebalance_lighting_variants.py \
  --distribution-report "<Phase 1 report.json>" --dry-run
```

`--dry-run` prints the per-bucket sampling weights, the resulting `lighting_times` allocation, and
flags any `no_real_baseline` buckets without writing anything. The `lighting_times` list feeds the
existing `spec103_build_synthetic_store.py` generator unchanged -- that generator retains sole
ownership of source-group/variant leak-safety tagging, and the plan's values are bare normalized
floats only (the input-contract check in `test_lighting_bucket_rebalancing.py` pins this).

## Phase 3 retrain and evaluate -- requires explicit go-ahead

`train_spec111_reconstruction.py` targets the **v50 lane** by delegating to the canonical
`v50_train_wdl_prior.py` entry point (default `--release v50.1`; the wrapped trainer enforces
`require_store_release`, so a non-v50 store fails closed). Spec 102's chain is BLOCKED on its M0
reharvest and is not a valid target. Running without `--confirm-run` only validates the
configuration and prints the delegated command -- it never starts a GPU run:

```powershell
uv run --directory wow-viewer/data-harvester python scripts/train_spec111_reconstruction.py \
  --rebalanced-plan "<Phase 2 plan.json>" --store "<mixed paired store>" \
  --output "<candidate.pt>" --baseline-checkpoint "<currently deployed checkpoint.pt>"
```

Only after the user explicitly authorizes the GPU run at that moment, re-run with `--confirm-run`.
After training, compute the held-out comparison through
`harvester.spec111.checkpoint_comparison.compare_checkpoints`; `promotion_decision` is True only
for a clear improvement -- regressed and inconclusive outcomes both keep the current checkpoint.
