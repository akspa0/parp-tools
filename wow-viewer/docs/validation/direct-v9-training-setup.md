# Direct V9 Training Setup

This is the current reproducible setup for training the direct `v9` terrain model from `wow-viewer` without routing the main corpus path back through harvested-dataset ownership.

All examples below assume a Bring Your Own Data workflow. Replace the placeholder paths with equivalents on your own machine.

The working shape today is:

1. Stage archive-backed game clients locally.
2. Build the main training cache from real game roots with `wow-viewer/scripts/run_v9_direct_pipeline.ps1`.
3. Build a separate development-map compatibility cache from `datasets/original_development/development`.
4. Split that development cache into PM4-bearing training additions and a non-overlapping development holdout.
5. Train `train_v9_optimized.py` with the merged corpus and the non-PM4 development holdout.

Important boundary:

- the direct wrapper is still the canonical way to build the main direct cache
- the wrapper still launches `train_v9.py` by default
- the current recommended PM4-mixed branch uses the wrapper for cache generation, then launches `train_v9_optimized.py` directly for the actual training run

This stays Bring Your Own Data. Do not ship client data, generated corpora, model weights, or model outputs derived from proprietary data.

## Why These Inputs

- `3_0_1_8303/Northrend` is now a required dataset source because it carries terrain and content patterns that were later copied into the development map.
- a development-map dataset root such as `datasets/original_development/development` is the current best held-out validation surface because it is sparse, important, and includes trusted tiles such as `development_0_0`.
- Early pre-release builds still have known minimap and compatibility gaps, so the current broad training flow should keep minimap and WDL gates relaxed unless a narrower experiment is explicitly proving those gates.

## Prerequisites

- A Python environment with the training dependencies installed.
- A local checkout of this repository.
- Access to the required game builds from your own archive mount, extracted client folders, or other lawful BYOD source.
- A fast local staging directory for repeated multi-build scans.
- An existing development-map dataset root for the held-out dev-eval cache.

The examples below use these PowerShell variables:

```powershell
$RepoRoot = 'C:/path/to/parp-tools'
$PythonExe = Join-Path $RepoRoot '.venv/Scripts/python.exe'
$ArchiveRoot = 'X:/path/to/WoWArchive/Mount'
$StagingRoot = Join-Path $RepoRoot 'output/tmp/wowarchive-clients'
$DevelopmentDatasetRoot = Join-Path $RepoRoot 'datasets/original_development/development'
$OutputRoot = Join-Path $RepoRoot 'output/ml-training'
```

## 1. Stage The Required Clients

The current recommended training set is:

- `0_5_3_3368`
- `0_5_5_3494`
- `0_7_0_3694`
- `3_0_1_8303`
- `3_3_5_12340`
- `4_0_0_11927`

Example staging command:

```powershell
$pairs = @(
  @{ Label='0_5_3_3368'; Source=(Join-Path $ArchiveRoot '0.X_Pre-Release_Windows_enUS_0.5.3.3368/World of Warcraft'); Target=(Join-Path $StagingRoot '0_5_3_3368/World of Warcraft') },
  @{ Label='0_5_5_3494'; Source=(Join-Path $ArchiveRoot '0.X_Pre-Release_Windows_enUS_0.5.5.3494/World of Warcraft'); Target=(Join-Path $StagingRoot '0_5_5_3494/World of Warcraft') },
  @{ Label='0_7_0_3694'; Source=(Join-Path $ArchiveRoot '0.X_Pre-Release_Windows_enUS_0.7.0.3694/World of Warcraft'); Target=(Join-Path $StagingRoot '0_7_0_3694/World of Warcraft') },
  @{ Label='3_0_1_8303'; Source=(Join-Path $ArchiveRoot '3.X_Pre-Release_Windows_enUS_3.0.1.8303/World of Warcraft'); Target=(Join-Path $StagingRoot '3_0_1_8303/World of Warcraft') },
  @{ Label='3_3_5_12340'; Source=(Join-Path $ArchiveRoot '3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft'); Target=(Join-Path $StagingRoot '3_3_5_12340/World of Warcraft') },
  @{ Label='4_0_0_11927'; Source=(Join-Path $ArchiveRoot '4.X_Beta_Windows_enUS_4.0.0.11927/World of Warcraft'); Target=(Join-Path $StagingRoot '4_0_0_11927/World of Warcraft') }
)

foreach ($pair in $pairs) {
    if (Test-Path $pair.Target) {
        Write-Host "[exists] $($pair.Label) => $($pair.Target)"
        continue
    }

    $parent = Split-Path -Parent $pair.Target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Write-Host "[copy] $($pair.Label)"
    Copy-Item -Path $pair.Source -Destination $pair.Target -Recurse -Force
}
```

If one build is not present under your archive mount, replace that `Source` with any equivalent local BYOD client copy. The important requirement is the build content, not the exact folder layout used above.

## 2. Build The Development-Map Compatibility Cache

The development map is still a compatibility-fed validation surface, so build its cache with the legacy compatibility builder rather than pretending it is already part of the direct client-root pipeline.

That root already includes the loose-development-derived object surfaces needed for evaluation, including `object_visibility_mask`, `pm4_mask`, `no_object_minimap`, and `terrain_only_minimap` on object-bearing tiles such as `development_16_32` and `development_31_36`.

```powershell
& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/build_v9_native_tensor_cache.py') `
  $DevelopmentDatasetRoot `
  --allow-harvested-dataset-compat `
  --output-dir (Join-Path $OutputRoot 'v9_dev_eval_original_development') `
  --overwrite
```

Expected output manifest:

```text
<output-root>/v9_dev_eval_original_development/v9_tensor_cache_manifest.json
```

That cache is the development-map compatibility source used for both of these roles:

- extracting PM4-bearing development tiles that should enter active training
- extracting a separate non-overlapping development holdout for checkpoint selection

## 3. Build The Main Direct Cache

Use the direct `wow-viewer` wrapper for the main training cache so the broad corpus still comes from real game roots.

```powershell
& (Join-Path $RepoRoot 'wow-viewer/scripts/run_v9_direct_pipeline.ps1') `
  -Mode audit `
  -WowArchiveOnly `
  -IncludeBuilds 0_5_3_3368,0_5_5_3494,0_7_0_3694,3_0_1_8303,3_3_5_12340,4_0_0_11927 `
  -NoRequireMinimap `
  -NoRequireWdl `
  -OutputDir (Join-Path $OutputRoot 'v9_direct_archive_core')
```

Why this cache shape:

- `3_0_1_8303` is explicit so `Northrend` is guaranteed to enter the corpus.
- `4_0_0_11927` is explicit because it is from the same era as the development map and still carries map-family data the reconstruction model needs.
- `-NoRequireMinimap -NoRequireWdl` keeps the current known early-build gating boundary explicit while the direct dataset path continues to mature.
- `-Mode audit` is used here because the wrapper currently has only `audit` and `train` modes; `audit` still performs scan, merge, audit, curate, and cache build, then stops after a trainer audit pass instead of launching the long run.

Expected direct output manifest:

```text
<output-root>/v9_direct_archive_core/cache/v9_tensor_cache_manifest.json
```

## 4. Compose The PM4-Mixed Training And Holdout Manifests

The current recommended `v9` branch does not leave the full development cache entirely outside training.

Instead it does this:

- all development entries with non-zero `pm4_mask_257` enter the training corpus
- the remaining non-PM4 development entries become the dev-eval holdout

Use the converter-native split command:

```powershell
$DirectManifest = Join-Path $OutputRoot 'v9_direct_archive_core/cache/v9_tensor_cache_manifest.json'
$DevManifest = Join-Path $OutputRoot 'v9_dev_eval_original_development/v9_tensor_cache_manifest.json'
$SplitDir = Join-Path $OutputRoot 'v9_direct_plus_devpm4_split'

dotnet run --project (Join-Path $RepoRoot 'wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj') -- `
  dataset-split-pm4 `
  --direct-manifest $DirectManifest `
  --development-manifest $DevManifest `
  --output-dir $SplitDir
```

This is the current practical answer to the data-shape mismatch between the broad direct corpus and the PM4-rich development tiles, and it now lives inside `wow-viewer` instead of a side Python helper.

## 5. Launch The Recommended PM4-Mixed Training Branch

Launch the optimized trainer directly against the merged manifest.

```powershell
& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/train_v9_optimized.py') `
  (Join-Path $OutputRoot 'v9_direct_plus_devpm4_split/v9_direct_plus_development_pm4_training_manifest.json') `
  --output-dir (Join-Path $OutputRoot 'runs/v9_pm4mix_fullsane') `
  --epochs 120 `
  --batch-size 4 `
  --train-workers 1 `
  --val-workers 1 `
  --use-compile true `
  --selection-metric dev_global_mae `
  --dev-eval-cache-manifest (Join-Path $OutputRoot 'v9_direct_plus_devpm4_split/v9_development_non_pm4_holdout_manifest.json')
```

Optional bounded run:

```powershell
  --target-curated-samples 1200
```

Use the full sane pool when you want the PM4-bearing development tiles guaranteed to stay in active training without extra curation pressure.

## 6. Resume A Paused Or Interrupted Optimized Run

If the optimized run stops after writing a normal checkpoint, rerun the same command with the same output directory. The optimized trainer now auto-resumes from `last_checkpoint.pt` when it finds one there:

```powershell
& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/train_v9_optimized.py') `
  (Join-Path $OutputRoot 'v9_direct_plus_devpm4_split/v9_direct_plus_development_pm4_training_manifest.json') `
  --output-dir (Join-Path $OutputRoot 'runs/v9_pm4mix_fullsane') `
  --epochs 120 `
  --batch-size 4 `
  --selection-metric dev_global_mae `
  --dev-eval-cache-manifest (Join-Path $OutputRoot 'v9_direct_plus_devpm4_split/v9_development_non_pm4_holdout_manifest.json') `
  --no-require-minimap `
  --no-require-wdl
```

If you want to point at a different checkpoint path explicitly, add:

```powershell
  --resume-from (Join-Path $OutputRoot 'runs/v9_pm4mix_fullsane/last_checkpoint.pt')
```

Current pause behavior:

- the optimized trainer does not stop at epoch 50 by default anymore
- it defaults to `--pause-on-stall-epochs 50`, which means it will checkpoint and stop cleanly only after crossing `50` epochs without a new best result

## What To Check In Logs

- The direct wrapper should print `3_0_1_8303 => ...` during client-root resolution.
- The scan phase should include `--map Northrend --build 3_0_1_8303`.
- the optimized trainer should print `Dev-eval holdout: <count> entries`
- the run should report `dev_eval tiles ... | model_mae ... | wdl_gain ...` on dev-eval epochs
- `BEST` should follow the configured selection metric, not plain validation loss alone
- the run directory should contain `best_model.pt`, `last_checkpoint.pt`, and `previews/`

## Current Boundaries

- The development map is still fed into `v9` through a compatibility-built cache, not as a first-class direct scan target.
- The PM4-mixed manifest composition step is currently a documented manual workflow, not yet a dedicated first-class command.
- `wow-viewer/scripts/run_v9_direct_pipeline.ps1` still launches `train_v9.py` when used in full `train` mode, so the wrapper alone is not yet the full current recommended PM4-mixed path.
- The remaining known early-build default-gate blocker is still `0_5_5_3494:EmeraldDream_24_25 -> missing_minimap_rgb_256`.
- Some `3.0.1.8303` texture reads still hit older shared-reader compatibility gaps, so `3_0_1_8303/Northrend` is important but not yet the same thing as full `3.0.1` format closure.
- `4_0_0_11927` should remain in the corpus even if you need to source it from a different local client copy than your other builds; the dataset requirement is build-era coverage, not one specific archive layout.
