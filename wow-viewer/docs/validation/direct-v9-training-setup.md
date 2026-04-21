# Direct V9 Training Setup

This is the current reproducible setup for training the direct `v9` terrain model from `wow-viewer` without routing the main corpus path back through harvested-dataset ownership.

All examples below assume a Bring Your Own Data workflow. Replace the placeholder paths with equivalents on your own machine.

The working shape today is:

1. Stage archive-backed game clients locally.
2. Build the main training cache from real game roots with `wow-viewer/scripts/run_v9_direct_pipeline.ps1`.
3. Build a separate development-map dev-eval cache from `datasets/original_development/development`.
4. Train `train_v9.py` with the direct cache as the main corpus and the development-map cache as the stable holdout used for checkpoint selection.

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

## 2. Build The Development-Map Dev-Eval Cache

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

That cache is the recommended held-out validation dataset for direct `v9` runs until the development-map path itself has a first-class direct shared-reader training flow.

## 3. Launch Direct Training

Use the direct `wow-viewer` wrapper for the main training cache so the corpus still comes from real game roots.

```powershell
& (Join-Path $RepoRoot 'wow-viewer/scripts/run_v9_direct_pipeline.ps1') `
  -Mode train `
  -WowArchiveOnly `
  -IncludeBuilds 0_5_3_3368,0_5_5_3494,0_7_0_3694,3_0_1_8303,3_3_5_12340,4_0_0_11927 `
  -NoRequireMinimap `
  -NoRequireWdl `
  -OutputDir (Join-Path $OutputRoot 'v9_direct_archive_core') `
  -TrainerArgs @(
      '--epochs','120',
      '--batch-size','8',
      '--target-curated-samples','128',
      '--selection-metric','dev_global_mae',
      '--dev-eval-cache-manifest',(Join-Path $OutputRoot 'v9_dev_eval_original_development/v9_tensor_cache_manifest.json'),
      '--pause-every-epochs','0'
  )
```

Why this launch shape:

- `3_0_1_8303` is explicit so `Northrend` is guaranteed to enter the corpus.
- `4_0_0_11927` is explicit because it is from the same era as the development map and still carries map-family data the reconstruction model needs.
- `--selection-metric dev_global_mae` makes the development-map holdout decide best-checkpoint selection instead of raw train or validation loss alone.
- `-NoRequireMinimap -NoRequireWdl` keeps the current known early-build gating boundary explicit while the direct dataset path continues to mature.

## 4. Resume A Paused Or Interrupted Run

If you keep periodic pauses enabled, or if the run stops after writing a normal checkpoint, resume from `last_checkpoint.pt`:

```powershell
& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/train_v9.py') `
  (Join-Path $OutputRoot 'v9_direct_archive_core/cache/v9_tensor_cache_manifest.json') `
  --output-dir (Join-Path $OutputRoot 'v9_direct_archive_core/train') `
  --resume-from (Join-Path $OutputRoot 'v9_direct_archive_core/train/last_checkpoint.pt') `
  --epochs 120 `
  --batch-size 8 `
  --selection-metric dev_global_mae `
  --dev-eval-cache-manifest (Join-Path $OutputRoot 'v9_dev_eval_original_development/v9_tensor_cache_manifest.json') `
  --no-require-minimap `
  --no-require-wdl
```

## What To Check In Logs

- The direct wrapper should print `3_0_1_8303 => ...` during client-root resolution.
- The scan phase should include `--map Northrend --build 3_0_1_8303`.
- `train_v9.py` should print a `Dev-eval holdout` line that points at the development-map cache manifest.
- Best-checkpoint selection should report `selection_metric=dev_global_mae` instead of falling back to plain `val_loss`.

## Current Boundaries

- The development map is still fed into `v9` as a separate compatibility-built dev-eval cache, not as a first-class direct scan target.
- The remaining known early-build default-gate blocker is still `0_5_5_3494:EmeraldDream_24_25 -> missing_minimap_rgb_256`.
- Some `3.0.1.8303` texture reads still hit older shared-reader compatibility gaps, so `3_0_1_8303/Northrend` is important but not yet the same thing as full `3.0.1` format closure.
- `4_0_0_11927` should remain in the corpus even if you need to source it from a different local client copy than your other builds; the dataset requirement is build-era coverage, not one specific archive layout.