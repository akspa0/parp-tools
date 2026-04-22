# V9 Runpod Training Guide

This guide documents the first cloud lane for the active `v9` trainer:

- portable bundle packaging
- GitHub-built GHCR image publication
- Runpod Pod launch
- checkpointed single-GPU training with clean resume behavior

This stays a Bring Your Own Data workflow. Do not ship client data, generated corpora, model weights, or model outputs derived from proprietary game data.

## What This Lane Covers

The active boundary is:

- package the training cache into a portable Linux-friendly bundle
- build the trainer image in GitHub Actions without local Docker
- launch the image on a Runpod Pod
- keep checkpoints and bundle data outside the image

This guide does not claim:

- multi-GPU or distributed training
- Runpod Serverless support
- image-embedded private datasets

## 1. Package A Portable Bundle

The portable bundle step rewrites `shard_path` entries to manifest-relative Linux-friendly paths and copies the referenced `.npz` shards into one self-contained bundle root.

Example using the current PM4-mixed split:

```powershell
$RepoRoot = 'i:/parp/parp-tools'
$PythonExe = Join-Path $RepoRoot '.venv/Scripts/python.exe'
$BundleRoot = Join-Path $RepoRoot 'output/ml-training/v9_run_bundle_20260422'

& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/package_v9_training_bundle.py') `
  --train-manifest (Join-Path $RepoRoot 'output/ml-training/cache/v9_direct_plus_devpm4_split_20260421/v9_direct_plus_development_pm4_training_manifest.json') `
  --dev-manifest (Join-Path $RepoRoot 'output/ml-training/cache/v9_direct_plus_devpm4_split_20260421/v9_development_non_pm4_holdout_manifest.json') `
  --output-dir $BundleRoot `
  --archive-format tar.gz
```

Expected bundle shape:

```text
v9_run_bundle_20260422/
  manifests/
    train_manifest.json
    dev_holdout_manifest.json
  cache/
    main/shards/...
    dev/shards/...
  metadata/
    bundle_summary.json
    source_manifests.json
```

Important runtime contract:

- the bundled manifests now use relative `shard_path` values
- `train_v9.py` and `train_v9_optimized.py` now resolve those paths relative to the manifest file itself
- that means the bundle remains launch-location-independent once copied to Linux

Use `--include-source-json` only when you explicitly want those JSON sidecars inside the bundle for debugging.

## 2. Build The GHCR Image

The repo now includes:

- Dockerfile: `gillijimproject_refactor/docker/v9-trainer/Dockerfile`
- entrypoint: `gillijimproject_refactor/docker/v9-trainer/entrypoint.sh`
- workflow: `.github/workflows/build-v9-trainer-image.yml`

The workflow builds a Linux `amd64` image from the current trainer scripts and publishes it to:

```text
ghcr.io/<github-owner>/parp-tools-v9-trainer
```

Default tags include:

- `sha-<commit>`
- `latest` on the default branch
- a branch tag on branch pushes

Recommended first use:

1. open the GitHub Actions workflow `Build V9 Trainer Image`
2. run it manually with `push_image=true`
3. note the resulting `sha-...` tag for the exact image you want Runpod to pull

## 3. Prepare Runpod Registry And Storage

For a private GHCR image, create a Runpod container-registry auth entry and keep its ID handy for Pod creation.

For bundle data, choose one of these first:

- attach a Runpod network volume and upload the unpacked bundle under `/workspace/data/v9_bundle`
- host the bundle archive at a private URL and let the container download it on boot

The container entrypoint supports both cases:

- mounted bundle: it trains immediately if `manifests/train_manifest.json` is already present
- downloaded bundle: set `V9_BUNDLE_DOWNLOAD_URL` and optional auth headers, and the entrypoint downloads and extracts the archive before launch

## 4. Launch A Runpod Pod

The launcher script uses the Runpod Pods REST API and the new container contract.

Example for a mounted bundle on a network volume:

```powershell
$env:RUNPOD_API_KEY = '<runpod-api-key>'

& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/runpod_launch_v9.py') `
  --image 'ghcr.io/<github-owner>/parp-tools-v9-trainer:sha-<commit>' `
  --container-registry-auth-id '<runpod-registry-auth-id>' `
  --name 'v9-pm4mix-trainer' `
  --run-name 'v9_pm4mix_cloud_20260422' `
  --gpu-type 'NVIDIA RTX A6000' `
  --gpu-type 'NVIDIA A40' `
  --network-volume-id '<runpod-network-volume-id>' `
  --epochs 120 `
  --batch-size 4 `
  --selection-metric dev_global_mae
```

Example for a bundle archive download:

```powershell
$env:RUNPOD_API_KEY = '<runpod-api-key>'
$env:V9_BUNDLE_AUTH_HEADER = 'Authorization: Bearer <private-download-token>'

& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/runpod_launch_v9.py') `
  --image 'ghcr.io/<github-owner>/parp-tools-v9-trainer:sha-<commit>' `
  --container-registry-auth-id '<runpod-registry-auth-id>' `
  --name 'v9-pm4mix-download' `
  --gpu-type 'NVIDIA RTX A6000' `
  --bundle-download-url 'https://example.invalid/private/v9_run_bundle_20260422.tar.gz' `
  --env-from-host V9_BUNDLE_AUTH_HEADER `
  --selection-metric dev_global_mae
```

The launcher writes a Pod request with:

- `imageName`
- `gpuTypeIds`
- `containerRegistryAuthId` when provided
- `networkVolumeId` or `volumeInGb`
- container env vars such as `V9_BUNDLE_ROOT`, `V9_TRAIN_MANIFEST`, `V9_OUTPUT_DIR`, and `V9_TRAINER_ARGS_JSON`

Use `--dry-run` to inspect the final payload before sending it to Runpod.

## 5. What The Container Runs

The image entrypoint launches the current optimized trainer:

```text
python train_v9_optimized.py <train-manifest> --output-dir <run-dir> ...
```

Default launcher behavior sets:

- `--epochs 120`
- `--batch-size 4`
- `--train-workers 1`
- `--val-workers 1`
- `--use-compile true`
- `--no-require-minimap`
- `--no-require-wdl`
- `--selection-metric dev_global_mae` when a dev manifest is enabled

Append additional trainer flags with repeated `--trainer-arg` tokens.

## 6. Resume Pattern

The operational pattern stays the same on Runpod as it does locally:

- keep the run output on mounted persistent storage
- relaunch the same command with the same output directory
- let `train_v9_optimized.py` auto-resume from `last_checkpoint.pt`

The image does not store checkpoints internally. They stay on the mounted volume under the chosen run directory.

## 7. Bounded First Success Criteria

Treat the first cloud slice as complete only when all of these are true:

1. the bundle packager can build a portable bundle without manual manifest editing
2. the image can launch `train_v9_optimized.py` against that bundle
3. Runpod can pull the image and start the Pod
4. checkpoints land on persistent storage
5. rerunning the same Pod command resumes from `last_checkpoint.pt`

Multi-GPU training remains a later task after explicit DDP support lands in the trainer.
