# V9 Runpod Pod Container Training Plan

## Intent

- move the active `v9` training lane off the local Windows workstation for serious runs
- package the trainer as a reproducible container image without requiring local Docker on this machine
- use Runpod `Pods`, not Runpod `Serverless`, as the first cloud execution target for long-running checkpointed training
- keep training data separate from the container image so corpora can be updated, bundled, versioned, and mounted independently
- make the first cloud path work on a single GPU first, then add multi-GPU support only after the trainer explicitly supports distributed execution

## Why This Boundary Is Right

- the current local run already proved the optimized trainer can make real progress, but it is too slow and too RAM-sensitive to be the long-range host for repeated heavy experiments
- the user does not want a Windows-local Docker or WSL-heavy setup just to build or test cloud training containers
- GitHub-hosted container builds remove the local Docker requirement entirely while still producing a reproducible image
- Runpod `Pods` are a better fit than `Serverless` for:
  - multi-hour or multi-day training
  - checkpoint persistence
  - shell access and debugging
  - mounted storage
  - direct `torchrun` or future DDP launch control
- the current training bundle is already close to portable:
  - manifests plus `.npz` shards
  - explicit split between train and development holdout manifests
- the main current portability gap is not the tensors themselves; it is the manifest path contract, which still points at absolute Windows paths

## Current Constraints

### Trainer

- `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v9_optimized.py` now has the desired resume behavior and stall-based pause behavior for long runs
- the current optimized trainer does **not** yet expose explicit distributed training hooks such as:
  - `torchrun`
  - `DistributedDataParallel`
  - `init_process_group`
  - rank or world-size aware dataset partitioning
- because of that, a multi-GPU Runpod target is a second-phase task, not the first execution slice

### Data

- the current PM4-mixed split manifests under `output/ml-training/cache/v9_direct_plus_devpm4_split_20260421/` are valid training surfaces for local runs
- those manifests currently store absolute Windows `shard_path` and `source_json` values
- that makes them non-portable as-is for cloud Linux paths
- the data therefore needs a packaging step that rewrites the manifest to a bundle-local relative-path contract

### Environment

- the user does not want local Docker Desktop or a large WSL + Ubuntu + Docker installation on this Windows machine
- the plan should therefore avoid any step that requires building the container locally
- the container image should be built in GitHub Actions and published to `ghcr.io`

## Core Recommendation

- first cloud lane:
  - portable training bundle
  - trainer container image
  - Runpod Pod launcher
  - single-GPU remote training

- do **not** begin with:
  - Runpod Serverless endpoint training
  - multi-GPU training
  - local Docker dependency

- treat the first success criterion as:
  - a clean cloud run that can pull the image, mount or download the data bundle, launch training, write checkpoints, and resume from them

## Target Architecture

### 1. Portable Training Bundle

- introduce a bundle packager that emits one self-contained training root, for example:

```text
v9_run_bundle/
  manifests/
    train_manifest.json
    dev_holdout_manifest.json
  cache/
    main/
      shards/...
    dev/
      shards/...
  metadata/
    bundle_summary.json
    source_manifests.json
```

- rewrite manifest fields so cloud training reads relative paths inside the bundle instead of machine-local Windows paths
- keep the bundle data-only; do not copy the trainer code into the bundle

### 2. Trainer Container

- add a Docker image rooted in this repo that includes:
  - Python runtime
  - PyTorch training dependencies
  - repository training scripts
  - a small entrypoint wrapper for `train_v9_optimized.py`
- keep the image generic enough to run against any mounted compatible bundle
- keep model outputs outside the image on mounted storage

### 3. GitHub-Based Image Build

- add GitHub Actions to:
  - build the training image
  - tag it deterministically
  - push it to `ghcr.io`
- use GitHub registry auth from Runpod instead of baking secrets into the repo
- this becomes the canonical image build path so local Docker is optional forever

### 4. Runpod Pod Launcher

- add a small launcher that can:
  - create a Pod from the Runpod API
  - point it at the GHCR image
  - attach a persistent volume or network volume
  - pass environment variables
  - start the training command
- prefer `Pods` over `Serverless` because the current workload is checkpointed training, not request-oriented handler execution

### 5. Remote Data Delivery

- support one or both of these first:
  - mounted persistent volume with the bundle already present
  - authenticated download of the bundle from a user-controlled API or object store

- the data lane should be secure and separate from the image:
  - the image is reusable
  - the bundle is private and replaceable

## Recommended Artifact Ownership

### Code

- training container and launch helpers should live in `gillijimproject_refactor` because the current trainer still lives there
- future long-range direct-ML ownership can still continue moving into `wow-viewer`, but this cloud-hosting slice is about operationalizing the current `v9` trainer

### Suggested File Surfaces

- `gillijimproject_refactor/docker/v9-trainer/Dockerfile`
- `gillijimproject_refactor/docker/v9-trainer/entrypoint.sh`
- `gillijimproject_refactor/src/WoWMapConverter/scripts/package_v9_training_bundle.py`
- `gillijimproject_refactor/src/WoWMapConverter/scripts/runpod_launch_v9.py`
- `.github/workflows/build-v9-trainer-image.yml`
- optional:
  - `gillijimproject_refactor/docs/V9_Runpod_Training_Guide.md`

## Bundle Contract Recommendation

### Inputs

- train split manifest:
  - currently `output/ml-training/cache/v9_direct_plus_devpm4_split_20260421/v9_direct_plus_development_pm4_training_manifest.json`
- dev holdout manifest:
  - currently `output/ml-training/cache/v9_direct_plus_devpm4_split_20260421/v9_development_non_pm4_holdout_manifest.json`

### Rewrite Rules

- convert `shard_path` to a bundle-relative path
- convert `source_json` to either:
  - bundle-relative debug JSON path when copied into the bundle, or
  - omit from the portable manifest if it is not required at training time
- preserve all curation and metric metadata needed by the current trainer
- record bundle provenance in `metadata/source_manifests.json`

### Output Guarantee

- the bundled manifests must be consumable on Linux without any path rewriting at launch time
- the bundle should be zip- or tar-friendly and safe to upload as a private artifact

## Trainer Container Recommendation

### First Image Goal

- support:
  - audit-only runs
  - normal training runs
  - auto-resume from `last_checkpoint.pt`
  - explicit `--resume-from`

### First Image Scope

- include:
  - `.venv-train`-equivalent Python dependencies
  - the repo training scripts
  - any lightweight shell helpers needed to launch training cleanly

- do not include:
  - private training data
  - checkpoints
  - giant cached outputs

### Entry Command Shape

- preferred pattern:
  - mount bundle under `/workspace/data/v9_bundle`
  - mount run output under `/workspace/runs/<run-name>`
  - launch:

```text
python train_v9_optimized.py /workspace/data/v9_bundle/manifests/train_manifest.json --dev-eval-cache-manifest /workspace/data/v9_bundle/manifests/dev_holdout_manifest.json --output-dir /workspace/runs/<run-name> ...
```

## Runpod Pod Recommendation

### First Execution Mode

- one Pod
- one GPU
- persistent volume for run outputs
- optional network volume or download step for the training bundle

### Why Not Serverless First

- the current workload is not a request/handler workload
- training needs checkpoint persistence, shell visibility, and flexible launch semantics
- Pods are the simpler and more honest execution model for this lane

### Why Not Multi-GPU First

- the trainer does not yet implement DDP or rank-aware launch behavior
- adding more GPUs before adding distributed support risks paying for idle hardware

## Security Recommendation

- keep the container image public only if the code is safe to expose; otherwise use private GHCR
- keep training bundles private
- pass registry credentials and bundle download credentials through:
  - GitHub secrets for image publication
  - Runpod environment variables or registry auth configuration for deployment
- do not hard-code tokens in scripts or checked-in workflow files

## Implementation Order

### Phase 1 - Portable Bundle

1. add `package_v9_training_bundle.py`
2. feed it the current train and dev manifests
3. copy or organize shard trees into one bundle root
4. rewrite manifests to relative Linux-friendly paths
5. validate that the bundled manifests still load in `train_v9_optimized.py` on a local dry run

### Phase 2 - Container Image

1. add `Dockerfile` and entrypoint wrapper
2. pin the trainer dependency set clearly
3. make the image run the existing trainer without changing the model contract
4. validate syntax and command wiring through CI

### Phase 3 - GitHub Image Build

1. add GitHub Actions workflow
2. build and publish to `ghcr.io`
3. document image tags and expected registry auth

### Phase 4 - Runpod Pod Launcher

1. add a small API launcher for Pod creation
2. allow image, GPU type, volume settings, and command overrides
3. support passing:
  - bundle path or download URL
  - output path
  - run name
  - training args
4. prove a first remote single-GPU training launch

### Phase 5 - Resume And Stop/Start Workflow

1. confirm checkpoints persist on mounted storage
2. confirm rerunning the same command auto-resumes cleanly
3. document the stop/start/resume operational pattern

### Phase 6 - Optional Multi-GPU Upgrade

1. add DDP support to `train_v9_optimized.py`
2. add `torchrun` launch support in the container entrypoint
3. add distributed-safe dataset partitioning and metric reduction
4. only then target `gpuCount > 1` Pods

## Validation Standard

- do not call this lane done until all of these are proven:
  1. the bundle can be created from the current local manifests without manual editing
  2. the bundled manifests are portable on Linux path semantics
  3. the container can launch the trainer against the bundle
  4. a Runpod Pod can pull the image and run training
  5. checkpoints persist and resume correctly across Pod restarts or relaunches

- distributed training is a separate proof level and should not be implied by the first cloud slice

## Immediate Recommendation

- implement the cloud lane in this exact order:
  1. portable bundle packager
  2. trainer Docker image
  3. GitHub Actions GHCR publish
  4. Runpod Pod launcher
  5. first single-GPU remote run

- keep the current local trainer as the development/debug lane only
- do not invest time in local Docker or WSL setup on this Windows machine unless another task later truly requires it
