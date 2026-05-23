# wow-viewer

`wow-viewer` is the active standalone codebase for:

- multi-era WoW terrain and client-data reading
- V16 terrain dataset generation and training
- format inspection for ADT, WDT, PM4, BLP, WMO, MDX, and M2
- Alpha/LK/Cataclysm terrain-domain conversion
- future viewer/runtime work that will eventually replace legacy `MdxViewer` ownership

The current high-value workflow is terrain AI, not legacy NPZ tooling.

## Primary Workflow

The modern V16 path is:

1. `WowViewer.Tool.Harvest harvest-stream`
2. `data-harvester/scripts/inspect_v16_harvest_samples.py`
3. `data-harvester/scripts/build_v16_dataset.py build --allow-zarr-write`
4. `build_v16_dataset.py validate-signals`
5. `inspect_v16_dataset.py --write-overview`
6. `build_v16_curation_manifest.py`
7. `validate_v16_training_ready.py`
8. `train_v16.py` or curated `train_v16_1_*.py`

Key docs:

- [data-harvester README](./data-harvester/README.md)
- [V16 terrain model spec](./docs/architecture/v16-terrain-model-spec-2026-05-16.md)
- [V16 harvest recovery plan](./docs/architecture/v16-harvest-recovery-plan-2026-05-17.md)
- [V16.2 sidecar signal-expansion spec](./specs/011-v16-2-patched-signal-expansion/spec.md)

## Current V16 Status

Finalized V16 stores currently exist for:

- `0_5_3_3368`
- `0_5_5_3494`
- `0_7_0_3694`
- `3_0_1_8303`
- `3_3_5_12340`
- `4_0_0_11927`

Current corpus truth:

- all six current `signal_validation.json` files pass
- visual QA artifacts exist under `wow-viewer/output/datasets/v16/inspection/`
- `0_7_0_3694` still carries the expected allowed warning for zero `has_holes_16`

Renderer-truth upgrade truth is narrower than the base V16 corpus truth:

- bounded MdxViewer capture proofs currently exist only for `0_5_3_3368` and
  `3_3_5_12340`
- current proof roots live under:
  - `output/tmp/mdxviewer_validation_smoke/`
  - `output/tmp/mdxviewer_validation_smoke_fix_wmo/`
  - `output/tmp/mdxviewer_validation_smoke_heightfilter/`
- do not treat those richer signals as six-build-validated yet
- V16.2 training is now the active lane — 7-channel input (3 minimap + 4 guidance) with object mask data patched directly into existing V16 stores
- first V16.2 normal run is in progress on all 6 builds

## Quick Start

### Build

```powershell
dotnet build .\wow-viewer\WowViewer.slnx -c Debug
```

### Build One V16 Store

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\inspect_v16_harvest_samples.py --build 3_3_5_12340 --maps Azeroth --kinds object placement

uv run python scripts\build_v16_dataset.py build --build 3_3_5_12340 --allow-zarr-write
```

### Validate Trainer Readiness

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\validate_v16_training_ready.py --build 3_3_5_12340
```

### Train

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 4000 `
  --train-epoch-tiles 1350 `
  --val-max-tiles 150 `
  --batch-size 72 `
  --gpu-duty-cycle 100 `
  --run-name v16_full_corpus_epoch_rotation_qc1
```

### Curated V16.1 Normal Training

```powershell
cd .\wow-viewer\data-harvester
uv run python -u scripts\build_v16_curation_manifest.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --profile normal_terrain_v16_1_1 `
  --workers -1 `
  --chunk-size 128 `
  --run-name normal_terrain_full_corpus_v16_1_1

uv run python -u scripts\train_v16_1_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ..\output\datasets\v16\curation\normal_terrain_full_corpus_v16_1_1 `
  --device auto `
  --batch-size 16 `
  --grad-accum-steps 1 `
  --target-vram-gb 12 `
  --autotune-batch-size `
  --train-max-tiles 400 `
  --train-epoch-tiles 128 `
  --bucket-sampling-profile v16_1_1_normal `
  --val-max-tiles 48 `
  --num-workers -1 `
  --epochs 50 `
  --run-name v16_1_1_normal_curated_pool400_epoch128_bs16
```

This is the preferred V16.1 pattern now: curate first, then train from the
manifest instead of raw tile rows, then keep epochs bounded with a curated
train pool and rotating per-epoch subsets. The V16.1.1 normal lane now also
records difficulty buckets in the manifest, biases epoch subsets toward harder
tiles, and strengthens intra-tile weighting with painted-transition-aware
hard-region emphasis while keeping terrain-valid masking authoritative. For the
current 16 GB card, treat `16 x 1` as the preferred starting point when VRAM
headroom is available; keep `8 x 1` as the safer fallback before dropping to
`4 x 2`, `2 x 4`, or `1 x 8`. The shared trainer can now also probe a batch
ladder automatically with `--target-vram-gb` plus `--autotune-batch-size`.

### V16.2 Training (7-Channel Input)

V16.2 extends V16.1.1 with 7-channel input (3 minimap RGB + 4 guidance channels:
object_filtered_mask, terrain_valid_mask, alpha_painted, mcly_any). Object mask
data is patched directly into existing V16 stores — no sidecar files.

```powershell
cd .\wow-viewer\data-harvester

uv run python -u scripts\train_v16_2_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --device auto `
  --batch-size 8 `
  --train-max-tiles 800 `
  --train-epoch-tiles 256 `
  --val-max-tiles 96 `
  --epochs 256 `
  --run-name v16_2_normal_all_builds_256ep
```

Resume:

```powershell
uv run python -u scripts\train_v16_2_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --device auto `
  --batch-size 8 `
  --epochs 512 `
  --resume-checkpoint ..\models\v16_2\normal\runs\v16_2_normal_all_builds_256ep\checkpoints\v16_2_normal_last.pt `
  --run-name v16_2_normal_resume
```

Available V16.2 task wrappers:
- `train_v16_2_normal.py` — normal prediction (first consumer of guidance channels)
- `train_v16_2_height.py` — height prediction
- `train_v16_2_holes.py` — hole mask prediction
- `train_v16_2_liquid.py` — liquid mask + type prediction
- `train_v16_2_texcomp.py` — texture decomposition + recomposition

Outputs go to `models/v16_2/<task>/runs/<run-name>/`.

For V16.1.x normal training, resume uses `--resume-checkpoint`. The older
`--resume-from auto` flow is only on `train_v16.py`.

```powershell
uv run python -u scripts\train_v16_1_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ..\output\datasets\v16\curation\normal_terrain_full_corpus_v16_1_1 `
  --device auto `
  --batch-size 16 `
  --grad-accum-steps 1 `
  --target-vram-gb 12 `
  --autotune-batch-size `
  --train-max-tiles 400 `
  --train-epoch-tiles 128 `
  --bucket-sampling-profile v16_1_1_normal `
  --val-max-tiles 48 `
  --epochs 100 `
  --num-workers -1 `
  --val-preview-interval 2 `
  --run-name v16_1_1_normal_curated_pool400_epoch128_bs16 `
  --resume-checkpoint ..\models\v16_1\normal\runs\v16_1_1_normal_curated_pool400_epoch128_bs16\checkpoints\v16_1_normal_last.pt
```

As with V16, `--epochs` is the new total ceiling for the run, not extra epochs
to add after the checkpoint. The shared V16.1 trainer now also extends the
cosine scheduler to that higher total when resuming, instead of staying pinned
to the original run ceiling.

## What Lives Here

### Shared WoW I/O

Core libraries under `src/core/` own format reading and writing for terrain and
client assets. This is where new shared-format work belongs.

### Terrain AI

The Python stack under `data-harvester/` owns:

- V16 dataset building
- signal validation
- visual QA
- trainer-readiness validation
- training
- inference
- dataset quality audits

### Conversion

The converter surfaces support:

- Alpha to LK terrain-domain conversion
- LK/Cataclysm to Alpha terrain-domain conversion
- real staged-client validation against legacy consumers where needed

## Important Rules

- Use staged clients under `output/tmp/wowarchive-clients/`, never `H:\CLIENTS`.
- For full dataset generation, use the harvest-first path, not legacy converter-side dataset commands.
- `wow-viewer` is the implementation owner; `gillijimproject_refactor` is reference/continuity/validation only.
- Spec Kit is expected for non-trivial repo work.

## Repository Layout

```text
wow-viewer/
  src/core/                 Shared libraries and runtime surfaces
  tools/                    C# CLI tools
  data-harvester/           Python dataset/training/inference stack
  docs/architecture/        Source-of-truth specs and plans
  specs/                    Spec Kit feature specs
  output/                   Local generated datasets, validation, and runs
```

## Data Policy

This repo is a local workflow surface for code, configs, manifests, and
training logic. It is not a distribution point for copyrighted client data,
harvested corpora, or model outputs derived from proprietary game assets.
