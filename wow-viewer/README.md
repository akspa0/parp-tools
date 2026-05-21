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
2. `data-harvester/scripts/build_v16_dataset.py build`
3. `build_v16_dataset.py validate-signals`
4. `inspect_v16_dataset.py --write-overview`
5. `validate_v16_training_ready.py`
6. `train_v16.py`

Key docs:

- [data-harvester README](./data-harvester/README.md)
- [V16 terrain model spec](./docs/architecture/v16-terrain-model-spec-2026-05-16.md)
- [V16 harvest recovery plan](./docs/architecture/v16-harvest-recovery-plan-2026-05-17.md)

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

## Quick Start

### Build

```powershell
dotnet build .\wow-viewer\WowViewer.slnx -c Debug
```

### Build One V16 Store

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\build_v16_dataset.py build --build 3_3_5_12340
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
