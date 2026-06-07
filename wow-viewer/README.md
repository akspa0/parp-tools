# wow-viewer

`wow-viewer` is the active standalone codebase for:

- multi-era WoW terrain and client-data reading
- V16 terrain dataset generation and training
- format inspection for ADT, WDT, PM4, BLP, WMO, MDX, and M2
- Alpha/LK/Cataclysm terrain-domain conversion
- future viewer/runtime work that will eventually replace legacy `MdxViewer` ownership

The current high-value workflow is the focused V18 terrain-reconstruction lane,
not the older broad-corpus V16 trainer flow.

## Primary Workflow

The active focused V18 path is:

1. prepare the focused V18 stores:
   - `wow-viewer/output/datasets/v18/0_5_3_3368.zarr`
   - `wow-viewer/output/datasets/v18/3_3_5_12340.zarr`
2. `data-harvester/scripts/build_v18_curation_manifest.py`
3. optionally `data-harvester/scripts/build_v18_tiny_manifest.py`
4. `data-harvester/scripts/train_v18_focus.py height`
5. `data-harvester/scripts/train_v18_focus.py normal`
6. `data-harvester/scripts/infer_v18_focus.py`

Key docs:

- [data-harvester README](./data-harvester/README.md)
- [spec 047 quickstart](./specs/047-v18-distill-corpus-open-source-loop/quickstart.md)
- [V18 focused architecture summary](./docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md)
- [V16 terrain model spec](./docs/architecture/v16-terrain-model-spec-2026-05-16.md)

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

The replacement `wow-viewer` validation-capture lane now has the same bounded
proof anchors, and it now emits the bounded derived artifact images too:

- command: `WowViewer.Tool.ValidationCapture capture --renderer`
- proof anchors:
  - `0_5_3_3368 / Azeroth_30_48`
  - `3_3_5_12340 / Azeroth_30_48`
- output roots:
  - `wow-viewer/output/tmp/validation-capture-gpu-viewer-style/`
  - `wow-viewer/output/tmp/validation-capture-gpu-viewer-style-335/`
- current boundary:
  - the tool now owns bounded four-variant capture through
    `ValidationWorldSceneAdapter` and `IValidationWorldSceneAdapter`
  - the tool now also writes compatible `images/<tile>_object_visibility_mask.png`
    and `images/<tile>_no_objects.png` outputs under the dataset root
  - it bypasses `WowViewerWorldScenePlanner`
  - it still reuses `WorldGpuPreviewRenderer` as a temporary backend
  - broader non-bounded renderer-truth automation and the current full dataset
    batch path still remain legacy-MdxViewer-owned
  - later renderer ownership still needs to replace the temporary
    `WorldGpuPreviewRenderer` backend reuse

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

### Focused V18 Curation

```powershell
cd .\wow-viewer\data-harvester

uv run python -u scripts\build_v18_curation_manifest.py `
  --run-name v18_focus_terrain_v1 `
  --workers -1 `
  --chunk-size 128 `
```

### Focused V18 Full Training

```powershell
cd .\wow-viewer\data-harvester

uv run python -u scripts\train_v18_focus.py height `
  --device cuda `
  --epochs 40 `
  --curation-manifest ..\output\datasets\v18\curation\v18_focus_terrain_v1 `
  --train-bucket-rotation-fraction 0.10 `
  --val-max-tiles 32 `
  --val-interval 1 `
  --run-name v18_height_focus_full_v1

uv run python -u scripts\train_v18_focus.py normal `
  --device cuda `
  --epochs 40 `
  --curation-manifest ..\output\datasets\v18\curation\v18_focus_terrain_v1 `
  --train-bucket-rotation-fraction 0.10 `
  --val-max-tiles 32 `
  --val-interval 1 `
  --run-name v18_normal_focus_full_v1
```

These are the current recommended full-session commands for the focused
two-build lane. They keep the active `047` defaults visible:

- focused manifest: `v18_focus_terrain_v1`
- restrained bucket rotation: `0.10`
- early stop: default patience `8`
- safer auto loader defaults for the focused base lane when `--num-workers`
  stays at `-1`

If you want a smaller scouting corpus instead of the full kept pool, derive one
with `build_v18_tiny_manifest.py` and then rerun the same train commands with
that manifest plus `--train-bucket-rotation-fraction 1.0`.

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
