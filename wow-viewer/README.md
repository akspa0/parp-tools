# wow-viewer

`wow-viewer` is the canonical home for shared World of Warcraft file I/O, dataset generation, and the **v10 Terrain AI** training pipeline. It is also the long-term target for runtime and viewer code extracted from the larger `parp-tools` workspace.

> **Viewer Status — On Hold**
> The desktop viewer shell (`WowViewer.App`) is paused while the v10 terrain model is brought to full training maturity. Once the model is trained and validated, viewer development will restart from a clean slate using the proven library stack below. Do not invest in the current viewer shell; it is transitional.

---

## What You Can Use Today

- **v10 Terrain AI pipeline** — end-to-end NPZ extraction, corpus building, dictionary mining, label generation, and PyTorch trainers.
- **Shared libraries** under `src/core` — canonical implementation target for new format and runtime work.
- **`WowViewer.Tool.Converter`** — conversion and dataset CLI; the primary surface for v10 workflows.
- **`WowViewer.Tool.Inspect`** — read-only inspection CLI for archive, BLP, M2, MDX, map, LIT, PM4, and WMO data.
- **Shared archive-backed virtual reads** with a persistent `WowViewer.Core.IO.Files` session cache.

---

## v10 Terrain AI

The v10 lane is the active development focus. It is organised in waves:

- **Wave 1** — Tensor-pack extraction and Stage 1 corpus building.
- **Wave 2** — Pattern mining (brush dictionaries, MCLY palettes, MCAL compositions, height profiles) and bounded classifier trainers.
- **Stage 2** — Multi-resolution terrain synthesis trainer.

### Wave 1 — Extraction & Stage 1 Corpus

Build minimap-backed NPZ shards from root ADTs:

```powershell
# Single-tile extraction
wowviewer-converter extract-v10-tensors --minimap-root <dir> <adt_path>

# Bulk Stage 1 corpus + manifest
wowviewer-converter dataset-build-v10-stage1 `
  --input-dir <root_adt_dir> `
  --output-dir <corpus_dir> `
  --minimap-root <minimap_dir> `
  --manifest <corpus_dir>/v10_stage1_manifest.json
```

Wave 1 writes shards containing:

| Signal | Shape | Description |
|--------|-------|-------------|
| `minimap_rgb_256` | 256×256×3 | Minimap texture |
| `height_257` | 257×257 | Full-resolution height |
| `height_65` | 65×65 | Mid-resolution height |
| `height_17` | 17×17 | Coarse height |
| `mcal_alpha_pack_256` | 256×256×4 | Chunk-level alpha layers |
| `mcly_texture_ids` | 16×16×4 | Per-chunk texture-layer IDs |
| `mcly_texture_names` | variable | MTEX path table |
| `ObjectMask257` | 257×257 | Placement-derived object mask |
| `ObjectPreciseMask257` | 257×257 | Precise object mask |
| `normal_257` | 257×257×3 | Normals |
| `hole_mask_16` | 16×16 | Hole bitmask |
| `mclq_liquid_mask_16` | 16×16 | Liquid presence |

### Wave 2 — Pattern Mining

All dictionary commands run natively inside `WowViewer.Tool.Converter` and consume the Stage 1 NPZ shard contract.

#### Anchor-aware brush mining
```powershell
wowviewer-converter mine-v10-brushes `
  --input-dir <corpus_dir> --placement-dir <corpus_dir> `
  --output-dir <out_dir> --anchor-mode hybrid
```
Emits `brush_dictionary.json` with object, terrain, and hybrid anchors.

#### MCLY texture-layer combination mining
```powershell
wowviewer-converter mine-v10-mcly `
  --input-dir <corpus_dir> --output-dir <out_dir>
```
Emits `mclay_dictionary.json` and `mcly_dictionary.json` with texture-path keyed palettes and conservative biome tags.

#### Reusable MCLY label manifest
```powershell
wowviewer-converter label-v10-mcly `
  --input <manifest.json> --dictionary <mclay_dictionary.json> `
  --output <label_manifest.json>
```
Emits `v10-mcly-label-manifest.v1` with per-tile 16×16 chunk label grids (`ignore_index = -100` for unretained combinations).

#### MCAL chunk composition mining
```powershell
wowviewer-converter mine-v10-mcal-compositions `
  --input-dir <corpus_dir> --output-dir <out_dir>
```
Emits `mcal_composition_dictionary.json` + `.npz` with averaged 64×64×4 centroids.

#### MCAL brush-stroke vocabulary mining
```powershell
wowviewer-converter mine-v10-mcal-brushes `
  --input-dir <corpus_dir> --output-dir <out_dir>
```
Emits `mcal_brush_dictionary.json` + `.npz` with per-layer 64×64 stamps and coarse shape-family labels.

#### Height-profile clustering
```powershell
wowviewer-converter mine-v10-height-profiles `
  --input-dir <corpus_dir> --output-dir <out_dir>
```
Emits `height_profile_dictionary.json` + `.npz` with normalised and absolute height archetypes.

### Trainers

Python trainers live in `scripts/` and consume the v10 NPZ shard contract directly.

| Trainer | Input | Output | Purpose |
|---------|-------|--------|---------|
| `curate_v10_training_shards.py` | v10/v9 manifests or NPZ dirs | curated manifest + report | Balance and filter shards before training |
| `train_v10_stage1_minimap2height.py` | Stage 1 manifest | `minimap2height.pt` | Baseline minimap → `height_17` regression |
| `train_v10_minimap_to_mclay.py` | Stage 1 manifest + `mclay_dictionary.json` | `minimap_to_mclay_classifier.pt` | Tile-level minimap → retained MCLY palette |
| `train_v10_minimap_to_mclay_grid.py` | Stage 1 manifest or label manifest | `minimap_to_mclay_grid_classifier.pt` | Chunk-grid minimap → 16×16 retained MCLY labels |
| `train_v10_stage2_terrain_synth.py` | Stage 1 manifest | `last.pt` | Multi-resolution height synthesis (17×17, 65×65, 257×257) with signal-dropout augmentation |

Example mixed-corpus curation from the all-version v9 cache plus native v10 shards:
```powershell
.venv\Scripts\python scripts\curate_v10_training_shards.py `
  <v9_tensor_cache_manifest.json> <v10_stage1_manifest.json> `
  --output output\ml-training\v10_curated\curated_manifest.json `
  --max-per-dataset 128
```

Current all-version Stage 2 training uses the proven v9 direct-cache shards for broad client coverage plus native v10 development shards for richer signals. Latest local CUDA output: `output/ml-training/v10_stage2_v9cache_native_dev_cuda_run2/checkpoints/best.pt`.

Example Stage 2 smoke test:
```powershell
.venv\Scripts\python scripts\train_v10_stage2_terrain_synth.py `
  <corpus_dir>\v10_stage1_manifest.json `
  --output-dir output\ml-training\v10_stage2 `
  --epochs 50 --device cuda --signal-dropout 0.15
```

---

## Shared Libraries

| Library | Contents |
|---------|----------|
| `WowViewer.Core` | Core contracts, maths primitives, dataset manifests, tensor-pack models |
| `WowViewer.Core.IO` | File readers, chunk parsers, archive virtualisation, ADT/WDT/WMO/BLP/DBC loaders |
| `WowViewer.Core.Runtime` | Runtime consumers and scene-building seams (world-session state, bridge code) |
| `WowViewer.Core.PM4` | PM4 parser, services, and research-facing contracts — the most mature format library today |

New format work and dataset contracts should land in `Core` / `Core.IO` first, then surface through `WowViewer.Tool.Converter` or `WowViewer.Tool.Inspect`.

---

## Tools

### Converter & Dataset CLI

Project: `tools/converter/WowViewer.Tool.Converter`

The primary surface for v10 workflows and general file detection / conversion.

```powershell
dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- --help
```

V10 commands:
- `extract-v10-tensors`
- `dataset-build-v10-stage1`
- `mine-v10-brushes`
- `mine-v10-mcly`
- `label-v10-mcly`
- `mine-v10-mcal-compositions`
- `mine-v10-mcal-brushes`
- `mine-v10-height-profiles`

Legacy / general commands still available:
- `detect`, `dataset-scan`, `dataset-merge`, `dataset-audit`, `dataset-curate`, `dataset-build-cache`
- `ml-corpus`, `ml-audit-signals`, `ml-harvest-brushes`, `ml-generate-controls`, `ml-repair-normalmaps`, `ml-synth-no-liquid`
- `export-tex-json`

### Inspect CLI

Project: `tools/inspect/WowViewer.Tool.Inspect`

Read-only format probing:

```powershell
dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- --help
```

Top-level areas: `archive`, `blp`, `m2`, `mdx`, `map`, `lit`, `pm4`, `wmo`.

### Desktop App (Paused)

Project: `src/viewer/WowViewer.App`

The desktop shell is **not the current focus**. It will be rebuilt from scratch once the v10 terrain model is fully trained. The existing shell exposes bounded CLI proofs (`m2-frame`, `mdx-gpu-frame`, `world-bootstrap`, etc.) but these are for transitional validation only.

---

## Prerequisites

- .NET 10 SDK
- PowerShell on Windows (or compatible shell)
- Python 3.11+ with PyTorch (for trainers only)
- Your own lawful game data, archives, or extracted client roots

## Bootstrap

```powershell
.\scripts\bootstrap.ps1
```

Optional evaluation repos:
```powershell
.\scripts\bootstrap.ps1 -IncludeOptional
```

## Build And Test

```powershell
# Full solution
dotnet build .\WowViewer.slnx -c Debug

# Tests (some fixtures may be missing in this checkout)
dotnet test .\WowViewer.slnx -c Debug
```

---

## Documentation Map

- [docs/validation/direct-v9-training-setup.md](docs/validation/direct-v9-training-setup.md) — legacy v9 direct dataset and training setup
- [docs/architecture/v10-stage2-terrain-synth-architecture-2026-04-27.md](docs/architecture/v10-stage2-terrain-synth-architecture-2026-04-27.md) — Stage 2 terrain model architecture, signal matrix, dataset composition, and validation-impact reference
- [docs/architecture/viewer-legacy-cutover-boundary-2026-04-17.md](docs/architecture/viewer-legacy-cutover-boundary-2026-04-17.md) — viewer ownership boundary (viewer restart planned post-v10)
- [docs/architecture/m2-native-client-research-2026-03-31.md](docs/architecture/m2-native-client-research-2026-03-31.md) — native-client M2 research
- [docs/architecture/m2/README.md](docs/architecture/m2/README.md) — M2 architecture and implementation handoff
- [docs/architecture/wdt-format-notes-2026-04-17.md](docs/architecture/wdt-format-notes-2026-04-17.md) — WDT format notes

---

## Data Policy

This repo is intended for Bring Your Own Data workflows.

Do not distribute proprietary game data, generated corpora derived from proprietary data, or trained model outputs that depend on copyrighted source assets.
