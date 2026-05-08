# wow-viewer

`wow-viewer` is the active standalone codebase for:

- reading World of Warcraft terrain and client data across multiple eras
- exporting model-ready NPZ terrain shards from real game clients
- inspecting formats like ADT, WDT, PM4, BLP, WMO, MDX, and M2
- converting Alpha-era maps to later formats
- serving as the foundation for the V14 terrain-model pipeline

It is not a toy viewer prototype anymore. The important part today is the data and format pipeline.

## What It Does

The project currently has four practical jobs:

1. **Shared WoW file I/O**
   Read Alpha, pre-release, Wrath, and Cataclysm terrain formats from staged clients.
2. **Dataset generation**
   Export NPZ shards with height, normals, alpha layers, liquid, masks, minimap, and placement-derived signals.
3. **Inspection and validation**
   Probe client files and generated outputs without opening the old legacy tools.
4. **Format conversion**
   Convert Alpha monolithic WDT terrain to LK ADT/WDT/WDL output.

## Current Status

### Done

- Unified terrain type system
- Native MPQ archive reader (`NativeMpqService`)
- Harvest/tensor-pack extraction for staged clients
- NPZ shard serialization
- Alpha placement export and resolved model names
- WL liquid fallback
- Minimap lookup via `md5translate`
- AlphaToLk conversion pipeline

### Validated

- `0.5.3.3368`
- `0.5.5.3494`
- `0.7.0.3694`
- `3.0.1.8303`
- `3.3.5.12340`
- `4.0.0.11927`

### Important Workflow Rule

For full dataset preparation, use the **harvest-first** path.

Use:

- `WowViewer.Tool.Harvest`
- `harvest-map-mpq`
- staged clients under `output\tmp\wowarchive-clients\...`

Do **not** use the older converter-side `dataset-scan` / `dataset-audit` / `dataset-build-cache` chain as the primary shard builder for V14 work. Those commands are legacy manifest/audit helpers, not the canonical full-signal extraction path.

## Supported Client Eras

`wow-viewer` is built around real client data, not mock assets.

| Era | Example build | Notes |
|---|---|---|
| Alpha | `0.5.3.3368`, `0.5.5.3494` | Monolithic WDT terrain |
| Pre-release | `0.7.0.3694` | Early retail-style terrain |
| Wrath pre-release | `3.0.1.8303` | Archive-backed validation target |
| Wrath retail | `3.3.5.12340` | Main LK validation target |
| Cataclysm beta | `4.0.0.11927` | Split ADT / MCCV-era target |

## Quick Start

### 1. Build

```powershell
dotnet build .\wow-viewer\WowViewer.slnx -c Debug
```

### 2. Prepare a Full Multi-Client Dataset

This is the canonical batch command.

```powershell
pwsh -File ".\wow-viewer\scripts\run_full_shard_batch.ps1" `
  -OutputDir "I:\parp\parp-tools\wow-viewer\output\datasets\full_shard_batch_staged_native"
```

This script:

- discovers maps from `Map.dbc`
- reads only staged clients from `output\tmp\wowarchive-clients\...`
- harvests NPZ shards with `harvest-map-mpq`
- writes a harvest manifest
- optionally selects validation samples from the harvested NPZ files
- optionally renders visualization PNGs from the sampled shards

### 3. Watch Progress

```powershell
Get-Content "I:\parp\parp-tools\wow-viewer\output\datasets\full_shard_batch_staged_native.log" -Wait -Tail 50
```

### 4. Visualize Harvested NPZ Shards

```powershell
cd .\wow-viewer\data-harvester
uv run python scripts\visualize_npz.py "I:\parp\parp-tools\wow-viewer\output\datasets\full_shard_batch_staged_native\validation_samples" --output-dir "I:\parp\parp-tools\wow-viewer\output\datasets\full_shard_batch_staged_native\visualizations"
```

## Output Layout

The modern dataset workflow writes under `wow-viewer\output\datasets\...`.

Typical structure:

```text
wow-viewer/output/datasets/full_shard_batch_staged_native/
  map_lists/
  manifests/
  shards/
    0_5_3_3368/
      Azeroth/
      Kalimdor/
    3_3_5_12340/
      Azeroth/
      Northrend/
      ...
  validation_samples/
  visualizations/
```

## Main Tools

### Harvest CLI

```powershell
dotnet run --project .\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj -c Debug -- <command>
```

Key commands:

- `harvest-map-mpq`
- `harvest-map`
- `harvest-tile`
- `extract-unified`
- `synthetic-minimap`

Examples:

```powershell
dotnet run --project .\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj -c Debug -- harvest-map-mpq --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_5_3494\World of Warcraft" --map Azeroth --output-dir "I:\parp\parp-tools\wow-viewer\output\datasets\demo\0_5_5_3494\Azeroth"
```

```powershell
dotnet run --project .\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj -c Debug -- extract-unified --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft" --map Azeroth --tile-x 32 --tile-y 32 --output "I:\parp\parp-tools\wow-viewer\output\datasets\single_tile\Azeroth_32_32.npz"
```

### Inspect CLI

```powershell
dotnet run --project .\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -c Debug -- <command>
```

Useful commands:

- `map inspect`
- `pm4 inspect`
- `pm4 audit`
- `blp inspect`
- `wmo inspect`
- `m2 inspect`
- `mdx inspect`

Example:

```powershell
dotnet run --project .\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -c Debug -- map inspect --input "I:\parp\parp-tools\wow-viewer\output\datasets\alpha_to_lk\Azeroth\Azeroth_32_32.adt"
```

### Converter CLI

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- <command>
```

Current important commands:

- `convert-alpha-to-lk`
- `dataset-list-maps`
- `detect`

Example:

```powershell
dotnet run --project .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug -- convert-alpha-to-lk --input "I:\parp\parp-tools\datasets\0_5_5_3494\Azeroth\Azeroth.wdt" --output "I:\parp\parp-tools\wow-viewer\output\datasets\alpha_to_lk\Azeroth"
```

## Data Signals in NPZ Shards

Depending on build and map contents, harvested shards can include:

- `height_257`
- `height_129`
- `height_65`
- `height_33`
- `height_17`
- `mcnr_normal_xyz`
- `mcal_alpha_pack_256`
- `mcly_texture_ids`
- `mcly_layer_mask`
- `hole_mask_16`
- `minimap_rgb_256`
- `mclq_surface_height`
- `mclq_type_mask`
- object and placement-derived masks
- metadata and provenance fields

This is the real training contract. The point is to preserve decoded game signals, not render pretty screenshots and pretend they are ground truth.

## AlphaToLk Conversion

The Alpha-to-LK terrain converter is validated.

Examples already proven:

- `0.5.5` Azeroth: `755/755` tiles
- `0.5.5` Kalimdor: `972/972` tiles
- `0.5.5` EmeraldDream: `256/256` tiles

Still open:

- AreaID crosswalk wiring
- split ADT output for later clients
- reverse `LkToAlpha` port

## Repository Layout

| Path | Purpose |
|---|---|
| `src/core/WowViewer.Core` | shared contracts, terrain models, dataset manifests |
| `src/core/WowViewer.Core.IO` | file readers, writers, archive access, DBC helpers |
| `src/core/WowViewer.Core.PM4` | PM4 library |
| `src/core/WowViewer.Core.Runtime` | runtime-side consumers |
| `tools/harvest` | canonical NPZ shard builder |
| `tools/inspect` | format inspection and validation |
| `tools/converter` | converters and some legacy helper commands |
| `data-harvester/scripts` | Python visualization and validation helpers |
| `output/datasets` | canonical dataset output root |

## What To Show People

If you want a short demo flow:

1. Run `harvest-map-mpq` on a staged client map.
2. Open one generated `.npz` with `visualize_npz.py`.
3. Show that the shard contains real decoded terrain signals, not screenshots.
4. Show `convert-alpha-to-lk` on an Alpha WDT.
5. Show `map inspect` validating the produced files.

That tells the actual story of the project much better than waving around stale one-off tools.

## Development Notes

- staged clients under `output\tmp\wowarchive-clients\...` are the canonical archive-backed inputs
- real dataset outputs belong under `wow-viewer\output\datasets\...`
- use `uv` for Python under `wow-viewer\data-harvester`
- bring your own lawful game data

## See Also

- `docs/architecture/wow-viewer-full-porting-roadmap.md`
- `docs/architecture/v14-model-and-refactor-plan-2026-05-06.md`
- workspace `AGENTS.md`

## Data Policy

Bring Your Own Data.

Do not distribute proprietary client data, harvested corpora, or derived outputs from copyrighted game assets.
