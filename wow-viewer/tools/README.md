# WoWViewer CLI Tooling

This is the canonical entry point for `wow-viewer` command-line tooling. The tools are thin
wrappers over the core libraries in `src/core/`; they should inspect, generate, convert, harvest,
or capture data without changing the base viewer, renderer, terrain loader, or format readers.

Use these commands from the repository root in PowerShell 7.

## Execution Pattern

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/<tool-folder>/<project>/<project>.csproj -c Debug -- `
  <command> [options]
```

For tool-specific syntax, run:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- --help
```

Client roots are runtime inputs. `H:\CLIENTS` is a known-good local library path in this workspace,
but portable scripts and tests should take client roots as parameters.

## Tool Map

| Folder | Project | Main Commands / Purpose |
|---|---|---|
| `inspect/` | `WowViewer.Tool.Inspect` | Multi-format inspection and research commands: `archive`, `assets`, `wtf`, `audio`, `blp`, `m2`, `mdx`, `map`, `adt`, `lit`, `light`, `pm4`, `pd4`, `wmo`, and `rosetta-generate`. |
| `converter/` | `WowViewer.Tool.Converter` | Dataset scans, map extraction, ML corpus helpers, Alpha/LK terrain conversion, WMO version conversion, M2/MDX conversion, and roundtrip validation. |
| `harvest/` | `WowViewer.Tool.Harvest` | Terrain tensor harvest, MPQ-backed map harvest, minimap synthesis, streaming harvest, map discovery, holes/tileset extraction, relief-to-map, object capture, and curation helpers. |
| `capture/` | `WowViewer.Tool.Capture` | Headless terrain tile render via `render`. |
| `validation-capture/` | `WowViewer.Tool.ValidationCapture` | Production scene capture, batch capture, and render profiling via `capture`, `capture-batch`, and `profile-render`. |
| `mask-validate/` | `WowViewer.Tool.MaskValidate` | ADT texture/mask validation for loose ADTs or MPQ-backed archive paths. |
| `wdl-read/` | `WowViewer.Tool.WdlRead` | WDL MARE read and synthetic WDL lattice generation via `read` and `synth`. |
| `wmo-minimap/` | `WowViewer.Tool.WmoMinimap` | Minimap archive probes and WMO minimap DBC-chain investigation. |
| `enrich/` | `WowViewer.Tool.V22Enrich` | V18 store asset enrichment stream generation for M2, WMO, and BLP references. |
| `scripts/` | PowerShell/Python helpers | Repo-local helper scripts. Keep long, GPU, or broad corpus runs operator-owned. |

The expanded historical command reference remains at [docs/CLI-TOOLS.md](../docs/CLI-TOOLS.md).
Prefer this README for current boundaries, project paths, and safety rules.

## Common Workflows

### Inspect Client Data

Inspect a model, map, archive, or PM4 file through the shared I/O layer:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  m2 inspect --archive-root "H:\CLIENTS\World of Warcraft 3.3.5a" --virtual-path "Creature\Murloc\Murloc.m2"
```

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  map inspect --archive-root "H:\CLIENTS\WoW-0.5.3.3368-Client"
```

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  pm4 inspect --input "World\Maps\Azeroth\Azeroth_32_48.pm4"
```

### Generate Rosetta Calibration Maps

`rosetta-generate` builds synthetic offline calibration maps with labelled terrain and object
placements for PM4/object matching work.

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  rosetta-generate `
  --client-root "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --output "output\rosetta_alpha" `
  --map-name "RosettaAlpha" `
  --format alpha `
  --pedestal-height 4.0 `
  --pedestal-bevel 12.5
```

Current Alpha Rosetta boundary:

- Alpha map bytes are written directly by `AlphaWdtWriter.Build(...)`.
- Rosetta minimap BLPs are generated separately under `Textures\Minimap\<mapName>\mapYY_XX.blp`.
- If a minimap screenshot shows the wrong object under the right label, first inspect minimap tile naming,
  MD5Translate behavior, pin placement, and YY_XX versus XX_YY capture assumptions before reopening map bytes.
- Do not change the base map renderer, terrain adapter, scene loader, culling policy, or protected writer
  to fix a generated minimap artifact.

### Convert Formats

The converter contains current and older dataset commands. For terrain format conversion, use the
current explicit command names:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug -- `
  convert-alpha-to-lk --input "World\Maps\Kalimdor\Kalimdor.wdt" --output "output\converted\Kalimdor_LK"
```

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug -- `
  convert-lk-to-alpha --input "output\converted\Development_LK" --output "output\converted\Development_Alpha.wdt"
```

Other conversion commands include `convert-split-adt-to-lk`, `convert-wmo-v17-to-v14`,
`convert-wmo-v14-to-v17`, `convert-m2-to-mdx`, `convert-mdx-to-m2`, and `validate-roundtrip`.

### Harvest Terrain, Minimap, and Dataset Signals

Use `WowViewer.Tool.Harvest` for terrain tensors, map discovery, minimap synthesis, and streaming
data extraction:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug -- `
  harvest-map-mpq `
  --client-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --map "Azeroth" `
  --output-dir "output\harvest\Azeroth"
```

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug -- `
  synthetic-minimap `
  --client-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --map "Azeroth" `
  --output-dir "output\minimap\Azeroth" `
  --per-tile `
  --whole-map
```

Long corpus harvests, GPU work, and training remain operator-owned. Prepare commands and dry-run checks,
then stop before launching broad jobs unless explicitly told to run them.

### Capture and Profile Rendering

Use `capture/` for a headless terrain tile render:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/capture/WowViewer.Tool.Capture/WowViewer.Tool.Capture.csproj -c Debug -- `
  render `
  --client-root "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --tile-name "Azeroth_30_48" `
  --output "output\captures\Azeroth_30_48.png"
```

Use `validation-capture/` when a spec needs production `WorldScene` proof or render profiling:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/WowViewer.Tool.ValidationCapture.csproj -c Debug -- `
  profile-render `
  --client-root "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --map-input "World\Maps\Kalimdor\Kalimdor.wdt" `
  --output "output\profiles\kalimdor-profile.json"
```

Capture output is evidence for the specific command, client root, map, camera, and options used.
It is not proof of unrelated renderer paths.

### Validate Masks and WDL Data

Validate an archive-backed ADT mask:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/mask-validate/WowViewer.Tool.MaskValidate/WowViewer.Tool.MaskValidate.csproj -c Debug -- `
  --archive "H:\CLIENTS\World of Warcraft 3.3.5a" "World\Maps\Azeroth\Azeroth_32_48.adt" "output\mask_validation"
```

Read WDL MARE data to NPZ:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead/WowViewer.Tool.WdlRead.csproj -c Debug -- `
  read --client-root "H:\CLIENTS\WoW-0.5.3.3368-Client" --map "Kalimdor" --output "output\wdl\kalimdor.npz"
```

### Probe Minimap/WMO Archive Data

Use this when investigating whether client archives provide minimap assets or DBC routing for WMOs:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/wmo-minimap/WowViewer.Tool.WmoMinimap/WowViewer.Tool.WmoMinimap.csproj -c Debug -- `
  list-minimap-blps --client-root "H:\CLIENTS\World of Warcraft 3.3.5a"
```

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/wmo-minimap/WowViewer.Tool.WmoMinimap/WowViewer.Tool.WmoMinimap.csproj -c Debug -- `
  probe-dbc-chain --client-root "H:\CLIENTS\World of Warcraft 3.3.5a"
```

### Enrich Dataset Stores

Generate a V22 enrichment stream from an existing V18 store:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/WowViewer.Tool.V22Enrich.csproj -c Debug -- `
  --v18-store "output\v18-store" `
  --client-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --output "output\enrichment\v22.stream" `
  --build-key "3_3_5_12340"
```

## Python Data Harvester

Python ML/data tooling lives in `wow-viewer/data-harvester/` and uses its own `uv` environment.
Run Python commands from that directory when imports depend on the package layout.

```powershell
cd I:/parp/parp-tools/wow-viewer/data-harvester
uv run python --help
```

Keep heavyweight training, CUDA jobs, and broad dataset generation user-launched unless the user
explicitly asks the agent to run a bounded job.

## Output Rules

- Write generated maps, captures, profiles, and datasets under explicit output directories.
- Do not write into a staged client root unless a command is explicitly designed for that and the
  operator asked for it.
- Record client root, build identity, command line, and output path when a result is used as evidence.
- Keep generated Rosetta map bytes and generated minimap artifacts conceptually separate.
- Prefer focused commands and fresh output folders when debugging generated data.

## Build and Test

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

For a single tool, build its project directly:

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug
```
