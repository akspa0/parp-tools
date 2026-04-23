# wow-viewer

`wow-viewer` is the new home for shared World of Warcraft file readers, runtime code, viewer tooling, and dataset-generation workflows that are being extracted from the larger `parp-tools` workspace.

This repo is in active migration. It already contains usable tooling, but it is not yet a drop-in replacement for every legacy viewer or exporter path.

## What You Can Use Today

- `WowViewer.App`: desktop shell plus bounded CLI proof commands for M2, MDX, and world-session inspection.
- `WowViewer.Tool.Inspect`: read-only inspection CLI for archive, BLP, M2, MDX, map, LIT, PM4, and WMO data.
- `WowViewer.Tool.Converter`: conversion and dataset CLI for file detection, direct dataset manifests, cache building, and ML helper workflows.
- Shared libraries under `src/core`: the canonical implementation target for new format and runtime work in this repo.

## Current Limits

- The desktop app is a real shell, not yet a finished replacement viewer.
- PM4 is the most mature library area today; other format families are at mixed levels of summary, parse, and runtime ownership.
- WMO shared I/O now includes version-aware material and embedded-group mesh document readers plus root portal and doodad ownership, and `WowViewer.App` now consumes that seam for a bounded standalone WMO batch preview that resolves material textures when available. This is still not the full world 3D renderer.
- Dataset and training flows are usable, but still documented as Bring Your Own Data workflows and still rely on some compatibility scripts downstream.

## Prerequisites

- .NET 10 SDK
- PowerShell on Windows or a compatible shell environment for bootstrap scripts
- Python only if you plan to use the training and dataset scripts documented under `docs/validation/`
- Your own lawful game data, archives, or extracted client roots

## Bootstrap

Clone the baseline dependencies into `libs/`:

```powershell
./scripts/bootstrap.ps1
```

Optional evaluation repos:

```powershell
./scripts/bootstrap.ps1 -IncludeOptional
```

Bash is also supported:

```bash
./scripts/bootstrap.sh
./scripts/bootstrap.sh --include-optional
```

## Build And Test

Build the full solution:

```powershell
dotnet build .\WowViewer.slnx -c Debug
```

Run the current test suites:

```powershell
dotnet test .\WowViewer.slnx -c Debug
```

## Main Entry Points

### Desktop App

Project: `src/viewer/WowViewer.App`

Open the desktop shell:

```powershell
dotnet run --project .\src\viewer\WowViewer.App\WowViewer.App.csproj --
```

The app also exposes bounded CLI commands for deterministic proofs and inspection-oriented captures, including:

- `viewer`
- `m2-frame`
- `m2-gpu-frame`
- `mdx-gpu-frame`
- `mdx-visual-regression`
- `world-bootstrap`
- `world-frame`
- `world-placement-audit`
- `m2-bounds`

Show usage:

```powershell
dotnet run --project .\src\viewer\WowViewer.App\WowViewer.App.csproj -- --help
```

### Inspect CLI

Project: `tools/inspect/WowViewer.Tool.Inspect`

The inspect tool is the read-only surface for probing supported formats and archives.

Top-level areas currently include:

- `archive`
- `blp`
- `m2`
- `mdx`
- `map`
- `lit`
- `pm4`
- `wmo`

Show usage:

```powershell
dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- --help
```

### Converter And Dataset CLI

Project: `tools/converter/WowViewer.Tool.Converter`

The converter tool handles file detection plus the direct dataset and ML helper workflows currently owned by `wow-viewer`.

Key commands include:

- `detect`
- `dataset-scan`
- `dataset-merge`
- `dataset-audit`
- `dataset-curate`
- `dataset-build-cache`
- `ml-corpus`
- `ml-audit-signals`
- `ml-harvest-brushes`
- `ml-generate-controls`
- `ml-repair-normalmaps`
- `ml-synth-no-liquid`
- `export-tex-json`

Show usage:

```powershell
dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- --help
```

## Training And Dataset Workflows

The current direct `v9` training workflow is documented separately in [docs/validation/direct-v9-training-setup.md](docs/validation/direct-v9-training-setup.md).

That document covers:

- staging BYOD client roots locally for repeated scans
- building the direct training cache from real game data with `run_v9_direct_pipeline.ps1`
- building a separate development-map compatibility cache
- splitting PM4-bearing development entries into a training subset and a non-overlapping holdout with `WowViewer.Tool.Converter dataset-split-pm4`
- launching `train_v9_optimized.py` with the merged corpus and the non-PM4 development holdout

## Repository Layout

- `src/viewer/WowViewer.App`: desktop shell and bounded viewer-facing CLI proofs
- `src/core/WowViewer.Core`: shared core contracts and format-independent primitives
- `src/core/WowViewer.Core.IO`: shared file readers and format loaders
- `src/core/WowViewer.Core.Runtime`: runtime consumers and scene-building seams
- `src/core/WowViewer.Core.PM4`: PM4 parser, services, and research-facing contracts
- `src/tools-shared/WowViewer.Tools.Shared`: shared tooling support code
- `tools/inspect/WowViewer.Tool.Inspect`: inspection CLI
- `tools/converter/WowViewer.Tool.Converter`: conversion and dataset CLI
- `tests/`: current automated tests
- `docs/`: user-facing and architecture-facing documentation

## Documentation Map

- [docs/validation/direct-v9-training-setup.md](docs/validation/direct-v9-training-setup.md): direct dataset and training setup
- [docs/architecture/viewer-legacy-cutover-boundary-2026-04-17.md](docs/architecture/viewer-legacy-cutover-boundary-2026-04-17.md): current viewer ownership boundary
- [docs/architecture/low-resolution-world-image-alignment-plan-2026-04-22.md](docs/architecture/low-resolution-world-image-alignment-plan-2026-04-22.md): low-resolution world-image alignment and streamed viewer verification plan
- [docs/architecture/audio-engine-plan-2026-04-21.md](docs/architecture/audio-engine-plan-2026-04-21.md): first audio-engine and game-engine subsystem plan
- [docs/architecture/m2/README.md](docs/architecture/m2/README.md): M2 architecture and implementation handoff
- [docs/architecture/m2-native-client-research-2026-03-31.md](docs/architecture/m2-native-client-research-2026-03-31.md): native-client M2 research notes
- [docs/architecture/wdt-format-notes-2026-04-17.md](docs/architecture/wdt-format-notes-2026-04-17.md): WDT notes and research context

## Data Policy

This repo is intended for Bring Your Own Data workflows.

Do not distribute proprietary game data, generated corpora derived from proprietary data, or trained model outputs that depend on copyrighted source assets.
