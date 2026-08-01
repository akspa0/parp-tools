# WoWViewer

Active development target inside `parp-tools`.

World viewer, CLI toolchain, shared format libraries, and data-harvester for staged World of Warcraft client data.

## Current focus

- **Spec 122 `122-dataset-curation` — canonical C# curation layer.** Consolidates dataset quality classification (difficulty/coverage/lighting buckets + height-normal-mismatch/non-finite/synthetic-fidelity findings) into `WowViewer.Core.Curation` + `WowViewer.Tool.Harvest curate` subcommand. Every tile gets a bucket + finding record; no tile is ever silently dropped. Real-data validated on PVPZone02 (64/64 tiles). This is the repo's first C# Parquet writer.
- **Synthesized minimap export** — `WowViewer.Tool.Harvest synthetic-minimap` composes terrain-only and _liquid minimap PNGs directly from client BLP textures + MCLY/MCAL/MCNR. Supports any time of day via `--time-hours` (default 12:00 noon). The shared solar direction (`TerrainSolarDirection`) holds a fixed NW bearing with cycling elevation, matching the traced 0.5.3.3368 native client behavior.
- **Lighting fixes (2026-08-01):** hillshade Y-axis inversion fixed in `SynthesizedTrainingService` (NW light was rendering as SW); object capture Z backlighting fixed in `ObjectCaptureShader` (DirectX-vs-OpenGL winding mismatch); taxi panel pop-up removed (broken ImGui popup with no title bar).
- **V50 dataset pipeline** — active terrain reconstruction lane using merged WDL prior + detailer architecture. V50.2 substrate adds lattice + object-mask arrays.

## Hard boundaries

- `wow-viewer/` owns new implementation work.
- `gillijimproject_refactor/` is read-only reference.
- Staged clients only: `output/tmp/wowarchive-clients/`.
- Any `H:\CLIENTS` reference is stale and must be removed.

## Build

```powershell
dotnet build wow-viewer/WowViewer.slnx -c Debug
dotnet test wow-viewer/WowViewer.slnx -c Debug
```

CI: `.github/workflows/wowviewer-build.yml` builds the solution on Windows and compile-checks
the cross-platform target on Linux + macOS on every push/PR touching `wow-viewer/`; the test
suite runs as an informational (non-blocking) job. A `v*` tag push (or manual dispatch with
`workflow_dispatch`) publishes self-contained binaries for **win-x64, linux-x64, osx-arm64, and
osx-x64** and, for tags, creates a GitHub Release with the notes from
[docs/releases/](docs/releases/). BLP decoding uses ImageSharp everywhere (the old GDI+ path
was Windows-only at runtime), so Linux/macOS builds are functional — with one known limitation:
native file dialogs are Windows-only; use the CLI automation flags (`--game-path`, `--build`,
`--world`) on other platforms. `WowViewer.Tool.ValidationCapture` is deliberately Windows-only
by design (GPU capture via a hidden window).

## Run viewer

```powershell
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

Normal startup path:

1. Open staged game folder.
2. Pick explicit client build.
3. Load world from viewer UI.

Automation path exists for capture/debug flows through `--game-path`, `--build`, `--world`, and capture flags.

## Data-harvester setup

```powershell
cd wow-viewer/data-harvester
uv sync
```

Use `uv run ...` from that directory for dataset, training, and inference work.

## Main surfaces

| Surface | Purpose | Path |
|---------|---------|------|
| Viewer app | 3D world viewer | `src/viewer/WoWViewer/` |
| Shared libraries | format/domain/runtime code | `src/core/` |
| CLI tools | inspect, convert, harvest, validation, animfarm | `tools/` |
| Tests | C# xUnit | `tests/` |
| Data harvester | Python dataset/training/inference | `data-harvester/` |
| Specs | feature packs | `specs/` |
| Architecture docs | long-form design notes | `docs/architecture/` |

## Canonical docs

- [AGENTS.md](/I:/parp/parp-tools/wow-viewer/AGENTS.md)
- [docs/DOCUMENTATION-STATUS.md](/I:/parp/parp-tools/wow-viewer/docs/DOCUMENTATION-STATUS.md)
- [docs/CLI-TOOLS.md](/I:/parp/parp-tools/wow-viewer/docs/CLI-TOOLS.md)
- [data-harvester/README.md](/I:/parp/parp-tools/wow-viewer/data-harvester/README.md)
- [docs/WoWViewer/USERGUIDE.md](/I:/parp/parp-tools/wow-viewer/docs/WoWViewer/USERGUIDE.md)
- [memory-bank/activeContext.md](/I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md)
- [memory-bank/progress.md](/I:/parp/parp-tools/wow-viewer/memory-bank/progress.md)

## Historical surfaces

- `specs/archived/` — closed or superseded.
- `specs/086-*` and `specs/087-*` — superseded by Spec 088; keep only as evidence.
- `plans/` — old planning notes unless a live spec points there.
- `docs/MdxViewer-legacy-documentation.tar.gz` — archive only.