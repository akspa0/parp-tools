# WoWViewer v0.5.2

Active development target inside `parp-tools`.

World viewer, CLI toolchain, shared format libraries, and data-harvester for staged World of Warcraft client data.

## Current release

**v0.5.2** — lighting corrections, synthesized-minimap time-of-day support, WMO overlay option, and UI cleanup. See [docs/releases/v0.5.2.md](docs/releases/v0.5.2.md) for the full release notes.

## Current focus

- **Spec 134 `134-v60-unified-dataset-model` — v60 unified dataset and shadow-first terrain model.** Consolidates all harvested builds (0.5.3 through 4.0.0.11927) into a single v60 Zarr store with all signals including `terrain_shadow_256` (the textureless lighting component). Includes the shadow→height model (`direct_cnn_v112` with 1-channel input) that learns the physical relationship between terrain shadow and terrain height.
- **Spec 133 `133-unbaked-minimap-decomposition` — terrain_shadow_256 signal.** The C# compositor now emits the textureless lighting term (Lambert N·L + ambient + cast shadows) as a separate float32 array alongside the existing `minimap_rgb`. The model can learn the shadow signal independently of the texture.
- **Spec 132 `132-terrain-brush-signature-classification` — three-tier classification.** Every tile classified as strong, normal, or weak signal with published criteria. Integrated into the archaeology pipeline and tile inventory.
- **Spec 131 `131-pm4-scene-graph-doodads` — PM4 scene graph restored.** Full hierarchical scene outliner (Blender-style tree view) with tile/CK24/Part hierarchy, MSLK linking summary, click-to-select.

## Hard boundaries

- `wow-viewer/` owns all new implementation work.
- `gillijimproject_refactor/` is read-only reference code.
- `H:\CLIENTS` is the approved client library for validation and extraction (AGENTS.md Rule 9).
- `output/tmp/wowarchive-clients/` is optional staging and may be pruned.
- The user runs all training, GPU work, and client-backed proof. The assistant prepares the exact commands.

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

- [AGENTS.md](AGENTS.md)
- [docs/DOCUMENTATION-STATUS.md](docs/DOCUMENTATION-STATUS.md)
- [docs/CLI-TOOLS.md](docs/CLI-TOOLS.md)
- [data-harvester/README.md](data-harvester/README.md)
- [docs/WoWViewer/USERGUIDE.md](docs/WoWViewer/USERGUIDE.md)
- [memory-bank/activeContext.md](memory-bank/activeContext.md)
- [memory-bank/progress.md](memory-bank/progress.md)

## Historical surfaces

- `specs/archived/` — closed or superseded.
- `specs/086-*` and `specs/087-*` — superseded by Spec 088; keep only as evidence.
- `plans/` — old planning notes unless a live spec points there.
- `docs/MdxViewer-legacy-documentation.tar.gz` — archive only.