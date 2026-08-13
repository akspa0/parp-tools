# WoWViewer v0.5.2

The active viewer, format-tool, and data-harvester project inside `parp-tools`.
It is a .NET 10 desktop viewer for inspecting staged World of Warcraft client data,
terrain, WMO/M2/MDX placements, minimap inputs, and capture paths.

## Current release truth

v0.5.2 is the current application release line. The About box, Windows build,
cross-platform build, and shared project identity report `0.5.2`.

The viewer is functional but not feature-complete across every client era. Runtime
claims below distinguish implemented routes from visual/client proof that is still
pending.

## Current viewer surfaces

- World loading uses configured client roots and explicit build selection.
- Terrain, WMO placements, doodads, liquids, WDL/WL* data, minimap surfaces, and
  lower status/runtime statistics are available through the viewer shell.
- The bottom action and status bars remain the stable home for scene toggles and
  compact runtime facts.
- The right workbench contains Model, World, and Tools pages. The active UI-overhaul
  work presents those main pages in a vertical rail; nested pages remain incremental.
- Capture Automation and Camera Path are under Tools > Utilities > Capture.
- Camera paths can be authored from the current camera, imported from loose/native
  camera assets, saved as JSON, and exported as native M2. JSON preserves position,
  target, timing, FOV, and roll. Camera-path keyboard authoring is opt-in and scoped
  to the active Capture page.
- Camera-path preloading is bounded to sampled tiles/objects along the path. It is
  not a whole-map residency guarantee.
- Help > Keyboard Shortcuts shows global and active-page controls.

## Client-era support boundary

| Client era | Current truth |
|---|---|
| Alpha 0.5.3 | Terrain/WDT/WMO/MDX/BLP/DBC reading and viewer routes exist; every MDX visual path is not yet release-proven. |
| 0.6.x–0.10.x | Split/early terrain and chunked model routes exist with partial client-specific coverage. |
| 1.12.1 | Era-specific M2 parsing/runtime route exists; broad in-viewer visual proof remains incomplete. |
| 2.x | Profiled embedded-native M2 route is implemented; material/visibility proof on real client scenes is still pending. |
| 3.0.x | Profiled early-M2 route is implemented; 3.0.x visual coverage is provisional. |
| 3.3.5 | Terrain/WMO/M2/PM4 routes are the strongest late-client implementation path; representative real-client proof remains the release gate. |
| 4.0.0.x | Terrain/WMO and Cataclysm-era PM4 paths exist; client-specific rendering and performance remain partial. |

This table is a support boundary, not a claim that every map or asset in an era
renders correctly. Real-client validation must record the configured client root,
build identity, and observed result.

## Lighting, horizon, and liquids

The viewer has client/build-aware lighting, terrain fog, WDL low-detail terrain,
WL*/liquid routes, and synthesized-minimap time-of-day controls. These paths are
still under visual audit across client eras. DBC/DBD layout data is authoritative;
fallback behavior is diagnostic compatibility only and is not a substitute for a
correct build schema. Night sky/stars and final WDL horizon presentation remain
open visual-proof work.

## UI overhaul status

Spec 145 extends the earlier UI consolidation work. It covers contextual keybinds,
shortcut help, vertical workbench navigation, bounded navigator/minimap layout,
wrapped logs, persistent utility windows, honest placeholder pages, and release
documentation. Existing viewer routes and bottom bars remain in place while those
changes are proven incrementally. See [specs/145-wow-ui-overhaul/spec.md](specs/145-wow-ui-overhaul/spec.md).

## Hard boundaries

- All new implementation work belongs in `wow-viewer/`.
- `gillijimproject_refactor/` is read-only reference code.
- Client roots are configuration. `H:\CLIENTS` is the approved known-good library;
  do not hardcode it into source or portable configuration.
- The user runs training, GPU work, and client-backed visual proof. Build output is
  not rendering or FPS signoff.
- Do not distribute proprietary client data, harvested corpora, or derived assets.

## Build and run

```powershell
dotnet build wow-viewer/WowViewer.slnx -c Debug
dotnet test wow-viewer/WowViewer.slnx -c Debug
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

The normal workflow is: open a configured game folder, choose the explicit client
build, then load a world or standalone asset. Capture/debug automation accepts
configured `--game-path`, `--build`, and `--world` arguments.

## Repository layout

| Surface | Purpose |
|---|---|
| `src/viewer/WoWViewer/` | Desktop viewer and shell |
| `src/core/` | Shared format/domain/runtime libraries |
| `tools/` | Inspect, convert, harvest, capture, and animation tools |
| `tests/` | C# tests |
| `data-harvester/` | Python dataset/training/inference workflow |
| `specs/` | Feature specifications and implementation plans |
| `docs/architecture/` | Evidence and design records |
| `memory-bank/` | Continuity state |

## Documentation

- [Release notes](docs/releases/v0.5.2.md)
- [Viewer user guide](docs/WoWViewer/USERGUIDE.md)
- [CLI tools](docs/CLI-TOOLS.md)
- [Alpha audio catalog: what it is and how to inspect it](docs/architecture/alpha-audio-catalog.md)
- [Audio engine plan](docs/architecture/audio-engine-plan-2026-04-21.md)
- [UI surface inventory](docs/architecture/viewer-ui-surface-inventory.md)
- [Memory bank](memory-bank/activeContext.md)
