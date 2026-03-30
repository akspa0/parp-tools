# Parp Tools — Gillijim Project (C# refactor)

WoW preservation, analysis, and visualization tooling. The primary active application in this tree is [MdxViewer](src/MdxViewer/), a .NET 10 / OpenGL 3.3 viewer for world data, models, WMOs, minimaps, PM4 overlay research, and format-conversion workflows.

This README is intentionally high level. The detailed viewer workflow lives in [src/MdxViewer/README.md](src/MdxViewer/README.md).

## Current state

- Primary viewer path: [src/MdxViewer](src/MdxViewer).
- Current viewer focus is not just standalone model viewing anymore. The app now acts as:
  - a world viewer for Alpha, Wrath, and selected Cataclysm-beta era client data
  - a PM4 inspection surface for the development-map workflow
  - a terrain and liquid debugging tool
  - a WMO and MDX/M2 inspection/export tool
  - a front end for several converter and validation utilities already in this repo

## v0.4.6.1 release snapshot

- `parp-tools WoW Viewer` `0.4.6.1` is the current release target in this tree.
- Recent viewer-facing changes that materially shape this release:
  - fullscreen and docked minimap interaction/camera-marker behavior was repaired and then runtime-confirmed by the user on the fixed development minimap dataset
  - taxi route inspection now has route picking, animated actor controls, asset override workflow, return-to-world flow, and saved override persistence
  - render-quality controls expose live texture filtering changes for already loaded assets
  - PM4 inspection/export workflows, minimap disk cache, and recent object-visibility tuning are all part of the active viewer path
  - PM4 overlay decoding and placement are now substantially closer to correct on the development map after the latest camera-window, tile-remap, empty-carrier, and linked-group placement fixes
  - PM4 hover information now has better WoW-styled tooltip display with clearer PM4 context for quick object inspection
  - the first rendering-performance slices now remove duplicate MDX scene walks and defer WMO doodad expansion, while a deeper render-layer/submission refactor remains the next likely seam
- Validation reality for the release target:
  - the minimap blocker now has targeted runtime user signoff on the real development minimap data
  - most other recent viewer slices are still build-validated only unless noted otherwise in the memory-bank files
  - there is still very little first-party automated regression coverage for the active viewer

## Version support

- Actively supported viewer range: `0.5.3` through `4.0.0.11927`.
- Additional terrain support exists for later `4.0.x` ADTs.
- The current codebase also includes untested support paths for later game data through `4.3.4`, especially in the split-ADT terrain pipeline.
- Some `5.x` data may already work in parts of the active pipeline, but this is still exploratory and not a signed-off support tier.
- `6.x+` support is future work; likely doable in phases via existing Warcraft.NET-era format coverage and map-upconversion paths, but still unresolved.
- Practical rule: `0.5.3` through `4.0.0.11927` is the documented support range; later `4.0.x` and `4.3.4` era data should be treated as promising but not yet broadly signed off.

## Quick start

### 1. Bootstrap vendored library dependencies

PowerShell:

```powershell
./setup-libs.ps1
```

### 2. Build the active viewer solution

```powershell
dotnet build .\src\MdxViewer\MdxViewer.sln -c Debug
```

The repository builds successfully in GitHub Actions across multiple platforms. The active viewer itself is developed around a .NET 10 and OpenGL 3.3 workflow and should not be described as Windows x64-only from a repository-build perspective.

### 3. Run the viewer

```powershell
dotnet run --project .\src\MdxViewer\MdxViewer.csproj
```

Optional flags:

```text
--verbose
--full-load
```

- `--verbose` enables console logging instead of suppressing noisy library output.
- `--full-load` loads all terrain tiles at startup instead of using the default AOI streaming path.

You can also pass a direct file path after the flags to open a loose asset immediately.

## How the current viewer is used

The old README flow that implied automatic build inference from a game path is stale. The current workflow is explicit.

### Open a base client

Use `File > Open Game Folder (MPQ)...`.

- The viewer opens a build-selection dialog before MPQ load.
- Saved base-client entries preserve both path and build identity.
- `Load Loose Map Folder Against Saved Base` is the intended workflow for patched overlays or mixed-source map debugging.

### Open worlds or loose assets

After a base client is open, you can:

- load WDT or ADT data through the in-app world/file browser
- open standalone WMO and MDX/M2 assets
- attach loose map folders on top of a saved MPQ base
- use the development-map PM4 tooling from the viewer UI

### Use the viewer as a debugging surface

The active viewer now includes UI and workflows for:

- fixed left/right sidebars by default, with dockable navigator/inspector panels as an opt-in `View` mode
- hideable chrome with `Tab`
- minimap zoom, pan, cached tiles, and guarded teleport
- repaired fullscreen minimap marker/click behavior on the development minimap dataset
- PM4 overlay inspection and PM4/WMO correlation browsing
- PM4 OBJ export
- terrain-hole rebuild override for inspection
- render-quality filtering controls for already loaded textures
- log viewer and perf windows

## Current MdxViewer capabilities

### World and terrain

- Alpha monolithic WDT terrain
- 0.6.0 split ADT terrain
- 3.3.5 split ADT terrain
- WMO-only map support
- AOI streaming with persistent caching
- minimap rendering with disk cache
- terrain alpha/shadow visualization and terrain-hole inspection toggles

### PM4 and development-map workflow

- map-wide PM4 overlay loading in the active world scene
- background PM4 decode/apply path instead of blocking UI load
- recovery for zero-CK24 PM4 object families
- offline PM4 OBJ export from the viewer
- PM4/WMO correlation report browsing and JSON export

### Object rendering

- standalone MDX and WMO viewing
- world-scene MDX/WMO placement rendering
- WMO doodads, liquids, and transparent material ordering fixes
- M2-family material follow-ups for UV/env-map and wrap/blend recovery

### Render quality

- a `Render Quality` window now exposes live texture filtering changes for already loaded assets
- current practical modes are `Nearest`, `Bilinear`, and `Trilinear`
- multisample object antialiasing is only available if the active GL context already exposes sample buffers
- current accepted direction: filtering matters, explicit MSAA work is not required right now

### Export and tooling

- GLB export for standalone assets and map tiles
- map converter UI
- WMO converter UI
- VLM export UI
- terrain texture transfer UI
- asset-catalog export already includes automated multi-angle model screenshots

## Documentation gaps still worth closing

- the READMEs now reflect the current viewer more accurately, but they still need a stronger visual walkthrough
- a screenshot guide now exists at [src/MdxViewer/docs/ui-screenshot-guide.md](src/MdxViewer/docs/ui-screenshot-guide.md), with a drop-folder path for candidate gallery images
- there is already automated screenshot capture infrastructure for asset-catalog exports; a separate automated pass for UI/menu showcase screenshots would be a reasonable follow-up if we want marketing-quality documentation assets

## Repository structure

### Primary active projects

- [src/MdxViewer](src/MdxViewer) — active viewer, renderer, PM4 inspection, export UI
- [src/WoWMapConverter](src/WoWMapConverter) — format and conversion library used by several tools
- [src/MDX-L_Tool](src/MDX-L_Tool) — MDX archaeology / parser utility
- [src/Pm4Research.Core](src/Pm4Research.Core) — standalone PM4 rediscovery and audit path

### Supporting areas

- `AlphaWDTAnalysisTool/`
- `DBCTool.V2/`
- `WoWRollback/`
- `reference_data/`
- `test_data/`
- `memory-bank/`

## Validation note

The active viewer tree has little first-party regression coverage. Build success matters, but it is not runtime signoff.

As of Mar 25, 2026:

- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-transpose-repair/` passed after the final minimap repair
- no automated tests were added for the latest viewer slices
- the fullscreen minimap blocker now has targeted real-data runtime user confirmation on the fixed development minimap dataset
- terrain, PM4, liquid, taxi, and most renderer follow-ups still require additional real-data runtime validation before being described as broadly verified

