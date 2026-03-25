# MdxViewer — WoW World Viewer

MdxViewer is the active viewer/debugging application in this repo. It is no longer just a standalone MDX/WMO viewer. It acts as a combined world viewer, PM4 inspection surface, terrain debugger, asset browser, and export front end.

## Runtime and build requirements

- .NET 10 SDK
- OpenGL 3.3 capable GPU/driver
- Alpha-Core SQL data for world npc and gameobject data (0.5.3-0.5.5, works on all versions of game data)

## Supported Data Files

- WMOv14/v16/v17 Reading and conversion support (0.5.3-4.0.0.11927)
- MDX/M2 reading (Partial conversion support)
- Alpha WDT terrain (0.5.3-0.5.5)
- Retail ADT terrain (ADTv18)
  - Support for MCLQ and MH2O chunk types
  - Support for later `4.0.x` ADT variants in the active terrain path
  - Untested support paths for later-era split-ADT variants up through `4.3.4`
- Minimap tile rendering and caching  


## Supported data range

- Documented support range: `0.5.3` through `4.0.0.11927`
- Additional support exists for later `4.0.x` ADTs
- The terrain pipeline also contains untested later-era support paths through `4.3.4`

## Build and run

From the repository root:

```powershell
dotnet build .\src\MdxViewer\MdxViewer.sln -c Debug
dotnet run --project .\src\MdxViewer\MdxViewer.csproj
```

Optional launch flags:

```text
--verbose
--full-load
--partial-load
```

- `--verbose` keeps console logging enabled.
- `--full-load` forces all tiles to load at startup.
- `--partial-load` is accepted for backward compatibility, but AOI streaming is already the default.

You can also pass a loose file path after the flags.

## v0.4.5 release snapshot

- `parp-tools WoW Viewer` `0.4.5` is the current release target in this tree.
- The previously broken fullscreen minimap/top-right Designer Island case is now fixed in the active branch and has runtime user confirmation on the fixed development minimap dataset.
- Other recent viewer slices such as taxi override workflow, object-culling tuning, and WMO baked-light prototyping should still be treated as build-validated unless a narrower runtime note says otherwise.

## Current startup workflow

The current usage path is explicit and UI-driven.

### Base client workflow

Use `File > Open Game Folder (MPQ)...`.

- The viewer asks for an explicit build selection before loading MPQs.
- Known-good base clients can be saved and reused.
- `Load Loose Map Folder Against Saved Base` is the intended path for overlay debugging and mixed-source development-map work.

### Loose asset workflow

Use `File > Open File...` or pass a file path on launch.

- standalone MDX/M2 models can be inspected directly
- standalone WMOs can be inspected directly
- world files can be opened after the relevant data source/base client is configured

## What the viewer currently supports

### World and terrain viewing

- Alpha monolithic WDT terrain
- 0.5.3 through 4.0.0.11927 era world map support with MPQ data sources or loose map folders
- WMO-only world maps
- AOI terrain streaming
- minimap tile rendering and cache reuse
- area names, POIs, taxi overlays (nodes and routes), fullscreen minimap, AlphaWDT <-> 3.3.5 ADT converter and lots more

### PM4 inspection and development-map workflows

- map-wide PM4 overlay loading
- background PM4 decode/apply path
- zero-CK24 family recovery for PM4 overlays
- PM4 alignment utilities
- PM4/WMO correlation report window
- PM4/WMO correlation JSON export
- PM4 OBJ set export from the viewer

### Rendering and material handling

- standalone MDX rendering with animation support
- standalone WMO rendering with doodads and liquids
- world-scene MDX and WMO placement rendering
- liquid ordering follow-ups so terrain liquids no longer always overdraw transparent model layers
- transparent geoset priority-plane ordering in MDX models
- M2-family UV/env-map and wrap/blend follow-ups

### UI and inspection workflows

- dockable navigator and inspector panels
- `Tab` hide-chrome mode
- floating log viewer
- floating perf window
- minimap window plus fullscreen minimap with `M`
- render-quality window
- PM4/WMO correlation window
- asset catalog window
- taxi route selection, route override workflow, and return-to-world flow

### Export and utilities

- GLB export for standalone assets
- GLB map-tile export
- map converter UI
- WMO converter UI
- VLM export UI
- terrain texture transfer UI

## Current controls

| Input | Action |
|------|--------|
| `W A S D` | Free-fly camera movement |
| `Q / E` | Vertical movement |
| Right mouse drag | Look around |
| Mouse wheel | Forward/back camera motion in the scene; zoom in minimap windows |
| `Tab` | Hide or restore UI chrome |
| `M` | Toggle fullscreen minimap |
| Triple-click same minimap tile | Teleport camera |
| Drag on minimap | Pan the minimap |

## Important current UI surfaces

### View menu

The `View` menu is now part of normal use, not an afterthought. It exposes:

- dock panels
- file browser
- model info
- terrain controls
- minimap
- log viewer
- perf
- render quality
- PM4/WMO correlation

### Render Quality

The active render-quality work is intentionally narrow and real.

- texture filtering modes: `Nearest`, `Bilinear`, `Trilinear`
- changes apply live to already loaded standalone models, WMOs, terrain textures, and world cached renderers
- object MSAA is only toggleable when the active OpenGL context actually provides sample buffers

Current accepted direction:

- filtering is the meaningful visual upgrade already landed
- explicit multisampled context work is not required right now

### Minimap

Current minimap behavior differs from older docs.

- teleport is guarded by triple-clicking the same tile
- drag-vs-click handling is stricter to avoid accidental teleports
- zoom, pan, and minimap window visibility persist in viewer settings
- decoded minimap tiles cache to disk under `output/cache/minimap`
- the `v0.4.5` repair closed the earlier fullscreen marker-alignment and top-right teleport regression on the fixed development minimap dataset

### Screenshot/export automation

- asset-catalog export already supports automated multi-angle model screenshots

### Terrain debugging

The viewer now supports terrain-hole inspection without mutating source data.

- terrain holes can be ignored during mesh rebuild for debugging
- this is viewer-side only
- ADT hole flags on disk are not edited by this feature

## SQL world population

The viewer can inject optional SQL-driven world spawns from an alpha-core checkout.

Expected root:

- `external/alpha-core`

Required SQL files:

- `etc/databases/world/world.sql`
- `etc/databases/dbc/dbc.sql`

In the viewer:

1. open the world-objects SQL population section
2. set the SQL root or use the detected submodule path
3. load spawns for the current map
4. select a SQL gameobject instance to inspect animation controls when available

## Current limits and boundaries

- the render-quality slice is sampler-quality control, not a general post-processing framework
