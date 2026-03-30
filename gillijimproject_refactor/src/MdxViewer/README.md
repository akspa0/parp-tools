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
- Some `5.x` data may already work in parts of the pipeline, but this is not a signed-off support tier yet.
- `6.x+` compatibility is future work and should be treated as research/integration backlog for now.

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

## v0.4.6.1 release snapshot

- `parp-tools WoW Viewer` `0.4.6.1` is the current release target in this tree.
- The previously broken fullscreen minimap/top-right Designer Island case is now fixed in the active branch and has runtime user confirmation on the fixed development minimap dataset.
- PM4 overlay decoding and placement are now much closer to correct on the development map after the latest runtime fixes.
- PM4 hover data display now uses a better WoW-styled info-tooltip path with clearer PM4 context for quick inspection.
- Other recent viewer slices such as taxi override workflow, object-culling tuning, and WMO baked-light prototyping should still be treated as build-validated unless a narrower runtime note says otherwise.
- Current render-performance work has started reducing duplicate scene walks and eager WMO doodad expansion, but a real render-layer/submission path is still the next major renderer seam.

## New User Quick Start (UI)

For first-time users, use this flow:

1. Use `File > Open Game Folder (MPQ)...`.
2. Choose your game folder root (the one containing `Data/`).
3. Pick the explicit client build in the build-selection dialog.
4. Load a world map from the left sidebar.
5. Use the right sidebar inspector for PM4 workbench, world objects, and map/debug controls.

Important:

- `Open File...` is mainly for direct standalone asset inspection (`.wmo`, `.mdx`, `.m2`) and loose files.
- For full world/map browsing, opening a base game folder first is the intended path.
- Fixed left/right sidebars are the startup default. Dock panels are opt-in from `View > Dock Panels`.

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

### PM4 terminology and evidence level

The PM4 data shown in the viewer is still partly reverse-engineered. Some labels are stable chunk names from wowdev. Some are local research aliases that are useful in the UI but are not proven original field names.

- `MSHD`
  - `MSHD` is the fixed-size PM4 or PD4 header chunk.
  - In the current corpus it decodes cleanly as eight `uint32` fields at offsets `0x00..0x1C`.
  - We have not closed the semantics of those fields yet.
  - The active confidence report still treats `MSHD` semantic meaning as unresolved, and the active `MdxViewer` PM4 path does not currently use `MSHD` to drive overlay placement or match scoring.
- `MSUR._0x1c` -> local alias: `CK24`
  - `CK24` is a viewer or research alias for the packed `MSUR` field at offset `0x1c`.
  - It is useful for grouping, coloring, and debugging PM4 objects.
  - `Ck24Type` means the upper byte of that packed field.
  - `Ck24ObjectId` means the low 16 bits of that packed field.
  - These are practical research labels, not confirmed official names.
- `MSLK._0x04` -> local alias: `LinkGroupObjectId`
  - This is a strong current grouping signal for many PM4 object families.
  - The viewer uses it as one clue when splitting or matching PM4 objects.
- `MPRL`
  - `MPRL` entries are linked position-reference records.
  - In the viewer they are mainly used as placement evidence and anchor hints, not as final proof of object identity.
- `MSUR.GroupKey`, `MSUR.AttributeMask`, `MSUR.MdosIndex`
  - These are local aliases for raw `MSUR` fields.
  - They are useful for debugging and grouping, but should be read as research shorthand rather than final format documentation.
- `uid` in PM4 match or correlation UI
  - `uid` is the matched world-placement unique id from `MODF` or `MDDF` candidates.
  - It does not currently mean that PM4 itself stores that same object uid.
  - The PM4 side gives us a PM4 object key like `CK24`, tile, part, and link evidence; the matched `uid` comes from nearby placed WMO or M2 records that the viewer ranks against that PM4 object.

Practical rule:

- Treat raw chunk names like `MSHD`, `MSLK`, `MSUR`, `MSCN`, and `MPRL` as stable.
- Treat local aliases like `CK24`, `GroupKey`, `AttributeMask`, `MdosIndex`, and `LinkGroupObjectId` as useful research vocabulary.
- Treat any undocumented PM4 field meaning as provisional unless a note here says the semantic confidence changed.

### Rendering and material handling

- standalone MDX rendering with animation support
- standalone WMO rendering with doodads and liquids
- world-scene MDX and WMO placement rendering
- liquid ordering follow-ups so terrain liquids no longer always overdraw transparent model layers
- transparent geoset priority-plane ordering in MDX models
- M2-family UV/env-map and wrap/blend follow-ups

### UI and inspection workflows

- fixed left/right sidebars are the startup default; dock panels are opt-in from the `View` menu
- PM4 workbench tabs (`Overlay`, `Selection`, `Correlation`) now keep their selected state instead of snapping back during normal clicking
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
- left sidebar
- right sidebar
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
- a dedicated screenshot guide for core viewer workflows now lives at [docs/ui-screenshot-guide.md](docs/ui-screenshot-guide.md)

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
