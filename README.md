# parp-tools

Preservation, conversion, archaeology, and visualization tooling for World of Warcraft data.

The primary active path in this repository is `gillijimproject_refactor`, with `parp-tools WoW Viewer` as the flagship application and `WoWMapConverter` / `Pm4Research.Core` as the core supporting libraries behind much of the current format and reconstruction work.

## Primary Active Project

### parp-tools WoW Viewer

`parp-tools WoW Viewer` is the active world viewer and research surface in this repository.

- Product name: `parp-tools WoW Viewer`
- Source path: `gillijimproject_refactor/src/MdxViewer`
- Current release line: `v0.4.5`
- Detailed project overview: `gillijimproject_refactor/README.md`
- Detailed viewer guide: `gillijimproject_refactor/src/MdxViewer/README.md`

This is no longer just a model viewer. The active app is already being used as a multi-era world viewer, terrain debugger, PM4 inspection surface, asset browser, export front end, SQL-driven spawn viewer, and conversion utility shell.

### What The Active Stack Already Does

- views world data from `0.5.3` through `4.0.0.11927`
- carries additional later-era terrain support paths into parts of `4.3.4`, though that band remains explicitly untested
- reads and converts WMO `v14`, `v16`, and `v17`
- supports MDX / M2 inspection and export workflows
- includes built-in map, terrain, WMO, VLM, and texture-transfer tooling in the active UI
- supports Alpha-Core SQL-driven NPC and gameobject population injection in the viewer
- provides PM4 overlay analysis, PM4/WMO correlation, and PM4 export tooling

### Conversion Coverage Worth Calling Out

The repo is stronger on conversion than the current top-level README used to suggest.

- `WoWMapConverter` is an active format and conversion library, not a side experiment
- the viewer ships built-in UI entry points for map conversion, WMO conversion, VLM export, and terrain texture transfer
- Alpha-era, Wrath-era, and `4.0.0.11927` era terrain workflows all exist in the active tree
- the repo contains real cross-era reconstruction work rather than only one-direction viewers or one-off exporters

## Supported Range

- Documented support range: `0.5.3` through `4.0.0.11927`
- Additional support exists for later `4.0.x` terrain variants
- Untested support paths also exist through parts of `4.3.4`

That range is the documented target, not a blanket claim that every subsystem has equal runtime signoff on every build. The active documentation distinguishes between build validation and real-data runtime validation, and that distinction matters here.

## Quick Start

From the repository root:

### 1. Bootstrap vendored dependencies

PowerShell:

```powershell
.\gillijimproject_refactor\setup-libs.ps1
```

### 2. Build the active viewer

```powershell
dotnet build .\gillijimproject_refactor\src\MdxViewer\MdxViewer.sln -c Debug
```

### 3. Run the active viewer

```powershell
dotnet run --project .\gillijimproject_refactor\src\MdxViewer\MdxViewer.csproj
```

Inside the viewer, the intended workflow is:

1. Open a base client through `File > Open Game Folder (MPQ)...`
2. Choose the correct client build when prompted
3. Open a world, loose map folder, or standalone asset from the UI

## Active Highlights

### Viewer and research workflow

- world viewing for Alpha, Wrath-era, and selected Cataclysm-era data paths
- terrain, liquid, minimap, taxi, and PM4 inspection workflows
- standalone WMO and MDX/M2 viewing plus GLB export
- SQL-driven world spawn loading from Alpha-Core
- render-quality controls, object inspection, and debugging utilities aimed at real dataset archaeology rather than pure presentation

### PM4 and reconstruction workflow

- standalone PM4 research library under `gillijimproject_refactor/src/Pm4Research.Core`
- active PM4 overlay loading inside the viewer
- PM4/WMO correlation reports and export
- PM4 OBJ export from the live viewer

### Built-in tooling

- map converter UI
- WMO converter UI
- VLM export UI
- terrain texture transfer UI
- asset-catalog export with automated multi-angle screenshot capture

## Repository Layout

### Active work

- `gillijimproject_refactor/`
	- Primary active development tree
	- Contains `parp-tools WoW Viewer`, `WoWMapConverter`, `Pm4Research.Core`, and the current documentation / memory-bank state

### Important active subprojects inside the refactor tree

- `gillijimproject_refactor/src/MdxViewer/`
	- Active viewer application
- `gillijimproject_refactor/src/WoWMapConverter/`
	- Format and conversion library used by the viewer and related tools
- `gillijimproject_refactor/src/Pm4Research.Core/`
	- Standalone PM4 decode and audit foundation
- `gillijimproject_refactor/src/MDX-L_Tool/`
	- MDX archaeology and parser utility
- `gillijimproject_refactor/WoWRollback/`
	- Supporting rollback, reconstruction, and UniqueID-oriented tooling that now feeds post-`v0.4.5` viewer planning directly

### Other top-level folders

- `parpToolbox/`
	- Older but still useful PM4 and related format research/tooling
- `PM4Tool/`
	- PM4-focused experiments and utilities
- `ADTPrefabTool/`
	- Separate tooling path, not the primary active viewer/converter path
- `archived_projects/`
	- Historical projects preserved for reference, not primary active work

## Screenshots And Capture

The repo still needs a stronger curated screenshot gallery.

- automated asset-catalog screenshot capture already exists in the active viewer pipeline
- broader UI, world-scene, and feature-showcase screenshot automation is still a follow-up task rather than a completed repo artifact

## Validation Reality

This repository does not currently have broad first-party automated regression coverage for the active viewer path.

What that means in practice:

- successful builds matter, but build success is not the same as runtime signoff
- some recent fixes, such as the `v0.4.5` minimap repair, have targeted real-data runtime confirmation
- many other areas are still build-validated only and should be described that way

If you need exact validation status for a recent viewer or terrain change, check the memory-bank files under `gillijimproject_refactor/memory-bank/` and `gillijimproject_refactor/src/MdxViewer/memory-bank/`.

## Documentation

Start with these files:

- `gillijimproject_refactor/README.md`
- `gillijimproject_refactor/src/MdxViewer/README.md`
- `gillijimproject_refactor/memory-bank/activeContext.md`
- `gillijimproject_refactor/memory-bank/progress.md`
- `gillijimproject_refactor/memory-bank/data-paths.md`

Those files are more current and more precise than older scattered notes elsewhere in the repository.

## Archived And Experimental Work

This repository also preserves older and experimental work, including:

- Alpha WDT analysis tooling
- PM4 decoding and reconstruction experiments
- older toolbox and exporter efforts
- work-in-progress rewrites and proof-of-concept tools

Those paths remain useful for archaeology and reference, but they should not be treated as the primary maintained surface unless a task explicitly targets them.

## Disclaimer

This project is not an official Blizzard Entertainment product and is not affiliated with or endorsed by Blizzard Entertainment or World of Warcraft.


