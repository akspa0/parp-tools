# parp-tools

Tooling for preservation, inspection, conversion, and visualization of World of Warcraft data.

This repository contains several generations of work. The primary active path today is the C# refactor under `gillijimproject_refactor`, with `parp-tools WoW Viewer` as the main user-facing application.

## Primary Active Project

### parp-tools WoW Viewer

`parp-tools WoW Viewer` is the active world viewer in this repository.

- Product name: `parp-tools WoW Viewer`
- Source path: `gillijimproject_refactor/src/MdxViewer`
- Current release line: `v0.4.5`
- Detailed project overview: `gillijimproject_refactor/README.md`
- Detailed viewer guide: `gillijimproject_refactor/src/MdxViewer/README.md`

The viewer is no longer just a standalone model viewer. It is currently used as a combined world viewer, terrain debugger, PM4 inspection surface, asset browser, export front end, and general-purpose research tool for multiple WoW client eras.

### Current Viewer Scope

- World viewing for Alpha, Wrath-era, and selected Cataclysm-beta data paths
- Terrain and liquid inspection
- WMO and MDX/M2 viewing
- Minimap rendering, cache reuse, and guarded teleport workflow
- PM4 overlay inspection and export workflows
- Asset browsing and GLB export utilities
- Converter and validation tool front end

### Supported Range

- Documented support range: `0.5.3` through `4.0.0.11927`
- Additional support exists for later `4.0.x` terrain variants
- Untested support paths also exist through parts of `4.3.4`

That does not mean every subsystem is equally validated across every supported build. The active repo documentation distinguishes between build validation and real-data runtime validation, and that distinction matters here.

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
	- Supporting tooling and modules used by some of the conversion and repair workflows

### Other top-level folders

- `parpToolbox/`
	- Older but still useful PM4 and related format research/tooling
- `PM4Tool/`
	- PM4-focused experiments and utilities
- `ADTPrefabTool/`
	- Separate tooling path, not the primary active viewer/converter path
- `archived_projects/`
	- Historical projects preserved for reference, not primary active work

## Current State Of The Repo

This is a research-heavy codebase with an active modernized path and a large amount of historical material.

The practical rules are:

- treat `gillijimproject_refactor` as the primary active tree
- treat `parp-tools WoW Viewer` as the main application in current use
- treat many older folders as reference material, experiments, or legacy tooling unless the task explicitly targets them

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


