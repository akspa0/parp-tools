# WoW Model Viewer (MdxViewer)

A C# 3D model viewer for World of Warcraft Alpha 0.5.3 assets. Renders **MDX** models and **WMO** world objects with an ImGui-based UI, and exports to **GLB** (binary glTF) with materials.

## Features

- **MDX Viewer** — Alpha 0.5.3 model rendering with per-geoset textures
- **WMO Viewer** — World Map Object rendering (v14 Alpha format) with per-group color coding
- **GLB Export** — Export loaded models to binary glTF with materials and textures via SharpGLTF
- **MPQ Data Source** — Browse and load models directly from MPQ archives using internal `(listfile)` entries — version-accurate for any client (Alpha, Classic, TBC, WotLK)
- **Loose File Scanning** — Automatically discovers loose model files in the game directory alongside MPQ contents
- **ImGui UI** — Menu bar, file browser with search/filter, model info panel, status bar
- **Free-Fly Camera** — WASD movement, Q/E vertical, right-click drag to look around
- **Replaceable Texture Resolver** — Maps replaceable texture IDs to actual BLP paths using DBC files
- **Geoset/Group Visibility** — Toggle individual geosets (MDX) or groups (WMO) on/off
- **Doodad Set Selection** — Switch between different doodad configurations in WMO models

## Requirements

- .NET 9.0 SDK
- Windows (StormLib/NativeMpqService dependency for MPQ reading)
- OpenGL 3.3+ capable GPU

## Build

```bash
cd src/MdxViewer
dotnet build
```

## Usage

### Launch with no arguments (GUI mode)
```bash
dotnet run --project src/MdxViewer
```
Opens the viewer with an empty viewport. Use the menu to load files.

### Launch with a file
```bash
dotnet run --project src/MdxViewer -- path/to/model.mdx
dotnet run --project src/MdxViewer -- path/to/building.wmo
```

### CLI GLB export (headless)
```bash
dotnet run --project src/MdxViewer -- path/to/model.mdx --export-glb output.glb
```

## Controls

| Input | Action |
|-------|--------|
| **Right drag** | Rotate camera (yaw/pitch) |
| **Scroll wheel** | Move camera forward/backward |
| **W / S** | Move camera forward/backward |
| **A / D** | Move camera left/right |
| **Q / E** | Move camera down/up |
| **W** | Toggle wireframe |
| **Esc** | Quit |

**Note:** The camera uses a free-fly system. Use WASD for movement, Q/E for vertical movement, and right-click drag to look around.

## Menu

### File
- **Open File...** — Load an MDX or WMO from disk by path
- **Open Game Folder (MPQ)...** — Point to a WoW install directory; loads all MPQ archives and populates the file browser
- **Export GLB...** — Export the currently loaded model to GLB with materials
- **Quit** — Exit the viewer

### View
- **Wireframe** — Toggle wireframe rendering
- **Reset Camera** — Reset camera to default position
- **File Browser** — Toggle the left-side file browser panel
- **Model Info** — Toggle the right-side model info panel

### Help
- **About** — Shows viewer information in the status bar

## Output Folder

All exported and cached files are written to an `output/` folder next to the executable:

```
output/
├── export/     # GLB exports
└── cache/      # Temporary files extracted from MPQ for loading
```

No files are written to `%TEMP%` or other system directories.

## Loading from MPQ Archives

1. **File > Open Game Folder (MPQ)...**
2. Enter the path to your WoW game directory (e.g. `H:\053-client`)
3. The viewer loads all MPQ archives and extracts their internal `(listfile)` entries
4. Loose files in `Data/` subdirectories (World, Creature, Dungeons, etc.) are also scanned
5. The file browser populates with all discovered files — only files that actually exist in this client version
6. Filter by extension (`.mdx`, `.wmo`, `.m2`, `.blp`), search by name
7. Double-click a file to load and render it

The viewer uses `WoWMapConverter.Core.Services.NativeMpqService` — a pure C# MPQ reader that handles encrypted, compressed, and multi-sector files without requiring StormLib. File lists come from MPQ-internal `(listfile)` entries, not external community listfiles, ensuring version accuracy.

### Tested with Alpha 0.5.3

```
8 MPQ archives loaded (42,060 files from internal listfiles)
1,866 loose files from Data/Dungeons/
43,926 total known files — all version-accurate
```

## File Browser Usage

The file browser panel (left side) provides powerful filtering and search capabilities:

### Extension Filter
- **Type dropdown**: Select file type to filter (`.mdx`, `.wmo`, `.m2`, `.blp`)
- Only files matching the selected extension are displayed

### Search Filter
- **Search box**: Type to filter files by name (case-insensitive)
- Searches match anywhere in the filename
- Real-time filtering as you type

### File List
- Shows filtered files with display names (filename only)
- Hover over a file to see the full virtual path
- **Double-click** to load and render the selected file
- Click to select (highlighted in blue)

### Status Information
- Displays the current data source name
- Shows the count of filtered files matching current criteria

**Note:** The file browser limits results to 5,000 files for performance. Use the search filter to narrow down results when browsing large archives.

## Status Bar

The status bar at the bottom of the window displays:
- Current loading status and progress messages
- Error messages when operations fail
- Information about loaded models (file name, geoset/group counts, vertex counts)
- Data source status (e.g., "No data source loaded" or "Loaded: MPQ Source")

## Window Title

The window title updates to show the currently loaded file:
- Default: "WoW Model Viewer"
- With file: "WoW Viewer - [filename]"

## Model Info Panel

When a model is loaded, the right panel shows:

**MDX models:**
- Version, name, geoset count, vertex/triangle counts
- Materials, textures (with paths or replaceable IDs)
- Bone count, animation sequences with time ranges

**WMO models:**
- Version, group count, vertex/triangle counts
- Materials, textures, doodad sets/defs
- Portal and light counts
- Per-group breakdown (name, vertex count, triangle count)

### Geoset/Group Visibility

For both MDX and WMO models, the Model Info panel includes a **Visibility** section with:
- **All On / All Off** buttons to toggle all geosets/groups at once
- Individual checkboxes for each geoset (MDX) or group (WMO) to control visibility

### Doodad Set Selection (WMO Only)

When viewing WMO models, a **Doodad Set** dropdown appears in the Model Info panel:
- Select different doodad sets to see different object placements
- Each set represents a different configuration of doodads within the WMO

## Replaceable Texture Resolver

The viewer includes a replaceable texture resolver that maps replaceable texture IDs to actual BLP texture paths using DBC (Database Client) files.

### How It Works

The resolver uses the following chain:
1. Model path → `CreatureModelData.ID`
2. `CreatureDisplayInfo.ModelID` → `TextureVariation`
3. Replaceable ID 1 = `TextureVariation[0]`, ID 2 = `TextureVariation[1]`, ID 11 = `TextureVariation[0]` (alias)

### Supported Replaceable IDs

| Replaceable ID | Texture Variation | Description |
|----------------|-------------------|-------------|
| 1 | TextureVariation[0] | Creature Skin 1 |
| 2 | TextureVariation[1] | Creature Skin 2 |
| 3 | TextureVariation[2] | Creature Skin 3 |
| 11 | TextureVariation[0] | Creature Skin 1 (alias) |
| 12 | TextureVariation[1] | Creature Skin 2 (alias) |
| 13 | TextureVariation[2] | Creature Skin 3 (alias) |

### Loading DBC Files

The resolver automatically loads DBC files from the MPQ archives when you open a game folder. It uses:
- `DBFilesClient\CreatureModelData.dbc` - Maps model paths to Model IDs
- `DBFilesClient\CreatureDisplayInfo.dbc` - Maps Model IDs to texture variations

The resolver supports any WoW version with appropriate DBD definitions and DBC files. Known build aliases:
- `0.5.3` → `0.5.3.3368`
- `0.5.5` → `0.5.5.3494`
- `0.6.0` → `0.6.0.3592`
- `3.3.5` → `3.3.5.12340`

### WoWDBDefs Requirements

For the replaceable texture resolver to work, the viewer needs WoWDBDefs definition files. The viewer searches for these in the following locations (in order):
1. `../../lib/WoWDBDefs/definitions` (relative to executable)
2. `./definitions` (next to executable)
3. `./WoWDBDefs/definitions` (next to executable)

If WoWDBDefs definitions are not found, the replaceable texture resolver will be unavailable, but the viewer will still function normally.

### Build Version Inference

The viewer automatically infers the WoW build version from the game folder path:
- Path contains `0.5.3` → Build `0.5.3.3368`
- Path contains `0.5.5` → Build `0.5.5.3494`
- Path contains `0.6.0` → Build `0.6.0.3592`
- Path contains `3.3.5` → Build `3.3.5.12340`
- Fallback: Checks for patch MPQ files with "patch" and "3" in the name

This inferred build is used to load the correct DBC definitions for replaceable texture resolution.

## Architecture

```
src/MdxViewer/
├── Program.cs                      # Entry point
├── ViewerApp.cs                    # Main app: window, GL, ImGui, camera, renderer lifecycle
├── DataSources/
│   ├── IDataSource.cs              # Abstraction for game data access
│   ├── LooseFileDataSource.cs      # Read from extracted files on disk
│   ├── MpqDataSource.cs            # MPQ + loose files via NativeMpqService
│   ├── MpqDBCProvider.cs           # DBC file reader from MPQ archives
│   └── ListfileDownloader.cs       # Optional community listfile downloader (for modern clients)
├── Rendering/
│   ├── ISceneRenderer.cs           # Common renderer interface
│   ├── Camera.cs                   # Free-fly camera (WASD + Q/E + right-drag)
│   ├── ModelRenderer.cs            # MDX model renderer (MdxRenderer)
│   ├── WmoRenderer.cs              # WMO renderer using WoWMapConverter.Core data model
│   └── ReplaceableTextureResolver.cs # Maps replaceable IDs to BLP paths via DBC
└── Export/
    └── GlbExporter.cs              # GLB export for MDX and WMO via SharpGLTF
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Silk.NET | 2.21.0 | OpenGL 3.3 windowing and rendering |
| Silk.NET.OpenGL.Extensions.ImGui | 2.21.0 | Dear ImGui integration |
| SixLabors.ImageSharp | 3.1.6 | Texture loading (PNG) |
| SharpGLTF.Toolkit | 1.0.2 | GLB binary export |
| DBCD | Latest | Direct DBC file reading for replaceable texture resolution |

| Project Reference | Purpose |
|-------------------|---------|
| MDX-L_Tool | MDX file parsing (MdxFile, geosets, materials, textures, bones) |
| WoWMapConverter.Core | WMO v14 parsing, MPQ archive service, listfile service |

## Supported Formats

| Format | Status | Notes |
|--------|--------|-------|
| MDX (Alpha 0.5.3) | ✅ Viewing + GLB export | Geosets, textures, materials |
| WMO v14 (Alpha) | ✅ Viewing + GLB export | Groups, materials, color-coded |
| WMO v17 (3.3.5) | 🔧 Planned | Requires root+group loader |
| M2 (modern) | 🔧 Planned | Requires M2 parser |
| MDX animations | 🔧 Planned | Bone hierarchy + keyframe playback |
| CASC storage | 🔧 Planned | For modern client data |
