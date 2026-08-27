# WoWViewer Comprehensive User Guide

**Application Version**: `v0.5.2.2`  
**Platform Support**: Windows x64, Linux x64, macOS (Apple Silicon arm64 / Intel x64)

`WoWViewer` is an interactive 3D desktop application for inspecting, analyzing, rendering, and reconciling World of Warcraft client data from pre-release Alpha 0.5.3 through Cataclysm 4.0.x.

---

## Table of Contents
1. [Getting Started](#1-getting-started)
2. [Command-Line Launch Arguments](#2-command-line-launch-arguments)
3. [User Interface Overview](#3-user-interface-overview)
4. [Camera Controls and Shortcuts](#4-camera-controls-and-shortcuts)
5. [World Viewing and Streaming](#5-world-viewing-and-streaming)
6. [Camera Path Studio](#6-camera-path-studio)
7. [Audio System and Positional Emitters](#7-audio-system-and-positional-emitters)
8. [Lighting, Atmosphere, and Time of Day](#8-lighting-atmosphere-and-time-of-day)
9. [PM4 Object Reconciliation Workbench (Spec 176)](#9-pm4-object-reconciliation-workbench-spec-176)
10. [Troubleshooting and FAQ](#10-troubleshooting-and-faq)

---

## 1. Getting Started

### Using Prebuilt Binaries
1. Download the self-contained archive for your platform from the [GitHub Releases Page](https://github.com/akspa0/parp-tools/releases).
2. Extract the archive into any directory. No .NET runtime installation is required.
3. Run `ParpToolsWoWViewer` (or `ParpToolsWoWViewer.exe` on Windows).

> [!NOTE]
> Native OS open-file dialogs are supported on Windows. On Linux and macOS, launch content using the `--game-path`, `--build`, and `--world` CLI options.

### Running from Source
Ensure [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) is installed:
```powershell
# Clone and build
git clone https://github.com/akspa0/parp-tools.git
cd parp-tools/wow-viewer
dotnet build WowViewer.slnx -c Debug

# Launch viewer
dotnet run --project src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

### Setting Up Client Data
`WoWViewer` reads game archives directly from game folders containing MPQ files or loose directory structures:
1. In the viewer, click **Open Client** in the top-left Navigator bar.
2. Select your game root (e.g. `H:\CLIENTS\World of Warcraft 3.3.5a` or `H:\CLIENTS\WoW-0.5.3.3368-Client`).
3. Select the detected client build from the **Build** dropdown.
4. Expand **World Maps** in the Navigator to load any world (e.g. `Azeroth`, `Kalimdor`, `Development`).

---

## 2. Command-Line Launch Arguments

You can pass command-line arguments to automate startup or load specific worlds directly:

| Argument | Description | Example |
|---|---|---|
| `--game-path <path>` | Path to staged game directory or MPQ archive root | `--game-path "H:\CLIENTS\WoW 3.3.5a"` |
| `--build <version>` | Pin the specific build version | `--build "3.3.5.12340"` |
| `--world <path>` | Virtual path or file path to WDT/ADT map | `--world "World\Maps\Azeroth\Azeroth.wdt"` |
| `--listfile <path>` | Supply an external custom listfile | `--listfile "listfile.csv"` |
| `--loose-map-overlay <dir>` | Overlay loose ADT/WDT files onto MPQ archives | `--loose-map-overlay "C:\LooseMaps"` |
| `--full-load` | Disable bounded streaming and load entire map into memory | `--full-load` |
| `--verbose` | Enable debug logging to console and log window | `--verbose` |
| `--capture-shot <name>` | Automatically take a screenshot after startup | `--capture-shot "stormwind_entrance"` |
| `--capture-output <dir>` | Directory to save captured screenshots | `--capture-output "output/captures"` |
| `--capture-after-frames <n>` | Frame delay before triggering screenshot | `--capture-after-frames 60` |
| `--exit-after-capture` | Terminate application immediately after capture | `--exit-after-capture` |

---

## 3. User Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WoWViewer                                                     [_][□][X]    │
├──────────────┬───────────────────────────────────────────────┬──────────────┤
│  NAVIGATOR   │                                               │  WORKBENCH   │
│  (Left Bar)  │               3D VIEWPORT                     │  (Right Bar) │
│              │                                               │              │
│ • Sources    │  [WASD + Mouse Look Camera]                   │ Destinations:│
│ • Build Pick │                                               │ • Quick      │
│ • File Tree  │  [Bounded Terrain Residency Ring: Radius 2-3] │ • Inspect    │
│ • World Maps │                                               │ • Scene      │
│ • Phase Maps │  [Dynamic Lighting, Sky, Fog, Audio Emitters] │ • Utilities  │
│              │                                               │ • Experiment │
├──────────────┴───────────────────────────────────────────────┴──────────────┤
│ [Scene Toggles]   [Subzone / Area Readout]   [Audio: ON/MUTED]   [FPS / ms] │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Left Navigator Sidebar
- **Sources & Builds**: Switch active client archives and select build definitions.
- **File Explorer**: Browse virtual paths inside MPQs (models, textures, sounds, databases).
- **World Maps**: One-click loading of all terrain maps discovered in the client.
- **Phase Map Selector**: Filter phased content for Cataclysm and later expansions.

### 2. Bottom Status and Action Bar
- **Scene Toggles**: Quick buttons for Terrain, WMOs, M2 Doodads, Liquids, Wireframe, Portals.
- **Location Readout**: Displays the current `AreaTable` Zone and Subzone name.
- **Master Audio Button**: Toggle master sound output (`AUDIO: ON` / `AUDIO: MUTED`).
- **Performance Readout**: Real-time FPS, frame time (ms), and draw call statistics.

### 3. Right Workbench Destinations
The right sidebar contains five primary destinations, each with specialized tool pages:

#### **Quick**
- Scene visibility switches, fast wireframe toggles, and performance summary.

#### **Inspect**
- **Context**: Information about the currently hovered or selected terrain vertex, model, or WMO.
- **Scene Investigation**: Deep hierarchy of resident ADT tiles, chunks, and submeshes.
- **MCNK / ADT**: Raw chunk headers, layer masks, shadow maps, and vertex height grids.
- **World Context**: Map properties, bounds, and global flags.
- **Archeology / Animations**: M2 skeletal animation player with sequence selection, scrubbing, speed control, and bone transform displays.
- **Actions**: Trigger manual garbage collection and reload current scene.

#### **Scene**
- **Placements**: Searchable list of all placed models (`MDDF`) and world models (`MODF`) on active tiles.
- **LOD**: Level-of-detail thresholds and distance cull distances.

#### **Utilities**
- **Minimap**: Interactive 2D map view with camera tracking and triple-click teleportation.
- **Audio**: Resident emitter list, `SoundEntries` preview player, gain sliders, and 3D emitter pins.
- **Capture**: Screenshot capture and Camera Path Authoring Studio.
- **Log Viewer**: Filterable application log stream with export capability.
- **Perf Profiler**: GPU and CPU frame breakdown.
- **Asset Catalog**: Global index of all enumerated client assets.
- **Taxi Paths**: Flight master trajectory visualization.

#### **Experimental**
- **PM4 Reconciliation**: Full Spec 176 PM4 guide matching, visual diffing, and ADT patching.
- **Terrain Lab**: Tensor extraction and normal/heightmap experiment workbench.
- **Converters**: On-the-fly Alpha $\leftrightarrow$ LK format converter.
- **Population**: Procedural doodad scatter testing.

---

## 4. Camera Controls and Shortcuts

### Global Navigation
| Key / Input | Action |
|---|---|
| `W` / `A` / `S` / `D` | Move camera forward, left, backward, right |
| `Q` / `E` | Move camera vertically down / up |
| **Right Mouse Button + Drag** | Look around (first-person mouse look) |
| **Shift (Hold)** | Camera speed boost ($3\times$) |
| **Mouse Wheel** | Adjust camera base movement speed |
| `Tab` | Toggle UI chrome visibility (fullscreen mode) |
| `I` | Toggle right Workbench sidebar |
| `M` | Toggle fullscreen Minimap overlay |
| `P` | Jump directly to PM4 Reconciliation tab |
| **Triple-Click on Minimap** | Teleport camera instantly to clicked tile |

---

## 5. World Viewing and Streaming

### Bounded Tile Admission
To ensure smooth frame rates and bounded memory usage, `WoWViewer` streams terrain using a camera-centered residency ring:
- **Inner Ring (Radius 2)**: 25 ADT tiles kept resident around the camera.
- **Directional Lookahead**: Additional tiles are prioritized in the forward view frustum.
- **WDL Underlay**: Distant horizons outside the high-detail ring render low-detail WDL terrain meshes, preventing abrupt popping.

---

## 6. Camera Path Studio

Located under **Utilities > Capture**, the Camera Path Studio lets you author smooth cinematic fly-throughs, record viewpoints, and export native WoW camera assets.

### Camera Path Keybindings (Active when Capture page is open)
| Key | Action |
|---|---|
| `K` | Insert camera keyframe at current playhead position and orientation |
| `U` | Update selected keyframe to current camera pose |
| `Delete` | Delete selected keyframe |
| `Space` | Play / Pause camera path animation |
| `Left` / `Right` Arrow | Select previous / next keyframe |
| `Ctrl + Left` / `Ctrl + Right` | Retime selected keyframe |
| `Z` / `X` | Roll camera counter-clockwise / clockwise |
| `Home` / `End` | Jump to start / end of path |
| `Ctrl + S` | Save camera path as portable `.json` project |
| `Ctrl + E` | Export camera path as a native `.m2` model |

---

## 7. Audio System and Positional Emitters

Located under **Utilities > Audio**, the OpenAL audio engine renders true 3D spatialized sound:
- **Positional Emitters**: As ADT tiles stream in, sound emitters declared in `MCSE` chunks and `MCNK` liquid flags are automatically registered.
- **3D Emitter Pins**: Enable visual pins in the viewport to see emitter positions:
  - 🟡 **Amber**: Positional sound effect (`MCSE`).
  - 🔵 **Cyan**: Liquid / water sound emitter.
  - 🟣 **Purple**: Ambient environmental emitter.
- **SoundEntries Previewer**: Search and audition any sound effect in `SoundEntries.dbc`.

---

## 8. Lighting, Atmosphere, and Time of Day

- **Alpha 0.5.3 World Clock**: The Alpha client advances on a 2,880-unit world cycle (24 real minutes). The lighting engine synchronizes terrain vertex lighting, ambient colors, directional sun/moon vectors, and skybox palettes.
- **Time Slider**: Located in **Quick** and **Inspect > World Context**, dragging the slider freezes the clock at a specific time of day for photography and inspection.
- **LIT & DBC Fallback**: Evaluates `.lit` lighting files when present, with automatic fallback to `Light.dbc` and `LightParams.dbc`.

---

## 9. PM4 Object Reconciliation Workbench (Spec 176)

Located under **Experimental > PM4**:

```
 ┌────────────────────────────────────────────────────────┐
 │ PM4 Guide: World\Maps\Development\Development_32_48.pm4 │
 │ Companion ADT: Development_32_48_obj0.adt              │
 ├────────────────────────────────────────────────────────┤
 │ [1. Parse PM4 Guide] ──> [2. Run Geometric Alignment]  │
 │                                                        │
 │ Proposals:                                             │
 │ • Seg 0x14A: Move HumanMale.m2 (+2.4m, rot: 45°)      │
 │ • Seg 0x14B: Substitute Unknown -> GoldMine.wmo       │
 │                                                        │
 │ [Preview Overlay]   [Apply Guarded Patch]   [Undo]     │
 └────────────────────────────────────────────────────────┘
```

### Step-by-Step Workflow:
1. **Load Companion Files**: Select the target PM4 guide and its corresponding `_obj0.adt` tile.
2. **Review Alignments**: The reconciliation engine compares PM4 convex hulls and bounding surfaces (`MSUR`, `MSPV`, `MSCN`) against placed objects in the ADT.
3. **Inspect Confidence**: Each proposed placement displays an alignment confidence score (`exp(-d/25)`).
4. **Guarded Apply**: Click **Apply** to write changes to `output/projects/<map>/<timestamp>/`. The operation verifies file hashes, creates a JSON provenance sidecar, and registers an undo step.
5. **Undo / Redo**: Press `Ctrl+Z` in the editor or click **Undo** to restore the previous state immediately.

---

## 10. Troubleshooting and FAQ

### Q: Why is terrain black or missing textures?
**A**: Ensure your client root points to the folder containing the `Data` directory (or loose `World\` folders). If the client uses MPQ archives, verify `ArchiveCatalog` has enumerated the listfiles.

### Q: Why are some dense city interiors slow?
**A**: In release `v0.5.2.2`, large multi-district WMOs (such as Stormwind) submit all interior groups simultaneously. Optimize performance by reducing the view distance slider in **Quick > LOD**.

### Q: Audio reports "OpenAL soft library missing"?
**A**: Ensure `soft_oal.dll` (Windows) or `libopenal.so` (Linux) / `libopenal.dylib` (macOS) is present in the application folder. The viewer continues normally with audio disabled if the library is not found.
