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
2. Select your client root containing the Data directory or loose game files.
3. Select the detected client build from the **Build** dropdown.
4. Expand **World Maps** in the Navigator to load any world (e.g. `Azeroth`, `Kalimdor`, `Development`).

---

## 2. Command-Line Launch Arguments

You can pass command-line arguments to automate startup or load specific worlds directly:

| Argument | Description | Example |
|---|---|---|
| `--game-path <path>` | Path to staged game directory or MPQ archive root | `--game-path "C:\YourClients\WoW"` |
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

The left **Navigator** loads clients, builds, files and maps. The center is the 3D viewport.
The right **Workbench** always shows **Quick / Inspector / Editor / Archaeology**, whether the
workspace profile at the top is Viewer, Editor or Archaeology. Switching profiles while Quick
is selected keeps Quick open and retains the current controls' values.

### Quick

**Fog End** is first, followed by the other fog controls, time of day, camera speed/FOV and ADT
budget. Quick uses the same setting owners as the full panels. Render quality, lighting, audio,
capture and performance tools are available here; expand the relevant section for detailed
controls. Section **[?]** buttons provide help without filling the sidebar with instructions.

Direct menu shortcuts include **View > Log Console...**, **View > Performance & Profiling...**,
**View > Lighting Diagnostics...**, **Tools > Taxi Routes...** and **Tools > Audio Settings...**.
The former Scene and Utilities top-level tabs have been replaced; their tools remain reachable.

### Inspector

- **Context** shows the selected WMO, WMO doodad, MDX/M2, PM4 object or WL liquid. A selected
  WMO exposes its doodad sets here. When terrain is the target, it uses a pinned chunk first,
  then the hovered chunk, then the chunk under the camera. With no resident chunk, it shows
  an active-world overview.
- **Placements** provides the searchable resident WMO/MDX hierarchy and camera framing.
- **LOD & Budget** contains terrain/WDL budget and object visibility controls.

To pin terrain, left-click a visible chunk when the click does not select a scene object or
belong to an active editing tool. Context displays its area name/ID, MCNK flags, holes, height
range, textures and alpha layers, shadow/MCCV presence, liquids and tile placement counts.
Use **Frame Chunk**, **Copy Texture Summary**, **Copy Coordinates** or **Clear Chunk Selection**.
Pins are scoped to the current terrain source and expire when their chunk is no longer resident.

The **MCNK Flag Overlay** controls are in the same Context surface: Impassable, River, Ocean,
Magma, Slime, Shadows, MCCV and BakedShadows. Enable the overlay and Impassable to use
**Highlight Diagonal Weak Corners**. Old ADT/MCNK investigation entry points lead here.

### Editor

The Editor contains **Tasks & Workspace**, **Converters**, **3D Object Library** and
**Imports & Exports**, plus retained authoring tools.

The **3D Object Library** reads a Rosetta manifest or library JSON. Enter **Manifest / Library
Path** and choose **Load**, use **Browse...**, or choose **Scan Default**. Search or filter M2/WMO
entries and sort by size or name. The selected entry shows bounds, span, volume, footprint and
calibration occurrences. **Inspect 3D Model in Viewport** opens the asset through the active
client; **Copy Model Path** and **Set as Active Placement Model** support placement workflows.
The available entries depend on the loaded manifest and client assets.

**Imports & Exports** groups synthesized minimap generation/export, GLB scene/collision/map-tile
export, and terrain alpha/heightmap/MCCV import and export. Choose Current Tile, Loaded Tiles or
Whole Map where the operation supports that scope. Opening this dashboard does not start a job.

### Archaeology

Analysis remains under **Weak Signal & Stratigraphy**, **UniqueId Timeline**, **Layers &
Provenance**, **Playback & Capture**, **PM4 Analysis** and **Cartography**. Cartography includes
its merged-data output actions; the UI consolidation does not itself implement additional map
composition capabilities.

**Playback & Capture** combines archaeology range playback with **Camera Paths & Video Capture**.
Use **Apply playback to next capture** to arm the next queued capture with the configured
archaeology playback range. In its Capture Automation selector, configure the `ffmpeg` executable,
output folder, container and frame rate, then use **Start Video Recording** / **Stop Video
Recording**. Switch the selector to Camera Path for **Play**, **Play + Video** and **Stop**. A
scene-only recording samples the viewport before ImGui; use `Tab` before recording only when a
full-window scene is desired. Verify the configured executable and output file on the active
machine before relying on a recording.

### Bottom bar and legacy layout

The bottom bar retains scene toggles, location/area readout, master audio and frame statistics.
The legacy non-tab layout remains supported through shared content; this guide describes the
default tab layout.

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

Located under **Quick > Capture** and **Archaeology > Playback & Capture**, the Camera Path Studio
lets you author smooth cinematic fly-throughs, record viewpoints, and export native WoW camera
assets.

Open **Tools > Panels > Camera Path** for a direct shortcut. Load the intended map, position the
camera and click **Add Current Camera Key**. Move to another viewpoint and add a second key,
then click **Play**. Interactive playback advances while terrain and objects stream. **Stop**
ends playback; **Loop** repeats the path.

**Preload path before capture** controls capture readiness. **Warm Path** prepares the route,
**Release Warmup** releases its residency, and **Play + Video** waits for enabled warmup before
recording. Paths retain their map/build binding; an unrelated map or build is rejected and
changing the active map during playback stops it.

### Taxi riding

Open **Tools > Taxi Routes...**, use **Load Taxi Paths** if needed, select a route and click
**Ride Selected Route**. Choose **Cockpit** or **Chase** in **Ride Camera Mode**. Starting
a ride enables taxi display and mount actors. Once attached, the selected ride continues to
advance even if route visibility or selection filters change; asset loading does not gate its
route position. Use **Detach Ride Camera** to return to free flight. Starting a camera path detaches
the taxi camera. A world-source change also detaches the ride.

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

Located under **Quick > Audio**, the OpenAL audio engine renders true 3D spatialized sound:
- **Positional Emitters**: As ADT tiles stream in, sound emitters declared in `MCSE` chunks and `MCNK` liquid flags are automatically registered.
- **3D Emitter Pins**: Enable visual pins in the viewport to see emitter positions:
  - 🟡 **Amber**: Positional sound effect (`MCSE`).
  - 🔵 **Cyan**: Liquid / water sound emitter.
  - 🟣 **Purple**: Ambient environmental emitter.
- **SoundEntries Previewer**: Search and audition any sound effect in `SoundEntries.dbc`.

---

## 8. Lighting, Atmosphere, and Time of Day

- **Alpha 0.5.3 World Clock**: The Alpha client advances on a 2,880-unit world cycle (24 real minutes). The lighting engine synchronizes terrain vertex lighting, ambient colors, directional sun/moon vectors, and skybox palettes.
- **Time Slider**: Located in **Quick** and **Quick > Lighting**, dragging the slider freezes the clock at a specific time of day for photography and inspection.
- **LIT & DBC Fallback**: Evaluates `.lit` lighting files when present, with automatic fallback to `Light.dbc` and `LightParams.dbc`.

---

## 9. PM4 Object Reconciliation Workbench (Spec 176)

Located under **Archaeology > PM4 Analysis**:

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

## 10. Phased Terrain Dual-Map Overlay (Spec 135 & 137)

In Cataclysm (4.x) and later expansions, game zones feature dynamic phases where sparse ADT tiles modify specific regions of a continent (e.g. `Gilneas`, `Gilneas2`, `GilneasPhase1` over `Azeroth`).

### Using Phased Map Overlays:
1. **Load Base Map**: Open the primary continent map (e.g. `Azeroth`) from the **World Maps** panel.
2. **Select Secondary Phase Map**:
   - In the **World Maps** sidebar, scroll to **Phased Terrain Secondary Overlay**.
   - Pick the phase overlay map from the **Select 2nd Map** dropdown (or type its folder name in the input box).
   - Click **Apply Overlay**.
3. **Seamless Tile Merging**: The viewer dynamically substitutes terrain geometry, MCNK chunk textures, and placed doodads from the phase map for all affected tiles while keeping unaffected continent tiles loaded.
4. **Minimap Synchronization**: The minimap automatically displays the secondary map's BLP tiles where available.
5. **Clear Overlay**: Click **Clear Overlay** to return instantly to the unmodified base continent terrain.

---

## 11. Rosetta Multi-Version Zarr Datastore & Cross-Era Loading (Spec 190)

The Rosetta Datastore is a unified, version-agnostic interchange format stored in Zarr v3 with globally content-addressed deduplicated assets in Parquet (`global_assets/catalog.parquet`).

### Cross-Era Asset Interoperability:
- **Transparent Format Shifting**: The viewer automatically shifts model extensions (`.mdx` $\leftrightarrow$ `.mdl` $\leftrightarrow$ `.m2`) across client eras.
  - When viewing an Alpha 0.5.3 map (`.mdx` doodad references) against a WotLK 3.3.5 client, models resolve seamlessly to `.m2`.
  - When viewing modern maps against older clients, `.m2` models resolve to `.mdx`/`.mdl` when available.

### Browsing the Rosetta library

Open **Editor > 3D Object Library** and load a manifest or library JSON as described above.
The former File > Load from Rosetta Datastore menu has been retired.

---

## 12. Troubleshooting and FAQ

### Q: Why is terrain black or missing textures?
**A**: Ensure your client root points to the folder containing the `Data` directory (or loose `World\` folders). If the client uses MPQ archives, verify `ArchiveCatalog` has enumerated the listfiles.

### Q: Why are some dense city interiors slow?
**A**: In release `v0.5.2.2`, large multi-district WMOs (such as Stormwind) submit all interior groups simultaneously. Optimize performance by reducing the view distance slider in **Inspector > LOD & Budget**.

### Q: Audio reports "OpenAL soft library missing"?
**A**: Ensure `soft_oal.dll` (Windows) or `libopenal.so` (Linux) / `libopenal.dylib` (macOS) is present in the application folder. The viewer continues normally with audio disabled if the library is not found.
