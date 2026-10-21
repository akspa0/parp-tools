# WoWViewer Comprehensive User Guide

**Application Version**: `v0.6.0.1`
**Platform Support**: Windows x64, Linux x64, macOS (Apple Silicon arm64 / Intel x64)

`WoWViewer` is an interactive 3D desktop application for inspecting, analyzing, rendering, and reconciling World of Warcraft client data.

**Bring your own data.** No game data is shipped. Open your own client install, MPQ set, CASC install or extracted map folder.

**Supported clients:** 0.5.3 – 0.11 **complete**; 0.12 – 5.x **partial**; WoW: Forever 1.60.1 (`wow_classic_beta`, CASC) **stable**.

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
9. [PM4 Inspection (Viewer)](#9-pm4-inspection-viewer)
10. [Phased Terrain Dual-Map Overlay](#10-phased-terrain-dual-map-overlay)
11. [Rosetta Multi-Version Zarr Datastore & Cross-Era Loading](#11-rosetta-multi-version-zarr-datastore--cross-era-loading)
12. [Modern CASC Clients: WoW: Forever](#12-modern-casc-clients-wow-forever)
13. [DAT v26 Terrain Project Files (alpha)](#13-dat-v26-terrain-project-files-alpha)
14. [Archaeology: Phase Maps and Map Layering](#14-archaeology-phase-maps-and-map-layering)
15. [Development Map Restoration and Experimental Research](#15-development-map-restoration-and-experimental-research)
16. [Troubleshooting and FAQ](#16-troubleshooting-and-faq)

---

## 1. Getting Started

### Using Prebuilt Binaries
1. Download the self-contained archive for your platform from the [GitHub Releases Page](https://github.com/akspa0/parp-tools/releases).
2. Extract the archive into any directory. No .NET runtime installation is required.
3. Run `ParpToolsWoWViewer` (or `ParpToolsWoWViewer.exe` on Windows).

> [!NOTE]
> Native OS open-file dialogs are supported on Windows. On Linux and macOS, launch content using the `--game-path`, `--build`, and `--world` CLI options.

The matching `ParpToolsWoWViewer-tools-<rid>.zip` is a separate download containing all nine self-contained
CLI apphosts and project manifests in one shared output directory; common libraries and data files are included
once. They can run without installing .NET. See the bundled
[`tools/README.md`](../../tools/README.md) for the Inspector, converter, harvest, validation, capture,
WDL, WMO and enrichment commands.

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
1. Use **File > Open Game Folder (MPQ)...** (or **File > Open CASC Install (local)...** for a Battle.net
   install) and browse to the client root with the shared path picker.
2. For a CASC install, pick the game version in the product list the picker shows.
3. In the left sidebar, open **Workspace** to see the active data source, and **World Overview** to see
   the discovered maps.
4. In the map list, click **Load** on the world you want (e.g. `Azeroth`, `Kalimdor`, `Development`).

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

The viewer is one shell with three destinations. The top bar carries the global actions, the bottom
bar switches destination, and each sidebar shows only the surfaces of the current destination.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ File   View   Help                                        ┌───────────────┐  │
│                                                           │    MINIMAP    │  │
│                                                           └───────────────┘  │
├───────────────┬──────────────────────────────────────────────┬───────────────┤
│ Left sidebar  │                                              │ Right sidebar │
│ (destination) │                 viewport                     │ (destination) │
├───────────────┴──────────────────────────────────────────────┴───────────────┤
│ Chat (bottom-left)                                          [Options ▾]      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Viewer │ Editor │ Archaeology │ Options                    FPS / ms          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Status bar: coordinates, tile, facing, audio and runtime stats               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### The shell

- **Top bar** — `File`, `View`, `Help`. The File menu opens data and settings, the View menu toggles
  the chrome and the sidebars and resets the camera or the shell layout, and the Help menu opens
  **Keyboard Shortcuts** (`F1`), **UI Reference** and **About**.
- **Left sidebar** — collapsible sections that depend on the destination (below).
- **Right sidebar** — the destination's tabs or page selector (below).
- **Minimap** — floats over the top-right of the viewport. It can be collapsed to a small button,
  hidden entirely, or docked into the top of the right sidebar; the choice is remembered. `M` shows
  or hides it.
- **Chat** — the bottom-left dock, which also carries the population controls.
- **Bottom bar** — the destination switcher (**Viewer | Editor | Archaeology**), the **Options**
  button, and the FPS / frame-time readout. The readout stays visible in every destination.
- **Status bar** — the strip at the very bottom: camera coordinates, tile and facing on the left,
  the runtime readout (or the last status message) and the audio mute toggle on the right. File
  operations report their result here.

### File menu

Every open action uses the shared path picker, so you browse to the target instead of typing a path:
**Open File...**, **Open Alpha WDT (loose map)...**, **Open Loose PM4 / PD4 File...**, **Open Game
Folder (MPQ)...**, **Open CASC Install (local)...**, **Open CASC Install (local + CDN fill)...**,
**Open DAT Terrain Folder...**, **Open Saved Game Folder**, **Attach Loose Map Folder...** and
**Load Loose Map Folder Against Saved Base**. The menu also holds **Save Current Game Folder As
Known-Good Base**, **Forget Known-Good Base**, **Settings...** and **Quit**.

The **View** menu holds **Reset Camera**, **Hide UI Chrome** (`Tab`), **Left Sidebar** (`[`),
**Right Sidebar** (`]`) and **Reset Shell Layout**.

### Viewer destination

The right sidebar has four tabs: **Inspector**, **World**, **Model**, **PM4**.

- **Inspector** — pages *Context*, *Scene Investigation*, *MCNK Flags*, *World Context*.
  - **Context** shows the selected WMO, WMO doodad, MDX/M2, PM4 object or WL liquid. A selected
    WMO exposes its doodad sets here. When terrain is the target, it uses a pinned chunk first,
    then the hovered chunk, then the chunk under the camera. With no resident chunk, it shows
    an active-world overview.
  - **Scene Investigation** holds the visual investigation toolbox, and **MCNK Flags** the MCNK
    flag overlay: Impassable, River, Ocean, Magma, Slime, Shadows, MCCV and BakedShadows. Enable
    the overlay and Impassable to use **Highlight Diagonal Weak Corners**.
  - **World Context** holds the hover/pick range, POI and AreaTrigger controls.
- **World** — pages *Objects*, *Placements*, *LOD & Budget*, *Exports*.
- **Model** — pages *Info*, *Animations*, *Actions & Exports*.
- **PM4** — the PM4 inspection surface, described in section 9.

To pin terrain, left-click a visible chunk when the click does not select a scene object or
belong to an active editing tool. Context displays its area name/ID, MCNK flags, holes, height
range, textures and alpha layers, shadow/MCCV presence, liquids and tile placement counts.
Use **Frame Chunk**, **Copy Texture Summary**, **Copy Coordinates** or **Clear Chunk Selection**.
Pins are scoped to the current terrain source and expire when their chunk is no longer resident.

The left sidebar shows **Workspace**, **World Overview**, **File Browser**, **Scene Tools**,
**Assets** and **Diagnostics**. **Scene Tools** has the *Taxi* and *Capture* pages; **Diagnostics**
opens the log and performance windows and shows the runtime statistics.

### Editor destination

The right sidebar is a page selector with **Placement & Objects**, **Terrain Tools**, **Data I/O**
and **Converters**.

- **Placement & Objects** — *Archaeology Tasks*, *Rosetta Object Library*, *Population*.
- **Terrain Tools** — the terrain lab, the terrain import/export actions, and the terrain file
  operations.
- **Data I/O** — *New Map Creator* and *Archaeology Imports*.
- **Converters** — *Map Converter* and *WMO Converter*.

The **Rosetta Object Library** reads a Rosetta manifest or library JSON. Enter **Manifest / Library
Path** and choose **Load**, use **Browse...**, or choose **Scan Default**. Search or filter M2/WMO
entries and sort by size or name. The selected entry shows bounds, span, volume, footprint and
calibration occurrences. **Inspect 3D Model in Viewport** opens the asset through the active
client; **Copy Model Path** and **Set as Active Placement Model** support placement workflows.
The available entries depend on the loaded manifest and client assets.

The left sidebar adds **Editor Workspace** (a shortcut list of the Editor pages) and
**Project Output** (the output root and the resolved output directory).

### Archaeology destination

The right sidebar is a page selector with **Weak Signal & Stratigraphy**, **UniqueId Timeline**,
**Layers & Provenance**, **Playback & Capture** and **Cartography**. PM4 is not an archaeology
page: the panel offers **Inspect PM4 in Viewer**, which switches to the Viewer's PM4 surface
(section 9). Cartography includes its merged-data output actions; the UI consolidation does not
itself implement additional map composition capabilities.

**Playback & Capture** combines archaeology range playback with **Camera Paths & Video Capture**.
Use **Apply playback to next capture** to arm the next queued capture with the configured
archaeology playback range. In its Capture Automation selector, configure the `ffmpeg` executable,
output folder, container and frame rate, then use **Start Video Recording** / **Stop Video
Recording**. Switch the selector to Camera Path for **Play**, **Play + Video** and **Stop**. A
scene-only recording samples the viewport before ImGui; use `Tab` before recording only when a
full-window scene is desired. Verify the configured executable and output file on the active
machine before relying on a recording.

The left sidebar adds **Archaeology Workspace** and **Layer Projects** (which points at
**Cartography** for the layer actions).

### Chat panel

The chat is a **docked panel**, not a floating window. It spans the scene between the left and
right sidebars and sits on the bottom edge of the scene area, so it never covers a sidebar, the
bottom bar or the status bar. It cannot be dragged or resized. **Collapse chat** reduces it to a
one-line bar and **Expand chat** restores it; the collapsed state is remembered between sessions.

### Options sub-menu

The **Options** button in the bottom bar (and the **Options** button in the chat header) opens a
sub-menu anchored to the right sidebar. It is grouped as:

| Group | Contents |
|---|---|
| Display | Render Quality, Lighting, Theme |
| Overlays | Alpha, Shadows, MCCV, Contours, Liquid, WL, Bounding Boxes, PM4 Overlay, Color Spawns |
| Layers | Terrain layer toggles, Holes, WDL |
| Grids | Tiles, Chunks, Cells |
| Objects | Object transparency, Terrain WF, M2/WMO WF, Anim, 3D Cursor, cursor model status and **Reload cursor model** |
| Audio | Audio output and per-zone audio settings |
| Simulation | Spawn BBs Only (Auto-Stream stays in the chat population controls) |

### Opening and exporting files

- **Open File...** (in the File menu and in the left sidebar **Workspace** section) opens the shared
  path picker, so you browse to the file instead of typing a path.
- Exports open the same picker for the output path, write the file, and report the written path in
  the status line. A failed export is reported instead of failing silently.
- The File menu no longer carries export/import trees. Export actions live with the destination
  that owns them, for example **Viewer > World > Exports**.
- **WC3 Beta data**: the left sidebar's **File Browser** section has an **Open WC3 Beta data** block
  that browses to a WC3 Beta install or extracted data folder (optionally with a listfile) and loads
  it as the active data source. The support level is labelled *WC3 Beta* in the viewer and in the
  [UI reference](UI-REFERENCE.md) because it targets clients without modern tooling.

### Control reference

Every documented control, its shortcut and its home surface are listed in the generated
[UI reference](UI-REFERENCE.md). The same list is searchable in the running viewer under
**Help > UI Reference**.

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
| `[` | Show or hide the left sidebar |
| `]` | Show or hide the right sidebar |
| `I` | Open the Viewer Inspector (right sidebar) |
| `M` | Show or hide the minimap |
| `P` | Open the Viewer PM4 inspection panel |
| `F1` | Open the keyboard shortcuts popup |
| `Esc` | Close the topmost popup |
| **Triple-Click on Minimap** | Teleport camera instantly to clicked tile |

> The scene input path also binds `M` to the fullscreen minimap overlay while terrain is loaded; both
> bindings exist in the current code.

---

## 5. World Viewing and Streaming

### Bounded Tile Admission
To ensure smooth frame rates and bounded memory usage, `WoWViewer` streams terrain using a camera-centered residency ring:
- **Inner Ring (Radius 2)**: 25 ADT tiles kept resident around the camera.
- **Directional Lookahead**: Additional tiles are prioritized in the forward view frustum.
- **WDL Underlay**: Distant horizons outside the high-detail ring render low-detail WDL terrain meshes, preventing abrupt popping.

---

## 6. Camera Path Studio

Located under **Viewer > Scene Tools > Capture** (left sidebar) and **Archaeology > Playback &
Capture**, the Camera Path Studio lets you author smooth cinematic fly-throughs, record viewpoints,
and export native WoW camera assets.

Open the **Capture** page and switch its page selector to **Camera Path**. Load the intended map,
position the camera and click **Add Current Camera Key**. Move to another viewpoint and add a second
key, then click **Play**. Interactive playback advances while terrain and objects stream. **Stop**
ends playback; **Loop** repeats the path.

**Preload path before capture** controls capture readiness. **Warm Path** prepares the route,
**Release Warmup** releases its residency, and **Play + Video** waits for enabled warmup before
recording. Paths retain their map/build binding; an unrelated map or build is rejected and
changing the active map during playback stops it.

### Taxi riding

Open **Viewer > Scene Tools > Taxi** in the left sidebar, use **Load Taxi Paths** if needed, select a
route and click **Ride Selected Route**. Choose **Cockpit** or **Chase** in **Ride Camera Mode**. Starting
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

Located under **Options > Audio**, the OpenAL audio engine renders true 3D spatialized sound:
- **Positional Emitters**: As ADT tiles stream in, sound emitters declared in `MCSE` chunks and `MCNK` liquid flags are automatically registered.
- **3D Emitter Pins**: Enable visual pins in the viewport to see emitter positions:
  - 🟡 **Amber**: Positional sound effect (`MCSE`).
  - 🔵 **Cyan**: Liquid / water sound emitter.
  - 🟣 **Purple**: Ambient environmental emitter.
- **SoundEntries Previewer**: Search and audition any sound effect in `SoundEntries.dbc`.

---

## 8. Lighting, Atmosphere, and Time of Day

- **Alpha 0.5.3 World Clock**: The Alpha client advances on a 2,880-unit world cycle (24 real minutes). The lighting engine synchronizes terrain vertex lighting, ambient colors, directional sun/moon vectors, and skybox palettes.
- **Time Slider**: In the terrain lighting controls (**Editor > Terrain Tools > Terrain Lab**, and **Archaeology > Weak Signal & Stratigraphy**), dragging the slider freezes the clock at a specific time of day for photography and inspection. The same block holds the **Automatic 24-minute cycle** switch.
- **Lighting Diagnostics**: **Options > Display > Lighting** reports the LIT source, the LIT entries, a LIT sample and the effective lighting.
- **LIT & DBC Fallback**: Evaluates `.lit` lighting files when present, with automatic fallback to `Light.dbc` and `LightParams.dbc`.

---

## 9. PM4 Inspection (Viewer)

PM4 inspection is a Viewer surface: the **PM4** tab of the right sidebar, with the **Viewer**
destination active. It has five pages — **Overlay**, **Selection**, **Info**, **Outliner** and
**Object Groups** — under a collapsible **PM4 Glossary / Evidence** summary of the record fields.

- **Overlay** — the overlay switches (PM4 Overlay, Solid Fill, X-Ray, Bounds, CK24 Bounds, Recovered
  placements, Placement-Z markers, MPRL refs, Centroids, MSCN nodes), a loose PM4/PD4 loader and
  **Reload PM4**. A loose file can also be opened from **File > Open Loose PM4 / PD4 File...**, which
  uses the shared path picker and reports the loaded path in the status line.
- **Selection**, **Info**, **Outliner** and **Object Groups** — the selection details, the decoded
  record information, the scene outliner and the object-group breakdown for the selected PM4 object.
  Click a PM4 object in the viewport to inspect it here.

### What is no longer a viewer surface

PM4 object **matching, alignment and reconciliation** are not viewer surfaces any more. They moved
to the offline patcher tool specified in Spec 281
([`specs/281-museum-adt-patcher`](../../specs/281-museum-adt-patcher/spec.md)), which pairs a folder
of WoWMuseum ADTs with a folder of original PM4 files and writes patched 3.3.5 LK ADTs with a
provenance report, without opening the viewer.

The viewer keeps inspection only. The **Archaeology** destination offers **Inspect PM4 in Viewer** as
a navigation shortcut to this surface; it does not duplicate it.

---

## 10. Phased Terrain Dual-Map Overlay

In Cataclysm (4.x) and later expansions, game zones feature dynamic phases where sparse ADT tiles modify specific regions of a continent (e.g. `Gilneas`, `Gilneas2`, `GilneasPhase1` over `Azeroth`).

### Using Phased Map Overlays:
1. **Load Base Map**: Open the primary continent map (e.g. `Azeroth`) from the map list in the left
   sidebar's **World Overview** section.
2. **Add the Phase Map**: In **Archaeology > Layers & Provenance**:
   - Pick the phase map from the **Add phase map...** list, or type its folder name in the **Manual
     map directory** field and press **Add**.
   - **Add DAT folder...** adds a folder of DAT terrain files as a donor layer instead.
3. **Compose the Layer**: Each layer in the stack can be enabled, locked against accidental edits,
   reordered with **Up** / **Down**, and aligned with its offset, rotation and mirror controls.
   **Only take what this phase actually has** gates which channels the layer contributes, and
   **Compose placed tiles only** restricts it to the donor tiles you picked.
4. **Seamless Tile Merging**: The viewer dynamically substitutes terrain geometry, MCNK chunk textures, and placed doodads from the phase map for all affected tiles while keeping unaffected continent tiles loaded.
5. **Minimap Synchronization**: The minimap automatically displays the secondary map's BLP tiles where available.
6. **Clear Overlay**: Use **Remove** on the layer (unlock it first if it is locked) to return instantly to the unmodified base continent terrain.

---

## 11. Rosetta Multi-Version Zarr Datastore & Cross-Era Loading

The Rosetta Datastore is a unified, version-agnostic interchange format stored in Zarr v3 with globally content-addressed deduplicated assets in Parquet (`global_assets/catalog.parquet`).

### Cross-Era Asset Interoperability:
- **Transparent Format Shifting**: The viewer automatically shifts model extensions (`.mdx` $\leftrightarrow$ `.mdl` $\leftrightarrow$ `.m2`) across client eras.
  - When viewing an Alpha 0.5.3 map (`.mdx` doodad references) against a WotLK 3.3.5 client, models resolve seamlessly to `.m2`.
  - When viewing modern maps against older clients, `.m2` models resolve to `.mdx`/`.mdl` when available.

### Browsing the Rosetta library

Open **Editor > Placement & Objects > Rosetta Object Library** and load a manifest or library JSON as
described above. The former File > Load from Rosetta Datastore menu has been retired.

---

## 12. Modern CASC Clients: WoW: Forever

Added in v0.5.4-alpha. Tested against **WoW: Forever**, the `wow_classic_beta` product, build 1.60.1.69876 (a 12.0-based client).

### Opening an install
1. **File → Open CASC Install (local + CDN fill)...** (or **(local)...** to read only what is on disk) and pick the folder that contains `.build.info` (for example the `World of Warcraft` folder the Battle.net launcher installed). Picking a game folder such as `_retail_` also works.
2. The **Open CASC Install** window lists every game version in `.build.info` (for example `1.60.1.69876 wow_classic_beta` and `1.15.9.69722 wow_classic_era`), and shows which ones have a game folder on disk. Select one.
   - **Fetch missing files from Blizzard's CDN** starts ticked when opened with **(local + CDN fill)** and unticked with **(local)**. It downloads files the build lists but the install does not have on disk, for the same build, and caches them under `output\cache\casc`. In the tested install most `wow_classic_beta` textures were not on disk, so without it terrain and models render untextured. With it off, no game data is downloaded; if the launcher has updated `.build.info` before downloading that build's manifests, those manifests alone are fetched and this is logged.
   - **Fall back to the other products for missing files** (off by default) reads a file from another listed version when the selected one lacks it. This mixes data across game versions.
   - The selected version's build number picks the DB2 definitions.
3. The map list is read from the build's `Map.db2` (for example **[451] Development Land**). Load a map as usual.

### What is supported
- Terrain tiles from the WDT's `MAID` table. Full-resolution tileset textures blended per chunk from `MDID`, up to 8 layers per chunk. The layer bar shows L4–L7 once loaded terrain uses them (Azeroth in 1.60.1 has chunks with 5 to 8 layers).
- M2 models (`MD21`): skins from `SFID`, textures from `TXID`.
- WMO buildings: groups from `GFID`, material textures from `MOMT` FileDataIDs, doodads from `MODI`.
- Area names, lights and liquid types from the build's DB2 tables, using WoWDBDefs.

### Loading speed
- The first visit to a map downloads its textures (with CDN fill). Downloads run in parallel in the background. Later visits read from the cache.
- Measured on `development`: the 460 WMO textures took 5.3 s to download in parallel, and about 1 s to read once cached.

### Known limitations
- Up to 8 texture layers per chunk (raised from 4 in v0.6.0-alpha). Layers whose texture FileDataID is 0 in the shipped data have no texture.
- Height-based texture blending (`MHID`) is not applied yet.
- The Azeroth WDT's `MAI2` chunk is not interpreted.

---

## 13. DAT v26 Terrain Project Files (alpha)

DAT v26 files are raw terrain project files: the uncompiled data client ADTs are built from. They shipped in the same `wow_classic_beta` build. Format details: [`docs/architecture/adt-v26-format.md`](../architecture/adt-v26-format.md). The reader and renderer were added in v0.5.4-alpha; decoded objects/normals/colours and the experimental writer were added in v0.6.0-alpha.

1. (Optional, for textures) open the CASC install first, as in section 12.
2. **File → Open DAT Terrain Folder...** and pick a folder of DAT v26 files. File names do not matter: files are recognised by content and placed by their `ALOC` tile coordinates. Heights, normals (`ANRM`), vertex colours (`ACVT`), texture layers and `ACDO` object placements render.
3. **Height scale**: the height divisor options (÷36 inches by default, plus 1×, ÷12 and ÷100) are implemented by the DAT terrain provider, and the default ÷36 treats heights as inches: −18,559.47 ÷ 36 = −515.54, matching the shipped ocean floor. No current menu entry exposes the height-scale submenu.
4. **Viewer > World > Exports** (section **DAT Conversion**) → **Export Nearby Tiles as DAT v26 (experimental)** writes the ADT tiles within 2 tiles of the camera (terrain, texture layers, vertex colours, normals, objects) to `output/dat_v26_export`. Reopen that folder with the step 2 command to compare.

Shadows (`ASHD`) are all zero in the corpus and are not rendered. The format has no liquid, hole or area data in this corpus, and it is not the project's save/interchange format. The horizontal scale relative to shipped ADTs is not established.

---

## 14. Archaeology: Phase Maps and Map Layering

The **Archaeology** destination of the right sidebar is where one map is shown over another. The
overlay is composed in **Layers & Provenance**.

- **Any map over any map.** The **Add phase map...** list shows every discovered map that is neither the
  loaded base map nor already a layer. Maps the client's own DBC/DB2 tables know are labelled with
  their map id, the rest as `[custom]`. Pick one to overlay it. Tiles present in the overlay map
  replace the base tiles; everything else stays the base map.
- **Any map directory.** The **Manual map directory** field accepts a map folder that is not in the
  list; **Add** adds it as a layer. **Add DAT folder...** adds a folder of DAT terrain files as a
  donor layer instead.
- **Clearing.** Use **Remove** on the layer to return to the unmodified base map.

Section 10 describes the overlay behaviour (terrain, chunk textures, doodads, minimap) in detail. Annotated
screenshots for this tab are planned for the integrated help in v0.6.1.

---

## 15. Development Map Restoration and Experimental Research

### Restoring the `development` map (v0.6.0, partial)

The `restore-unauthored` PM4 command combines the `335-dev` base tiles, original development placements,
PM4 collision data, the prepared association ledger, and minimap colors for untextured tiles. It writes
monolithic LK 3.3.5 ADTs into a separate output tree and validates each written tile by reading it back.
The command does not create a WDT; open the base `development.wdt` with the generated tree attached as a
loose-map overlay.

Follow the step-by-step [Development Map Restoration guide](development-map-restoration.md) for input
layout, safe smoke/full-batch commands, output validation, viewer launch, and troubleshooting. The result
is partial; in particular, museum objects on adjacent tiles that are not recovered by the current PM4
resolver remain a known gap.

### Terrain reconstruction research (experimental, unstable)

The Zarr datasets, elevation/trestle models and ComfyUI nodes under `data-harvester/` reconstruct terrain from
images with little human interaction. This work is **unexplored and unstable**, needs your own data and a GPU,
and is intentionally not documented here beyond this note.

---

## 16. Troubleshooting and FAQ

### Q: A CASC map shows untextured terrain or models?
**A**: The install probably lacks the texture data on disk. Reopen it with **File → Open CASC Install (local + CDN fill)...**.

### Q: Why is terrain black or missing textures?
**A**: Ensure your client root points to the folder containing the `Data` directory (or loose `World\` folders). If the client uses MPQ archives, verify `ArchiveCatalog` has enumerated the listfiles.

### Q: Why are some dense city interiors slow?
**A**: In large multi-district WMOs (such as Stormwind), interior groups can be budget-intensive. Optimize performance by reducing the view distance slider in **Viewer > World > LOD & Budget**.

### Q: Audio reports "OpenAL soft library missing"?
**A**: Ensure `soft_oal.dll` (Windows) or `libopenal.so` (Linux) / `libopenal.dylib` (macOS) is present in the application folder. The viewer continues normally with audio disabled if the library is not found.
