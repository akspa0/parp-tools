# WoWViewer User Guide

Covers the current app (v0.5.2) only.

## Start

### From a release build

Download the archive for your platform from the [releases page](https://github.com/akspa0/parp-tools/releases),
unzip it, and run `ParpToolsWoWViewer`. The builds are self-contained — no .NET install needed.

Native file dialogs are Windows-only. On Linux and macOS, point the viewer at content with
`--game-path`, `--build`, and `--world` instead of using the open dialogs.

### From source

```powershell
cd wow-viewer
dotnet run --project .\src\viewer\WoWViewer\WoWViewer.csproj -c Debug
```

### Normal startup flow

1. Open a staged game folder.
2. Pick the explicit build.
3. Load a world from the left Navigator sidebar.
4. Use the right workbench for inspection, scene, utility, and experimental surfaces.

## Client roots

- The configured client library (e.g. `H:\CLIENTS`) is the approved data source.
- `output/tmp/wowarchive-clients/` is optional staging and may be pruned.
- Client roots are passed as CLI arguments or chosen in the UI, never hardcoded in source.

## Launch flags

| Flag | Purpose |
|---|---|
| `--verbose` | Keep detailed logging on |
| `--game-path <clientRoot>` | Open a staged client directly |
| `--build <buildVersion>` | Pin the build on startup |
| `--world <path-or-virtual-path>` | Open a world or asset after startup |
| `--listfile <path>` | Supply an explicit listfile |
| `--loose-map-overlay <dir>` | Overlay loose map files on the staged client |
| `--full-load` / `--partial-load` | Override the default bounded tile admission |
| `--capture-shot <name>` | Queue a saved shot |
| `--capture-output <dir>` | Override the capture root |
| `--capture-after-frames <n>` | Delay the queued capture by n frames |
| `--exit-after-capture` | Close after the queued capture |

`--full-load` bypasses normal camera-based streaming and loads the whole map. It exists for stress
and capture work; it is not the normal path and will use far more memory.

## Navigating the UI

The shell is a left Navigator sidebar, a 3D viewport, a bottom action/status bar, and a right
workbench.

- **Left Navigator** owns all loading: sources, files, world maps, and the Phase Map selector.
- **Right workbench** has five destinations, each with its own page dropdown:

| Destination | Contains |
|---|---|
| **Quick** | Common toggles for the loaded scene |
| **Inspect** | Context, Scene Investigation, MCNK / ADT, World Context, Archeology, Animations, Actions |
| **Scene** | Placements, LOD |
| **Utilities** | Minimap, Audio, Capture, Log Viewer, Perf, Asset Catalog, Taxi |
| **Experimental** | Terrain Lab, PM4, Converters, Population |

- **Bottom bar** keeps scene toggles, the area/subzone readout, and the `AUDIO: ON` / `AUDIO: MUTED`
  button.
- **Help > Keyboard Shortcuts** lists global bindings plus the ones for the active page.

## Keyboard shortcuts

Global:

| Input | Action |
|---|---|
| `W A S D` | Move the free-fly camera (hold `Shift` to boost) |
| `Q` / `E` | Move vertically |
| right mouse drag | Look |
| `Tab` | Show or hide viewer chrome |
| `I` | Show or hide the right workbench |
| `M` | Toggle the fullscreen minimap (terrain loaded) |
| `P` | Focus PM4 tools when available |
| triple-click same minimap tile | Teleport the camera |

Inspect page:

| Input | Action |
|---|---|
| `Left` / `Right` / `Space` | Step or play the loaded model animation |

Capture page (camera-path authoring is scoped to this page, so these keys are inert elsewhere):

| Input | Action |
|---|---|
| `K` | Add a camera-path key at the playhead |
| `U` | Update the selected key |
| `Delete` | Delete the selected key |
| `Space` | Play or pause the path |
| `Left` / `Right` | Select previous or next key |
| `Ctrl+Left` / `Ctrl+Right` | Retime the selected key |
| `Ctrl+Up` / `Ctrl+Down` | Nudge the playhead |
| `Home` / `End` | Jump to path start or end |
| `Z` / `X` | Roll the camera (hold `Shift` for a larger step) |
| `Ctrl+S` | Save the path as JSON |
| `Ctrl+E` | Export the path as native M2 |

## World viewing

### Terrain streaming

The viewer streams a bounded set of ADTs around the camera rather than loading a whole map:

- A **retained window** keeps a camera-centered square resident (radius 2 by default, 3 maximum).
- A **directional selector** picks which tiles get detailed geometry — it protects the largest
  complete camera-centered square the budget allows (3×3 from 9 tiles, 5×5 at 25), then spends the
  rest forward.
- Distance admission measures the nearest point on a tile's bounds, so standing at a tile edge does
  not drop that tile.

If terrain seems to be missing, raise the tile budget before assuming a load failure. Retained
neighbors keep their WDL low-detail underlay until detailed terrain actually submits.

### Time of day

Alpha 0.5.3 lighting runs on the native world clock — 2,880 units per cycle, 24 real minutes. It
advances automatically. Moving the time slider freezes the clock at that value until you resume it.
Light DBC, LIT, sky, and audio all read the same value each frame.

### WMO interiors

WMO visibility uses portal traversal from the camera's group. Interiors are entered when the camera
is inside any group's local bounds. The decision is deliberately fail-open: broken geometry or a
boundary camera keeps surfaces drawn rather than dropping them, so expect over-draw before
under-draw.

### Audio

Audio is under **Utilities > Audio**. Playback needs OpenAL; if the native library is absent the
viewer reports it and continues without sound.

- Resident MCSE / MCNK emitters play positionally as tiles stream in and release on unload.
- Preview any resident `SoundEntries` ID at the camera, with master and emitter gain controls.
- An off-by-default marker overlay draws one 3D pin per resident emitter: amber MCSE, cyan MCNK
  water, purple MCNK environment.
- The bottom-bar `AUDIO` button mutes everything through the master bus.
- **Automatic ZoneMusic playback is muted by policy.** Resolution and diagnostics are still shown;
  the row indirection is not yet proven, so it does not auto-play. MIDI and DLS are reported as
  unsupported rather than guessed.

## Synthesized minimap export

**Utilities > Minimap → Synthesized Terrain Minimap Export** composes terrain-only minimap PNGs
directly from the client's BLP textures plus MCLY/MCAL/MCNR/MCSH data. It does not read a shipped
minimap image.

| Control | Purpose |
|---|---|
| Client root | Staged client directory |
| Map name | Map directory name (e.g. `Kalimdor`) |
| Time of day | Hour + minute; controls sun elevation. Default 12:00 (noon, full-bright) |
| Tile resolution | Per-tile PNG resolution (default 256) |
| Write per-tile PNGs | Emit one terrain-only PNG per tile |
| Write one stitched map PNG | Emit one stitched whole-map PNG |
| Include WMO geometry | Composite placed WMOs onto tiles (experimental, needs headless GL, default off) |
| Bake MCSH shadows | Preview the client's baked MCSH map; normal output omits it as a separate signal |

The solar bearing stays fixed north-west and only elevation cycles with time of day, matching the
traced 0.5.3.3368 client. That is why changing the time changes brightness and contrast, not shadow
direction. The sun is *frozen* at the requested time — synthetic output never advances a live clock,
and the manifest records `timeOfDayMode=frozen`. Map LIT and Light DBC profiles belong to the viewer
and are deliberately never applied here.

Terrain renders with Lambert hillshading. Analytic cast shadows are an addition the original client
never had and default off for the Alpha era. Output lands in
`output/synthesized-minimaps/<map>/tod-<time>/` with a `synthesis-manifest.json`.

Equivalent CLI:

```powershell
WowViewer.Tool.Harvest synthetic-minimap `
  --client-root <clientRoot> --map Kalimdor --output-dir <dir> `
  --time-hours 1800 --per-tile --whole-map
```

See [CLI-TOOLS.md](../CLI-TOOLS.md) for the full flag set, including `--authored-reference`,
`--match-time`, `--tile-list`, and `--score`.

## Camera paths and capture

**Utilities > Capture** owns both capture automation and camera paths.

- Author keys from the current camera, or import from loose and in-client `.m2` camera assets
  (MD20 `0x100` early and `0x109+` modern layouts are both supported).
- Save as JSON — position, target, timing, FOV, and roll are preserved — or export as native M2.
- Path preload is bounded to the swept tile footprint along the path plus the configured radius. It
  is not a whole-map residency guarantee.

## PM4 work

**Experimental > PM4** provides overlay inspection, the scene graph tree, region-aware selection and
export, and the wall-mesh toggle. PM4 coordinate frames are solved and confirmed as of v0.5.2, so
tiles align without per-object fitting.

## Troubleshooting

### Viewer opens but the map does not load

- Verify the staged client root.
- Verify the chosen build matches the staged data.
- Check console/log output with `--verbose`, or use Utilities > Log Viewer.

### Nearby terrain or buildings pop in and out

- Raise the retained radius and detail tile budget in the streaming controls.
- This class of bug was heavily reworked in v0.5.2; if it still reproduces, note the tile
  coordinates and camera heading — that is the useful report.

### No sound

- Confirm OpenAL is available; the Audio page reports backend status.
- Confirm the `AUDIO` bottom-bar button is not muted.
- ZoneMusic does not auto-play by design.

### Capture flow fails

- Verify the output path exists.
- Verify startup flags point at staged data.
- Verify the saved shot name exists before using `--capture-shot`.

## Related docs

- [Viewer README](../../README.md)
- [Release notes — v0.5.2](../releases/v0.5.2.md)
- [CLI tools](../CLI-TOOLS.md)
- [Documentation status](../DOCUMENTATION-STATUS.md)
- [Alpha audio catalog](../architecture/alpha-audio-catalog.md)
