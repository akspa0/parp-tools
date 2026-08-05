# WoWViewer User Guide

Short guide. Current app only.

## Start

### From source

```powershell
cd wow-viewer
dotnet run --project .\src\viewer\WoWViewer\WoWViewer.csproj -c Debug
```

### Normal startup flow

1. Open staged game folder.
2. Pick explicit build.
3. Load world from viewer UI.
4. Use minimap, world, terrain, PM4, and diagnostics surfaces as needed.

## Client roots

- The configured client library (e.g. `H:\CLIENTS`) is the approved data source.
- `output/tmp/wowarchive-clients/` is optional staging and may be pruned.
- Client roots are passed as CLI arguments, never hardcoded in source.

## Useful launch flags

| Flag | Purpose |
|------|------|
| `--verbose` | keep detailed logging on |
| `--game-path <clientRoot>` | open staged client directly |
| `--build <buildVersion>` | pin build on startup |
| `--world <path-or-virtual-path>` | open world or asset after startup |
| `--capture-shot <name>` | queue saved shot |
| `--capture-output <dir>` | override capture root |
| `--exit-after-capture` | close after queued capture |

## Core controls

| Input | Action |
|------|------|
| `W A S D` | move camera |
| `Q / E` | vertical move |
| right mouse drag | look |
| mouse wheel | move / zoom depending on panel |
| `Tab` | hide or restore UI chrome |
| `M` | toggle fullscreen minimap |
| triple-click same minimap tile | teleport camera |

## Current viewer surfaces

### Synthesized minimap export

The **Synthesized Terrain Minimap Export** dialog (Tools menu → Synthesized Minimap) composes
terrain-only minimap PNGs directly from the client's BLP textures plus MCLY/MCAL/MCNR/MCSH data.
It does not read a shipped minimap image. Settings:

| Control | Purpose |
|---------|---------|
| Client root | Staged client directory (e.g. a build under `output/tmp/wowarchive-clients/`) |
| Map name | Map directory name (e.g. `Kalimdor`) |
| Time of day | Hour + minute; controls the sun elevation. Default 12:00 (noon, full-bright) |
| Tile resolution | Per-tile PNG resolution (default 256) |
| Write per-tile PNGs | Emit one terrain-only PNG per tile |
| Write one stitched map PNG | Emit one stitched whole-map PNG |
| Include WMO geometry | Composite placed WMO buildings onto the tiles (experimental, defaults off) |
| Bake MCSH shadows | Include the terrain-side static shadow map. Without it, only Lambert hillshading (no cast shadows) |

The solar direction keeps a fixed north-west bearing and only cycles elevation with time of day,
matching the 0.5.3.3368 client. Output goes to `output/synthesized-minimaps/<map>/tod-<time>/`
with a `synthesis-manifest.json`.

Equivalent CLI:

```powershell
WowViewer.Tool.Harvest synthetic-minimap `
  --client-root <clientRoot> --map Kalimdor --output-dir <dir> `
  --time-hours 1800 --per-tile --whole-map
```

### World viewing

- staged-client world loading
- terrain, WMOs, M2/MDX, liquids, PM4 overlays
- minimap + fullscreen minimap
- capture automation

### PM4 work

- PM4 overlay inspection
- PM4 Object Match window
- PM4/WMO Correlation window
- region-aware selection/export/debug surfaces

### Model and animation work

- standalone asset inspection
- model animation playback controls
- capture/export support where implemented

## Troubleshooting

### Viewer opens but map does not load

- verify staged client root
- verify chosen build matches staged data
- check console/log output with `--verbose`

### Capture flow fails

- verify output path exists
- verify startup flags point at staged data
- verify saved shot name exists before using `--capture-shot`

### PM4 windows missing

- use current Tools menu surfaces
- if working in legacy compatibility lane, remember latest slice is source-landed but legacy compile proof is still blocked outside that slice

## Related docs

- [../../README.md](/I:/parp/parp-tools/wow-viewer/README.md)
- [../../docs/CLI-TOOLS.md](/I:/parp/parp-tools/wow-viewer/docs/CLI-TOOLS.md)
- [../../docs/DOCUMENTATION-STATUS.md](/I:/parp/parp-tools/wow-viewer/docs/DOCUMENTATION-STATUS.md)
