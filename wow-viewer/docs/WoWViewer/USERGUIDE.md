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

- Use staged clients only under `output/tmp/wowarchive-clients/`.
- Any `H:\CLIENTS` path is stale and wrong.

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
