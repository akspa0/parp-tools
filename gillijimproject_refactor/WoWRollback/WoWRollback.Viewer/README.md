# WoWRollback.Viewer

## Overview
Static web assets (HTML/JS/CSS) for the WoWRollback interactive viewer.
- Loads overlays, placements, terrain CSVs, meshes (GLB)
- Works with any static file server

## Quick Start
After generating outputs (via `WoWRollback.Cli analyze-map-adts` or the Orchestrator), serve the `viewer/` directory:
```powershell
# Using the built-in server
dotnet run --project ..\WoWRollback.Cli -- serve-viewer --viewer-dir analysis_output\viewer --port 8080
```

## Customize
- Edit assets in this folder to change look and behavior (styles, UI, plugins).
- Overlays and data files are loaded from the generated output structure.

## See Also
- Root `README.md` → “Serve Viewer (Built-in HTTP Server)”
- `../WoWRollback.ViewerModule/README.md` for server details
