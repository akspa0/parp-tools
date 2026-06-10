# WoWRollback.ViewerModule

## Overview
Lightweight ASP.NET Core (Kestrel) static file server used by the toolkit to host the web viewer. This module backs the `serve-viewer` command in `WoWRollback.Cli`.

## Quick Start
Serve an existing viewer directory:
```powershell
dotnet run --project ..\WoWRollback.Cli -- serve-viewer --viewer-dir analysis_output\viewer --port 8080
```

## Notes
- Proper MIME types for `.webp`, `.json`, `.geojson` are configured.
- `--no-browser` disables auto-open.
- Cross-platform; suitable for local use and small demos.

## See Also
- Root `README.md` → “Serve Viewer (Built-in HTTP Server)”
- `../docs/V...` → `VIEWER_OVERLAY_BUG_FIXES.md` and `VIEWER_FIX_PLAN.md`
