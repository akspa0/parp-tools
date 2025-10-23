# WoW Rollback Viewer

This is the interactive web viewer for WoW Rollback comparison results.

## Running the Viewer

The viewer uses JavaScript ES6 modules and requires a web server to function properly. It **will not work** if you open the HTML file directly (file:// protocol).

### Option 1: Python HTTP Server (Recommended)

Navigate to the viewer directory and run:

```powershell
cd rollback_outputs\comparisons\<comparison_name>\viewer
python -m http.server 8080
```

Then open: http://localhost:8080

### Option 2: Node.js HTTP Server

```powershell
cd rollback_outputs\comparisons\<comparison_name>\viewer
npx http-server -p 8080
```

Then open: http://localhost:8080

### Option 3: Live Server (VS Code Extension)

1. Install "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

## Features

### Index View (index.html)
- **Version Selector**: Choose which version to view
- **Map Selector**: Choose which map to explore
- **Tile Grid**: Click any tile to view details
- **Search**: Find specific tiles by coordinates (e.g., "30_30" or "r30c30")

### Tile View (tile.html)
- **Minimap Display**: Shows the minimap for the selected version/tile
- **Version Switcher**: Toggle between different versions
- **Diff Mode**: Compare two versions side-by-side
  - Added objects (green)
  - Removed objects (red)
  - Moved objects (orange)
  - Changed objects (purple)
- **Object List**: View all objects in the current tile

## Per-Version Minimap Structure

The viewer now properly separates minimaps by version:

```
viewer/
  minimap/
    0.5.3.3368/
      Kalimdor/
        Kalimdor_30_30.png
    0.5.5.3494/
      Kalimdor/
        Kalimdor_30_30.png
```

This prevents mixing tiles between versions (e.g., when 0.5.5 has tiles that 0.5.3 doesn't).

## Data Files

- **index.json**: Catalog of all maps, tiles, and versions
- **config.json**: Viewer configuration (thresholds, defaults)
- **overlays/{map}/tile_r{row}_c{col}.json**: Object placement data per tile
- **diffs/{map}/tile_r{row}_c{col}.json**: Diff data between versions

## Browser Compatibility

Requires a modern browser with ES6 module support:
- Chrome 61+
- Firefox 60+
- Safari 11+
- Edge 79+
