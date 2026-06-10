# Per-Tile UI Implementation Plan

**Goal**: Implement per-tile UniqueID selection UI and persistence for advanced Rollback mode.

---

## 🎯 Overview

### Two-Mode System

1. **Simple Mode** (index.html) - ✅ **ALREADY WORKING**
   - Map-wide range selection
   - Uses `id_ranges_by_map.csv`
   - Quick and easy

2. **Advanced Mode** (tile.html) - 🔨 **IMPLEMENT NOW**
   - Per-tile granular control
   - 64x64 tile grid interface
   - Uses `id_ranges_by_tile.csv`
   - Power user mode

---

## 📋 Step 1: Generate Per-Tile CSV Data

### File: `WoWRollback.Core/Services/Analysis/UniqueIdRangeCsvWriter.cs`

**Add new method:**
public void WritePerTileRangesCsv(
    List<PlacementRecord> placements,
    string outputPath)
{
    var lines = new List<string>();
    lines.Add("TileRow,TileCol,MinUniqueID,MaxUniqueID,Count,ModelType,CommonFlags");
    
    // Group by tile
    var byTile = placements
        .GroupBy(p => (p.TileRow, p.TileCol))
        .OrderBy(g => g.Key.TileRow)
        .ThenBy(g => g.Key.TileCol);
    
    foreach (var tileGroup in byTile)
    {
        var (row, col) = tileGroup.Key;
        var tilePlacements = tileGroup.ToList();
        
        // Cluster within tile (10K ranges)
        var ranges = ClusterIntoRanges(tilePlacements, clusterSize: 10000);
        
        foreach (var range in ranges)
        {
            // Determine model type (M2 or WMO)
            var firstPlacement = tilePlacements
                .First(p => p.UniqueId >= range.MinUniqueId && p.UniqueId <= range.MaxUniqueId);
            
            string modelType = firstPlacement.ModelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase) 
                ? "WMO" 
                : "M2";
            
            lines.Add($"{row},{col},{range.MinUniqueId},{range.MaxUniqueId},{range.Count},{modelType}");
        }
    }
    
    File.WriteAllLines(outputPath, lines);
}
```

**Integration point:**
- Call from existing analysis pipeline after map-wide CSV generation
- Output to: `cached_maps/analysis/{version}/{map}/csv/id_ranges_by_tile.csv`

---

## 📋 Step 2: Update Pipeline to Generate Per-Tile CSVs

### File: Search for where `id_ranges_by_map.csv` is currently generated

**Add after map-wide generation:**

```csharp
// Generate per-tile ranges (for advanced mode)
var perTileCsvPath = Path.Combine(csvOutputDir, "id_ranges_by_tile.csv");
csvWriter.WritePerTileRangesCsv(allPlacements, perTileCsvPath);
logger.LogInformation("Generated per-tile ranges: {Path}", perTileCsvPath);
```

---

## 📋 Step 3: Copy Per-Tile CSVs to Viewer

### File: `rebuild-and-regenerate.ps1`

**Update the CSV copy section (around line 622):**

```powershell
# Copy CSV data for sedimentary layers (UniqueID filtering)
Write-Host "Copying CSV data for UniqueID filtering..." -ForegroundColor Cyan
foreach ($ver in $Versions) {
    foreach ($mapName in $Maps) {
        $csvSource = Join-Path $PSScriptRoot "cached_maps\analysis\$ver\$mapName\csv"
        if (Test-Path $csvSource) {
            $csvDest = Join-Path $viewerDir "cached_maps\analysis\$ver\$mapName\csv"
            if (-not (Test-Path $csvDest)) {
                New-Item -Path $csvDest -ItemType Directory -Force | Out-Null
            }
            
            # Copy map-wide ranges (simple mode)
            Copy-Item -Path (Join-Path $csvSource "id_ranges_by_map.csv") `
                      -Destination $csvDest -Force -ErrorAction SilentlyContinue
            
            # Copy per-tile ranges (advanced mode)  ← NEW
            Copy-Item -Path (Join-Path $csvSource "id_ranges_by_tile.csv") `
                      -Destination $csvDest -Force -ErrorAction SilentlyContinue
            
            Copy-Item -Path (Join-Path $csvSource "unique_ids_all.csv") `
                      -Destination $csvDest -Force -ErrorAction SilentlyContinue
            
            Write-Host "  ✓ Copied CSVs for $ver/$mapName" -ForegroundColor DarkGray
        }
    }
}
```

---

## 📋 Step 4: Build tile.html UI

### File: `ViewerAssets/tile.html`

**Key components:**

1. **64x64 Tile Grid** (Canvas-based)
2. **Range Selection Modal** (Reuse checkbox code from sedimentary-layers-csv.js)
3. **Color Coding**:
   - 🟢 Green: All ranges selected
   - 🟡 Yellow: Partial selection
   - 🔴 Red: No ranges selected
   - ⚫ Gray: Empty tile

### File: `ViewerAssets/js/tile-selector.js` (NEW)

```javascript
class TileSelector {
    constructor() {
        this.currentMap = null;
        this.currentVersion = null;
        this.tileData = new Map(); // row_col → ranges
        this.tileSelections = new Map(); // row_col → selected ranges
        this.canvas = document.getElementById('tileGrid');
        this.ctx = this.canvas.getContext('2d');
    }
    
    async loadPerTileCsv(map, version) {
        const path = `cached_maps/analysis/${version}/${map}/csv/id_ranges_by_tile.csv`;
        const response = await fetch(path);
        const text = await response.text();
        
        this.tileData.clear();
        
        const lines = text.split('\n').slice(1); // Skip header
        lines.forEach(line => {
            const [row, col, min, max, count, type] = line.split(',');
            const key = `${row}_${col}`;
            
            if (!this.tileData.has(key)) {
                this.tileData.set(key, []);
            }
            
            this.tileData.get(key).push({
                min: parseInt(min),
                max: parseInt(max),
                count: parseInt(count),
                type: type.trim()
            });
        });
        
        this.renderGrid();
    }
    
    renderGrid() {
        const tileSize = this.canvas.width / 64;
        
        for (let row = 0; row < 64; row++) {
            for (let col = 0; col < 64; col++) {
                const key = `${row}_${col}`;
                const ranges = this.tileData.get(key) || [];
                const selections = this.tileSelections.get(key) || [];
                
                // Determine color
                let color;
                if (ranges.length === 0) {
                    color = '#444'; // Gray - empty
                } else if (selections.length === 0) {
                    color = '#f44'; // Red - nothing selected
                } else if (selections.length === ranges.length) {
                    color = '#4f4'; // Green - all selected
                } else {
                    color = '#ff4'; // Yellow - partial
                }
                
                this.ctx.fillStyle = color;
                this.ctx.fillRect(col * tileSize, row * tileSize, tileSize, tileSize);
                
                // Draw grid lines
                this.ctx.strokeStyle = '#000';
                this.ctx.strokeRect(col * tileSize, row * tileSize, tileSize, tileSize);
            }
        }
    }
    
    onTileClick(x, y) {
        const tileSize = this.canvas.width / 64;
        const col = Math.floor(x / tileSize);
        const row = Math.floor(y / tileSize);
        const key = `${row}_${col}`;
        
        const ranges = this.tileData.get(key);
        if (!ranges || ranges.length === 0) return;
        
        this.openRangeSelectionModal(row, col, ranges);
    }
    
    openRangeSelectionModal(row, col, ranges) {
        const modal = document.getElementById('rangeModal');
        const content = document.getElementById('rangeCheckboxes');
        
        content.innerHTML = `<h3>Tile [${row}, ${col}]</h3>`;
        
        const key = `${row}_${col}`;
        const currentSelections = this.tileSelections.get(key) || [];
        
        ranges.forEach((range, index) => {
            const checked = currentSelections.includes(range.min);
            content.innerHTML += `
                <label>
                    <input type="checkbox" 
                           data-min="${range.min}" 
                           data-max="${range.max}"
                           ${checked ? 'checked' : ''}>
                    ${range.min}-${range.max} (${range.count} objects, ${range.type})
                </label><br>
            `;
        });
        
        modal.style.display = 'block';
        
        // Save button handler
        document.getElementById('saveRanges').onclick = () => {
            this.saveSelections(row, col, ranges);
            modal.style.display = 'none';
        };
    }
    
    saveSelections(row, col, ranges) {
        const key = `${row}_${col}`;
        const checkboxes = document.querySelectorAll('#rangeCheckboxes input[type=checkbox]');
        const selected = [];
        
        checkboxes.forEach(cb => {
            if (cb.checked) {
                selected.push(parseInt(cb.dataset.min));
            }
        });
        
        this.tileSelections.set(key, selected);
        this.saveToLocalStorage();
        this.renderGrid();
    }
    
    saveToLocalStorage() {
        const config = {
            map: this.currentMap,
            version: this.currentVersion,
            mode: 'advanced',
            tiles: {}
        };
        
        this.tileSelections.forEach((selected, key) => {
            const ranges = this.tileData.get(key) || [];
            config.tiles[key] = {
                selectedRanges: ranges.filter(r => selected.includes(r.min))
            };
        });
        
        localStorage.setItem('rollback_config', JSON.stringify(config));
        console.log('Saved config to localStorage:', config);
    }
    
    loadFromLocalStorage() {
        const stored = localStorage.getItem('rollback_config');
        if (!stored) return;
        
        const config = JSON.parse(stored);
        if (config.mode !== 'advanced') return;
        if (config.map !== this.currentMap) return;
        if (config.version !== this.currentVersion) return;
        
        this.tileSelections.clear();
        
        Object.keys(config.tiles).forEach(key => {
            const tile = config.tiles[key];
            const selectedMins = tile.selectedRanges.map(r => r.min);
            this.tileSelections.set(key, selectedMins);
        });
        
        this.renderGrid();
    }
}
```

---

## 📋 Step 5: Create Export Configuration Page

### File: `ViewerAssets/export.html` (NEW)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Rollback Configuration Manager</title>
    <link rel="stylesheet" href="styles/styles.css">
</head>
<body>
    <h1>🔧 Rollback Configuration Manager</h1>
    
    <div id="configSummary">
        <h2>Current Configuration</h2>
        <p>Map: <span id="map"></span></p>
        <p>Version: <span id="version"></span></p>
        <p>Mode: <span id="mode"></span></p>
        <p>Objects to keep: <span id="keepCount"></span></p>
        <p>Objects to remove: <span id="removeCount"></span></p>
        <p>Affected tiles: <span id="tileCount"></span></p>
        <p>Created: <span id="created"></span></p>
        <p>Last Modified: <span id="modified"></span></p>
    </div>
    
    <div id="exportButtons">
        <h3>Export Options</h3>
        <button onclick="downloadConfig()">📥 Download Configuration JSON</button>
        <button onclick="downloadReport()">📊 Download Detailed Report</button>
        <button onclick="copyToClipboard()">📋 Copy Config to Clipboard</button>
        
        <h3>Import Options</h3>
        <input type="file" id="importFile" accept=".json" style="display:none" onchange="importConfig(event)">
        <button onclick="document.getElementById('importFile').click()">📤 Import Configuration</button>
        
        <h3>Patch Generation (CLI)</h3>
        <button onclick="copyCliCommand()">⌨️ Copy CLI Command</button>
        <p class="hint">Copy this command and run it in PowerShell to generate patched files</p>
    </div>
    
    <div id="configPreview">
        <h3>Configuration Preview</h3>
        <pre id="configJson"></pre>
    </div>
    
    <script src="js/export-config.js"></script>
</body>
</html>
```

### File: `ViewerAssets/js/export-config.js` (NEW)

```javascript
function loadConfig() {
    const config = JSON.parse(localStorage.getItem('rollback_config') || '{}');
    
    // Add metadata if missing
    if (!config.metadata) {
        config.metadata = {
            created: new Date().toISOString(),
            modified: new Date().toISOString(),
            version: '1.0',
            generator: 'WoWRollback Viewer'
        };
        localStorage.setItem('rollback_config', JSON.stringify(config));
    }
    
    document.getElementById('map').textContent = config.map || 'N/A';
    document.getElementById('version').textContent = config.version || 'N/A';
    document.getElementById('mode').textContent = config.mode || 'simple';
    document.getElementById('created').textContent = 
        config.metadata?.created ? new Date(config.metadata.created).toLocaleString() : 'N/A';
    document.getElementById('modified').textContent = 
        config.metadata?.modified ? new Date(config.metadata.modified).toLocaleString() : 'N/A';
    
    // Calculate statistics
    let keepCount = 0;
    let removeCount = 0;
    let tileCount = 0;
    
    if (config.mode === 'advanced' && config.tiles) {
        tileCount = Object.keys(config.tiles).length;
        Object.values(config.tiles).forEach(tile => {
            keepCount += tile.selectedRanges.reduce((sum, r) => sum + r.count, 0);
        });
    } else if (config.mode === 'simple' && config.selectedRanges) {
        keepCount = config.selectedRanges.reduce((sum, r) => sum + r.count, 0);
    }
    
    document.getElementById('keepCount').textContent = keepCount;
    document.getElementById('tileCount').textContent = tileCount;
    
    // Display config JSON
    document.getElementById('configJson').textContent = 
        JSON.stringify(config, null, 2);
}

function downloadConfig() {
    const config = JSON.parse(localStorage.getItem('rollback_config') || '{}');
    
    // Update modified timestamp
    config.metadata = config.metadata || {};
    config.metadata.modified = new Date().toISOString();
    
    const filename = `rollback_${config.map}_${config.version}_${config.mode}.json`;
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function downloadReport() {
    const config = JSON.parse(localStorage.getItem('rollback_config') || '{}');
    
    let report = `# WoWRollback Detailed Report\n\n`;
    report += `**Generated**: ${new Date().toLocaleString()}\n\n`;
    report += `## Configuration Summary\n\n`;
    report += `- **Map**: ${config.map || 'N/A'}\n`;
    report += `- **Version**: ${config.version || 'N/A'}\n`;
    report += `- **Mode**: ${config.mode || 'N/A'}\n`;
    report += `- **Created**: ${config.metadata?.created ? new Date(config.metadata.created).toLocaleString() : 'N/A'}\n`;
    report += `- **Last Modified**: ${config.metadata?.modified ? new Date(config.metadata.modified).toLocaleString() : 'N/A'}\n\n`;
    
    if (config.mode === 'advanced' && config.tiles) {
        report += `## Per-Tile Selections\n\n`;
        report += `**Total tiles configured**: ${Object.keys(config.tiles).length}\n\n`;
        
        Object.keys(config.tiles).sort().forEach(tileKey => {
            const tile = config.tiles[tileKey];
            const [row, col] = tileKey.split('_');
            report += `### Tile [${row}, ${col}]\n\n`;
            report += `**Selected Ranges**: ${tile.selectedRanges.length}\n\n`;
            
            if (tile.selectedRanges.length > 0) {
                report += `| Min UniqueID | Max UniqueID | Count | Type |\n`;
                report += `|--------------|--------------|-------|------|\n`;
                tile.selectedRanges.forEach(range => {
                    report += `| ${range.min} | ${range.max} | ${range.count} | ${range.type} |\n`;
                });
                report += `\n`;
            }
        });
    } else if (config.mode === 'simple' && config.selectedRanges) {
        report += `## Map-Wide Selections\n\n`;
        report += `**Selected Ranges**: ${config.selectedRanges.length}\n\n`;
        report += `| Min UniqueID | Max UniqueID | Count |\n`;
        report += `|--------------|--------------|-------|\n`;
        config.selectedRanges.forEach(range => {
            report += `| ${range.min} | ${range.max} | ${range.count} |\n`;
        });
        report += `\n`;
    }
    
    report += `## Invisible Model Mappings\n\n`;
    report += `- **M2 (Doodads)**: Objects will be replaced with \`SPELLS\\Invisible.m2\`\n`;
    report += `- **WMO (Buildings)**: Objects will be replaced with \`world\\wmo\\dungeon\\test\\test.wmo\`\n\n`;
    
    report += `## Next Steps\n\n`;
    report += `1. Download the JSON configuration file\n`;
    report += `2. Run the CLI command to generate patched files:\n\n`;
    report += `\`\`\`powershell\n`;
    report += generateCliCommand() + `\n`;
    report += `\`\`\`\n\n`;
    report += `3. Test patched files in WoW Alpha ${config.version} client\n`;
    report += `4. Keep this report for lineage tracking\n`;
    
    const filename = `rollback_report_${config.map}_${config.version}_${Date.now()}.md`;
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function copyToClipboard() {
    const config = localStorage.getItem('rollback_config');
    navigator.clipboard.writeText(config).then(() => {
        alert('Configuration copied to clipboard!');
    });
}

function importConfig(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const imported = JSON.parse(e.target.result);
            
            // Validate config
            if (!imported.map || !imported.version || !imported.mode) {
                alert('Invalid configuration file!');
                return;
            }
            
            // Update metadata
            imported.metadata = imported.metadata || {};
            imported.metadata.imported = new Date().toISOString();
            imported.metadata.originalCreated = imported.metadata.created;
            imported.metadata.created = new Date().toISOString();
            
            // Save to localStorage
            localStorage.setItem('rollback_config', JSON.stringify(imported));
            
            alert(`Configuration imported successfully!\nMap: ${imported.map}\nVersion: ${imported.version}\nMode: ${imported.mode}`);
            
            // Reload page to reflect changes
            location.reload();
        } catch (err) {
            alert('Error importing configuration: ' + err.message);
        }
    };
    reader.readAsText(file);
}

function generateCliCommand() {
    const config = JSON.parse(localStorage.getItem('rollback_config') || '{}');
    
    let cmd = `dotnet run --project WoWRollback.Cli -- rollback patch-alpha-wdt `;
    cmd += `\\\n  --wdt "test_data/${config.version}/tree/World/Maps/${config.map}/${config.map}.wdt" `;
    cmd += `\\\n  --config "rollback_${config.map}_${config.version}_${config.mode}.json" `;
    cmd += `\\\n  --output "patched/${config.version}/${config.map}/${config.map}.wdt"`;
    
    return cmd;
}

function copyCliCommand() {
    const cmd = generateCliCommand();
    navigator.clipboard.writeText(cmd).then(() => {
        alert('CLI command copied to clipboard!');
    });
}

window.onload = loadConfig;
```

---

## 📋 Step 6: Navigation Integration

### Update `ViewerAssets/index.html`

Add buttons to switch modes:

```html
<div id="modeSelector">
    <button onclick="location.href='index.html'">Simple Mode</button>
    <button onclick="location.href='tile.html'">Advanced Mode (Per-Tile)</button>
    <button onclick="location.href='export.html'">Export Configuration</button>
</div>
```

---

## 📊 Implementation Order

1. ✅ **Step 1**: Add `WritePerTileRangesCsv()` method (30 min)
2. ✅ **Step 2**: Integrate into analysis pipeline (15 min)
3. ✅ **Step 3**: Update `rebuild-and-regenerate.ps1` CSV copy (10 min)
4. 🔨 **Step 4**: Build `tile.html` + `tile-selector.js` (3-4 hours)
5. 🔨 **Step 5**: Create `export.html` + `export-config.js` (1-2 hours)
6. ✅ **Step 6**: Add navigation buttons (15 min)

**Total estimated time**: 5-7 hours

---

## ✅ Success Criteria

- [ ] Per-tile CSVs generated automatically
- [ ] tile.html displays 64x64 grid
- [ ] Clicking tile opens range selection modal
- [ ] Selections persist in localStorage
- [ ] Grid colors update based on selections
- [ ] export.html shows config summary
- [ ] Can download JSON config file
- [ ] Supports both M2 and WMO model types

---

## 🎯 Next Phase: AlphaWDT Patching

After UI is complete, implement backend patching:
- Read rollback config JSON
- Parse AlphaWDT file
- Replace model paths with invisible models:
  - M2: `SPELLS\Invisible.m2`
  - WMO: `world\wmo\dungeon\test\test.wmo`
- Write patched WDT
- Test in Alpha client
