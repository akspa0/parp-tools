# Flags Tracking & Visual Representation - IMMEDIATE PRIORITY

**Status**: 🚨 **URGENT** - Implement ASAP  
**User Request**: "This should be implemented sooner than later"

---

## 🎯 Goal

Track placement flags and use them to **visually differentiate objects in the viewer** with colors and shapes.

**Purpose**:
- **At-a-glance understanding** of object types and properties
- **Color-coded markers** based on flags
- **Different shapes** for different asset types (M2, WMO, liquids, etc.)
- **Interactive legend** showing what each color/shape means
- **No clicking required** to understand what you're looking at

---

## 🔍 Background

**User Clarification**: Flags are primarily for **visualization and analysis**, not rollback mechanism.

**Benefits**:
1. **Visual Differentiation**: Color-code by flags (hidden, particle emitters, collision-only, etc.)
2. **Shape Coding**: Different shapes for M2 vs WMO vs liquids
3. **Better Analysis**: Understand object distribution at a glance
4. **Legend System**: Sidebar legend shows color/shape meanings
5. **Enhanced Popups**: Detailed flag info on click

---

## 📋 Implementation Tasks

### Task 1: Add Flags to Data Model ⚡ IMMEDIATE

**File**: Search for `PlacementRecord` or equivalent placement data class

**Add flags field**:
```csharp
public class PlacementRecord
{
    public uint UniqueId { get; set; }
    public string ModelPath { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public ushort Scale { get; set; }
    
    // ADD THIS:
    public ushort Flags { get; set; }  // From MDDF/MODF chunk
    
    public int TileRow { get; set; }
    public int TileCol { get; set; }
}
```

**Parse from WDT/ADT**:
- MDDF offset 16: Flags (2 bytes, ushort)
- MODF offset 14: Flags (2 bytes, ushort)

---

### Task 2: Update CSV Export ⚡ IMMEDIATE

**File**: `WoWRollback.Core/Services/Analysis/UniqueIdRangeCsvWriter.cs`

**Add flags column to ALL CSVs**:

```csharp
// unique_ids_all.csv
public void WriteDetailedCsv(List<PlacementRecord> placements, string outputPath)
{
    var lines = new List<string>();
    lines.Add("UniqueID,ModelPath,TileRow,TileCol,PosX,PosY,PosZ,RotX,RotY,RotZ,Scale,Flags");
    
    foreach (var p in placements.OrderBy(x => x.UniqueId))
    {
        lines.Add($"{p.UniqueId},{p.ModelPath},{p.TileRow},{p.TileCol}," +
                 $"{p.Position.X},{p.Position.Y},{p.Position.Z}," +
                 $"{p.Rotation.X},{p.Rotation.Y},{p.Rotation.Z}," +
                 $"{p.Scale},0x{p.Flags:X4}");
    }
    
    File.WriteAllLines(outputPath, lines);
}

// id_ranges_by_map.csv
public void WriteMapRangesCsv(...)
{
    lines.Add("MinUniqueID,MaxUniqueID,Count,CommonFlags");
    
    foreach (var range in ranges)
    {
        // Find most common flags value in this range
        var commonFlags = placements
            .Where(p => p.UniqueId >= range.Min && p.UniqueId <= range.Max)
            .GroupBy(p => p.Flags)
            .OrderByDescending(g => g.Count())
            .First().Key;
        
        lines.Add($"{range.Min},{range.Max},{range.Count},0x{commonFlags:X4}");
    }
}

// id_ranges_by_tile.csv
public void WritePerTileRangesCsv(...)
{
    lines.Add("TileRow,TileCol,MinUniqueID,MaxUniqueID,Count,ModelType,CommonFlags");
    
    // ... (add commonFlags calculation per range)
}
```

---

### Task 3: Visual Marker System ⚡ IMMEDIATE

**File**: `ViewerAssets/js/map.js` or equivalent marker rendering

**Implement color/shape coding**:

```javascript
// Marker shape definitions
const MARKER_SHAPES = {
    M2: 'circle',
    WMO: 'square',
    LIQUID: 'triangle',
    PARTICLE: 'star',
    SOUND: 'diamond'
};

// Color coding based on flags
function getMarkerColor(placement) {
    // Reference: ADT_v18.md MDDF/MODF flags
    
    if (placement.flags & 0x0001) return '#FF0000'; // Red - Unknown flag
    if (placement.flags & 0x0002) return '#00FF00'; // Green - Unknown flag
    if (placement.flags & 0x0004) return '#0000FF'; // Blue - Unknown flag
    if (placement.flags & 0x0008) return '#FFFF00'; // Yellow - Unknown flag
    // ... (populate based on ADT_v18.md research)
    
    return '#808080'; // Gray - default/no flags
}

function getMarkerShape(placement) {
    if (placement.modelPath.endsWith('.m2')) return MARKER_SHAPES.M2;
    if (placement.modelPath.endsWith('.wmo')) return MARKER_SHAPES.WMO;
    // Add more types as discovered
    return MARKER_SHAPES.M2; // default
}

function createMarker(placement) {
    const color = getMarkerColor(placement);
    const shape = getMarkerShape(placement);
    
    // Use Leaflet.awesome-markers or similar for custom shapes
    const marker = L.marker([placement.y, placement.x], {
        icon: L.AwesomeMarkers.icon({
            icon: shape,
            markerColor: color,
            prefix: 'fa' // Font Awesome for shapes
        })
    });
    
    return marker;
}
```

---

### Task 4: Interactive Legend ⚡ IMMEDIATE

**File**: `ViewerAssets/index.html` + `ViewerAssets/js/legend.js` (NEW)

**Add legend sidebar**:

```html
<div id="legendPanel" class="sidebar-panel">
    <h3>Map Legend</h3>
    
    <div class="legend-section">
        <h4>Object Types (Shapes)</h4>
        <div class="legend-item">
            <span class="shape-circle"></span> M2 Models (Doodads)
        </div>
        <div class="legend-item">
            <span class="shape-square"></span> WMO Objects (Buildings)
        </div>
        <div class="legend-item">
            <span class="shape-triangle"></span> Liquids (Water/Lava/etc)
        </div>
        <div class="legend-item">
            <span class="shape-star"></span> Particle Emitters
        </div>
        <div class="legend-item">
            <span class="shape-diamond"></span> Sound Emitters
        </div>
    </div>
    
    <div class="legend-section">
        <h4>Flag States (Colors)</h4>
        <div class="legend-item">
            <span class="color-swatch" style="background: #808080"></span> Normal (no special flags)
        </div>
        <div class="legend-item">
            <span class="color-swatch" style="background: #FF0000"></span> Flag 0x0001 (TBD)
        </div>
        <div class="legend-item">
            <span class="color-swatch" style="background: #00FF00"></span> Flag 0x0002 (TBD)
        </div>
        <div class="legend-item">
            <span class="color-swatch" style="background: #0000FF"></span> Flag 0x0004 (TBD)
        </div>
        <!-- Add more as flags are researched -->
    </div>
    
    <div class="legend-section">
        <h4>Statistics</h4>
        <div id="legendStats">
            <p>Total Objects: <span id="totalObjects">0</span></p>
            <p>M2 Models: <span id="m2Count">0</span></p>
            <p>WMO Objects: <span id="wmoCount">0</span></p>
            <p>Flagged Objects: <span id="flaggedCount">0</span></p>
        </div>
    </div>
</div>
```

```javascript
// legend.js
class LegendManager {
    constructor() {
        this.flagCounts = {};
        this.typeCounts = {};
    }
    
    updateStatistics(placements) {
        this.typeCounts = {
            m2: 0,
            wmo: 0,
            liquid: 0,
            other: 0
        };
        
        this.flagCounts = {};
        
        placements.forEach(p => {
            // Count by type
            if (p.modelPath.endsWith('.m2')) this.typeCounts.m2++;
            else if (p.modelPath.endsWith('.wmo')) this.typeCounts.wmo++;
            else this.typeCounts.other++;
            
            // Count by flags
            if (p.flags !== 0) {
                this.flagCounts[p.flags] = (this.flagCounts[p.flags] || 0) + 1;
            }
        });
        
        this.render();
    }
    
    render() {
        document.getElementById('totalObjects').textContent = 
            Object.values(this.typeCounts).reduce((a, b) => a + b, 0);
        document.getElementById('m2Count').textContent = this.typeCounts.m2;
        document.getElementById('wmoCount').textContent = this.typeCounts.wmo;
        document.getElementById('flaggedCount').textContent = 
            Object.values(this.flagCounts).reduce((a, b) => a + b, 0);
    }
}
```

---

### Task 5: Update Viewer Popups ⚡ IMMEDIATE

**File**: `ViewerAssets/js/map.js` or equivalent marker click handler

**Display flags in object info popup**:

```javascript
function onMarkerClick(placement) {
    const flagsHex = '0x' + placement.flags.toString(16).padStart(4, '0');
    const flagsDecoded = decodeFlags(placement.flags);
    
    const popup = `
        <b>Object Info</b><br>
        UniqueID: ${placement.uniqueId}<br>
        Model: ${placement.modelPath}<br>
        Position: (${placement.x}, ${placement.y}, ${placement.z})<br>
        <b>Flags: ${flagsHex}</b><br>
        ${flagsDecoded.map(f => `  - ${f}`).join('<br>')}
    `;
    
    marker.bindPopup(popup).openPopup();
}

function decodeFlags(flags) {
    const decoded = [];
    
    // Known flag bits (research needed for exact meanings)
    if (flags & 0x0001) decoded.push('Flag 0x0001 (Unknown)');
    if (flags & 0x0002) decoded.push('Flag 0x0002 (Unknown)');
    if (flags & 0x0004) decoded.push('Flag 0x0004 (Unknown)');
    if (flags & 0x0008) decoded.push('Flag 0x0008 (Unknown)');
    if (flags & 0x0010) decoded.push('Flag 0x0010 (Unknown)');
    if (flags & 0x0020) decoded.push('Flag 0x0020 (Unknown)');
    if (flags & 0x0040) decoded.push('Flag 0x0040 (Possible Hidden?)');
    if (flags & 0x0080) decoded.push('Flag 0x0080 (Unknown)');
    // ... etc
    
    if (decoded.length === 0) {
        decoded.push('No flags set');
    }
    
    return decoded;
}
```

---

### Task 6: Flag Research ⚡ HIGH PRIORITY

**Primary Reference**: `lib/Warcraft.NET/Files/ADT/Chunks/`

**Secondary Reference**: `reference_data/wowdev.wiki/ADT_v18.md`

**Strategy**: Extract flag enums from Warcraft.NET, cross-reference with ADT_v18.md

**Extract flag definitions**:

1. **MDDF (M2 Doodad) Flags**:
   - Open `lib/Warcraft.NET/Files/ADT/Chunks/MDDF.cs`
   - Extract flag enum definitions
   - Cross-reference with ADT_v18.md

2. **MODF (WMO Object) Flags**:
   - Open `lib/Warcraft.NET/Files/ADT/Chunks/MODF.cs`
   - Extract flag enum definitions
   - Cross-reference with ADT_v18.md

3. **MHDR Flags** - Already visible at ADT_v18.md L80-104
4. **MCNK Flags** - Chunk-level flags from Warcraft.NET
5. **MH2O Flags** - Liquids flags from Warcraft.NET
6. **MCLQ Flags** - Old liquids chunk flags

**Create flag mapping document**:

`docs/ADT_FLAGS_REFERENCE.md`:
```markdown
# ADT Flag Definitions (from ADT_v18.md)

## MHDR Flags
- 0x0001: mhdr_MFBO (contains MFBO chunk)
- 0x0002: mhdr_northrend (Northrend map)

## MDDF Flags (M2 Placements)
- 0x???? - TBD (extract from ADT_v18.md)

## MODF Flags (WMO Placements)
- 0x???? - TBD (extract from ADT_v18.md)

## Color/Shape Mapping for Viewer

| Flag | Color | Meaning | Shape |
|------|-------|---------|-------|
| 0x0000 | Gray | Normal | Circle/Square |
| 0x???? | Red | ??? | Circle/Square |
...
```

---

### Task 7: Noggit Research 🔬 MEDIUM PRIORITY

**File**: Look at Noggit source code

**Search in Noggit codebase**:
1. **"show hidden models"** or **"hidden"** or **"visibility"**
2. **MDDF parsing** - see what flags field is called
3. **MODF parsing** - see what flags field is called
4. **Flag constants** - find #define or enum for flag bits

**Key files to check**:
- `ModelManager.h/cpp`
- `MapChunk.h/cpp`
- `World.h/cpp`
- `MapView.cpp` (rendering logic)

**Cross-reference with ADT_v18.md findings**

---

## 📊 Flag Bit Research Template

Create `docs/flags-research.md`:

```markdown
# Model Placement Flags Research

## MDDF Flags (M2 Doodads)

Offset 16, 2 bytes (ushort)

| Bit | Hex Value | Name | Description | Source |
|-----|-----------|------|-------------|--------|
| 0   | 0x0001    | ?    | Unknown     | ?      |
| 1   | 0x0002    | ?    | Unknown     | ?      |
| 2   | 0x0004    | ?    | Unknown     | ?      |
| 3   | 0x0008    | ?    | Unknown     | ?      |
| 4   | 0x0010    | ?    | Unknown     | ?      |
| 5   | 0x0020    | ?    | Unknown     | ?      |
| 6   | 0x0040    | HIDDEN? | Possible visibility flag | Noggit? |
| 7   | 0x0080    | ?    | Unknown     | ?      |
| ... | ...       | ...  | ...         | ...    |

## MODF Flags (WMO World Objects)

Offset 14, 2 bytes (ushort)

| Bit | Hex Value | Name | Description | Source |
|-----|-----------|------|-------------|--------|
| ... | ...       | ...  | ...         | ...    |

## Test Results

### Test 1: Flag 0x0040 in Alpha Client
- Result: ?
- Notes: ?

### Test 2: ...
```

---

## 🌊 Additional Data Tracking (Future Enhancement)

**User Request**: "Maybe even track things like MH2O liquids data... older WLW/WLM/WLQ files and MCLQ chunks"

### Liquids Tracking Expansion

**Current State**: ✅ We already have "simple ADT liquids flag tracking that works great"

**Future Enhancements**:

1. **MH2O Modern Liquids** (Wrath+)
   - Parse MH2O chunk from ADT
   - Extract liquid types (water, lava, slime, ocean)
   - Track liquid flags and rendering properties
   - Display as triangle markers in viewer

2. **MCLQ Old Liquids** (Alpha/Vanilla)
   - Parse MCLQ subchunk in MCNK
   - Extract liquid heights and types
   - Correlate with modern MH2O where applicable
   - Show in viewer with different color per type

3. **WLW/WLM/WLQ Legacy Files** (Pre-MCLQ)
   - Parse standalone liquid files
   - Extract liquid mesh data
   - Convert to modern format for viewer
   - Document format in reference_data

**Visualization Strategy**:
- **Shape**: Triangles (pointing up for water, down for lava)
- **Color**: Blue (water), Red (lava), Green (slime), Cyan (ocean)
- **Opacity**: Adjust based on liquid flags (shallow vs deep)

**Data Model Extension**:
```csharp
public class LiquidRecord
{
    public LiquidType Type { get; set; } // Water, Lava, Slime, Ocean
    public Vector2 Position { get; set; } // Chunk position
    public float MinHeight { get; set; }
    public float MaxHeight { get; set; }
    public ushort Flags { get; set; }
    public int TileRow { get; set; }
    public int TileCol { get; set; }
}
```

---

## 🚀 Priority Order

1. **⚡ IMMEDIATE**: Add flags field to PlacementRecord
2. **⚡ IMMEDIATE**: Parse flags from MDDF/MODF
3. **⚡ IMMEDIATE**: Add flags to CSV exports
4. **⚡ IMMEDIATE**: Implement visual marker system (colors/shapes)
5. **⚡ IMMEDIATE**: Create interactive legend
6. **⚡ IMMEDIATE**: Display flags in viewer popups
7. **🔬 HIGH**: Extract flag definitions from ADT_v18.md
8. **🔬 HIGH**: Create ADT_FLAGS_REFERENCE.md
9. **🔬 MEDIUM**: Research Noggit flag usage
10. **📝 MEDIUM**: Document all flag meanings
11. **🌊 FUTURE**: Expand liquids tracking (MH2O, WLW/WLM/WLQ, MCLQ)

---

## ✅ Success Criteria

### Data Pipeline
- [ ] Flags field added to PlacementRecord
- [ ] Flags parsed from MDDF/MODF chunks
- [ ] Flags visible in `unique_ids_all.csv`
- [ ] CommonFlags tracked in range CSVs
- [ ] ModelType + Flags in per-tile CSVs

### Viewer Visualization  
- [ ] Markers color-coded by flags
- [ ] Different shapes for M2 vs WMO vs liquids
- [ ] Interactive legend showing shape/color meanings
- [ ] Legend updates statistics (object counts by type/flag)
- [ ] Viewer popups show flags (hex + decoded)
- [ ] At-a-glance understanding without clicking

### Documentation
- [ ] Flag definitions extracted from ADT_v18.md
- [ ] ADT_FLAGS_REFERENCE.md created
- [ ] Color/shape mapping documented
- [ ] Noggit flag usage researched
- [ ] All known flag meanings documented

---

## 🎯 Ultimate Goals

### Short-term: Visual Analysis
- **At-a-glance object type identification** via shape
- **At-a-glance flag state identification** via color
- **Statistical overview** in legend panel
- **Detailed info on demand** via popups

### Long-term: Enhanced Rollback
- Understanding flags may reveal simpler rollback methods
- Flags could indicate special object behavior
- May enable flag-based filtering in addition to UniqueID filtering

---

**Start with Task 1 (add flags field) - implement NOW!**
