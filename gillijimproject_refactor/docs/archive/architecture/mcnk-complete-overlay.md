# MCNK Complete Terrain Overlay Design

## Purpose

Extract and visualize **all** MCNK chunk flags and metadata from ADT files, including terrain properties, liquid types, holes, and area boundaries.

---

## ADT Structure Reference

### MCNK Chunk (Map Chunk) - Complete

From `reference_data/wowdev.wiki/ADT_v18.md`:

```c
struct SMChunk {
  struct {
    uint32_t has_mcsh : 1;              // bit 0  - has shadow map (MCSH subchunk)
    uint32_t impass : 1;                // bit 1  - impassible terrain
    uint32_t lq_river : 1;              // bit 2  - contains river liquid
    uint32_t lq_ocean : 1;              // bit 3  - contains ocean liquid
    uint32_t lq_magma : 1;              // bit 4  - contains magma liquid
    uint32_t lq_slime : 1;              // bit 5  - contains slime liquid
    uint32_t has_mccv : 1;              // bit 6  - has vertex colors (MCCV subchunk)
    uint32_t unknown_0x80 : 1;          // bit 7  - unknown
    uint32_t : 7;                       // bits 8-14 - unused
    uint32_t do_not_fix_alpha_map : 1;  // bit 15 - alpha map format flag
    uint32_t high_res_holes : 1;        // bit 16 - use 64-bit hole map
    uint32_t : 15;                      // bits 17-31 - unused
  } flags;
  
  uint32_t IndexX;                     // Chunk column (0-15)
  uint32_t IndexY;                     // Chunk row (0-15)
  uint32_t nLayers;                    // Number of texture layers (max 4)
  uint32_t nDoodadRefs;                // Number of doodad references
  uint32_t ofsHeight;                  // Offset to MCVT (height) subchunk
  uint32_t ofsNormal;                  // Offset to MCNR (normals) subchunk
  uint32_t ofsLayer;                   // Offset to MCLY (layers) subchunk
  uint32_t ofsRefs;                    // Offset to MCRF (refs) subchunk
  uint32_t ofsAlpha;                   // Offset to MCAL (alpha) subchunk
  uint32_t sizeAlpha;                  // Size of alpha data
  uint32_t ofsShadow;                  // Offset to MCSH (shadow) subchunk
  uint32_t sizeShadow;                 // Size of shadow data
  uint32_t areaid;                     // AreaTable.dbc ID ← TARGET for AreaID overlay
  uint32_t nMapObjRefs;                // Number of WMO references
  uint16_t holes_low_res;              // 4×4 hole bitmap (16 bits)
  uint16_t unknown_but_used;           // Unknown field
  uint2_t[8][8] ReallyLowQualityTextureingMap;  // Texture detail map
  uint1_t[8][8] noEffectDoodad;        // Doodad disable bitmap
  uint32_t ofsSndEmitters;             // Offset to sound emitters
  uint32_t nSndEmitters;               // Number of sound emitters
  uint32_t ofsLiquid;                  // Offset to liquid data
  uint32_t sizeLiquid;                 // Size of liquid data
  C3Vector position;                   // World position of chunk center
  uint32_t ofsMCCV;                    // Offset to vertex colors
  uint32_t ofsMCLV;                    // Offset to vertex lighting
  uint32_t unused;
  
  // If high_res_holes flag is set:
  uint64_t holes_high_res;             // 8×8 hole bitmap (at offset 0x14, overrides ofsHeight/ofsNormal)
};
```

---

## Data Pipeline Design

### Stage 1: Extraction (AlphaWDTAnalysisTool)

#### Expanded CSV Schema: `<map>_mcnk_terrain.csv`

```csv
map,tile_row,tile_col,chunk_row,chunk_col,flags_raw,has_mcsh,impassible,lq_river,lq_ocean,lq_magma,lq_slime,has_mccv,high_res_holes,areaid,num_layers,has_holes,hole_type,hole_bitmap_hex,hole_count,position_x,position_y,position_z
Azeroth,31,34,0,0,0x00000002,false,true,false,false,false,false,false,false,1519,3,false,none,0x0000,0,-1470.834,206.87,150.2
Azeroth,31,34,0,1,0x00010046,false,false,true,true,false,false,false,false,1519,2,true,low_res,0x00F0,4,-1404.167,206.87,150.2
```

#### Column Definitions:

| Column | Type | Description |
|--------|------|-------------|
| `map` | string | Map name |
| `tile_row` | int | ADT tile row (0-63) |
| `tile_col` | int | ADT tile column (0-63) |
| `chunk_row` | int | MCNK chunk row within tile (0-15) |
| `chunk_col` | int | MCNK chunk column within tile (0-15) |
| `flags_raw` | hex | Full 32-bit flags value |
| `has_mcsh` | bool | Has shadow map data |
| `impassible` | bool | Terrain is impassible |
| `lq_river` | bool | Contains river liquid |
| `lq_ocean` | bool | Contains ocean liquid |
| `lq_magma` | bool | Contains magma liquid |
| `lq_slime` | bool | Contains slime liquid |
| `has_mccv` | bool | Has vertex colors |
| `high_res_holes` | bool | Uses 64-bit hole map |
| `areaid` | int | AreaTable.dbc ID (0 if unmapped) |
| `num_layers` | int | Number of texture layers (0-4) |
| `has_holes` | bool | Has any terrain holes |
| `hole_type` | enum | "none", "low_res", "high_res" |
| `hole_bitmap_hex` | hex | Hole bitmap (16 or 64 bits) |
| `hole_count` | int | Number of hole bits set |
| `position_x` | float | World X coordinate of chunk center |
| `position_y` | float | World Y coordinate of chunk center |
| `position_z` | float | World Z coordinate of chunk center |

#### Additional CSV: Shadow Maps

**File**: `<map>_mcnk_shadows.csv`

```csv
map,tile_row,tile_col,chunk_row,chunk_col,shadow_bitmap_base64
Azeroth,31,34,0,0,AAAA//8AAAD//wAA...
Azeroth,31,34,0,1,/////wAAAAD//wAA...
```

**Columns**:
- `shadow_bitmap_base64`: Base64-encoded 64×64 bit array (512 bytes = 4096 bits)

**Why separate CSV?**
- Shadow data is large (512 bytes per chunk × 256 chunks = 128KB per tile)
- Optional overlay (not everyone needs shadows)
- Can be loaded on-demand

---

#### Implementation: `McnkTerrainExtractor.cs`

```csharp
namespace AlphaWdtAnalyzer.Core;

public sealed class McnkTerrainExtractor
{
    public static List<McnkTerrainEntry> ExtractTerrain(WdtAlphaScanner wdt)
    {
        var results = new List<McnkTerrainEntry>();
        
        foreach (var adtNum in wdt.AdtNumbers)
        {
            var adt = new AdtAlpha(wdt.WdtPath, wdt.AdtMhdrOffsets[adtNum], adtNum);
            var tileX = adt.GetXCoord();
            var tileY = adt.GetYCoord();
            
            for (int chunkRow = 0; chunkRow < 16; chunkRow++)
            {
                for (int chunkCol = 0; chunkCol < 16; chunkCol++)
                {
                    var mcnk = adt.GetMcnkChunk(chunkRow, chunkCol);
                    
                    // Parse flags (offset 0x00, 4 bytes)
                    uint flags = BitConverter.ToUInt32(mcnk, 0x00);
                    
                    // Extract all flag bits
                    bool hasMcsh = (flags & 0x1) != 0;
                    bool impassible = (flags & 0x2) != 0;
                    bool lqRiver = (flags & 0x4) != 0;
                    bool lqOcean = (flags & 0x8) != 0;
                    bool lqMagma = (flags & 0x10) != 0;
                    bool lqSlime = (flags & 0x20) != 0;
                    bool hasMccv = (flags & 0x40) != 0;
                    bool highResHoles = (flags & 0x10000) != 0;
                    
                    // Parse other fields
                    uint indexX = BitConverter.ToUInt32(mcnk, 0x04);
                    uint indexY = BitConverter.ToUInt32(mcnk, 0x08);
                    uint nLayers = BitConverter.ToUInt32(mcnk, 0x0C);
                    uint areaId = BitConverter.ToUInt32(mcnk, 0x34);
                    
                    // Parse chunk position (offset 0x68, 12 bytes = 3 floats)
                    float posX = BitConverter.ToSingle(mcnk, 0x68);
                    float posY = BitConverter.ToSingle(mcnk, 0x6C);
                    float posZ = BitConverter.ToSingle(mcnk, 0x70);
                    
                    // Parse holes
                    string holeType = "none";
                    string holeBitmap = "0x0000";
                    int holeCount = 0;
                    
                    if (highResHoles)
                    {
                        ulong holes = BitConverter.ToUInt64(mcnk, 0x14);
                        holeType = "high_res";
                        holeBitmap = $"0x{holes:X16}";
                        holeCount = CountBits(holes);
                    }
                    else
                    {
                        ushort holes = BitConverter.ToUInt16(mcnk, 0x3C);
                        if (holes != 0)
                        {
                            holeType = "low_res";
                            holeBitmap = $"0x{holes:X4}";
                            holeCount = CountBits(holes);
                        }
                    }
                    
                    results.Add(new McnkTerrainEntry(
                        Map: wdt.MapName,
                        TileRow: tileY,
                        TileCol: tileX,
                        ChunkRow: chunkRow,
                        ChunkCol: chunkCol,
                        FlagsRaw: $"0x{flags:X8}",
                        HasMcsh: hasMcsh,
                        Impassible: impassible,
                        LqRiver: lqRiver,
                        LqOcean: lqOcean,
                        LqMagma: lqMagma,
                        LqSlime: lqSlime,
                        HasMccv: hasMccv,
                        HighResHoles: highResHoles,
                        AreaId: (int)areaId,
                        NumLayers: (int)nLayers,
                        HasHoles: holeCount > 0,
                        HoleType: holeType,
                        HoleBitmapHex: holeBitmap,
                        HoleCount: holeCount,
                        PositionX: posX,
                        PositionY: posY,
                        PositionZ: posZ
                    ));
                }
            }
        }
        
        return results;
    }
    
    private static int CountBits(ulong value)
    {
        int count = 0;
        while (value != 0)
        {
            count += (int)(value & 1);
            value >>= 1;
        }
        return count;
    }
}

public record McnkTerrainEntry(
    string Map,
    int TileRow, int TileCol,
    int ChunkRow, int ChunkCol,
    string FlagsRaw,
    bool HasMcsh,
    bool Impassible,
    bool LqRiver,
    bool LqOcean,
    bool LqMagma,
    bool LqSlime,
    bool HasMccv,
    bool HighResHoles,
    int AreaId,
    int NumLayers,
    bool HasHoles,
    string HoleType,
    string HoleBitmapHex,
    int HoleCount,
    float PositionX,
    float PositionY,
    float PositionZ
);
```

---

## Overlay Categories

The complete MCNK data supports **multiple overlay types**:

### 1. Terrain Properties Overlay
**Flags**: `impassible`, `has_mccv`, `num_layers`

**Visualization**:
- Impassible areas: Red overlay
- Vertex-colored areas: Blue tint
- Multi-layer texture areas: Yellow border

### 1b. Shadow Map Overlay (MCSH)
**Data**: 64×64 bitmap of baked shadows per chunk

**Visualization**:
- Render as semi-transparent grayscale overlay
- Dark areas = shadowed
- Light areas = lit
- One bitmap per chunk (16×16 chunks per tile = 16 shadow bitmaps)
- Composite into single canvas per tile for performance

### 2. Liquid Overlay
**Flags**: `lq_river`, `lq_ocean`, `lq_magma`, `lq_slime`

**Visualization**:
- Rivers: Light blue
- Oceans: Deep blue
- Magma: Orange-red
- Slime: Green

### 3. Holes Overlay
**Data**: `holes_low_res`, `holes_high_res`

**Visualization**:
- Holes: Black rectangles
- Grid resolution: 4×4 or 8×8 depending on type

### 4. AreaID Boundary Overlay
**Data**: `areaid`

**Visualization**:
- Chunk outlines colored by AreaID
- Gradient between different areas
- Popup shows area name from AreaTable.dbc

---

## Stage 2: Transformation (WoWRollback.Core)

### Multi-Category Overlay JSON

```json
{
  "map": "Azeroth",
  "tile": {"row": 31, "col": 34},
  "minimap": {"width": 512, "height": 512},
  "chunk_size": 32,
  "layers": [
    {
      "version": "0.5.3",
      "terrain_properties": {
        "impassible": [{"row": 0, "col": 0}],
        "shadow_mapped": [{"row": 1, "col": 1}],
        "vertex_colored": [{"row": 2, "col": 2}]
      },
      "liquids": {
        "river": [{"row": 3, "col": 3}],
        "ocean": [{"row": 4, "col": 4}],
        "magma": [],
        "slime": []
      },
      "holes": [
        {
          "row": 5,
          "col": 5,
          "type": "low_res",
          "holes": [0, 1, 4, 5]
        }
      ],
      "area_ids": {
        "chunks": [
          {"row": 0, "col": 0, "area_id": 1519, "area_name": "Stormwind City"},
          {"row": 0, "col": 1, "area_id": 1519, "area_name": "Stormwind City"}
        ],
        "boundaries": [
          {
            "from_area": 1519,
            "to_area": 12,
            "edge": [[0, 5], [0, 6]]
          }
        ]
      }
    }
  ]
}
```

---

## Stage 3: Visualization (ViewerAssets)

### Module Structure

```
ViewerAssets/js/
├── overlays/
│   ├── terrainPropertiesLayer.js  # Impassible, shadows, vertex colors
│   ├── liquidsLayer.js            # River, ocean, magma, slime
│   ├── holesLayer.js              # Terrain holes
│   └── areaIdLayer.js             # Area boundaries
└── overlayManager.js              # Coordinate all overlay types
```

### UI Control Panel

```html
<div class="overlay-controls">
  <h3>Terrain Overlays</h3>
  
  <!-- Terrain Properties -->
  <div class="overlay-group">
    <label>
      <input type="checkbox" id="showTerrainProps" checked>
      Terrain Properties
    </label>
    <div class="indent" id="terrainPropsOptions">
      <label><input type="checkbox" id="showImpassible" checked> Impassible</label>
      <label><input type="checkbox" id="showShadows"> Shadow Mapped</label>
      <label><input type="checkbox" id="showVertexColors"> Vertex Colored</label>
    </div>
  </div>
  
  <!-- Liquids -->
  <div class="overlay-group">
    <label>
      <input type="checkbox" id="showLiquids">
      Liquids
    </label>
    <div class="indent" id="liquidsOptions">
      <label><input type="checkbox" id="showRiver" checked> Rivers</label>
      <label><input type="checkbox" id="showOcean" checked> Oceans</label>
      <label><input type="checkbox" id="showMagma" checked> Magma</label>
      <label><input type="checkbox" id="showSlime" checked> Slime</label>
    </div>
  </div>
  
  <!-- Holes -->
  <div class="overlay-group">
    <label>
      <input type="checkbox" id="showHoles">
      Terrain Holes
    </label>
  </div>
  
  <!-- Area Boundaries -->
  <div class="overlay-group">
    <label>
      <input type="checkbox" id="showAreaBoundaries">
      Area Boundaries
    </label>
    <div class="indent" id="areaOptions">
      <label><input type="checkbox" id="showAreaNames" checked> Show Names</label>
      <label><input type="checkbox" id="colorByArea" checked> Color By Area</label>
    </div>
  </div>
</div>
```

---

## Color Scheme

### Terrain Properties
- **Impassible**: `rgba(255, 0, 0, 0.3)` - Red
- **Shadow Mapped**: `rgba(128, 128, 128, 0.2)` - Gray
- **Vertex Colored**: `rgba(0, 128, 255, 0.2)` - Blue

### Liquids
- **River**: `rgba(64, 164, 223, 0.4)` - Light blue
- **Ocean**: `rgba(0, 64, 164, 0.5)` - Deep blue
- **Magma**: `rgba(255, 69, 0, 0.6)` - Orange-red
- **Slime**: `rgba(0, 255, 0, 0.4)` - Green

### Holes
- **Holes**: `rgba(0, 0, 0, 0.7)` - Black

### Area Boundaries
- **Edge Lines**: `rgba(255, 255, 0, 0.8)` - Yellow, 2px width
- **Area Fill**: Hash colors by AreaID with `opacity: 0.1`

---

## AreaID Integration

### Leverage Existing AlphaWDTAnalysisTool Data

The tool already generates `areaid_verify_*.csv` in verbose mode:

```csv
tile,chunk_y,chunk_x,alpha_areaid,alpha_zone,alpha_sub,chosen_lk_area,reason
31_34,0,0,360960,5,3840,1519,"strict CSV numeric match"
```

**Strategy**: 
1. Extract AreaID from MCNK directly (always available)
2. Optionally consume `areaid_verify_*.csv` for mapping metadata
3. Load AreaTable.dbc names from viewer config (JSON lookup)

### AreaID Visualization Options

1. **Chunk Coloring**: Hash AreaID to consistent color
2. **Boundary Lines**: Detect chunk neighbors with different AreaID
3. **Labels**: Render area name at chunk centers
4. **Popups**: Show AreaID, zone/sub breakdown, and mapped LK area

---

## Performance Considerations

### Data Volume
- 16×16 chunks per tile = 256 chunks
- All overlays combined = ~1KB JSON per tile
- Lazy load per visible tile (8×8 viewport = ~8KB)

### Rendering Strategy
1. **Layer Groups**: One L.layerGroup per overlay type
2. **Visibility Toggles**: Enable/disable entire layer groups
3. **Debouncing**: 500ms delay on pan/zoom
4. **Aggressive Unloading**: Remove tiles >2 tiles from viewport

---

## Testing Checklist

- [ ] Extract all MCNK flags correctly
- [ ] AreaID extraction matches existing verify CSV
- [ ] CSV validates against expanded schema
- [ ] JSON overlay builds with all categories
- [ ] Terrain properties render correctly
- [ ] Liquid types render with correct colors
- [ ] Holes render at correct positions
- [ ] AreaID boundaries detect correctly
- [ ] Area names display in popups
- [ ] UI toggles work for all overlay types
- [ ] Performance < 100ms per tile (all overlays)
- [ ] Multi-overlay combinations work

---

## Future Enhancements

1. **MCVT Height Map**: Contour lines from height data
2. **MCNR Normals**: Shading based on normals
3. **MCAL Alpha Maps**: Texture blend visualization
4. **MCLY Layers**: Show texture layer assignments
5. **Sound Emitters**: Audio source locations
6. **Doodad Density**: Heatmap of placement counts
