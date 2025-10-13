# Technical Context - WoWRollback Implementation

## Runtime Environment
- **Target Framework**: .NET 9.0
- **Platform**: Windows x64
- **Build System**: dotnet CLI with MSBuild
- **Dependencies**: 
  - `GillijimProject.WowFiles` (Alpha/LK ADT parsing)
  - `System.Text.Json` (Configuration)

## Project Structure
```
WoWRollback/
├── WoWRollback.Core/           # Shared libraries
│   ├── Models/                 # Data models (PlacementRange, etc.)
│   └── Services/              # Core services
│       ├── AlphaWdtAnalyzer   # Alpha WDT archaeological analysis
│       ├── AdtPlacementAnalyzer # LK ADT preservation analysis  
│       ├── RangeScanner       # Map-level aggregation
│       └── Config/            # Configuration management
├── WoWRollback.Cli/           # Console application
├── memory-bank/               # Project documentation
└── docs/                      # Usage and specifications
```

## Key Dependencies
- **GillijimProject.WowFiles**: Existing C# port of WoW file format parsers
- **Alpha Classes**: `WdtAlpha`, `AdtAlpha`, `McnkAlpha`
- **LK Classes**: `Mhdr`, `Mddf`, `Modf` (for LK ADT analysis)

## Data Flow Architecture
```
Alpha WDT Files → AlphaWdtAnalyzer → Archaeological Ranges → CSV Output
LK ADT Files → AdtPlacementAnalyzer → Preservation Ranges → CSV Output  
Configuration → RangeSelector → Rollback Filtering → Modified ADTs
```

## File Format Considerations

### Alpha WDT Structure
- **WDT File**: Contains tile map and references to individual ADT files
- **ADT Files**: Individual tiles with MCNK chunks containing placement data
- **UniqueID Storage**: Located within chunk structures (format needs verification)

### LK ADT Structure  
- **MHDR**: Header with chunk offsets
- **MDDF**: M2 (doodad) placement data with UniqueIDs at offset +4
- **MODF**: WMO (building) placement data with UniqueIDs at offset +4
- **Entry Sizes**: MDDF=36 bytes, MODF=64 bytes per entry

## Output Formats

### Archaeological CSV Schema
```csv
map,tile_row,tile_col,kind,count,min_unique_id,max_unique_id,file
```

### Session Management
- **Output Root**: `rollback_outputs/` (default)
- **Session Directories**: `session_YYYYMMDD_HHMMSS/`
- **Per-Map Files**: `<session>/<map>/alpha_ranges_by_map_<map>.csv`

## Implementation Gaps

### Alpha Parsing (Critical)
The `AlphaWdtAnalyzer` currently has placeholder implementations:
- Alpha MCNK chunk structure analysis
- UniqueID field locations within Alpha format
- WDT tile presence detection

### Required Research
- Study existing `AlphaWDTAnalysisTool` parsing logic
- Verify Alpha format compatibility with current parsers
- Map Alpha chunk structures to UniqueID locations

## Error Handling Strategy
- **Resilient Analysis**: Continue processing if individual files fail
- **Logging**: Console output with archaeological terminology
- **Skip-if-Missing**: Don't abort on missing optional files

## Performance Considerations
- **Batch Processing**: Handle entire map directories
- **Memory Management**: Process files individually, not all in memory
- **Parallel Processing**: Future enhancement for large datasets

## Integration Points
- **AlphaWDTAnalysisTool**: Reference existing parsing logic
- **GillijimProject.WowFiles**: Leverage existing format parsers
- **Configuration System**: JSON/YAML for rollback rules

## MPQ Archive Infrastructure (Added 2025-10-12)

### Existing Infrastructure
Located in `lib/WoWTools.Minimaps/StormLibWrapper/`:
- **MpqArchive.cs** - C# wrapper for StormLib (open/read MPQ files)
- **MPQReader.cs** - Extract files from MPQ archives
- **DirectoryReader.cs** - Auto-detect and sort patch chain
- **MpqArchive.AddPatchArchives()** - Automatic patch application

### WoW File Resolution Priority (CRITICAL)
WoW reads files in this exact order:
1. **Loose files in Data/ subfolders** (HIGHEST priority)
2. **Patch MPQs** (patch-3.MPQ > patch-2.MPQ > patch.MPQ)
3. **Base MPQs** (lowest priority)

**Why This Matters:**
- Players exploited loose file overrides for model swapping (giant campfire = escape geometry)
- `md5translate.txt` can exist in BOTH MPQ and `Data/textures/Minimap/md5translate.txt`
- **Any archive reader MUST check filesystem BEFORE MPQ**

### Implementation Requirements
```csharp
// Required abstraction (not yet implemented)
interface IArchiveSource {
    bool FileExists(string path);
    Stream OpenFile(string path);
}

// Priority wrapper (MUST implement)
class PrioritizedArchiveSource : IArchiveSource {
    // 1. Check loose file in Data/ folder FIRST
    // 2. If not found, delegate to MpqArchive
}
```

### Patch Chain Handling
- `DirectoryReader.cs` automatically detects and sorts patch MPQs
- Higher-numbered patches override lower (patch-3 > patch-2 > patch-1 > base)
- `MpqArchive.AddPatchArchives()` applies patch chain automatically

### Current Gap
- StormLibWrapper exists but not integrated with WoWRollback
- No loose file priority layer implemented
- No IArchiveSource abstraction

## WDT and Map Type Detection (Added 2025-10-12)

### WDT File Structure
```
WDT File:
├── MPHD chunk (header with flags)
├── MAIN chunk (64x64 tile grid, flags indicate which ADTs exist)
└── MWMO chunk (WMO filename for WMO-only maps)
```

### Map Types
1. **ADT-based**: Normal terrain (Azeroth, Kalimdor, Outland)
2. **WMO-only**: Instances with single WMO (Karazhan, Scarlet Monastery, Deadmines)
3. **Battlegrounds**: Special handling (Alterac Valley, Warsong Gulch)

### Detection Logic
- **MPHD.GlobalWMO flag** - Indicates WMO-only map (no ADTs)
- **MAIN grid entries** - Each has flags indicating if ADT exists
- **MWMO chunk** - Contains WMO path for instances

**Benefit:** Prevents scanning for non-existent ADT files, handles instances correctly

### Current Gap
- `analyze-map-adts` assumes all ADTs exist
- No WDT pre-check before ADT processing
- Can't handle Karazhan and other WMO-only maps

## MCNK Terrain Data (Added 2025-10-12)

### MCNK Structure
Each ADT has 256 MCNK chunks (16x16 grid). Each chunk has subchunks:
- **MCVT** - 145 vertex heights (enables height map overlays)
- **MCNR** - Normal vectors (lighting/shading)
- **MCLY** - Texture layers (up to 4 per chunk with blend modes)
- **MCAL** - Alpha maps (texture blending)
- **MCLQ** - Liquid data (water/lava/slime types, heights, flags)
- **MCRF** - Doodad/WMO references in this chunk
- **MCSH** - Shadow map (baked shadows)
- **MCSE** - Sound emitters (ambient sound)

### Current Implementation
`AdtTerrainExtractor.cs` only extracts basic MCNK header:
- AreaID, Flags, TextureLayers count, HasLiquids, HasHoles, IsImpassible

**Missing:** All subchunk data (MCVT, MCNR, MCLY, MCAL, MCLQ, MCRF, MCSH, MCSE)

### Planned Enhancement
New module `WoWRollback.DetailedAnalysisModule` will extract full MCNK data for:
- Height map overlays (MCVT heatmaps)
- Texture distribution overlays (MCLY)
- Liquid region overlays (MCLQ)
- Impassable terrain overlays (MCNK flags)
- Area boundary overlays (AreaID changes)

## Archaeological Terminology Integration
- **Console Output**: Uses archaeological language ("excavation", "preservation", "artifacts")
- **Variable Naming**: "sedimentary layers", "volumes of work", "ancient developers"
- **Documentation**: Consistently applies archaeological metaphor
