# Product Context - WoWRollback.RollbackTool

## Why This Exists

WoW map files contain object placements (M2 models and WMOs) tagged with **UniqueIDs**. These IDs increase monotonically as content is added during development. By analyzing UniqueID ranges and selectively "burying" objects above a threshold, we can create historical snapshots of the game world.

**Problem Solved**: Users want to see what WoW maps looked like at different stages of development without manually removing thousands of objects.

## How It Works

### Three-Phase Workflow

#### Phase 1: Analysis
```
WDT File → Scan all ADT tiles → Extract all UniqueIDs → Generate statistics
```
**Output**: JSON with min/max UniqueIDs, percentile thresholds, placement counts

#### Phase 2: Overlay Generation
```
Analysis JSON → Generate PNG overlays per range → Color-code objects → Build manifest
```
**Output**: Pre-rendered minimap overlays showing green (kept) and red (buried) objects for each threshold

#### Phase 3: Rollback
```
WDT File + Threshold + Options → Modify placements → Fix terrain → Write output + MD5
```
**Output**: Modified WDT with objects buried, terrain holes cleared, shadows optionally removed

## User Experience Flow

### Step 1: Point at Map
```powershell
WoWRollback analyze --input World/Maps/Azeroth/Azeroth.wdt
```
**Result**: Shows UniqueID distribution and suggests thresholds

### Step 2: Open Viewer
```
open analysis/azeroth/viewer/index.html
```
**Result**: Visual slider showing what each rollback threshold looks like

### Step 3: Roll Back
```powershell
WoWRollback rollback --input Azeroth.wdt --output rollback/Azeroth.wdt --max-uniqueid 10000 --clear-holes
```
**Result**: Modified map ready to drop into WoW client

## Technical Approach

### Object Burial Strategy
- Modify Z coordinate in MDDF/MODF chunks
- Set Z = -5000.0 (deep underground, never rendered)
- Keeps object data intact, just moves it out of sight

### Terrain Hole Fix
- MCNK chunks have a `Holes` field (16 bits = 4x4 grid of 2x2 areas)
- When a WMO is buried, clear its hole mask
- Prevents "holes to void" where buildings used to be

### Shadow Removal (Optional)
- MCSH chunks contain baked shadow maps
- Can look weird when objects are removed
- Option to zero out MCSH data

## Why This Architecture?

### Pre-generated Overlays (Not On-the-Fly)
- **Performance**: No processing during viewing
- **Simplicity**: Pure HTML+JS, works anywhere
- **Reliability**: Generate once, view forever

### Separate Analysis and Modification
- **Safety**: Never modify original files during analysis
- **Flexibility**: Try different thresholds without re-scanning
- **Transparency**: See what will happen before doing it

### Reuse Existing Infrastructure
- **AlphaWDTAnalysisTool**: Already has complete ADT parsing
- **gillijimproject-csharp**: Proven WoW file format library
- **Don't Reinvent**: Build on what works

## Known Limitations

- Only works on extracted WDT files (not in MPQs... yet)
- Doesn't modify _obj0.adt or _obj1.adt files (terrain objects)
- MCNK spatial calculation assumes flat terrain (good enough for most cases)
- Pre-generation requires disk space (1-2 MB per map)

## Future Enhancements

- MPQ reading/writing support
- Batch processing (all maps at once)
- Diff mode (compare two rollback points)
- 3D preview (Three.js viewer with terrain mesh)
