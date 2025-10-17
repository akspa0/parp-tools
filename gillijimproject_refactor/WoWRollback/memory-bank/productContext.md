# Product Context

## Why This Project Exists

### The Problem
World of Warcraft's early Alpha builds (0.5.3-0.6.0) contain unique map content and design decisions that were later modified or removed. This content is:
- **Trapped in obsolete formats**: Alpha ADT/WDT files use different structures than modern formats
- **Difficult to analyze**: No modern tools support Alpha format directly
- **At risk of loss**: Alpha clients and data are increasingly rare
- **Mixed with later additions**: Hard to distinguish original content from later iterations

### The Solution
WoWRollback provides:
1. **Direct Alpha analysis**: Parse Alpha ADT files without conversion
2. **Bidirectional conversion**: Move between Alpha ↔ Lich King formats
3. **Surgical object control**: Selectively preserve/remove objects by UniqueID ranges
4. **Modern visualization**: Interactive web viewer for exploring map content

## How It Works

### Analysis Pipeline
```
Alpha ADT Files → Parser → Extract Placements → Cluster Analysis → CSV/JSON Output → Web Viewer
```

### Conversion Pipeline
```
Alpha WDT/ADT → Reader → Converter → LK ADT Writer → Lich King Format
LK ADT (3 files) → Reader → Converter → Alpha ADT Writer → Alpha Format (monolithic)
```

### Patching Pipeline
```
User Selection (UI) → Rollback Config → ADT Patcher → Replace Model Paths → Patched Files
```

## User Experience Goals

### For Analysts
- **Fast exploration**: Analyze loose ADT files without full conversion
- **Rich data**: Extract 26K+ object placements with spatial clustering
- **Visual context**: See objects overlaid on minimaps in web viewer
- **Export options**: CSV, JSON, GLB (3D meshes)

### For Converters
- **One-command conversion**: Simple CLI for Alpha ↔ LK conversion
- **Validation**: Built-in checks for format parity and data integrity
- **Transparency**: Detailed logging of what was converted/skipped
- **Reversibility**: Round-trip conversions preserve original data

### For Patchers
- **Granular control**: Per-tile or map-wide object selection
- **Visual selection**: Interactive UI for choosing UniqueID ranges
- **Safe patching**: Always preserve originals, validate outputs
- **Client testing**: Patched files load in original game clients

## Key Workflows

### Workflow 1: Analyze Loose ADTs
```powershell
# Analyze Alpha ADT files directly
dotnet run --project WoWRollback.Cli -- analyze-map-adts \
  --map development \
  --map-dir "test_data/development/World/Maps/development/" \
  --out "analysis_output"

# Start web viewer
dotnet run --project WoWRollback.Cli -- serve-viewer
```

**Result**: Interactive viewer with 26K+ placements, spatial clusters, 3D terrain

### Workflow 2: Alpha → LK Conversion
```powershell
# Convert Alpha WDT to LK format
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic \
  --lk-dir "path/to/lk/map" \
  --lk-wdt "path/to/map.wdt" \
  --map Azeroth \
  --out "project_output/"
```

**Result**: Monolithic Alpha WDT with embedded terrain-only ADTs

### Workflow 3: LK → Alpha Conversion
```powershell
# Convert LK ADT back to Alpha
dotnet run --project WoWRollback.AdtConverter -- convert-map-terrain \
  --lk-dir "path/to/lk/map" \
  --out "output/alpha_map" \
  --format alpha
```

**Result**: Alpha-format ADTs loadable in 0.5.3 client

### Workflow 4: Selective Rollback (Future)
```
1. Generate UniqueID ranges (per-tile or map-wide)
2. Open web viewer, select ranges to preserve
3. Export rollback configuration (JSON)
4. Apply configuration to patch ADT files
5. Test patched files in Alpha client
```

**Result**: Custom map with only selected objects visible

## Problems We Solve

### Problem 1: Format Incompatibility
**Before**: Alpha ADTs can't be opened in modern tools (Noggit, WoW Model Viewer)  
**After**: Convert to LK format, use modern tools, convert back to Alpha

### Problem 2: Mixed Content Layers
**Before**: Can't distinguish Alpha-era objects from later additions  
**After**: UniqueID range analysis reveals sedimentary layers of development

### Problem 3: No Visualization
**Before**: Must load Alpha client to see map content  
**After**: Web viewer shows placements, clusters, and 3D terrain instantly

### Problem 4: All-or-Nothing Preservation
**Before**: Keep entire map or lose it all  
**After**: Selectively preserve specific object ranges, create custom rollbacks

## Design Principles

### 1. Preservation First
- Never modify original files
- Always work on copies
- Validate before and after operations
- Log all transformations

### 2. Format Fidelity
- Preserve chunk structures exactly
- Maintain byte-level compatibility
- Respect client expectations
- Document all assumptions

### 3. Transparency
- Detailed logging of all operations
- Clear error messages
- Validation reports
- Diff-able outputs

### 4. Usability
- Simple CLI commands
- Interactive web viewer
- One-command workflows
- Sensible defaults

## Success Stories

### Story 1: Terrain Parity Achieved
- MCAL/MCLY bidirectional conversion working
- Alpha → LK → Alpha round-trip preserves texture layers
- Noggit-compatible alpha map decoding implemented

### Story 2: Viewer Functionality
- 26K+ object placements visualized
- Spatial clustering reduces visual clutter
- 3D terrain meshes (GLB) generated
- Built-in HTTP server (no Python needed)

### Story 3: Format Conversion Pipeline
- Alpha WDT → LK ADT conversion working
- LK ADT → Alpha ADT reverse conversion implemented
- Monolithic WDT packing functional
- Validation and comparison tools operational
