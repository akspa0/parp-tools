# Project Brief

## Vision
WoWRollback is a digital archaeology toolkit for World of Warcraft map content, enabling analysis, conversion, and selective preservation of objects across different game versions (Alpha 0.5.x through Lich King 3.3.5a and beyond).

## Core Mission
Provide surgical control over WoW map content by:
1. **Analyzing** loose ADT files without conversion
2. **Converting** between Alpha ↔ Lich King formats bidirectionally
3. **Patching** maps to selectively preserve/remove objects by UniqueID ranges
4. **Visualizing** terrain and object placements in an interactive web viewer

## Primary Use Cases

### 1. Digital Archaeology
- Extract and analyze object placements from Alpha builds (0.5.3-0.6.0)
- Identify design patterns, prefab usage, and development layers
- Track asset evolution across game versions

### 2. Format Conversion
- **Alpha → LK**: Convert Alpha WDT/ADT to Lich King format for modern tooling
- **LK → Alpha**: Convert Lich King ADT back to Alpha for authentic client testing
- Preserve terrain geometry, textures, and model placements across conversions

### 3. Selective Rollback
- Generate UniqueID ranges per tile or map-wide
- UI for selecting which object ranges to preserve
- Patch ADT/WDT files to replace unwanted objects with invisible models
- Create custom "rolled back" versions preserving only selected content

### 4. Visualization & Analysis
- Interactive web viewer with tile-based navigation
- 3D terrain mesh extraction (GLB format)
- Spatial clustering and pattern detection
- Cross-tile duplicate filtering

## Key Constraints
- **Bidirectional parity**: Alpha ↔ LK conversions must be reversible with minimal data loss
- **Format fidelity**: Preserve chunk structures, offsets, and file integrity
- **Client compatibility**: Converted files must load in original game clients
- **Non-destructive**: Always preserve original files, work on copies

## Success Metrics
- ✅ Converted ADTs load in target client without crashes
- ✅ Terrain geometry matches source (visually identical)
- ✅ MCAL/MCLY texture layers preserve parity across conversions
- ✅ Object placements maintain correct positions and rotations
- ✅ Patched files replace objects with invisible models successfully
- ✅ Viewer displays 26K+ object placements with spatial clustering

## Non-Goals
- Not a map editor (use Noggit for that)
- Not a model converter (M2/WMO conversion out of scope)
- Not a texture converter (BLP handling minimal)
- Not a gameplay simulator (client-side only)

## Target Users
- WoW development historians and researchers
- Private server developers needing format conversion
- Map designers studying early WoW design patterns
- Community members preserving Alpha content
