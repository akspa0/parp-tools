# Quake 3 BSP Testing Guide

## Quick Testing Commands

### Command Line Testing
```bash
# Original Quake 3 / ioquake3
q3.exe +map test.bsp

# With developer mode
q3.exe +devmap test.bsp +give all

# Modern engines (OpenArena, etc.)
openarena +devmap test
spearmint +devmap test
```

### In-Game Console Testing
1. Launch engine and press `~` for console
2. Type: `devmap test` (without .bsp extension)

### File Directory Structure
```
baseq3/
├── maps/
│   └── test.bsp
├── scripts/
│   └── test.shader
└── textures/
    └── textures/
```

### Verification Commands
```
/devmap 1              # Developer mode
/camera                # Camera toggle
/coord                 # Show coordinates
/lightmapstats         # Lightmap info
/visstats              # Visibility stats
```

### Common Issues & Solutions

**Map doesn't load:**
- Check file permissions
- Verify BSP file is in `maps/` directory
- Check console for error messages

**Textures missing:**
- Ensure textures are in `textures/textures/` directory
- Verify shader scripts are in `scripts/` directory

**Invalid BSP format:**
- Check engine supports Quake 3 BSP version 46
- Verify all lumps are properly aligned

### Engine Compatibility

| Engine | BSP Version | Command | Notes |
|--------|-------------|---------|-------|
| Quake 3 | 46 | `q3.exe +map` | Original |
| ioquake3 | 46 | `ioquake3 +map` | Modern fork |
| OpenArena | 46 | `openarena +devmap` | Based on ioquake3 |
| Spearmint | 46 | `spearmint +devmap` | Modern development |