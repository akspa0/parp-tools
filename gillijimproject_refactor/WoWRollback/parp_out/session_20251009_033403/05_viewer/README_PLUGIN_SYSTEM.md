# Plugin System Implementation - Complete

**Date**: 2025-10-08  
**Status**: ✅ Phase 1 & 2 Complete

---

## Quick Start

### Test the Plugin System
1. Open `test-plugin-system.html` in a browser
2. You should see:
   - 64x64 ADT grid overlay
   - Grid toggle control
   - Test buttons for coordinate system
3. Open browser console to see logs
4. Click "Run Coordinate Tests" to verify transforms

### Use in Main Viewer
1. Open `index.html` in a browser
2. Look for "Show ADT Grid" checkbox in sidebar
3. Toggle to show/hide the grid overlay

---

## What's Implemented

### Core System ✅
```
js/core/
├── CoordinateSystem.js    - All coordinate transforms
├── OverlayPlugin.js       - Base plugin class
└── PluginManager.js       - Plugin lifecycle management
```

### Plugins ✅
```
js/plugins/
├── GridPlugin.js          - 64x64 ADT grid (working)
├── M2Plugin.js            - M2 doodads (ready for data)
└── WMOPlugin.js           - WMO objects (ready for data)
```

### Testing ✅
```
js/tests/
└── CoordinateSystem.test.js  - 8 comprehensive tests
```

### Integration ✅
- `main.js` - Minimal working entry point
- `index.html` - Grid toggle added
- `test-plugin-system.html` - Standalone test page

---

## Architecture

### Plugin Lifecycle
```
Register → Load → Enable → Show → [Active] → Hide → Disable → Destroy
```

### Coordinate System
```javascript
// World coordinates (yards)
worldX, worldY → Tile (row, col)

// Leaflet coordinates
lat, lng ← Tile (row, col)

// Elevation visualization
worldZ → color brightness & marker size
```

### Data Flow
```
WoW Files → C# Exporter → JSON/WebP
                              ↓
                         CoordinateSystem
                              ↓
                         PluginManager
                              ↓
                    [Grid, M2, WMO Plugins]
                              ↓
                         Leaflet Map
```

---

## File Structure

```
WoWRollback.Viewer/assets/
├── js/
│   ├── core/
│   │   ├── CoordinateSystem.js      ✅ Complete
│   │   ├── OverlayPlugin.js         ✅ Complete
│   │   └── PluginManager.js         ✅ Complete
│   ├── plugins/
│   │   ├── GridPlugin.js            ✅ Complete
│   │   ├── M2Plugin.js              ✅ Complete (needs data)
│   │   └── WMOPlugin.js             ✅ Complete (needs data)
│   ├── tests/
│   │   └── CoordinateSystem.test.js ✅ Complete
│   ├── main.js                      ✅ Minimal working version
│   └── state.js                     ✅ Updated for WebP
├── index.html                       ✅ Grid toggle added
├── test-plugin-system.html          ✅ Test page
├── PLUGIN_SYSTEM_STATUS.md          ✅ Status tracking
├── WEBP_MIGRATION.md                ✅ Performance docs
└── README_PLUGIN_SYSTEM.md          ✅ This file
```

---

## How to Use

### Adding a New Plugin

1. **Create plugin file** in `js/plugins/`:
```javascript
import { OverlayPlugin } from '../core/OverlayPlugin.js';

export class MyPlugin extends OverlayPlugin {
    constructor(map, coordSystem) {
        super('myPlugin', 'My Plugin', map, coordSystem);
    }
    
    async loadVisibleData(bounds, zoom) {
        // Load and render data for visible area
    }
}
```

2. **Register in main.js**:
```javascript
import { MyPlugin } from './plugins/MyPlugin.js';

const myPlugin = new MyPlugin(map, coordSystem);
pluginManager.register(myPlugin);
```

3. **Add UI control** in `index.html`:
```html
<label>
    <input type="checkbox" id="myPluginToggle"> My Plugin
</label>
```

4. **Wire up in setupUI()**:
```javascript
document.getElementById('myPluginToggle').addEventListener('change', (e) => {
    const plugin = pluginManager.get('myPlugin');
    if (e.target.checked) {
        plugin.onEnable();
        plugin.onShow();
    } else {
        plugin.onHide();
        plugin.onDisable();
    }
});
```

---

## Testing

### Run Coordinate Tests
```javascript
// In browser console
window.runCoordinateSystemTests()
```

### Test Plugin Lifecycle
```javascript
// In test-plugin-system.html
testPluginLifecycle()
```

### Log Plugin State
```javascript
// In test-plugin-system.html
logPluginState()
```

---

## Performance

### WebP Migration ✅
- All minimap tiles now export as WebP
- **25-35% smaller** file sizes
- **Lower memory usage** in browser
- Ready for 2.5D/3D future

### Lazy Loading
- Plugins only load data for visible tiles
- Tiles unloaded when out of viewport
- Proximity-based loading for M2/WMO

### Optimization Tips
- Keep plugin `loadVisibleData()` fast
- Cache loaded tiles to avoid re-fetching
- Use `clearLayers()` when hiding plugins
- Leverage `CoordinateSystem` for all transforms

---

## Next Steps

### Phase 3: Data Integration (Week 2)
- [ ] Wire M2Plugin to actual JSON data
- [ ] Wire WMOPlugin to actual JSON data
- [ ] Add data export from C# pipeline
- [ ] Test with real placement data

### Phase 4: UI & State (Week 2)
- [ ] Add M2/WMO toggle controls
- [ ] Options modals for plugin settings
- [ ] State persistence (localStorage)
- [ ] Preset views

### Phase 5: Advanced Plugins (Week 3)
- [ ] TerrainPlugin (properties overlay)
- [ ] LiquidsPlugin (water/lava/etc)
- [ ] AreaBoundariesPlugin
- [ ] HolesPlugin

### Phase 6: 3D Integration (Month 2+)
- [ ] Evaluate WebWowViewerCpp integration
- [ ] Or implement 2.5D isometric view
- [ ] Sync 2D/3D views
- [ ] See `09-webwowviewercpp-integration.md`

---

## Success Criteria

- [x] CoordinateSystem works correctly
- [x] Plugin lifecycle functions
- [x] Grid aligns with tiles
- [x] Minimap tiles load
- [x] M2/WMO plugins created
- [x] Test page works
- [x] Grid toggle in main UI
- [ ] M2/WMO plugins show real data
- [ ] UI controls functional
- [ ] State persists across sessions

---

## Resources

### Documentation
- `08-overlay-plugin-system.md` - Original plan
- `09-webwowviewercpp-integration.md` - Future 3D integration
- `PLUGIN_SYSTEM_STATUS.md` - Current status
- `WEBP_MIGRATION.md` - Performance optimizations

### Code
- `js/core/` - Core plugin system
- `js/plugins/` - Plugin implementations
- `js/tests/` - Test suite
- `test-plugin-system.html` - Standalone test page

### External
- Leaflet: https://leafletjs.com/
- WebWowViewerCpp: `lib/WebWowViewerCpp`
- wow.tools: https://wow.tools/mv/

---

## Troubleshooting

### Grid doesn't show
- Check browser console for errors
- Verify `GridPlugin` is registered
- Check `gridToggle` checkbox is checked
- Ensure Leaflet is loaded

### Coordinate tests fail
- Check console for specific failures
- Verify `CoordinateSystem` constants are correct
- Compare with original plan values

### Plugins don't load
- Check `PluginManager` is initialized
- Verify plugins are registered before `loadAll()`
- Check browser console for import errors

---

**Status**: Phase 1 & 2 complete. Ready for data integration and UI polish!
