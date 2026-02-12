# Testing the Plugin System

## Quick Start

### 1. Start the Test Server

From the `WoWRollback` directory:

```powershell
# Start server on default port 8081
.\serve-plugin-test.ps1

# Start server and open browser automatically
.\serve-plugin-test.ps1 -OpenBrowser

# Use a different port
.\serve-plugin-test.ps1 -Port 8080
```

### 2. Open Test Pages

Once the server is running:

**Plugin System Test:**
- URL: http://localhost:8081/test-plugin-system.html
- Features:
  - Grid overlay visualization
  - Coordinate system tests
  - Plugin lifecycle tests
  - State inspection

**Main Viewer:**
- URL: http://localhost:8081/index.html
- Features:
  - Full viewer with grid toggle
  - Minimap tiles (if data available)
  - All existing viewer features

### 3. Run Tests

In the browser console:

```javascript
// Run coordinate system tests
window.runCoordinateSystemTests()

// Test plugin lifecycle
testPluginLifecycle()

// Log current plugin state
logPluginState()
```

## What to Look For

### ✅ Success Indicators

1. **Grid Displays**
   - 64x64 grid overlay visible on map
   - Grid lines are straight and aligned
   - Grid covers entire 64x64 tile space

2. **Coordinate Tests Pass**
   - Click "Run Coordinate Tests" button
   - All 8 tests should pass
   - Check console for detailed results

3. **Plugin Toggle Works**
   - Check/uncheck "Show ADT Grid"
   - Grid should show/hide smoothly
   - No console errors

4. **Console Logs Clean**
   - Look for initialization messages
   - No red error messages
   - Plugin registration confirmed

### ❌ Common Issues

**Grid doesn't appear:**
- Check browser console for errors
- Verify Leaflet loaded (check Network tab)
- Ensure JavaScript modules are enabled

**Module import errors:**
- Make sure you're using a web server (not file://)
- Check that all .js files exist in correct locations
- Verify import paths are correct

**Coordinate tests fail:**
- Check console for specific test failures
- Verify CoordinateSystem constants
- Compare with expected values in test file

## Browser Console Commands

### Inspect Plugin System

```javascript
// Access global objects
map              // Leaflet map instance
coordSystem      // CoordinateSystem instance
pluginManager    // PluginManager instance
gridPlugin       // GridPlugin instance

// Get plugin info
gridPlugin.getConfig()
pluginManager.plugins

// Test coordinates
coordSystem.worldToTile(0, 0)        // Should be {row: 32, col: 32}
coordSystem.tileToWorld(32, 32)      // Should be ~{worldX: 0, worldY: 0}
coordSystem.tileToLatLng(30, 35)     // Convert to Leaflet coords
```

### Manual Plugin Control

```javascript
// Enable/disable grid
gridPlugin.onEnable()
gridPlugin.onShow()
gridPlugin.onHide()
gridPlugin.onDisable()

// Change grid appearance
gridPlugin.setGridColor('#FF0000')   // Red grid
gridPlugin.setGridWeight(2)          // Thicker lines
gridPlugin.toggleTileLabels(false)   // Hide labels
```

## Test Checklist

### Phase 1: Basic Functionality
- [ ] Server starts without errors
- [ ] Test page loads in browser
- [ ] Grid overlay displays
- [ ] Grid toggle works
- [ ] No console errors

### Phase 2: Coordinate System
- [ ] Run coordinate tests
- [ ] All 8 tests pass
- [ ] World ↔ Tile conversions correct
- [ ] Tile ↔ Lat/Lng conversions correct
- [ ] Elevation helpers work

### Phase 3: Plugin Lifecycle
- [ ] Plugin registers successfully
- [ ] Enable/disable works
- [ ] Show/hide works
- [ ] State persists (localStorage)
- [ ] Cleanup on destroy

### Phase 4: Integration
- [ ] Works in main viewer (index.html)
- [ ] Grid toggle in sidebar works
- [ ] No conflicts with existing code
- [ ] Performance is acceptable

## Performance Metrics

### Expected Performance
- **Initial load**: < 1 second
- **Grid render**: < 100ms
- **Toggle response**: Instant
- **Memory usage**: < 50MB

### Check Performance

```javascript
// Measure grid render time
console.time('grid-render');
gridPlugin.renderGrid();
console.timeEnd('grid-render');

// Check memory (Chrome DevTools)
// Performance → Memory → Take snapshot
```

## Debugging Tips

### Enable Verbose Logging

```javascript
// In browser console
localStorage.setItem('debug', 'true');
location.reload();
```

### Check Module Loading

```javascript
// Verify modules loaded
console.log('CoordinateSystem:', typeof CoordinateSystem);
console.log('PluginManager:', typeof PluginManager);
console.log('GridPlugin:', typeof GridPlugin);
```

### Inspect Leaflet Layers

```javascript
// List all layers
map.eachLayer(layer => console.log(layer));

// Count layers
let count = 0;
map.eachLayer(() => count++);
console.log('Total layers:', count);
```

## Next Steps

After successful testing:

1. **Wire M2/WMO plugins** to real data
2. **Add more UI controls** for plugin options
3. **Implement state persistence**
4. **Build additional plugins** (Terrain, Liquids, etc.)
5. **Optimize performance** for large datasets

## Troubleshooting

### Server won't start
- Check if port is already in use
- Try a different port: `.\serve-plugin-test.ps1 -Port 8082`
- Verify Python is installed: `python --version`

### Browser shows blank page
- Check browser console for errors
- Verify server is running
- Try hard refresh: Ctrl+Shift+R

### Modules won't load
- Ensure using http:// not file://
- Check Network tab for 404 errors
- Verify file paths are correct

### Grid appears but is misaligned
- Check CoordinateSystem constants
- Verify tileBounds() calculation
- Compare with original plan values

---

**Ready to test!** Run `.\serve-plugin-test.ps1 -OpenBrowser` and start exploring! 🚀
