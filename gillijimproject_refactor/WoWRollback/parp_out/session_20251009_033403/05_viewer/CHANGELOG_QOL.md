# Viewer Quality of Life Improvements

## Date: 2025-10-03

### Issues Fixed

1. **Map Switching Broken**
   - **Problem**: Tiles from previous maps persisted when switching maps, creating visual mish-mash
   - **Fix**: Added map change detection in `state.js` and `main.js` to clear tile cache on map switch

2. **Inefficient Caching**
   - **Problem**: Every overlay load had cache-busting timestamp, preventing actual caching
   - **Fix**: Removed per-load cache-busting in `overlayLoader.js`, only clear cache on version/map change

3. **Popup Closes on Pan**
   - **Problem**: Default Leaflet behavior closed popups when panning map
   - **Fix**: Added persistent popup support with `autoClose: false` and `closeOnClick: false` options

4. **Resource-Intensive Object Marker Loading**
   - **Problem**: Full overlay reload on every pan/zoom with no debouncing
   - **Fix**: Added 500ms debounce to `updateObjectMarkers()` to batch rapid viewport changes

5. **Tile Layer Memory Leak**
   - **Problem**: Tiles loaded but never aggressively unloaded when out of viewport
   - **Fix**: Added aggressive unloading of tiles >2 tiles away from viewport

6. **Small, Hard-to-Read Popups**
   - **Problem**: Popups were cramped and difficult to read
   - **Fix**: Increased popup size (min-width 280px, max-width 350px) with better formatting

### Files Modified

#### `state.js`
- Added `lastVersion` and `lastMap` tracking
- Updated `setVersion()` and `setMap()` to track previous values
- Added `cacheBust` on map change (not just version change)

#### `overlayLoader.js`
- Removed per-call cache-busting with `Date.now()`
- Simplified caching logic to use paths directly
- Cache now cleared only on version/map change

#### `main.js`
- Added `lastMap`, `currentPopup`, and `pendingOverlayLoad` tracking variables
- Enhanced `onStateChange()` to detect both version AND map changes
- Implemented 500ms debouncing for overlay loading via `updateObjectMarkers()`
- Split overlay loading into debounced wrapper + actual update function
- Added aggressive tile unloading (>2 tiles from viewport) in `refreshMinimapTiles()`
- Enhanced popup HTML with better formatting and more information
- Added persistent popup support with `autoClose: false` options
- Added popup close event tracking
- Added click handlers to track current open popup

#### `styles.css`
- Added `.persistent-popup` styling for dark theme consistency
- Styled close button with hover effects
- Improved popup visual appearance

### Performance Improvements

1. **Reduced Network Requests**: Smart caching eliminates redundant overlay fetches
2. **Lower Memory Usage**: Aggressive tile unloading frees memory for distant tiles
3. **Smoother Panning**: Debouncing prevents overlay thrashing during navigation
4. **Better UX**: Persistent popups allow exploration without losing context

### Usage Notes

- **Popups** now stay open during map pan/zoom until explicitly closed via X button or clicking another object
- **Overlay loading** is debounced 500ms - rapid panning won't trigger excessive loads
- **Tile unloading** happens automatically for tiles >2 tiles outside viewport
- **Map switching** now properly clears all cached data and tiles
