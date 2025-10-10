# Viewer Rebuild Architecture Plan

## Problem Statement

The current viewer is broken due to coordinate transformation hell. We're trying to map:
- WoW world coordinates (floating point, centered at 0,0)
- To Leaflet lat/lng (tile indices 0-63)
- Through multiple broken transformation functions
- With pixel coordinates bolted on as an afterthought

**Result:** Objects render in wrong positions, black markers, random tile placement.

## Root Cause

**We're using Leaflet's internal coordinate system when we shouldn't be.**

The POC viewer worked because it:
1. Converted world coords to pixels in the pipeline
2. Used pixels directly for positioning
3. Had ONE simple function: `pixelToLatLng(row, col, pixelX, pixelY)`

## Solution: Canvas-Based Rendering

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ C# Pipeline (MapMasterIndexWriter)                          │
│                                                              │
│ 1. Read ADT placement (world coords or 0,0,0)              │
│ 2. Calculate tile-local offset (0-533.33 yards)            │
│ 3. Convert to pixel coords (0-512 pixels)                  │
│ 4. Output: { tileX, tileY, pixelX, pixelY, ... }          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Viewer (Canvas Overlay)                                      │
│                                                              │
│ 1. Load placements per tile                                 │
│ 2. For each object:                                         │
│    - canvasX = tileCol * 512 + pixelX                      │
│    - canvasY = tileRow * 512 + pixelY                      │
│ 3. Draw circle/square at (canvasX, canvasY)                │
│                                                              │
│ NO COORDINATE TRANSFORMATIONS                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Principle

**Leaflet is ONLY used for:**
- Tile grid management
- Pan/zoom controls
- Base map rendering

**Leaflet is NEVER used for:**
- Object positioning (use canvas pixels)
- Coordinate transformations (done in pipeline)
- Lat/lng conversions (not needed)

## Implementation Plan

### Phase 1: Fix Pipeline Output ✅

Already done:
- Added `pixelX` and `pixelY` to PlacementRecord
- Calculated from tile-local offsets

### Phase 2: Simplify Viewer

**Remove:**
- ❌ `CoordinateSystem.js` (all transformation functions)
- ❌ `worldToLatLng()` calls
- ❌ Elevation color/radius functions
- ❌ Complex lat/lng calculations

**Keep:**
- ✅ Leaflet map for pan/zoom
- ✅ Plugin system (but simplified)
- ✅ DataAdapter (but simplified)

**Add:**
- ✅ Canvas overlay for object rendering
- ✅ Simple pixel-based positioning

### Phase 3: Canvas Renderer

```javascript
class ObjectRenderer {
    constructor(map) {
        this.map = map;
        this.canvas = L.canvas({ pane: 'overlayPane' });
        this.objects = [];
    }
    
    addObject(obj) {
        // obj: { tileX, tileY, pixelX, pixelY, type, ... }
        this.objects.push(obj);
    }
    
    render() {
        this.objects.forEach(obj => {
            // Direct pixel calculation
            const canvasX = obj.tileX * 512 + obj.pixelX;
            const canvasY = obj.tileY * 512 + obj.pixelY;
            
            // Convert to Leaflet point ONLY for rendering
            const point = this.map.latLngToContainerPoint([
                obj.tileY + obj.pixelY / 512,
                obj.tileX + obj.pixelX / 512
            ]);
            
            // Draw on canvas
            if (obj.type === 'WMO') {
                this.drawSquare(point.x, point.y, '#FF9800');
            } else {
                this.drawCircle(point.x, point.y, '#2196F3');
            }
        });
    }
    
    drawCircle(x, y, color) {
        const ctx = this.canvas._ctx;
        ctx.fillStyle = color;
        ctx.strokeStyle = '#000';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    
    drawSquare(x, y, color) {
        const ctx = this.canvas._ctx;
        ctx.fillStyle = color;
        ctx.strokeStyle = '#000';
        ctx.fillRect(x - 4, y - 4, 8, 8);
        ctx.strokeRect(x - 4, y - 4, 8, 8);
    }
}
```

## Testing Strategy

1. **Verify pixel coords in pipeline output**
   - Check master index JSON has pixelX/pixelY
   - Values should be 0-512

2. **Test canvas rendering**
   - Load single tile
   - Render objects at pixel positions
   - Verify they appear in correct locations

3. **Compare with POC**
   - Same data should render at same positions
   - Orange squares for WMO
   - Blue circles for M2

## Success Criteria

- ✅ Objects render at correct pixel positions
- ✅ No coordinate transformation bugs
- ✅ Proper colors (blue/orange)
- ✅ Popups show correct info
- ✅ Performance: 60fps with 1000+ objects
- ✅ Code is simple and maintainable

## Migration Path

1. Keep current viewer as backup
2. Create new simplified viewer in parallel
3. Test with single map (Kalidar)
4. Once proven, replace old viewer
5. Delete all broken coordinate transformation code

## Files to Create

- `assets/js/core/ObjectRenderer.js` - Canvas-based rendering
- `assets/js/core/SimpleDataAdapter.js` - Load placements only
- `assets/simple-viewer.html` - New clean viewer

## Files to Delete

- `CoordinateSystem.js` - broken transformations
- Complex elevation functions
- Unused overlay code

## Timeline

- Phase 1: ✅ Done (pixel coords in pipeline)
- Phase 2: 2 hours (clean up viewer, remove bad code)
- Phase 3: 3 hours (canvas renderer)
- Phase 4: 1 hour (testing)

**Total: ~6 hours for working viewer**
