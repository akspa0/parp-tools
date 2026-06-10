# Viewer UX Fixes - Phase 3 Enhancement

**Date**: 2025-10-05  
**Status**: Planning  
**Goal**: Fix critical UX issues discovered during Phase 2 validation

---

## Issues Discovered

### 1. **M2 Marker Overlap at Auto-Zoom** 🔴 HIGH PRIORITY
**Symptom**: When zooming out, M2 markers stack on top of each other, making it impossible to select individual objects.

**Root Cause**: 
- Many M2 models placed at same/similar coordinates
- No marker clustering implemented
- Markers use fixed size regardless of zoom level
- No collision detection between markers

**Impact**: 
- User discovered many M2s in 0.5.5 they didn't know existed
- Cannot interact with overlapping markers
- Frustrating exploration experience

---

### 2. **Glitchy Popup Bubbles** 🔴 HIGH PRIORITY
**Symptom**: 
- Popups glitch when multiple open simultaneously
- Clicking on multiple markers causes UI instability
- Popups overlap each other
- No awareness of other open popups

**Root Cause**:
- Leaflet default popup behavior (single popup mode)
- No popup collision avoidance
- No popup positioning logic
- Simultaneous popup animations conflict

**Impact**:
- Cannot explore multiple objects at once
- UI feels janky and unpredictable
- Discourages multi-object comparison

---

### 3. **Non-Functional "Sedimentary Layers"** 🟡 MEDIUM PRIORITY
**Symptom**: 
- UI panel exists in sidebar
- "Filter by UniqueID range..." search box present
- No functionality implemented
- Never worked

**Root Cause**:
- HTML/CSS exists
- No JavaScript implementation
- No UniqueID filtering logic
- Panel is decorative only

**Impact**:
- Cannot isolate objects by UniqueID
- Cannot explore object placement patterns
- Wasted UI real estate

---

## Proposed Solutions

### Solution 1: Marker Clustering 🎯

**Approach**: Implement smart marker clustering using Leaflet.markercluster

**Features**:
1. **Auto-cluster** when markers overlap
2. **Cluster icons** show count of contained markers
3. **Zoom to cluster** expands on click
4. **Spiderfy mode** spreads markers in circle when zoomed
5. **Dynamic sizing** based on zoom level

**Implementation**:
```javascript
// Add Leaflet.markercluster plugin
import MarkerClusterGroup from 'leaflet.markercluster';

const m2Clusters = L.markerClusterGroup({
    maxClusterRadius: 40, // px - adjust based on zoom
    spiderfyOnMaxZoom: true,
    showCoverageOnHover: false,
    zoomToBoundsOnClick: true,
    
    // Custom cluster icon
    iconCreateFunction: function(cluster) {
        const count = cluster.getChildCount();
        let className = 'm2-cluster-small';
        if (count > 50) className = 'm2-cluster-large';
        else if (count > 10) className = 'm2-cluster-medium';
        
        return L.divIcon({
            html: `<div><span>${count}</span></div>`,
            className: `marker-cluster ${className}`,
            iconSize: L.point(40, 40)
        });
    }
});

// Add markers to cluster group instead of map directly
m2Clusters.addLayer(marker);
map.addLayer(m2Clusters);
```

**CSS**:
```css
.marker-cluster {
    background-clip: padding-box;
    border-radius: 50%;
    text-align: center;
    font-weight: bold;
}

.m2-cluster-small {
    background-color: rgba(181, 226, 140, 0.6);
    width: 30px;
    height: 30px;
}

.m2-cluster-medium {
    background-color: rgba(241, 211, 87, 0.6);
    width: 40px;
    height: 40px;
}

.m2-cluster-large {
    background-color: rgba(253, 156, 115, 0.6);
    width: 50px;
    height: 50px;
}
```

**Benefits**:
- ✅ Solves overlap problem
- ✅ Shows count of objects
- ✅ Smooth zoom-to-explore UX
- ✅ Performance improvement (fewer DOM elements)

---

### Solution 2: Smart Popup Management 🎯

**Approach**: Implement multi-popup mode with collision avoidance

**Features**:
1. **Multiple popups** can be open simultaneously
2. **Collision detection** moves popups to avoid overlap
3. **Z-index stacking** brings clicked popup to front
4. **Close button** on each popup for manual management
5. **Max popup limit** (e.g., 5 simultaneous)

**Implementation**:

```javascript
class SmartPopupManager {
    constructor(map) {
        this.map = map;
        this.openPopups = new Set();
        this.maxPopups = 5;
    }
    
    openPopup(marker, content) {
        // Close oldest if at max
        if (this.openPopups.size >= this.maxPopups) {
            const oldest = this.openPopups.values().next().value;
            this.closePopup(oldest);
        }
        
        // Create custom popup
        const popup = L.popup({
            closeButton: true,
            autoClose: false, // KEY: Don't auto-close
            closeOnClick: false, // KEY: Keep open on map click
            maxWidth: 300,
            className: 'smart-popup'
        })
        .setContent(this.buildPopupContent(content))
        .setLatLng(marker.getLatLng());
        
        marker.bindPopup(popup);
        marker.openPopup();
        
        this.openPopups.add(popup);
        
        // Position adjustment for collision avoidance
        this.adjustPopupPosition(popup);
        
        // Bring to front on click
        popup.on('click', () => this.bringToFront(popup));
        popup.on('remove', () => this.openPopups.delete(popup));
    }
    
    adjustPopupPosition(popup) {
        // Get popup DOM element
        const popupEl = popup.getElement();
        if (!popupEl) return;
        
        const rect = popupEl.getBoundingClientRect();
        let adjusted = false;
        
        // Check collision with other popups
        for (const other of this.openPopups) {
            if (other === popup) continue;
            
            const otherEl = other.getElement();
            if (!otherEl) continue;
            
            const otherRect = otherEl.getBoundingClientRect();
            
            if (this.rectsOverlap(rect, otherRect)) {
                // Move popup down and right
                popup.options.offset = [20, 40];
                popup.update();
                adjusted = true;
                break;
            }
        }
        
        // Recursively adjust if still colliding
        if (adjusted) {
            setTimeout(() => this.adjustPopupPosition(popup), 100);
        }
    }
    
    rectsOverlap(r1, r2) {
        return !(r1.right < r2.left || 
                r1.left > r2.right || 
                r1.bottom < r2.top || 
                r1.top > r2.bottom);
    }
    
    bringToFront(popup) {
        const el = popup.getElement();
        if (!el) return;
        
        // Reset z-index of all popups
        for (const other of this.openPopups) {
            const otherEl = other.getElement();
            if (otherEl) otherEl.style.zIndex = 1000;
        }
        
        // Bring clicked popup to front
        el.style.zIndex = 1001;
    }
    
    closePopup(popup) {
        popup.remove();
        this.openPopups.delete(popup);
    }
    
    closeAll() {
        for (const popup of this.openPopups) {
            popup.remove();
        }
        this.openPopups.clear();
    }
}

// Usage
const popupManager = new SmartPopupManager(map);

marker.on('click', () => {
    popupManager.openPopup(marker, {
        name: 'Model Name',
        uniqueId: 12345,
        type: 'M2'
    });
});
```

**CSS**:
```css
.smart-popup {
    transition: all 0.2s ease;
}

.smart-popup .leaflet-popup-content-wrapper {
    border: 2px solid #333;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.smart-popup.front {
    z-index: 1001 !important;
}
```

**Benefits**:
- ✅ Multiple popups work smoothly
- ✅ No overlap or glitchiness
- ✅ Better exploration UX
- ✅ Clear visual hierarchy

---

### Solution 3: Functional Sedimentary Layers 🎯

**Approach**: Implement UniqueID filtering and layer isolation

**Features**:
1. **Auto-populate** UniqueID list from loaded objects
2. **Range filtering** (e.g., "1000-2000")
3. **Individual selection** (checkboxes per UniqueID)
4. **Highlight mode** (dim non-selected objects)
5. **Exclusive mode** (hide non-selected objects)

**Implementation**:

```javascript
class SedimentaryLayersManager {
    constructor(map, objectManager) {
        this.map = map;
        this.objectManager = objectManager;
        this.uniqueIds = new Map(); // UniqueID -> [markers]
        this.activeFilters = new Set();
        this.mode = 'highlight'; // 'highlight' or 'exclusive'
        
        this.initUI();
    }
    
    initUI() {
        const panel = document.getElementById('layersPanel');
        const searchInput = document.getElementById('layersSearch');
        const listContainer = document.getElementById('layersList');
        
        // Populate UniqueID list
        this.populateList(listContainer);
        
        // Search filtering
        searchInput.addEventListener('input', (e) => {
            this.filterList(e.target.value);
        });
        
        // Mode toggle
        const modeToggle = this.createModeToggle();
        panel.querySelector('.layers-header').appendChild(modeToggle);
    }
    
    registerMarker(marker, uniqueId) {
        if (!this.uniqueIds.has(uniqueId)) {
            this.uniqueIds.set(uniqueId, []);
        }
        this.uniqueIds.get(uniqueId).push(marker);
    }
    
    populateList(container) {
        container.innerHTML = '';
        
        // Sort UniqueIDs numerically
        const sortedIds = Array.from(this.uniqueIds.keys()).sort((a, b) => a - b);
        
        for (const uniqueId of sortedIds) {
            const markers = this.uniqueIds.get(uniqueId);
            const item = this.createLayerItem(uniqueId, markers.length);
            container.appendChild(item);
        }
    }
    
    createLayerItem(uniqueId, count) {
        const div = document.createElement('div');
        div.className = 'layer-item';
        div.dataset.uniqueId = uniqueId;
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `layer-${uniqueId}`;
        checkbox.checked = true;
        
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                this.activeFilters.delete(uniqueId);
            } else {
                this.activeFilters.add(uniqueId);
            }
            this.applyFilters();
        });
        
        const label = document.createElement('label');
        label.htmlFor = `layer-${uniqueId}`;
        label.innerHTML = `
            <span class="layer-id">UniqueID ${uniqueId}</span>
            <span class="layer-count">(${count})</span>
        `;
        
        div.appendChild(checkbox);
        div.appendChild(label);
        
        return div;
    }
    
    createModeToggle() {
        const select = document.createElement('select');
        select.id = 'layerMode';
        select.innerHTML = `
            <option value="highlight">Highlight</option>
            <option value="exclusive">Exclusive</option>
        `;
        
        select.addEventListener('change', (e) => {
            this.mode = e.target.value;
            this.applyFilters();
        });
        
        return select;
    }
    
    applyFilters() {
        if (this.activeFilters.size === 0) {
            // No filters: show all normally
            this.showAll();
            return;
        }
        
        for (const [uniqueId, markers] of this.uniqueIds.entries()) {
            const isFiltered = this.activeFilters.has(uniqueId);
            
            for (const marker of markers) {
                if (this.mode === 'exclusive') {
                    // Exclusive mode: hide filtered objects
                    if (isFiltered) {
                        marker.setOpacity(0);
                        marker.options.interactive = false;
                    } else {
                        marker.setOpacity(1);
                        marker.options.interactive = true;
                    }
                } else {
                    // Highlight mode: dim filtered objects
                    if (isFiltered) {
                        marker.setOpacity(0.2);
                    } else {
                        marker.setOpacity(1);
                    }
                }
            }
        }
    }
    
    showAll() {
        for (const markers of this.uniqueIds.values()) {
            for (const marker of markers) {
                marker.setOpacity(1);
                marker.options.interactive = true;
            }
        }
    }
    
    filterList(query) {
        const items = document.querySelectorAll('.layer-item');
        
        if (!query) {
            items.forEach(item => item.style.display = '');
            return;
        }
        
        // Support range syntax: "1000-2000"
        const rangeMatch = query.match(/^(\d+)-(\d+)$/);
        
        if (rangeMatch) {
            const [, min, max] = rangeMatch.map(Number);
            items.forEach(item => {
                const id = parseInt(item.dataset.uniqueId);
                item.style.display = (id >= min && id <= max) ? '' : 'none';
            });
        } else {
            // Simple substring match
            items.forEach(item => {
                const id = item.dataset.uniqueId;
                item.style.display = id.includes(query) ? '' : 'none';
            });
        }
    }
}

// Usage
const layersManager = new SedimentaryLayersManager(map, objectManager);

// Register each marker
layersManager.registerMarker(marker, uniqueId);
```

**CSS**:
```css
.layers-panel {
    width: 300px;
    max-height: 400px;
    overflow-y: auto;
}

.layer-item {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    border-bottom: 1px solid #ddd;
}

.layer-item:hover {
    background-color: #f5f5f5;
}

.layer-id {
    font-weight: bold;
    margin-left: 8px;
}

.layer-count {
    color: #666;
    margin-left: 4px;
    font-size: 0.9em;
}

.layers-filter {
    padding: 8px;
    border-bottom: 2px solid #333;
}

.layers-filter input {
    width: 100%;
    padding: 4px 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
}
```

**Benefits**:
- ✅ Finally functional!
- ✅ Explore by UniqueID
- ✅ Discover object placement patterns
- ✅ Isolate specific object groups

---

## Implementation Plan

### Phase 3A: Marker Clustering (Week 1)
1. Add Leaflet.markercluster dependency
2. Refactor M2 marker creation to use cluster groups
3. Style cluster icons
4. Test with dense M2 areas (0.5.5 maps)
5. Adjust cluster radius based on zoom

### Phase 3B: Popup Management (Week 1)
1. Create `SmartPopupManager` class
2. Implement collision detection
3. Add z-index management
4. Test multi-popup scenarios
5. Polish animations

### Phase 3C: Sedimentary Layers (Week 2)
1. Create `SedimentaryLayersManager` class
2. Populate UniqueID list from objects
3. Implement range filtering
4. Add highlight/exclusive modes
5. Test with large object counts

### Phase 3D: Integration & Testing (Week 2)
1. Integrate all three systems
2. Test together
3. Performance profiling
4. User testing
5. Documentation

---

## Success Criteria

### Marker Clustering
- [ ] No overlapping markers at any zoom level
- [ ] Clusters show correct counts
- [ ] Spiderfy works smoothly
- [ ] Performance: handles 1000+ markers

### Popup Management
- [ ] 5 popups can be open simultaneously
- [ ] No glitches or flicker
- [ ] Collision avoidance works
- [ ] Z-index stacking correct

### Sedimentary Layers
- [ ] All UniqueIDs listed correctly
- [ ] Range filtering works (e.g., "1000-2000")
- [ ] Highlight mode dims correctly
- [ ] Exclusive mode hides correctly
- [ ] Search is fast (<100ms)

---

## Future Enhancements

1. **Marker Styles by UniqueID**
   - Different colors per UniqueID range
   - Custom icons for special objects

2. **Object Grouping**
   - Group by model name
   - Group by placement pattern

3. **Heatmap Mode**
   - Show object density as heatmap
   - Identify high-traffic areas

4. **Export Selected**
   - Export filtered objects as CSV
   - Copy UniqueID list to clipboard

---

**Status**: ⏳ Ready for Implementation  
**Priority**: HIGH (blocks exploration UX)  
**Estimated Time**: 2 weeks
