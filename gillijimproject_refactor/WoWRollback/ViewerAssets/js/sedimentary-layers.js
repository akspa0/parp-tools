/**
 * Sedimentary Layers Manager
 * 
 * Implements UniqueID filtering for WoW object markers.
 * Allows exploration of object placement patterns by isolating specific UniqueID ranges.
 */

export class SedimentaryLayersManager {
    constructor(map) {
        this.map = map;
        this.uniqueIdToMarkers = new Map(); // UniqueID -> [markers]
        this.activeFilters = new Set(); // UniqueIDs currently being filtered
        this.mode = 'dim'; // 'dim', 'hide', or 'show'
        this.isInitialized = false;
        
        this.initUI();
    }
    
    initUI() {
        const panel = document.getElementById('layersPanel');
        if (!panel) {
            console.warn('[SedimentaryLayers] Panel not found');
            return;
        }
        
        const searchInput = document.getElementById('layersSearch');
        const listContainer = document.getElementById('layersList');
        const toggle = document.getElementById('layersPanelToggle');
        
        if (!searchInput || !listContainer || !toggle) {
            console.warn('[SedimentaryLayers] Required elements not found');
            return;
        }
        
        // Collapse/expand panel
        toggle.addEventListener('click', () => {
            const content = panel.querySelector('.layers-content');
            if (content.style.display === 'none') {
                content.style.display = '';
                toggle.textContent = '−';
            } else {
                content.style.display = 'none';
                toggle.textContent = '+';
            }
        });
        
        // Search input with range support
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            if (!query) {
                this.clearFilters();
                return;
            }
            
            this.handleSearch(query);
        });
        
        // Create mode selector
        this.createModeSelector(panel);
        
        this.isInitialized = true;
        console.log('[SedimentaryLayers] Initialized');
    }
    
    handleSearch(query) {
        // Support range syntax: "1000-2000"
        const rangeMatch = query.match(/^(\d+)-(\d+)$/);
        
        if (rangeMatch) {
            const min = parseInt(rangeMatch[1]);
            const max = parseInt(rangeMatch[2]);
            this.filterByRange(min, max);
        } else if (/^\d+$/.test(query)) {
            // Single UniqueID
            const id = parseInt(query);
            this.filterById(id);
        } else {
            // Invalid syntax
            this.updateStatus(`Invalid syntax. Use: "1234" or "1000-2000"`);
        }
    }
    
    createModeSelector(panel) {
        const header = panel.querySelector('.layers-header');
        if (!header) return;
        
        const modeContainer = document.createElement('div');
        modeContainer.style.display = 'inline-block';
        modeContainer.style.marginLeft = '10px';
        
        const label = document.createElement('label');
        label.textContent = 'Mode: ';
        label.style.fontSize = '11px';
        label.style.marginRight = '4px';
        
        const select = document.createElement('select');
        select.id = 'layerMode';
        select.style.fontSize = '11px';
        select.style.padding = '2px 4px';
        select.innerHTML = `
            <option value="dim">Dim Others</option>
            <option value="hide">Hide Others</option>
            <option value="show">Show Only</option>
        `;
        
        select.addEventListener('change', (e) => {
            this.mode = e.target.value;
            this.applyFilters();
        });
        
        modeContainer.appendChild(label);
        modeContainer.appendChild(select);
        header.appendChild(modeContainer);
    }
    
    registerMarker(marker, uniqueId) {
        if (!this.uniqueIdToMarkers.has(uniqueId)) {
            this.uniqueIdToMarkers.set(uniqueId, []);
        }
        this.uniqueIdToMarkers.get(uniqueId).push(marker);
        
        // Store uniqueId on marker for reference
        marker.options.uniqueId = uniqueId;
    }
    
    filterByRange(min, max) {
        if (min > max) {
            [min, max] = [max, min]; // Swap if reversed
        }
        
        this.activeFilters.clear();
        
        // Filter OUT objects outside the range
        for (const uniqueId of this.uniqueIdToMarkers.keys()) {
            if (uniqueId < min || uniqueId > max) {
                this.activeFilters.add(uniqueId);
            }
        }
        
        this.applyFilters();
        
        const showing = this.uniqueIdToMarkers.size - this.activeFilters.size;
        this.updateStatus(`Range ${min}-${max}: showing ${showing} UniqueIDs`);
    }
    
    filterById(id) {
        this.activeFilters.clear();
        
        // Filter OUT all other IDs
        for (const uniqueId of this.uniqueIdToMarkers.keys()) {
            if (uniqueId !== id) {
                this.activeFilters.add(uniqueId);
            }
        }
        
        if (this.uniqueIdToMarkers.has(id)) {
            const count = this.uniqueIdToMarkers.get(id).length;
            this.applyFilters();
            this.updateStatus(`UniqueID ${id}: ${count} object(s)`);
        } else {
            this.updateStatus(`UniqueID ${id}: not found`);
        }
    }
    
    applyFilters() {
        if (this.activeFilters.size === 0) {
            this.clearFilters();
            return;
        }
        
        for (const [uniqueId, markers] of this.uniqueIdToMarkers.entries()) {
            const isFiltered = this.activeFilters.has(uniqueId);
            
            for (const marker of markers) {
                if (this.mode === 'hide') {
                    // Hide: completely invisible and non-interactive
                    if (isFiltered) {
                        marker.setOpacity(0);
                        if (marker.options) marker.options.interactive = false;
                    } else {
                        marker.setOpacity(1);
                        if (marker.options) marker.options.interactive = true;
                    }
                } else if (this.mode === 'dim') {
                    // Dim: visible but faded
                    if (isFiltered) {
                        marker.setOpacity(0.1);
                    } else {
                        marker.setOpacity(1);
                    }
                } else if (this.mode === 'show') {
                    // Show only: same as hide
                    if (isFiltered) {
                        marker.setOpacity(0);
                        if (marker.options) marker.options.interactive = false;
                    } else {
                        marker.setOpacity(1);
                        if (marker.options) marker.options.interactive = true;
                    }
                }
            }
        }
    }
    
    clearFilters() {
        this.activeFilters.clear();
        
        // Restore all markers to full visibility
        for (const markers of this.uniqueIdToMarkers.values()) {
            for (const marker of markers) {
                marker.setOpacity(1);
                if (marker.options) marker.options.interactive = true;
            }
        }
        
        const totalIds = this.uniqueIdToMarkers.size;
        const totalMarkers = Array.from(this.uniqueIdToMarkers.values())
            .reduce((sum, markers) => sum + markers.length, 0);
        
        this.updateStatus(`Showing all: ${totalIds} UniqueIDs (${totalMarkers} objects)`);
    }
    
    updateStatus(message) {
        const listContainer = document.getElementById('layersList');
        if (!listContainer) return;
        
        listContainer.innerHTML = `
            <div class="layer-status">
                ${message || 'Ready'}
            </div>
        `;
    }
    
    getStats() {
        const totalIds = this.uniqueIdToMarkers.size;
        const totalMarkers = Array.from(this.uniqueIdToMarkers.values())
            .reduce((sum, markers) => sum + markers.length, 0);
        const filtered = this.activeFilters.size;
        
        return {
            totalUniqueIds: totalIds,
            totalMarkers: totalMarkers,
            filteredUniqueIds: filtered,
            showingUniqueIds: totalIds - filtered
        };
    }
}
