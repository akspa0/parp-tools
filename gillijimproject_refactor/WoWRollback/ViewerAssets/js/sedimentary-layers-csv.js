/**
 * Sedimentary Layers Manager - Enhanced with CSV Loading
 * 
 * Loads pre-generated UniqueID range CSVs and provides:
 * - Checkbox UI for range selection
 * - Per-tile filtering
 * - Visual object counts
 */

export class SedimentaryLayersManagerCSV {
    constructor(map, state) {
        this.map = map;
        this.state = state;
        this.uniqueIdToMarkers = new Map(); // UniqueID -> [markers]
        this.tileToMarkers = new Map(); // "row_col" -> [markers]
        this.ranges = []; // Loaded from CSV: [{min, max, count, enabled}]
        this.mode = 'dim'; // 'dim', 'hide', or 'show'
        this.currentTileOnly = false; // Filter to current tile
        this.isInitialized = false;
        this.isFiltering = false; // Prevent recursive filtering
        this.filterTimeout = null; // Debounce timer
        
        this.initUI();
    }
    
    async initUI() {
        const panel = document.getElementById('layersPanel');
        if (!panel) {
            console.warn('[SedimentaryLayersCSV] Panel not found');
            return;
        }
        
        const searchInput = document.getElementById('layersSearch');
        const listContainer = document.getElementById('layersList');
        const toggle = document.getElementById('layersPanelToggle');
        
        if (!searchInput || !listContainer || !toggle) {
            console.warn('[SedimentaryLayersCSV] Required elements not found');
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
        
        // Manual search still works (fallback)
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            if (!query) {
                this.clearFilters();
                return;
            }
            this.handleManualSearch(query);
        });
        
        // Create mode selector
        this.createModeSelector(panel);
        
        // Create tile filter toggle
        this.createTileFilterToggle(panel);
        
        // Load button
        this.createLoadButton(listContainer);
        
        this.isInitialized = true;
        console.log('[SedimentaryLayersCSV] Initialized');
    }
    
    createModeSelector(panel) {
        const header = panel.querySelector('.layers-header');
        if (!header) return;
        
        const modeContainer = document.createElement('div');
        modeContainer.style.display = 'inline-block';
        modeContainer.style.marginLeft = '10px';
        
        const select = document.createElement('select');
        select.id = 'layerMode';
        select.style.fontSize = '11px';
        select.style.padding = '2px 4px';
        select.innerHTML = `
            <option value="dim">Dim</option>
            <option value="hide">Hide</option>
            <option value="show">Show Only</option>
        `;
        
        select.addEventListener('change', (e) => {
            this.mode = e.target.value;
            this.applyFilters();
        });
        
        modeContainer.appendChild(select);
        header.appendChild(modeContainer);
    }
    
    createTileFilterToggle(panel) {
        const header = panel.querySelector('.layers-header');
        if (!header) return;
        
        const container = document.createElement('div');
        container.style.display = 'inline-block';
        container.style.marginLeft = '10px';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = 'tileFilterToggle';
        checkbox.title = 'Filter to current tile only';
        
        const label = document.createElement('label');
        label.htmlFor = 'tileFilterToggle';
        label.textContent = 'Current Tile';
        label.style.fontSize = '11px';
        label.style.marginLeft = '4px';
        label.style.cursor = 'pointer';
        
        checkbox.addEventListener('change', (e) => {
            this.currentTileOnly = e.target.checked;
            this.applyFilters();
        });
        
        container.appendChild(checkbox);
        container.appendChild(label);
        header.appendChild(container);
    }
    
    createLoadButton(container) {
        container.innerHTML = `
            <div style="padding: 20px; text-align: center;">
                <button id="loadRangesBtn" style="
                    padding: 10px 20px;
                    background: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 13px;
                ">Load UniqueID Ranges</button>
                <div id="loadStatus" style="margin-top: 10px; font-size: 11px; color: #888;"></div>
            </div>
        `;
        
        const btn = document.getElementById('loadRangesBtn');
        btn.addEventListener('click', () => this.loadCSVRanges());
    }
    
    async loadCSVRanges() {
        const statusDiv = document.getElementById('loadStatus');
        const btn = document.getElementById('loadRangesBtn');
        
        try {
            statusDiv.textContent = 'Loading...';
            btn.disabled = true;
            
            // Get current version and map from state
            const version = this.state.selectedVersion;
            const mapName = this.state.selectedMap;
            
            if (!version || !mapName) {
                statusDiv.textContent = 'Error: No map selected';
                return;
            }
            
            // Build CSV path - CSVs are copied into viewer directory by rebuild script
            const csvPath = `cached_maps/analysis/${version}/${mapName}/csv/id_ranges_by_map.csv`;
            
            console.log('[SedimentaryLayersCSV] Loading from:', csvPath);
            
            const response = await fetch(csvPath);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const csvText = await response.text();
            this.parseCSV(csvText);
            this.renderRangeCheckboxes();
            
            statusDiv.textContent = `Loaded ${this.ranges.length} ranges`;
            
        } catch (err) {
            console.error('[SedimentaryLayersCSV] Load failed:', err);
            statusDiv.textContent = `Error: ${err.message}`;
            btn.disabled = false;
        }
    }
    
    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        
        this.ranges = [];
        
        for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split(',');
            if (parts.length >= 4) {
                this.ranges.push({
                    map: parts[0],
                    min: parseInt(parts[1]),
                    max: parseInt(parts[2]),
                    count: parseInt(parts[3]),
                    enabled: true // All enabled by default
                });
            }
        }
        
        console.log(`[SedimentaryLayersCSV] Parsed ${this.ranges.length} ranges`);
    }
    
    renderRangeCheckboxes() {
        const listContainer = document.getElementById('layersList');
        listContainer.innerHTML = '';
        
        const header = document.createElement('div');
        header.style.cssText = 'padding: 8px; background: #333; font-weight: bold; font-size: 12px; display: flex; justify-content: space-between;';
        header.innerHTML = `
            <span>Range</span>
            <span>Count</span>
        `;
        listContainer.appendChild(header);
        
        const rangesDiv = document.createElement('div');
        rangesDiv.style.cssText = 'max-height: 400px; overflow-y: auto;';
        
        this.ranges.forEach((range, index) => {
            const item = document.createElement('div');
            item.style.cssText = `
                padding: 6px 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #444;
                font-size: 11px;
                cursor: pointer;
                background: ${range.enabled ? '#2a2a2a' : '#1a1a1a'};
            `;
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = range.enabled;
            checkbox.style.marginRight = '8px';
            
            const label = document.createElement('label');
            label.textContent = `${range.min} - ${range.max}`;
            label.style.flex = '1';
            label.style.cursor = 'pointer';
            
            const countSpan = document.createElement('span');
            countSpan.textContent = range.count.toLocaleString();
            countSpan.style.color = '#888';
            countSpan.style.fontSize = '10px';
            
            checkbox.addEventListener('change', (e) => {
                range.enabled = e.target.checked;
                item.style.background = range.enabled ? '#2a2a2a' : '#1a1a1a';
                this.applyFilters();
            });
            
            item.addEventListener('click', (e) => {
                if (e.target !== checkbox) {
                    checkbox.checked = !checkbox.checked;
                    checkbox.dispatchEvent(new Event('change'));
                }
            });
            
            item.appendChild(checkbox);
            item.appendChild(label);
            item.appendChild(countSpan);
            rangesDiv.appendChild(item);
        });
        
        listContainer.appendChild(rangesDiv);
        
        // Add summary
        const summary = document.createElement('div');
        summary.style.cssText = 'padding: 8px; background: #333; font-size: 11px; text-align: center;';
        const totalCount = this.ranges.reduce((sum, r) => sum + r.count, 0);
        summary.innerHTML = `
            <div>Total: ${totalCount.toLocaleString()} objects</div>
            <button id="applyFilterBtn" style="margin-top: 8px; padding: 6px 12px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Apply Filter Now
            </button>
        `;
        listContainer.appendChild(summary);
        
        // Apply filters immediately
        this.applyFilters();
        
        // Manual filter button
        document.getElementById('applyFilterBtn').addEventListener('click', () => {
            console.log('[SedimentaryLayersCSV] Manual filter triggered');
            this.applyFilters();
        });
    }
    
    handleManualSearch(query) {
        // Keep manual search as fallback
        const rangeMatch = query.match(/^(\d+)-(\d+)$/);
        
        if (rangeMatch) {
            const min = parseInt(rangeMatch[1]);
            const max = parseInt(rangeMatch[2]);
            this.filterByRange(min, max);
        } else if (/^\d+$/.test(query)) {
            const id = parseInt(query);
            this.filterById(id);
        }
    }
    
    registerMarker(marker, uniqueId, tileRow, tileCol) {
        // Register by UniqueID
        if (!this.uniqueIdToMarkers.has(uniqueId)) {
            this.uniqueIdToMarkers.set(uniqueId, []);
        }
        this.uniqueIdToMarkers.get(uniqueId).push(marker);
        
        // Register by tile
        const tileKey = `${tileRow}_${tileCol}`;
        if (!this.tileToMarkers.has(tileKey)) {
            this.tileToMarkers.set(tileKey, []);
        }
        this.tileToMarkers.get(tileKey).push(marker);
        
        // Store metadata
        marker.options.uniqueId = uniqueId;
        marker.options.tileRow = tileRow;
        marker.options.tileCol = tileCol;
        
        // Debug: log first few registrations
        if (this.uniqueIdToMarkers.size <= 5) {
            console.log(`[SedimentaryLayersCSV] Registered marker UID=${uniqueId} tile=${tileRow},${tileCol}`);
        }
    }
    
    applyFilters() {
        // Debounce to avoid excessive calls
        clearTimeout(this.filterTimeout);
        this.filterTimeout = setTimeout(() => this._applyFiltersNow(), 100);
    }
    
    _applyFiltersNow() {
        // Prevent recursive filtering
        if (this.isFiltering) {
            console.log('[SedimentaryLayersCSV] Already filtering, skipping...');
            return;
        }
        
        if (this.ranges.length === 0) {
            // No ranges loaded, show all
            console.log('[SedimentaryLayersCSV] No ranges loaded, skipping filter');
            return;
        }
        
        this.isFiltering = true;
        
        const enabledRanges = this.ranges.filter(r => r.enabled);
        console.log(`[SedimentaryLayersCSV] Filtering with ${enabledRanges.length}/${this.ranges.length} enabled ranges`);
        console.log(`[SedimentaryLayersCSV] Registered markers: ${this.uniqueIdToMarkers.size} UniqueIDs, total markers: ${Array.from(this.uniqueIdToMarkers.values()).reduce((sum, arr) => sum + arr.length, 0)}`);
        
        let hiddenCount = 0;
        let shownCount = 0;
        
        for (const [uniqueId, markers] of this.uniqueIdToMarkers.entries()) {
            const inEnabledRange = enabledRanges.some(r => uniqueId >= r.min && uniqueId <= r.max);
            
            for (const marker of markers) {
                let shouldShow = inEnabledRange;
                
                // Apply tile filter if enabled
                if (this.currentTileOnly && shouldShow) {
                    const bounds = this.map.getBounds();
                    const center = this.map.getCenter();
                    const tileRow = Math.floor(center.lat);
                    const tileCol = Math.floor(center.lng);
                    
                    shouldShow = marker.options.tileRow === tileRow && 
                                 marker.options.tileCol === tileCol;
                }
                
                this.applyMarkerVisibility(marker, shouldShow);
                
                if (shouldShow) {
                    shownCount++;
                } else {
                    hiddenCount++;
                }
            }
        }
        
        console.log(`[SedimentaryLayersCSV] Filter applied: ${shownCount} shown, ${hiddenCount} hidden/dimmed`);
        
        // Reset filtering flag
        this.isFiltering = false;
    }
    
    applyMarkerVisibility(marker, shouldShow) {
        // Use setStyle for CircleMarker/Rectangle (Path objects)
        if (this.mode === 'hide' || this.mode === 'show') {
            if (shouldShow) {
                marker.setStyle({ fillOpacity: marker._originalOpacity || 0.85, opacity: 1 });
                if (marker.options) marker.options.interactive = true;
            } else {
                // Store original opacity first time
                if (!marker._originalOpacity) {
                    marker._originalOpacity = marker.options.fillOpacity || 0.85;
                }
                marker.setStyle({ fillOpacity: 0, opacity: 0 });
                if (marker.options) marker.options.interactive = false;
            }
        } else if (this.mode === 'dim') {
            if (shouldShow) {
                marker.setStyle({ fillOpacity: marker._originalOpacity || 0.85, opacity: 1 });
                if (marker.options) marker.options.interactive = true;
            } else {
                if (!marker._originalOpacity) {
                    marker._originalOpacity = marker.options.fillOpacity || 0.85;
                }
                marker.setStyle({ fillOpacity: 0.2, opacity: 0.3 });
                if (marker.options) marker.options.interactive = false;
            }
        }
    }
    
    filterByRange(min, max) {
        if (min > max) [min, max] = [max, min];
        
        for (const [uniqueId, markers] of this.uniqueIdToMarkers.entries()) {
            const inRange = uniqueId >= min && uniqueId <= max;
            for (const marker of markers) {
                this.applyMarkerVisibility(marker, inRange);
            }
        }
    }
    
    filterById(id) {
        for (const [uniqueId, markers] of this.uniqueIdToMarkers.entries()) {
            const matches = uniqueId === id;
            for (const marker of markers) {
                this.applyMarkerVisibility(marker, matches);
            }
        }
    }
    
    clearFilters() {
        // Reset all ranges to enabled
        this.ranges.forEach(r => r.enabled = true);
        
        // Show all markers
        for (const markers of this.uniqueIdToMarkers.values()) {
            for (const marker of markers) {
                marker.setOpacity(1);
                if (marker.options) marker.options.interactive = true;
            }
        }
        
        if (this.ranges.length > 0) {
            this.renderRangeCheckboxes();
        }
    }
    
    getStats() {
        const totalIds = this.uniqueIdToMarkers.size;
        const totalMarkers = Array.from(this.uniqueIdToMarkers.values())
            .reduce((sum, markers) => sum + markers.length, 0);
        const enabledRanges = this.ranges.filter(r => r.enabled).length;
        
        return {
            totalUniqueIds: totalIds,
            totalMarkers: totalMarkers,
            totalRanges: this.ranges.length,
            enabledRanges: enabledRanges
        };
    }
}
