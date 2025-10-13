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
        this.mode = 'show'; // 'dim', 'hide', or 'show' - default to 'show' (show only checked)
        this.currentTileOnly = false; // Filter to current tile
        this.isInitialized = false;
        this.isFiltering = false; // Prevent recursive filtering
        this.filterTimeout = null; // Debounce timer
        this.showAllTiles = false; // Toggle for showing tiles without data
        
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
        
        // Show all tiles toggle
        const showAllTilesCheckbox = document.getElementById('showAllTiles');
        if (showAllTilesCheckbox) {
            showAllTilesCheckbox.addEventListener('change', (e) => {
                this.showAllTiles = e.target.checked;
                this.updateMinimapTileVisibility();
            });
        }
        
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
            <option value="show" selected>Show Only</option>
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
        
        // Auto-reload CSV when map changes
        this.lastLoadedMap = null;
        this.lastLoadedVersion = null;
        
        // Watch for map/version changes
        this.checkForMapChange();
    }
    
    checkForMapChange() {
        // Poll for map changes (since state doesn't have events)
        setInterval(() => {
            const currentMap = this.state.selectedMap;
            const currentVersion = this.state.selectedVersion;
            
            // Debug logging
            console.log(`[SedimentaryLayersCSV] Poll: current=${currentMap}/${currentVersion}, last=${this.lastLoadedMap}/${this.lastLoadedVersion}, ranges=${this.ranges.length}`);
            
            if (currentMap && currentVersion && 
                (currentMap !== this.lastLoadedMap || currentVersion !== this.lastLoadedVersion)) {
                
                console.log(`[SedimentaryLayersCSV] Map changed to ${currentMap} (${currentVersion}), reloading CSV...`);
                
                // Auto-reload if ranges were previously loaded
                if (this.ranges.length > 0) {
                    this.loadCSVRanges();
                } else {
                    console.log('[SedimentaryLayersCSV] No ranges loaded yet, skipping auto-reload');
                }
            }
        }, 1000); // Check every second
    }
    
    async loadCSVRanges() {
        console.log('[SedimentaryLayersCSV] loadCSVRanges() called');
        
        const statusDiv = document.getElementById('loadStatus');
        const btn = document.getElementById('loadRangesBtn');
        
        try {
            if (statusDiv) statusDiv.textContent = 'Loading...';
            if (btn) btn.disabled = true;
            
            // Get current version and map from state
            const version = this.state.selectedVersion;
            const mapName = this.state.selectedMap;
            
            console.log('[SedimentaryLayersCSV] Loading CSV for:', mapName, version);
            
            if (!version || !mapName) {
                const msg = 'Error: No map selected';
                console.error('[SedimentaryLayersCSV]', msg);
                if (statusDiv) statusDiv.textContent = msg;
                if (btn) btn.disabled = false;
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
            
            // Track what was loaded
            this.lastLoadedMap = mapName;
            this.lastLoadedVersion = version;
            
            if (statusDiv) statusDiv.textContent = `Loaded ${this.ranges.length} ranges for ${mapName}`;
            console.log('[SedimentaryLayersCSV] Successfully loaded', this.ranges.length, 'ranges');
            
        } catch (err) {
            console.error('[SedimentaryLayersCSV] Load failed:', err);
            if (statusDiv) statusDiv.textContent = `Error: ${err.message}`;
            if (btn) btn.disabled = false;
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
        
        // Sort ranges by min UniqueID (smallest to largest)
        this.ranges.sort((a, b) => a.min - b.min);
        
        console.log(`[SedimentaryLayersCSV] Parsed ${this.ranges.length} ranges (sorted by min)`);
    }
    
    renderRangeCheckboxes() {
        const listContainer = document.getElementById('layersList');
        listContainer.innerHTML = '';
        
        // Add control buttons (reload + bulk selection)
        const controlButtons = document.createElement('div');
        controlButtons.style.cssText = 'padding: 8px; background: #2a2a2a; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;';
        controlButtons.innerHTML = `
            <button id="reloadRangesBtn" style="
                padding: 4px 12px;
                background: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                font-weight: bold;
            ">🔄 Reload Ranges</button>
            <button id="selectAllBtn" style="
                padding: 4px 12px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
            ">Select All</button>
            <button id="deselectAllBtn" style="
                padding: 4px 12px;
                background: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
            ">Deselect All</button>
        `;
        listContainer.appendChild(controlButtons);
        
        // Add event listeners for control buttons
        const reloadBtn = document.getElementById('reloadRangesBtn');
        if (reloadBtn) {
            reloadBtn.addEventListener('click', () => {
                console.log('[SedimentaryLayersCSV] 🔄 Reload Ranges button clicked!');
                console.log('[SedimentaryLayersCSV] Current state:', this.state.selectedMap, this.state.selectedVersion);
                this.loadCSVRanges();
            });
            console.log('[SedimentaryLayersCSV] Reload button event listener attached');
        } else {
            console.error('[SedimentaryLayersCSV] Reload button not found!');
        }
        
        document.getElementById('selectAllBtn').addEventListener('click', () => {
            this.ranges.forEach(r => r.enabled = true);
            this.renderRangeCheckboxes();
        });
        
        document.getElementById('deselectAllBtn').addEventListener('click', () => {
            this.ranges.forEach(r => r.enabled = false);
            this.renderRangeCheckboxes();
        });
        
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
            // Extract tile info from map field if available (format: "mapName_(row_col)")
            const tileMatch = range.map?.match(/_\((\d+)_(\d+)\)$/);
            const tileInfo = tileMatch ? ` [${tileMatch[1]},${tileMatch[2]}]` : '';
            label.textContent = `${range.min} - ${range.max}${tileInfo}`;
            label.style.flex = '1';
            label.style.cursor = 'pointer';
            label.title = range.map || `Range ${range.min}-${range.max}`;
            
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
        
        // Register by tile - use same format as minimap keys: r{row}_c{col}
        const tileKey = `r${tileRow}_c${tileCol}`;
        if (!this.tileToMarkers.has(tileKey)) {
            this.tileToMarkers.set(tileKey, []);
        }
        this.tileToMarkers.get(tileKey).push(marker);
        
        // Store metadata
        marker.options.uniqueId = uniqueId;
        marker.options.tileRow = tileRow;
        marker.options.tileCol = tileCol;
        
        // Auto-apply current filter to newly registered marker
        if (this.ranges.length > 0) {
            this.applyFilterToSingleMarker(marker, uniqueId);
        }
        
        // Debug: log registrations to track data
        const totalRegistered = Array.from(this.uniqueIdToMarkers.values()).reduce((sum, arr) => sum + arr.length, 0);
        if (totalRegistered % 100 === 0) {
            console.log(`[SedimentaryLayersCSV] Registered ${totalRegistered} markers across ${this.uniqueIdToMarkers.size} UniqueIDs, ${this.tileToMarkers.size} tiles`);
        }
    }
    
    getRegistrationStats() {
        const tileStats = new Map();
        for (const [tileKey, markers] of this.tileToMarkers.entries()) {
            tileStats.set(tileKey, markers.length);
        }
        return {
            totalMarkers: Array.from(this.uniqueIdToMarkers.values()).reduce((sum, arr) => sum + arr.length, 0),
            uniqueIds: this.uniqueIdToMarkers.size,
            tiles: this.tileToMarkers.size,
            tileStats: tileStats
        };
    }
    
    applyFilterToSingleMarker(marker, uniqueId) {
        // Apply current filter state to a single marker immediately
        const enabledRanges = this.ranges.filter(r => r.enabled);
        const inEnabledRange = enabledRanges.some(r => uniqueId >= r.min && uniqueId <= r.max);
        
        let shouldShow = inEnabledRange;
        
        // Apply tile filter if enabled
        if (this.currentTileOnly && shouldShow) {
            const bounds = this.map.getBounds();
            const center = this.map.getCenter();
            const tileRow = Math.floor(window.latToRow ? window.latToRow(center.lat) : center.lat);
            const tileCol = Math.floor(center.lng);
            
            shouldShow = marker.options.tileRow === tileRow && 
                         marker.options.tileCol === tileCol;
        }
        
        this.applyMarkerVisibility(marker, shouldShow);
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
        
        // If no ranges enabled, reset minimap and show all markers
        if (enabledRanges.length === 0) {
            console.log('[SedimentaryLayersCSV] No ranges enabled, showing all markers and resetting minimap');
            for (const markers of this.uniqueIdToMarkers.values()) {
                for (const marker of markers) {
                    this.applyMarkerVisibility(marker, true);
                }
            }
            this.resetMinimapVisibility();
            this.isFiltering = false;
            return;
        }
        
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
                    const tileRow = Math.floor(window.latToRow ? window.latToRow(center.lat) : center.lat);
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
        
        // Update minimap tile visibility based on enabled ranges
        this.updateMinimapTileVisibility();
        
        // Reset filtering flag
        this.isFiltering = false;
    }
    
    updateMinimapTileVisibility() {
        // Track which tiles have objects in enabled ranges
        const tilesWithData = new Set();
        const enabledRanges = this.ranges.filter(r => r.enabled);
        
        console.log('[SedimentaryLayersCSV] updateMinimapTileVisibility called');
        console.log('[SedimentaryLayersCSV] Total UniqueIDs registered:', this.uniqueIdToMarkers.size);
        console.log('[SedimentaryLayersCSV] Enabled ranges:', enabledRanges.length);
        
        // Build complete set of tiles with data BEFORE updating opacity
        for (const [uniqueId, markers] of this.uniqueIdToMarkers.entries()) {
            const inEnabledRange = enabledRanges.some(r => uniqueId >= r.min && uniqueId <= r.max);
            if (inEnabledRange) {
                for (const marker of markers) {
                    const tileKey = `r${marker.options.tileRow}_c${marker.options.tileCol}`;
                    tilesWithData.add(tileKey);
                }
            }
        }
        
        console.log('[SedimentaryLayersCSV] Tiles with data:', Array.from(tilesWithData).sort());
        console.log('[SedimentaryLayersCSV] Tiles registered in system:', Array.from(this.tileToMarkers.keys()).sort());
        console.log('[SedimentaryLayersCSV] window.minimapImages exists?', !!window.minimapImages);
        console.log('[SedimentaryLayersCSV] window.minimapImages size:', window.minimapImages?.size);
        console.log('[SedimentaryLayersCSV] window.minimapImages keys:', window.minimapImages ? Array.from(window.minimapImages.keys()).sort() : 'N/A');
        
        // Store this for future tile loads
        this._currentTilesWithData = tilesWithData;
        
        // Update visibility for ALL currently loaded minimap tiles
        // minimapImages is a Map with key format "r{row}_c{col}" -> L.ImageOverlay
        if (window.minimapImages && window.minimapLayer) {
            let shown = 0;
            let hidden = 0;
            
            console.log('[SedimentaryLayersCSV] Available minimap keys:', Array.from(window.minimapImages.keys()).sort());
            
            // Update all loaded tiles - HIDE tiles without data, SHOW tiles with data or if showAllTiles enabled
            window.minimapImages.forEach((imageOverlay, key) => {
                const hasData = tilesWithData.has(key);
                
                if (hasData || this.showAllTiles || enabledRanges.length === 0) {
                    // Show tile (has data, user toggled show all, or no filter active)
                    if (!window.minimapLayer.hasLayer(imageOverlay)) {
                        imageOverlay.addTo(window.minimapLayer);
                    }
                    imageOverlay.setOpacity(1.0);
                    shown++;
                } else {
                    // Hide tile (no data and user wants to hide)
                    if (window.minimapLayer.hasLayer(imageOverlay)) {
                        window.minimapLayer.removeLayer(imageOverlay);
                    }
                    hidden++;
                }
            });
            
            console.log(`[SedimentaryLayersCSV] Updated minimap visibility: ${shown} shown, ${hidden} hidden (${tilesWithData.size} tiles have data, showAllTiles=${this.showAllTiles})`);
            
            // Hook into minimap tile loading to apply visibility to newly loaded tiles
            this._hookMinimapTileLoading();
        } else {
            console.warn('[SedimentaryLayersCSV] window.minimapImages or window.minimapLayer not available');
        }
    }
    
    _hookMinimapTileLoading() {
        // Ensure we only hook once
        if (this._minimapHooked) return;
        this._minimapHooked = true;
        
        const self = this;
        
        // Intercept when new minimap tiles are added
        if (window.minimapImages && !window.minimapImages._visibilityHooked) {
            window.minimapImages._visibilityHooked = true;
            
            const originalMapSet = window.minimapImages.set.bind(window.minimapImages);
            window.minimapImages.set = function(key, imageOverlay) {
                // Call original set
                const result = originalMapSet(key, imageOverlay);
                
                // Apply visibility to newly loaded tile
                if (self._currentTilesWithData && self.ranges.length > 0) {
                    const hasData = self._currentTilesWithData.has(key);
                    const enabledRanges = self.ranges.filter(r => r.enabled);
                    
                    if (hasData || self.showAllTiles || enabledRanges.length === 0) {
                        // Tile should be visible - it's already added by main.js
                        imageOverlay.setOpacity(1.0);
                    } else {
                        // Tile should be hidden - remove it
                        if (window.minimapLayer && window.minimapLayer.hasLayer(imageOverlay)) {
                            window.minimapLayer.removeLayer(imageOverlay);
                        }
                    }
                    console.log(`[SedimentaryLayersCSV] Applied visibility to newly loaded tile ${key}: ${hasData ? 'visible' : 'hidden'}`);
                }
                
                return result;
            };
            
            console.log('[SedimentaryLayersCSV] Hooked minimap tile loading for visibility control');
        }
    }
    
    resetMinimapVisibility() {
        // Reset all minimap tiles to visible with full opacity
        if (window.minimapImages && window.minimapLayer) {
            window.minimapImages.forEach((imageOverlay, key) => {
                if (!window.minimapLayer.hasLayer(imageOverlay)) {
                    imageOverlay.addTo(window.minimapLayer);
                }
                imageOverlay.setOpacity(1.0);
            });
            console.log('[SedimentaryLayersCSV] Reset all minimap tiles to visible');
        }
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
