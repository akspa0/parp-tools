// Overlay Manager
// Coordinates loading and rendering of all terrain overlays

import { TerrainPropertiesLayer } from './terrainPropertiesLayer.js';
import { LiquidsLayer } from './liquidsLayer.js';
import { HolesLayer } from './holesLayer.js';
import { AreaIdLayer } from './areaIdLayer.js';
// import { ShadowMapLayer } from './shadowMapLayer.js'; // Disabled - needs reimplementation

export class OverlayManager {
    constructor(map) {
        this.map = map;
        
        // Initialize all overlay layers
        this.layers = {
            terrainProperties: new TerrainPropertiesLayer(map),
            liquids: new LiquidsLayer(map),
            holes: new HolesLayer(map),
            areaIds: new AreaIdLayer(map),
            // shadowMaps: new ShadowMapLayer(map, null, null) // Disabled - needs reimplementation
        };

        // Track loaded tile data
        this.loadedTiles = new Map(); // key: "r{row}_c{col}" -> overlay data
        
        // Debounce timer
        this.loadTimer = null;
        this.loadDelay = 500; // ms
    }

    // Show/hide specific layer
    showLayer(layerName) {
        if (this.layers[layerName]) {
            this.layers[layerName].show();
            this.renderVisibleTiles();
        }
    }

    hideLayer(layerName) {
        if (this.layers[layerName]) {
            this.layers[layerName].clear();
            this.layers[layerName].hide();
        }
    }

    // Clear all layers
    clearAll() {
        Object.values(this.layers).forEach(layer => layer.clear());
    }

    // Clear all data (use when switching maps)
    clearAllData() {
        this.clearAll();
        this.loadedTiles.clear();
    }

    // Load overlays for visible tiles (debounced)
    loadVisibleOverlays(mapName, version) {
        if (!mapName || !version) return;

        // Debounce to avoid excessive loading during pan/zoom
        clearTimeout(this.loadTimer);
        this.loadTimer = setTimeout(() => {
            this._loadVisibleOverlaysNow(mapName, version);
        }, this.loadDelay);
    }

    async _loadVisibleOverlaysNow(mapName, version) {
        const bounds = this.map.getBounds();
        const visibleTiles = this.getVisibleTiles(bounds);

        // Clear layers
        this.clearAll();

        // Load and render each visible tile
        for (const tile of visibleTiles) {
            await this.loadAndRenderTile(mapName, version, tile.row, tile.col);
        }

        // Cleanup: remove tiles that are far from view
        this.cleanupDistantTiles(visibleTiles);
    }

    async loadAndRenderTile(mapName, version, tileRow, tileCol) {
        const tileKey = `r${tileRow}_c${tileCol}`;
        
        // Shadow maps disabled - needs reimplementation
        // if (this.layers.shadowMaps && 
        //     (this.layers.shadowMaps.mapName !== mapName || this.layers.shadowMaps.version !== version)) {
        //     this.layers.shadowMaps.mapName = mapName;
        //     this.layers.shadowMaps.version = version;
        // }
        
        // Check if already loaded
        if (this.loadedTiles.has(tileKey)) {
            this.renderTile(this.loadedTiles.get(tileKey), tileRow, tileCol);
            return;
        }

        // Load overlay JSON
        const overlayPath = `overlays/${version}/${mapName}/terrain_complete/tile_${tileKey}.json`;
        
        try {
            const response = await fetch(overlayPath);
            if (!response.ok) {
                console.warn(`Overlay not found: ${overlayPath}`);
                return;
            }

            const data = await response.json();
            
            // Cache the data
            this.loadedTiles.set(tileKey, data);
            
            // Render the tile
            this.renderTile(data, tileRow, tileCol);
            
            // Shadow maps disabled - needs reimplementation
            // if (this.layers.shadowMaps) {
            //     await this.layers.shadowMaps.loadTile(tileRow, tileCol);
            // }
            
        } catch (error) {
            console.error(`Failed to load overlay ${overlayPath}:`, error);
        }
    }

    renderTile(tileData, tileRow, tileCol) {
        if (!tileData || !tileData.layers || tileData.layers.length === 0) return;

        const layer = tileData.layers[0]; // Use first layer (single version)

        // Render each overlay type
        if (layer.terrain_properties) {
            this.layers.terrainProperties.render(layer.terrain_properties, tileRow, tileCol);
        }

        if (layer.liquids) {
            this.layers.liquids.render(layer.liquids, tileRow, tileCol);
        }

        if (layer.holes) {
            this.layers.holes.render(layer.holes, tileRow, tileCol);
        }

        if (layer.area_ids) {
            this.layers.areaIds.render(layer.area_ids, tileRow, tileCol);
        }
    }

    renderVisibleTiles() {
        // Re-render all loaded tiles that are visible
        const bounds = this.map.getBounds();
        const visibleTiles = this.getVisibleTiles(bounds);

        this.clearAll();

        visibleTiles.forEach(tile => {
            const tileKey = `r${tile.row}_c${tile.col}`;
            const data = this.loadedTiles.get(tileKey);
            if (data) {
                this.renderTile(data, tile.row, tile.col);
            }
        });
    }

    getVisibleTiles(bounds) {
        const tiles = [];
        
        const latS = bounds.getSouth();
        const latN = bounds.getNorth();
        const west = bounds.getWest();
        const east = bounds.getEast();

        // Convert to tile coordinates
        const rowNorth = this.latToRow(latN);
        const rowSouth = this.latToRow(latS);
        const minRow = Math.floor(Math.min(rowNorth, rowSouth));
        const maxRow = Math.ceil(Math.max(rowNorth, rowSouth));
        const minCol = Math.floor(west);
        const maxCol = Math.ceil(east);

        // Clamp to valid range (0-63)
        for (let r = Math.max(0, minRow); r <= Math.min(63, maxRow); r++) {
            for (let c = Math.max(0, minCol); c <= Math.min(63, maxCol); c++) {
                tiles.push({ row: r, col: c });
            }
        }

        return tiles;
    }

    latToRow(lat) {
        return 63 - lat;
    }

    cleanupDistantTiles(visibleTiles) {
        // Remove tiles that are > 2 tiles away from view
        const visibleSet = new Set(visibleTiles.map(t => `r${t.row}_c${t.col}`));
        
        for (const [tileKey, data] of this.loadedTiles.entries()) {
            if (!visibleSet.has(tileKey)) {
                // Check distance
                const match = tileKey.match(/r(\d+)_c(\d+)/);
                if (match) {
                    const row = parseInt(match[1]);
                    const col = parseInt(match[2]);
                    
                    const minDist = Math.min(
                        ...visibleTiles.map(t => 
                            Math.abs(t.row - row) + Math.abs(t.col - col)
                        )
                    );
                    
                    if (minDist > 2) {
                        this.loadedTiles.delete(tileKey);
                    }
                }
            }
        }
    }

    // Set layer options
    setLayerOption(layerName, optionKey, value) {
        if (this.layers[layerName] && this.layers[layerName].setOption) {
            this.layers[layerName].setOption(optionKey, value);
            this.renderVisibleTiles();
        }
    }

    setLayerOpacity(layerName, opacity) {
        if (this.layers[layerName] && this.layers[layerName].setOpacity) {
            this.layers[layerName].setOpacity(opacity);
            this.renderVisibleTiles();
        }
    }
}
