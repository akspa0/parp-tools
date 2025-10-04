// Overlay Manager
// Coordinates loading and rendering of all terrain overlays

import { TerrainPropertiesLayer } from './terrainPropertiesLayer.js';
import { LiquidsLayer } from './liquidsLayer.js';
import { HolesLayer } from './holesLayer.js';
import { AreaIdLayer } from './areaIdLayer.js';
import { ShadowMapLayer } from './shadowMapLayer.js';

export class OverlayManager {
    constructor(map) {
        this.map = map;

        // Initialize all overlay layers
        this.layers = {
            terrainProperties: new TerrainPropertiesLayer(map),
            liquids: new LiquidsLayer(map),
            holes: new HolesLayer(map),
            areaIds: new AreaIdLayer(map),
            shadowMaps: new ShadowMapLayer(map, null, null) // mapName/version set later
        };

        // Track loaded tile data for interactive overlays
        this.loadedTiles = new Map(); // key: "r{row}_c{col}" -> overlay data

        // Debounce timer for batch loading
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
            this.layers[layerName].hide();
        }
    }

    clearInteractiveLayers() {
        Object.values(this.layers).forEach(layer => layer.clear());
    }

    clearAll() {
        Object.values(this.layers).forEach(layer => layer.clear());
        this.loadedTiles.clear();
    }

    // Load overlays for visible tiles (debounced)
    loadVisibleOverlays(mapName, version) {
        if (this.loadTimer) {
            clearTimeout(this.loadTimer);
        }

        this.loadTimer = setTimeout(() => {
            this.doLoadVisibleOverlays(mapName, version);
        }, this.loadDelay);
    }

    async doLoadVisibleOverlays(mapName, version) {
        const bounds = this.map.getBounds();
        const visibleTiles = this.getVisibleTiles(bounds);
        this.clearAll();

        for (const tile of visibleTiles) {
            await this.loadAndRenderTile(mapName, version, tile.row, tile.col);
        }

        this.cleanupDistantTiles(visibleTiles);
    }

    async loadAndRenderTile(mapName, version, tileRow, tileCol) {
        const tileKey = `r${tileRow}_c${tileCol}`;

        // Shadow maps use JSON metadata; keep loading them regardless
        if (this.layers.shadowMaps) {
            if (this.layers.shadowMaps.mapName !== mapName || this.layers.shadowMaps.version !== version) {
                this.layers.shadowMaps.mapName = mapName;
                this.layers.shadowMaps.version = version;
            }

            await this.layers.shadowMaps.loadTile(tileRow, tileCol);
        }

        if (this.loadedTiles.has(tileKey)) {
            this.renderTile(this.loadedTiles.get(tileKey), tileRow, tileCol);
            return;
        }

        const overlayPath = `overlays/${version}/${mapName}/terrain_complete/tile_${tileKey}.json`;

        try {
            const response = await fetch(overlayPath);
            if (!response.ok) {
                return;
            }

            const data = await response.json();
            this.loadedTiles.set(tileKey, data);
            this.renderTile(data, tileRow, tileCol);
        } catch (error) {
            console.error(`Failed to load overlay ${overlayPath}:`, error);
        }
    }

    renderTile(tileData, tileRow, tileCol) {
        if (!tileData || !tileData.layers || tileData.layers.length === 0) {
            return;
        }

        const layer = tileData.layers[0];

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

        const rowNorth = this.latToRow(latN);
        const rowSouth = this.latToRow(latS);
        const minRow = Math.floor(Math.min(rowNorth, rowSouth));
        const maxRow = Math.ceil(Math.max(rowNorth, rowSouth));
        const minCol = Math.floor(west);
        const maxCol = Math.ceil(east);

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
        const visibleSet = new Set(visibleTiles.map(t => `r${t.row}_c${t.col}`));

        for (const [tileKey] of this.loadedTiles.entries()) {
            if (!visibleSet.has(tileKey)) {
                const match = tileKey.match(/r(\d+)_c(\d+)/);
                if (match) {
                    const row = parseInt(match[1], 10);
                    const col = parseInt(match[2], 10);

                    const minDist = Math.min(
                        ...visibleTiles.map(t => Math.abs(t.row - row) + Math.abs(t.col - col))
                    );

                    if (minDist > 2) {
                        this.loadedTiles.delete(tileKey);
                    }
                }
            }
        }
    }
}
