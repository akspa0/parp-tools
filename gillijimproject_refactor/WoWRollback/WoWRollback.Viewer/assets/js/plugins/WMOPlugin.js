import { OverlayPlugin } from '../core/OverlayPlugin.js';

/**
 * WMOPlugin - Displays WMO object placements
 * Loads data from JSON and renders as square markers
 */
export class WMOPlugin extends OverlayPlugin {
    constructor(map, coordSystem) {
        super('wmo', 'WMO Objects', map, coordSystem);
        
        this.color = '#FF9800';
        this.baseSize = 0.006; // Size in lat/lng units
        this.data = null;
        this.loadedTiles = new Set();
    }
    
    async onLoad(version, mapName) {
        // WMO data will be loaded per-tile on demand
        console.log(`[WMOPlugin] Ready to load data for ${mapName} (${version})`);
    }
    
    async loadVisibleData(bounds, zoom) {
        if (!this.enabled || !this.visible) return;
        
        // Get visible tiles
        const tiles = this.getVisibleTiles(bounds);
        
        // Load data for each visible tile
        for (const tile of tiles) {
            const key = `${tile.row}_${tile.col}`;
            if (!this.loadedTiles.has(key)) {
                await this.loadTileData(tile.row, tile.col);
                this.loadedTiles.add(key);
            }
        }
    }
    
    async loadTileData(row, col) {
        try {
            // TODO: Replace with actual data path from state
            // For now, this is a placeholder
            console.log(`[WMOPlugin] Loading tile ${row}_${col}`);
            
            // Example data structure:
            // {
            //   "placements": [
            //     {
            //       "uniqueId": 54321,
            //       "modelPath": "World\\Azeroth\\Buildings\\HumanFarm\\HumanFarm.wmo",
            //       "worldX": 2345.67,
            //       "worldY": -1234.56,
            //       "worldZ": 56.78,
            //       "flags": 0
            //     }
            //   ]
            // }
            
            // For testing, we'll just log that we tried to load
            // Real implementation will fetch JSON and call renderPlacements()
            
        } catch (error) {
            console.warn(`[WMOPlugin] Failed to load tile ${row}_${col}:`, error);
        }
    }
    
    renderPlacements(placements, row, col) {
        placements.forEach(placement => {
            // Convert world coordinates to lat/lng
            const tile = this.coords.worldToTile(placement.worldX, placement.worldY);
            const latLng = this.coords.tileToLatLng(tile.row, tile.col);
            
            // Use elevation for visual cues
            const elevationColor = this.coords.getElevationColor(placement.worldZ, this.color);
            const squareSize = this.getScaledSquareSize();
            
            // Create square bounds
            const bounds = [
                [latLng.lat - squareSize, latLng.lng - squareSize],
                [latLng.lat + squareSize, latLng.lng + squareSize]
            ];
            
            const square = L.rectangle(bounds, {
                color: '#000',
                weight: 1,
                fillColor: elevationColor,
                fillOpacity: 0.85
            });
            
            // Add popup with details
            const popupContent = `
                <div style="min-width: 250px;">
                    <strong>${placement.modelPath || 'WMO Object'}</strong><br>
                    <div style="margin-top: 8px; font-size: 12px;">
                        <strong>UID:</strong> ${placement.uniqueId || 'N/A'}<br>
                        <strong>Position:</strong><br>
                        &nbsp;&nbsp;X: ${placement.worldX.toFixed(2)}<br>
                        &nbsp;&nbsp;Y: ${placement.worldY.toFixed(2)}<br>
                        &nbsp;&nbsp;Z: ${placement.worldZ.toFixed(2)}<br>
                        <strong>Elevation:</strong> ${placement.worldZ.toFixed(2)} yards<br>
                        ${placement.flags ? `<strong>Flags:</strong> 0x${placement.flags.toString(16).padStart(4, '0')}` : ''}
                    </div>
                </div>
            `;
            
            square.bindPopup(popupContent, {
                maxWidth: 350,
                closeButton: true
            });
            
            square.addTo(this.map);
            this.layers.push(square);
        });
    }
    
    getVisibleTiles(bounds) {
        const sw = this.coords.latLngToTile(bounds.getSouth(), bounds.getWest());
        const ne = this.coords.latLngToTile(bounds.getNorth(), bounds.getEast());
        
        const tiles = [];
        for (let row = sw.row; row <= ne.row; row++) {
            for (let col = sw.col; col <= ne.col; col++) {
                if (row >= 0 && row < 64 && col >= 0 && col < 64) {
                    tiles.push({ row, col });
                }
            }
        }
        return tiles;
    }
    
    getScaledSquareSize() {
        const zoom = this.map.getZoom();
        // Scale from 0.7x at zoom 0 to 2.5x at zoom 12
        const scale = 0.7 + (zoom / 12) * 1.8;
        return this.baseSize * scale;
    }
    
    onDisable() {
        super.onDisable();
        this.loadedTiles.clear();
    }
    
    setColor(color) {
        this.color = color;
        // TODO: Update existing markers
    }
    
    setBaseSize(size) {
        this.baseSize = size;
        // TODO: Update existing markers
    }
}
