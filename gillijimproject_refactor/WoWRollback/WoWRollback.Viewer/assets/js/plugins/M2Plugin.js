import { OverlayPlugin } from '../core/OverlayPlugin.js';

/**
 * M2Plugin - Displays M2 doodad placements
 * Loads data from JSON and renders as circle markers
 */
export class M2Plugin extends OverlayPlugin {
    constructor(map, coordSystem) {
        super('m2', 'M2 Doodads', map, coordSystem);
        
        this.color = '#2196F3';
        this.baseRadius = 5;
        this.data = null;
        this.loadedTiles = new Set();
    }
    
    async onLoad(version, mapName) {
        // M2 data will be loaded per-tile on demand
        console.log(`[M2Plugin] Ready to load data for ${mapName} (${version})`);
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
            console.log(`[M2Plugin] Loading tile ${row}_${col}`);
            
            // Example data structure:
            // {
            //   "placements": [
            //     {
            //       "uniqueId": 12345,
            //       "modelPath": "World\\Azeroth\\Elwynn\\PassiveObjects\\Trees\\ElwynnTree01.m2",
            //       "worldX": 1234.56,
            //       "worldY": -789.12,
            //       "worldZ": 42.34,
            //       "flags": 0
            //     }
            //   ]
            // }
            
            // For testing, we'll just log that we tried to load
            // Real implementation will fetch JSON and call renderPlacements()
            
        } catch (error) {
            console.warn(`[M2Plugin] Failed to load tile ${row}_${col}:`, error);
        }
    }
    
    renderPlacements(placements, row, col) {
        placements.forEach(placement => {
            // Convert world coordinates to lat/lng
            const tile = this.coords.worldToTile(placement.worldX, placement.worldY);
            const latLng = this.coords.tileToLatLng(tile.row, tile.col);
            
            // Adjust for position within tile (if we have pixel coords)
            // For now, use tile center
            
            // Use elevation for visual cues
            const elevationColor = this.coords.getElevationColor(placement.worldZ, this.color);
            const elevationRadius = this.coords.getElevationRadius(placement.worldZ, this.baseRadius);
            
            const marker = L.circleMarker([latLng.lat, latLng.lng], {
                radius: elevationRadius,
                color: '#000',
                weight: 1,
                fillColor: elevationColor,
                fillOpacity: 0.8
            });
            
            // Add popup with details
            const popupContent = `
                <div style="min-width: 250px;">
                    <strong>${placement.modelPath || 'M2 Doodad'}</strong><br>
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
            
            marker.bindPopup(popupContent, {
                maxWidth: 350,
                closeButton: true
            });
            
            marker.addTo(this.map);
            this.layers.push(marker);
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
    
    onDisable() {
        super.onDisable();
        this.loadedTiles.clear();
    }
    
    setColor(color) {
        this.color = color;
        // TODO: Update existing markers
    }
    
    setBaseRadius(radius) {
        this.baseRadius = radius;
        // TODO: Update existing markers
    }
}
