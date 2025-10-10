import { OverlayPlugin } from '../core/OverlayPlugin.js';

/**
 * M2Plugin - Displays M2 (doodad) object placements
 * Loads placement data from JSON and renders markers on the map
 */
export class M2Plugin extends OverlayPlugin {
    constructor(map, coordSystem, dataAdapter = null, options = {}) {
        super('m2', 'M2 Objects', map, coordSystem);
        
        this.dataAdapter = dataAdapter;
        this.color = options.color || '#FF00FF';
        this.baseRadius = options.baseRadius || 4;
        this.showLabels = options.showLabels !== false;
        this.opacity = options.opacity || 0.8;
        this.zIndex = options.zIndex || 600;
        this.clusterRadius = options.clusterRadius || 50;
        
        this.placements = new Map(); // tile -> placements
        this.loadedTiles = new Set();
    }
    
    async onLoad(version, mapName) {
        console.log(`[M2Plugin] Ready to load M2 placements for ${mapName} v${version}`);
        // M2 data will be loaded per-tile on demand via loadVisibleData
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
            let placements = [];
            
            // Use DataAdapter if available
            if (this.dataAdapter && this.dataAdapter.masterIndex) {
                const tilePlacements = this.dataAdapter.getTilePlacements(col, row, 'M2');
                placements = tilePlacements.map(p => this.dataAdapter.convertPlacement(p));
                console.log(`[M2Plugin] Loaded ${placements.length} M2 objects for tile ${row}_${col} from DataAdapter`);
            } else {
                // Fall back to loading from JSON files
                try {
                    const response = await fetch(`overlays/0.5.3/Azeroth/m2_placements_${row}_${col}.json`);
                    if (response.ok) {
                        const data = await response.json();
                        placements = data.placements || [];
                        console.log(`[M2Plugin] Loaded ${placements.length} M2 objects for tile ${row}_${col} from JSON`);
                    }
                } catch (fetchError) {
                    // Silently fail for missing tile data
                }
            }
            
            if (placements.length > 0) {
                this.renderPlacements(placements, row, col);
            }
            
        } catch (error) {
        }
    }
    
    renderPlacements(placements, row, col) {
        placements.forEach(placement => {
            // Use pixel coordinates (0-512 within tile) to calculate position
            // This matches the POC viewer approach
            const pixelX = placement.pixelX || 256; // Default to center if missing
            const pixelY = placement.pixelY || 256;
            const lat = row + (pixelY / 512);
            const lng = col + (pixelX / 512);
            
            const marker = L.circleMarker([lat, lng], {
                radius: 4,
                color: '#000',
                weight: 1,
                fillColor: '#2196F3', // Blue like POC viewer
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
        if (this.visible) {
            this.onHide();
            this.onShow();
        }
    }
    
    setBaseRadius(radius) {
        this.baseRadius = radius;
        if (this.visible) {
            this.onHide();
            this.onShow();
        }
    }
}
