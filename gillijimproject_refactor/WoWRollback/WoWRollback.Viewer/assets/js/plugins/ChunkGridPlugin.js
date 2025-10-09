import { OverlayPlugin } from '../core/OverlayPlugin.js';

/**
 * ChunkGridPlugin - Displays 16×16 chunk grid within each tile
 * Each tile has 256 chunks (16×16), each chunk is 33.33 yards
 */
export class ChunkGridPlugin extends OverlayPlugin {
    constructor(map, coordSystem, options = {}) {
        super('chunk-grid', 'Chunk Grid', map, coordSystem);
        
        this.gridColor = options.gridColor || '#00FFFF';
        this.gridWeight = options.gridWeight || 0.5;
        this.gridOpacity = options.gridOpacity || 0.4;
        this.showLabels = options.showLabels || false;
        this.labelColor = options.labelColor || '#00FFFF';
        this.zIndex = options.zIndex || 450;
        
        this.visibleTiles = new Set();
    }
    
    async onLoad(version, mapName) {
        console.log('[ChunkGridPlugin] Loaded');
    }
    
    async loadVisibleData(bounds, zoom) {
        if (!this.enabled || !this.visible) return;
        
        // Get visible tiles
        const tiles = this.getVisibleTiles(bounds);
        const tileKeys = new Set(tiles.map(t => `${t.row}_${t.col}`));
        
        // Remove chunks from tiles that are no longer visible
        for (const key of this.visibleTiles) {
            if (!tileKeys.has(key)) {
                this.removeTileChunks(key);
                this.visibleTiles.delete(key);
            }
        }
        
        // Add chunks for newly visible tiles
        for (const tile of tiles) {
            const key = `${tile.row}_${tile.col}`;
            if (!this.visibleTiles.has(key)) {
                this.renderChunksForTile(tile.row, tile.col);
                this.visibleTiles.add(key);
            }
        }
    }
    
    renderChunksForTile(row, col) {
        // Get tile bounds in lat/lng
        const tileBounds = this.coords.tileBounds(row, col);
        const [[minLat, minLng], [maxLat, maxLng]] = tileBounds;
        
        // Calculate chunk size in lat/lng units
        const chunkLatSize = (maxLat - minLat) / 16;
        const chunkLngSize = (maxLng - minLng) / 16;
        
        // Draw 16×16 grid
        for (let chunkRow = 0; chunkRow <= 16; chunkRow++) {
            // Horizontal lines
            const lat = minLat + chunkRow * chunkLatSize;
            const line = L.polyline(
                [[lat, minLng], [lat, maxLng]],
                {
                    color: this.gridColor,
                    weight: this.gridWeight,
                    opacity: this.gridOpacity,
                    interactive: false
                }
            );
            line.addTo(this.map);
            line._tileKey = `${row}_${col}`;
            this.layers.push(line);
        }
        
        for (let chunkCol = 0; chunkCol <= 16; chunkCol++) {
            // Vertical lines
            const lng = minLng + chunkCol * chunkLngSize;
            const line = L.polyline(
                [[minLat, lng], [maxLat, lng]],
                {
                    color: this.gridColor,
                    weight: this.gridWeight,
                    opacity: this.gridOpacity,
                    interactive: false
                }
            );
            line.addTo(this.map);
            line._tileKey = `${row}_${col}`;
            this.layers.push(line);
        }
        
        // Optionally add chunk coordinate labels
        if (this.showLabels) {
            for (let chunkRow = 0; chunkRow < 16; chunkRow++) {
                for (let chunkCol = 0; chunkCol < 16; chunkCol++) {
                    const lat = minLat + (chunkRow + 0.5) * chunkLatSize;
                    const lng = minLng + (chunkCol + 0.5) * chunkLngSize;
                    
                    const labelDiv = document.createElement('div');
                    labelDiv.textContent = `${chunkRow},${chunkCol}`;
                    labelDiv.style.cssText = `
                        color: ${this.labelColor};
                        font-size: 10px;
                        text-align: center;
                        opacity: ${this.gridOpacity};
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                        pointer-events: none;
                    `;
                    
                    const marker = L.marker([lat, lng], {
                        icon: L.divIcon({
                            html: labelDiv.outerHTML,
                            className: 'chunk-label-icon',
                            iconSize: null,
                            iconAnchor: [0, 0]
                        }),
                        interactive: false
                    });
                    
                    marker.addTo(this.map);
                    marker._tileKey = `${row}_${col}`;
                    this.layers.push(marker);
                }
            }
        }
    }
    
    removeTileChunks(tileKey) {
        // Remove all layers associated with this tile
        this.layers = this.layers.filter(layer => {
            if (layer._tileKey === tileKey) {
                layer.remove();
                return false;
            }
            return true;
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
        this.visibleTiles.clear();
    }
    
    setGridColor(color) {
        this.gridColor = color;
        if (this.visible) {
            this.onHide();
            this.onShow();
        }
    }
    
    setShowLabels(show) {
        this.showLabels = show;
        if (this.visible) {
            this.onHide();
            this.onShow();
        }
    }
}
