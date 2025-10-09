import { OverlayPlugin } from '../core/OverlayPlugin.js';

/**
 * DensityHeatmapPlugin - Visualizes object density per chunk
 * Shows heat colors based on object count in each chunk
 */
export class DensityHeatmapPlugin extends OverlayPlugin {
    constructor(map, coordSystem, dataAdapter, options = {}) {
        super('density-heatmap', 'Density Heatmap', map, coordSystem);
        
        this.dataAdapter = dataAdapter;
        this.minColor = options.minColor || '#00FF0020';
        this.maxColor = options.maxColor || '#FF000080';
        this.threshold = options.threshold || 1; // Min objects to show
        this.showCounts = options.showCounts !== false;
        this.zIndex = options.zIndex || 420;
        
        this.visibleTiles = new Set();
        this.colorStops = this.generateColorGradient();
    }
    
    async onLoad(version, mapName) {
        console.log('[DensityHeatmapPlugin] Loaded');
    }
    
    async loadVisibleData(bounds, zoom) {
        if (!this.enabled || !this.visible) return;
        if (!this.dataAdapter || !this.dataAdapter.idRanges) {
            console.warn('[DensityHeatmapPlugin] No data adapter or ID ranges loaded');
            return;
        }
        
        const tiles = this.getVisibleTiles(bounds);
        const tileKeys = new Set(tiles.map(t => `${t.row}_${t.col}`));
        
        // Remove heatmaps from tiles that are no longer visible
        for (const key of this.visibleTiles) {
            if (!tileKeys.has(key)) {
                this.removeTileHeatmap(key);
                this.visibleTiles.delete(key);
            }
        }
        
        // Add heatmaps for newly visible tiles
        for (const tile of tiles) {
            const key = `${tile.row}_${tile.col}`;
            if (!this.visibleTiles.has(key)) {
                this.renderHeatmapForTile(tile.row, tile.col);
                this.visibleTiles.add(key);
            }
        }
    }
    
    renderHeatmapForTile(row, col) {
        // Get chunk density data
        const density = this.dataAdapter.getTileChunkDensity(col, row);
        if (!density) return;
        
        // Get tile bounds
        const tileBounds = this.coords.tileBounds(row, col);
        const [[minLat, minLng], [maxLat, maxLng]] = tileBounds;
        
        // Calculate chunk size
        const chunkLatSize = (maxLat - minLat) / 16;
        const chunkLngSize = (maxLng - minLng) / 16;
        
        // Find max density for normalization
        let maxDensity = 0;
        density.forEach(rowData => {
            rowData.forEach(count => {
                if (count > maxDensity) maxDensity = count;
            });
        });
        
        if (maxDensity === 0) return; // No objects in this tile
        
        // Render heat rectangles for each chunk
        for (let chunkY = 0; chunkY < 16; chunkY++) {
            for (let chunkX = 0; chunkX < 16; chunkX++) {
                const count = density[chunkY][chunkX];
                
                if (count < this.threshold) continue;
                
                // Calculate chunk bounds
                const lat1 = minLat + chunkY * chunkLatSize;
                const lat2 = lat1 + chunkLatSize;
                const lng1 = minLng + chunkX * chunkLngSize;
                const lng2 = lng1 + chunkLngSize;
                
                // Get color based on density
                const color = this.getHeatColor(count, maxDensity);
                
                // Create rectangle
                const rect = L.rectangle(
                    [[lat1, lng1], [lat2, lng2]],
                    {
                        color: 'none',
                        fillColor: color,
                        fillOpacity: 0.6,
                        interactive: true
                    }
                );
                
                // Add popup with details
                rect.bindPopup(`
                    <div style="font-size: 12px;">
                        <strong>Chunk Density</strong><br>
                        Tile: ${row}_${col}<br>
                        Chunk: ${chunkX},${chunkY}<br>
                        <strong>Objects: ${count}</strong><br>
                        <em>(${((count / maxDensity) * 100).toFixed(1)}% of tile max)</em>
                    </div>
                `);
                
                rect.addTo(this.map);
                rect._tileKey = `${row}_${col}`;
                this.layers.push(rect);
                
                // Optionally add count label
                if (this.showCounts && count >= 5) {
                    const centerLat = (lat1 + lat2) / 2;
                    const centerLng = (lng1 + lng2) / 2;
                    
                    const labelDiv = document.createElement('div');
                    labelDiv.textContent = count;
                    labelDiv.style.cssText = `
                        color: #FFFFFF;
                        font-size: 10px;
                        font-weight: bold;
                        text-align: center;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.9);
                        pointer-events: none;
                    `;
                    
                    const marker = L.marker([centerLat, centerLng], {
                        icon: L.divIcon({
                            html: labelDiv.outerHTML,
                            className: 'density-label-icon',
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
    
    getHeatColor(count, maxCount) {
        // Normalize to 0-1
        const normalized = Math.min(count / maxCount, 1);
        
        // Use a color gradient: green (low) -> yellow -> orange -> red (high)
        if (normalized < 0.25) {
            return `rgba(0, 255, 0, ${0.2 + normalized * 0.8})`;
        } else if (normalized < 0.5) {
            return `rgba(128, 255, 0, ${0.4 + normalized * 0.6})`;
        } else if (normalized < 0.75) {
            return `rgba(255, 128, 0, ${0.5 + normalized * 0.5})`;
        } else {
            return `rgba(255, 0, 0, ${0.6 + normalized * 0.4})`;
        }
    }
    
    generateColorGradient() {
        // Pre-generate color stops for smooth gradient
        const stops = [];
        for (let i = 0; i <= 100; i++) {
            const ratio = i / 100;
            stops.push(this.interpolateColor(ratio));
        }
        return stops;
    }
    
    interpolateColor(ratio) {
        // Green -> Yellow -> Orange -> Red
        if (ratio < 0.33) {
            const local = ratio / 0.33;
            return `rgba(${Math.floor(local * 255)}, 255, 0, ${0.3 + ratio * 0.5})`;
        } else if (ratio < 0.66) {
            const local = (ratio - 0.33) / 0.33;
            return `rgba(255, ${Math.floor(255 - local * 127)}, 0, ${0.4 + ratio * 0.4})`;
        } else {
            const local = (ratio - 0.66) / 0.34;
            return `rgba(255, ${Math.floor(128 - local * 128)}, 0, ${0.5 + ratio * 0.4})`;
        }
    }
    
    removeTileHeatmap(tileKey) {
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
    
    setThreshold(value) {
        this.threshold = value;
        if (this.visible) {
            this.onHide();
            this.onShow();
        }
    }
    
    setShowCounts(show) {
        this.showCounts = show;
        if (this.visible) {
            this.onHide();
            this.onShow();
        }
    }
}
