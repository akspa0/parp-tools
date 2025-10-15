/**
 * Shadow Map Overlay Layer
 * Renders baked terrain shadows from MCSH data
 */
export class ShadowMapLayer {
    constructor(map, mapName, version) {
        this.map = map;
        this.mapName = mapName;
        this.version = version;
        this.overlays = new Map();
        this.visible = false;
        this.opacity = 0.5;
    }

    async loadTile(row, col) {
        const key = `${row}_${col}`;
        if (this.overlays.has(key)) {
            return this.overlays.get(key);
        }

        const url = `/overlays/${this.version}/${this.mapName}/shadow_map/tile_r${row}_c${col}.json`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) {
                // No shadow data for this tile - that's okay
                return null;
            }
            
            const data = await response.json();
            
            // Create canvas for shadow rendering
            const canvas = this.renderShadowMap(data);
            
            // Convert to Leaflet ImageOverlay
            const bounds = this.calculateTileBounds(row, col);
            const overlay = L.imageOverlay(canvas.toDataURL(), bounds, {
                opacity: this.opacity,
                interactive: false,
                className: 'shadow-overlay'
            });
            
            this.overlays.set(key, overlay);
            
            if (this.visible) {
                overlay.addTo(this.map);
            }
            
            return overlay;
        } catch (err) {
            console.debug(`No shadow data for tile ${row},${col}`);
            return null;
        }
    }

    renderShadowMap(data) {
        // Create 1024×1024 canvas (16 chunks × 64 pixels each)
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');
        
        // Draw each chunk's shadow map
        data.chunks.forEach(chunk => {
            const chunkPixelX = chunk.x * 64;
            const chunkPixelY = chunk.y * 64;
            
            // Draw 64×64 shadow pixels
            const imageData = ctx.createImageData(64, 64);
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const index = (y * 64 + x) * 4;
                    const shadow = chunk.shadow[y][x];  // 0-5
                    
                    // Black with varying alpha
                    // 0 = opaque black (full shadow)
                    // 5 = transparent (no shadow/fully lit)
                    const alpha = 255 - (shadow * 51);  // 0→255, 5→0
                    
                    imageData.data[index + 0] = 0;      // R
                    imageData.data[index + 1] = 0;      // G
                    imageData.data[index + 2] = 0;      // B
                    imageData.data[index + 3] = alpha;  // A (inverted)
                }
            }
            
            ctx.putImageData(imageData, chunkPixelX, chunkPixelY);
        });
        
        return canvas;
    }

    calculateTileBounds(row, col) {
        // Calculate lat/lng bounds for this ADT tile
        // Matches minimap tile bounds
        const tileSize = 533.33333;
        const minLat = -(row * tileSize);
        const maxLat = -((row + 1) * tileSize);
        const minLng = col * tileSize;
        const maxLng = (col + 1) * tileSize;
        
        return [[maxLat, minLng], [minLat, maxLng]];
    }

    show() {
        this.visible = true;
        this.overlays.forEach(overlay => {
            if (overlay) overlay.addTo(this.map);
        });
    }

    hide() {
        this.visible = false;
        this.overlays.forEach(overlay => {
            if (overlay) overlay.remove();
        });
    }

    setOpacity(opacity) {
        this.opacity = opacity;
        this.overlays.forEach(overlay => {
            if (overlay) overlay.setOpacity(opacity);
        });
    }

    clear() {
        this.hide();
        this.overlays.clear();
    }
}
