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

        const basePath = `/overlays/${this.version}/${this.mapName}/shadow_map/tile_r${row}_c${col}`;
        const metadataUrl = `${basePath}.json`;

        try {
            const metadataResponse = await fetch(metadataUrl, { cache: 'no-store' });
            if (!metadataResponse.ok) {
                return null;
            }

            const metadata = await metadataResponse.json();
            const imageUrl = `/overlays/${this.version}/${this.mapName}/shadow_map/${metadata.overview}`;
            const bounds = this.calculateTileBounds(row, col);
            const overlay = L.imageOverlay(imageUrl, bounds, {
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
