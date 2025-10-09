/**
 * Canonical coordinate system for WoW maps
 * Handles 3D world coordinates with 2D projection
 * ALL overlays MUST use these transforms
 */
export class CoordinateSystem {
    constructor(config) {
        this.coordMode = config.coordMode || 'wowtools';
        
        // Constants
        this.TILE_SIZE = 533.33333;        // yards
        this.MAP_HALF_SIZE = 17066.66656;  // yards
        this.CHUNK_SIZE = 33.33333;        // yards
        this.CHUNKS_PER_TILE = 16;
        this.TILE_PIXEL_SIZE = 512;
        
        // Z-coordinate ranges (for visualization)
        this.MIN_ELEVATION = -500;  // Typical min (deep water)
        this.MAX_ELEVATION = 2000;  // Typical max (mountains)
    }
    
    // World → Tile
    worldToTile(worldX, worldY) {
        const row = Math.floor(32 - (worldY / this.TILE_SIZE));
        const col = Math.floor(32 - (worldX / this.TILE_SIZE));
        return { row, col };
    }
    
    // Tile → World (center of tile)
    tileToWorld(row, col) {
        // Tile (32,32) should map to world (0,0)
        // Each tile is TILE_SIZE yards, centered on its coordinates
        const worldX = (32 - col) * this.TILE_SIZE;
        const worldY = (32 - row) * this.TILE_SIZE;
        return { worldX, worldY };
    }
    
    // World → Chunk
    worldToChunk(worldX, worldY) {
        const chunkX = Math.floor((32 * 16) - (worldX / this.CHUNK_SIZE));
        const chunkY = Math.floor((32 * 16) - (worldY / this.CHUNK_SIZE));
        return { chunkX, chunkY };
    }
    
    // Tile + Pixel → World
    tilePixelToWorld(row, col, px, py) {
        const tileWorld = this.tileToWorld(row, col);
        const offsetX = (px / this.TILE_PIXEL_SIZE - 0.5) * this.TILE_SIZE;
        const offsetY = (py / this.TILE_PIXEL_SIZE - 0.5) * this.TILE_SIZE;
        return {
            worldX: tileWorld.worldX + offsetX,
            worldY: tileWorld.worldY + offsetY
        };
    }
    
    // Leaflet lat/lng → Tile
    latLngToTile(lat, lng) {
        const row = this.coordMode === 'wowtools' ? (63 - lat) : lat;
        const col = lng;
        return { row: Math.floor(row), col: Math.floor(col) };
    }
    
    // Tile → Leaflet lat/lng
    tileToLatLng(row, col) {
        const lat = this.coordMode === 'wowtools' ? (63 - row) : row;
        const lng = col;
        return { lat, lng };
    }
    
    // Leaflet bounds for tile
    tileBounds(row, col) {
        // WoW tile (row, col) needs Y-flip to match grid
        // WoW row 0 (North) → Leaflet lat 63-64
        // WoW row 63 (South) → Leaflet lat 0-1
        const lat1 = 63 - row;      // Top edge
        const lat2 = 64 - row;      // Bottom edge (one more)
        return [
            [Math.min(lat1, lat2), col],
            [Math.max(lat1, lat2), col + 1]
        ];
    }
    
    /**
     * Convert WoW world coordinates to Leaflet lat/lng
     * WoW: X-axis=North↔South (positive=North), Y-axis=West↔East (positive=West)
     * worldX = worldWest (WoW Y-axis)
     * worldY = worldNorth (WoW X-axis)
     * Formula from WoWDev: tileIndex = floor((32 - (coordinate / 533.33333)))
     */
    worldToLatLng(worldX, worldY) {
        // Calculate which tile the coordinates fall in
        const wowTileX = Math.floor(32 - (worldY / this.TILE_SIZE)); // North-South tile index
        const wowTileY = Math.floor(32 - (worldX / this.TILE_SIZE)); // West-East tile index
        
        // Calculate the north/west edge coordinates of this tile
        const tileNorthEdge = (32 - wowTileX) * this.TILE_SIZE;
        const tileWestEdge = (32 - wowTileY) * this.TILE_SIZE;
        
        // Calculate offset from tile edges (in yards)
        const offsetFromNorth = tileNorthEdge - worldY;
        const offsetFromWest = tileWestEdge - worldX;
        
        // Convert to fraction (0-1 within tile)
        const fractionY = offsetFromNorth / this.TILE_SIZE;
        const fractionX = offsetFromWest / this.TILE_SIZE;
        
        // Leaflet coordinate with Y-flip: row 0 = top (North)
        const row = 63 - wowTileX;
        const col = wowTileY;
        
        return {
            lat: row + fractionY,
            lng: col + fractionX
        };
    }
    
    // Elevation visualization helpers
    normalizeElevation(z) {
        // Normalize Z to 0-1 range for visualization
        return (z - this.MIN_ELEVATION) / (this.MAX_ELEVATION - this.MIN_ELEVATION);
    }
    
    getElevationColor(z, baseColor) {
        // Adjust color brightness based on elevation
        const normalized = this.normalizeElevation(z);
        const brightness = 0.5 + (normalized * 0.5); // 50-100% brightness
        return this.adjustColorBrightness(baseColor, brightness);
    }
    
    getElevationRadius(z, baseRadius) {
        // Adjust marker size based on elevation
        const normalized = this.normalizeElevation(z);
        return baseRadius * (0.7 + normalized * 0.6); // 70-130% of base
    }
    
    adjustColorBrightness(hex, brightness) {
        const rgb = parseInt(hex.slice(1), 16);
        const r = Math.floor(((rgb >> 16) & 0xFF) * brightness);
        const g = Math.floor(((rgb >> 8) & 0xFF) * brightness);
        const b = Math.floor((rgb & 0xFF) * brightness);
        return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
    }
}
