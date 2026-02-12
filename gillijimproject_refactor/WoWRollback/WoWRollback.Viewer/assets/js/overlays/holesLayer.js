// Holes Overlay Layer
// Renders terrain holes as black rectangles in a 4×4 grid per chunk

export class HolesLayer {
    constructor(map) {
        this.map = map;
        this.layerGroup = L.layerGroup();
        this.visible = false;
        this.opacity = 0.7;
    }

    show() {
        if (!this.visible) {
            this.layerGroup.addTo(this.map);
            this.visible = true;
        }
    }

    hide() {
        if (this.visible) {
            this.map.removeLayer(this.layerGroup);
            this.visible = false;
        }
    }

    clear() {
        this.layerGroup.clearLayers();
    }

    render(holesData, tileRow, tileCol) {
        if (!holesData || !holesData.holes || !this.visible) return;

        holesData.holes.forEach(chunkHoles => {
            this.renderChunkHoles(chunkHoles, tileRow, tileCol);
        });
    }

    renderChunkHoles(chunkHoles, tileRow, tileCol) {
        const { row, col, type, holes } = chunkHoles;
        
        if (!holes || holes.length === 0) return;

        // Render each hole in the 4×4 grid
        holes.forEach(holeIndex => {
            this.renderHole(tileRow, tileCol, row, col, holeIndex, type);
        });
    }

    renderHole(tileRow, tileCol, chunkRow, chunkCol, holeIndex, type) {
        const bounds = this.getHoleBounds(tileRow, tileCol, chunkRow, chunkCol, holeIndex, type);
        
        const rect = L.rectangle(bounds, {
            color: '#000000',
            fillColor: '#000000',
            fillOpacity: this.opacity,
            weight: 1,
            interactive: true
        });

        rect.bindPopup(`
            <b>Terrain Hole</b><br>
            Tile: ${tileRow}, ${tileCol}<br>
            Chunk: ${chunkRow}, ${chunkCol}<br>
            Hole Index: ${holeIndex}<br>
            Type: ${type}
        `);

        this.layerGroup.addLayer(rect);
    }

    getHoleBounds(tileRow, tileCol, chunkRow, chunkCol, holeIndex, type) {
        const chunkSize = 32 / 512; // Normalized to 512-pixel minimap
        
        // For low-res holes, use 4×4 grid (16 holes per chunk)
        const holeSize = chunkSize / 4;
        
        // Calculate hole position within chunk (4×4 grid, row-major order)
        const holeRow = Math.floor(holeIndex / 4);
        const holeCol = holeIndex % 4;
        
        // Calculate chunk base position
        const chunkNorth = 63 - tileRow - (chunkRow * chunkSize);
        const chunkWest = tileCol + (chunkCol * chunkSize);
        
        // Calculate hole position
        const north = chunkNorth - (holeRow * holeSize);
        const south = north - holeSize;
        const west = chunkWest + (holeCol * holeSize);
        const east = west + holeSize;
        
        return [[south, west], [north, east]];
    }

    setOpacity(opacity) {
        this.opacity = opacity;
    }
}
