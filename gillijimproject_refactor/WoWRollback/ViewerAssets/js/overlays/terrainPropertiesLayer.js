// Terrain Properties Overlay Layer
// Renders impassible, vertex-colored, and multi-layer chunks

export class TerrainPropertiesLayer {
    constructor(map) {
        this.map = map;
        this.layerGroup = L.layerGroup();
        this.visible = false;
        this.options = {
            showImpassible: true,
            showVertexColored: true,
            showMultiLayer: true,
            opacity: 0.4
        };
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

    render(terrainData, tileRow, tileCol) {
        if (!terrainData || !this.visible) return;

        const { impassible, vertex_colored, multi_layer } = terrainData;

        // Render impassible chunks (red)
        if (this.options.showImpassible && impassible) {
            impassible.forEach(chunk => {
                this.renderChunk(chunk, tileRow, tileCol, {
                    color: '#F44336',
                    fillColor: '#F44336',
                    fillOpacity: this.options.opacity,
                    weight: 1,
                    title: 'Impassible'
                });
            });
        }

        // Render vertex-colored chunks (blue)
        if (this.options.showVertexColored && vertex_colored) {
            vertex_colored.forEach(chunk => {
                this.renderChunk(chunk, tileRow, tileCol, {
                    color: '#2196F3',
                    fillColor: '#2196F3',
                    fillOpacity: this.options.opacity,
                    weight: 1,
                    title: 'Vertex Colored (MCCV)'
                });
            });
        }

        // Render multi-layer chunks (yellow)
        if (this.options.showMultiLayer && multi_layer) {
            multi_layer.forEach(chunk => {
                this.renderChunk(chunk, tileRow, tileCol, {
                    color: '#FFC107',
                    fillColor: '#FFC107',
                    fillOpacity: this.options.opacity * 0.5, // Less opaque
                    weight: 2,
                    title: `${chunk.layers} Texture Layers`
                });
            });
        }
    }

    renderChunk(chunk, tileRow, tileCol, style) {
        const bounds = this.getChunkBounds(tileRow, tileCol, chunk.row, chunk.col);
        
        const rect = L.rectangle(bounds, {
            color: style.color,
            fillColor: style.fillColor,
            fillOpacity: style.fillOpacity,
            weight: style.weight,
            interactive: true
        });

        rect.bindPopup(`
            <b>${style.title}</b><br>
            Tile: ${tileRow}, ${tileCol}<br>
            Chunk: ${chunk.row}, ${chunk.col}<br>
            ${chunk.layers ? `Layers: ${chunk.layers}<br>` : ''}
        `);

        this.layerGroup.addLayer(rect);
    }

    getChunkBounds(tileRow, tileCol, chunkRow, chunkCol) {
        // Each tile is 64x64 game units
        // Each chunk is 32x32 game units (16x16 chunks per tile)
        // Leaflet coords: [lat, lng] where lat increases downward
        
        const chunkSize = 32 / 512; // Normalized to 512-pixel minimap
        
        // Calculate position
        const north = 63 - tileRow - (chunkRow * chunkSize);
        const south = north - chunkSize;
        const west = tileCol + (chunkCol * chunkSize);
        const east = west + chunkSize;
        
        return [[south, west], [north, east]];
    }

    setOpacity(opacity) {
        this.options.opacity = opacity;
        // Re-render if visible
        if (this.visible) {
            // Would need to store current data to re-render
            // For now, user needs to pan/zoom to update
        }
    }

    setOption(key, value) {
        this.options[key] = value;
    }
}
