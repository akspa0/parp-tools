// Liquids Overlay Layer
// Renders river, ocean, magma, and slime chunks

export class LiquidsLayer {
    constructor(map) {
        this.map = map;
        this.layerGroup = L.layerGroup();
        this.visible = false;
        this.options = {
            showRiver: true,
            showOcean: true,
            showMagma: true,
            showSlime: true,
            opacity: 0.5
        };
        
        // Color scheme for liquid types
        this.liquidColors = {
            river: { color: '#40A4DF', fillColor: '#40A4DF', label: 'River' },
            ocean: { color: '#0040A4', fillColor: '#0040A4', label: 'Ocean' },
            magma: { color: '#FF4500', fillColor: '#FF4500', label: 'Magma' },
            slime: { color: '#00FF00', fillColor: '#00FF00', label: 'Slime' }
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

    render(liquidsData, tileRow, tileCol) {
        if (!liquidsData || !this.visible) return;

        // Render each liquid type
        for (const [liquidType, chunks] of Object.entries(liquidsData)) {
            if (liquidType === 'version') continue; // Skip version field
            
            const optionKey = `show${liquidType.charAt(0).toUpperCase() + liquidType.slice(1)}`;
            if (!this.options[optionKey] || !chunks || chunks.length === 0) continue;

            const colors = this.liquidColors[liquidType];
            if (!colors) continue;

            chunks.forEach(chunk => {
                this.renderChunk(chunk, tileRow, tileCol, {
                    ...colors,
                    fillOpacity: this.options.opacity,
                    weight: 1
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
            <b>${style.label} Liquid</b><br>
            Tile: ${tileRow}, ${tileCol}<br>
            Chunk: ${chunk.row}, ${chunk.col}
        `);

        this.layerGroup.addLayer(rect);
    }

    getChunkBounds(tileRow, tileCol, chunkRow, chunkCol) {
        const chunkSize = 32 / 512; // Normalized to 512-pixel minimap
        
        const north = 63 - tileRow - (chunkRow * chunkSize);
        const south = north - chunkSize;
        const west = tileCol + (chunkCol * chunkSize);
        const east = west + chunkSize;
        
        return [[south, west], [north, east]];
    }

    setOpacity(opacity) {
        this.options.opacity = opacity;
    }

    setOption(key, value) {
        this.options[key] = value;
    }
}
