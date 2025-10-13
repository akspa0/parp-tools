// AreaID Overlay Layer
// Renders area boundaries as lines and optionally fills areas with color

export class AreaIdLayer {
    constructor(map) {
        this.map = map;
        this.layerGroup = L.layerGroup();
        this.visible = false;
        this.options = {
            showBoundaries: true,
            showLabels: true,
            showFill: false,
            lineOpacity: 0.8,
            fillOpacity: 0.1
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

    render(areaData, tileRow, tileCol) {
        if (!areaData || !this.visible) return;

        const { chunks, boundaries } = areaData;

        // Render area fills (optional)
        if (this.options.showFill && chunks) {
            this.renderAreaFills(chunks, tileRow, tileCol);
        }

        // Render boundaries
        if (this.options.showBoundaries && boundaries) {
            this.renderBoundaries(boundaries, tileRow, tileCol);
        }

        // Render labels (optional)
        if (this.options.showLabels && chunks) {
            this.renderLabels(chunks, tileRow, tileCol);
        }
    }

    renderAreaFills(chunks, tileRow, tileCol) {
        // Group chunks by area
        const byArea = new Map();
        chunks.forEach(chunk => {
            if (!byArea.has(chunk.area_id)) {
                byArea.set(chunk.area_id, {
                    name: chunk.area_name,
                    chunks: []
                });
            }
            byArea.get(chunk.area_id).chunks.push(chunk);
        });

        // Render each area
        byArea.forEach((area, areaId) => {
            const color = this.hashColor(areaId);
            
            area.chunks.forEach(chunk => {
                const bounds = this.getChunkBounds(tileRow, tileCol, chunk.row, chunk.col);
                
                const rect = L.rectangle(bounds, {
                    color: color,
                    fillColor: color,
                    fillOpacity: this.options.fillOpacity,
                    weight: 0,
                    interactive: false
                });

                this.layerGroup.addLayer(rect);
            });
        });
    }

    renderBoundaries(boundaries, tileRow, tileCol) {
        boundaries.forEach(boundary => {
            const line = this.getBoundaryLine(
                tileRow, tileCol,
                boundary.chunk_row, boundary.chunk_col,
                boundary.edge
            );

            const polyline = L.polyline(line, {
                color: '#FFD700', // Gold color
                weight: 3,
                opacity: this.options.lineOpacity,
                interactive: true
            });

            polyline.bindPopup(`
                <b>Area Boundary</b><br>
                From: ${boundary.from_name} (${boundary.from_area})<br>
                To: ${boundary.to_name} (${boundary.to_area})<br>
                Edge: ${boundary.edge}
            `);

            this.layerGroup.addLayer(polyline);
        });
    }

    renderLabels(chunks, tileRow, tileCol) {
        // Group by area and find center of each area
        const byArea = new Map();
        chunks.forEach(chunk => {
            if (!byArea.has(chunk.area_id)) {
                byArea.set(chunk.area_id, {
                    name: chunk.area_name,
                    chunks: [],
                    sumRow: 0,
                    sumCol: 0
                });
            }
            const area = byArea.get(chunk.area_id);
            area.chunks.push(chunk);
            area.sumRow += chunk.row;
            area.sumCol += chunk.col;
        });

        // Render label at center of each area
        byArea.forEach((area, areaId) => {
            const centerRow = area.sumRow / area.chunks.length;
            const centerCol = area.sumCol / area.chunks.length;
            
            const center = this.getChunkCenter(tileRow, tileCol, centerRow, centerCol);
            
            const label = L.marker(center, {
                icon: L.divIcon({
                    className: 'area-label',
                    html: `<div style="
                        background: rgba(0,0,0,0.7);
                        color: #FFD700;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-size: 11px;
                        font-weight: bold;
                        white-space: nowrap;
                        border: 1px solid #FFD700;
                    ">${area.name}</div>`,
                    iconSize: [null, null]
                }),
                interactive: false
            });

            this.layerGroup.addLayer(label);
        });
    }

    getBoundaryLine(tileRow, tileCol, chunkRow, chunkCol, edge) {
        const chunkSize = 32 / 512;
        
        const chunkNorth = 63 - tileRow - (chunkRow * chunkSize);
        const chunkSouth = chunkNorth - chunkSize;
        const chunkWest = tileCol + (chunkCol * chunkSize);
        const chunkEast = chunkWest + chunkSize;

        switch (edge) {
            case 'north':
                return [[chunkNorth, chunkWest], [chunkNorth, chunkEast]];
            case 'east':
                return [[chunkNorth, chunkEast], [chunkSouth, chunkEast]];
            case 'south':
                return [[chunkSouth, chunkWest], [chunkSouth, chunkEast]];
            case 'west':
                return [[chunkNorth, chunkWest], [chunkSouth, chunkWest]];
            default:
                return [];
        }
    }

    getChunkBounds(tileRow, tileCol, chunkRow, chunkCol) {
        const chunkSize = 32 / 512;
        
        const north = 63 - tileRow - (chunkRow * chunkSize);
        const south = north - chunkSize;
        const west = tileCol + (chunkCol * chunkSize);
        const east = west + chunkSize;
        
        return [[south, west], [north, east]];
    }

    getChunkCenter(tileRow, tileCol, chunkRow, chunkCol) {
        const chunkSize = 32 / 512;
        
        const north = 63 - tileRow - (chunkRow * chunkSize);
        const south = north - chunkSize;
        const west = tileCol + (chunkCol * chunkSize);
        const east = west + chunkSize;
        
        return [(north + south) / 2, (west + east) / 2];
    }

    hashColor(areaId) {
        // Generate consistent color from area ID
        const hue = (areaId * 137.508) % 360; // Golden angle
        return `hsl(${hue}, 70%, 50%)`;
    }

    setOption(key, value) {
        this.options[key] = value;
    }
}
