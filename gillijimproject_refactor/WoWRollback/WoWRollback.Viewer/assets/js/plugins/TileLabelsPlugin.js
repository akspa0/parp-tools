import { OverlayPlugin } from '../core/OverlayPlugin.js';

/**
 * TileLabelsPlugin - Displays clickable tile labels (e.g., "32_32")
 * Shows WoW tile coordinates in a readable format
 */
export class TileLabelsPlugin extends OverlayPlugin {
    constructor(map, coordSystem, options = {}) {
        super('tile-labels', 'Tile Labels', map, coordSystem);
        
        this.labelColor = options.labelColor || '#FFFFFF';
        this.labelSize = options.labelSize || '14px';
        this.labelOpacity = options.labelOpacity || 0.7;
        this.showBackground = options.showBackground !== false;
        this.backgroundColor = options.backgroundColor || 'rgba(0, 0, 0, 0.5)';
        this.zIndex = options.zIndex || 500;
        
        this.labels = [];
    }
    
    async onLoad(version, mapName) {
        console.log('[TileLabelsPlugin] Loaded');
    }
    
    onShow() {
        super.onShow();
        this.renderLabels();
    }
    
    onHide() {
        super.onHide();
        this.clearLabels();
    }
    
    async loadVisibleData(bounds, zoom) {
        // Labels are always rendered for all tiles
        // Could optimize to only show labels in viewport if needed
    }
    
    renderLabels() {
        this.clearLabels();
        
        // Create a label for each tile
        for (let row = 0; row < 64; row++) {
            for (let col = 0; col < 64; col++) {
                const bounds = this.coordSystem.tileBounds(row, col);
                const center = [
                    (bounds[0][0] + bounds[1][0]) / 2,
                    (bounds[0][1] + bounds[1][1]) / 2
                ];
                
                // Create div marker for the label
                const labelDiv = document.createElement('div');
                labelDiv.className = 'tile-label-marker';
                labelDiv.textContent = `${row}_${col}`;
                labelDiv.style.cssText = `
                    color: ${this.labelColor};
                    font-size: ${this.labelSize};
                    font-weight: bold;
                    text-align: center;
                    pointer-events: auto;
                    cursor: pointer;
                    user-select: none;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                    ${this.showBackground ? `
                        background: ${this.backgroundColor};
                        padding: 2px 6px;
                        border-radius: 3px;
                    ` : ''}
                `;
                
                // Add click handler
                labelDiv.addEventListener('click', () => {
                    console.log(`[TileLabelsPlugin] Clicked tile: ${row}_${col}`);
                    this.onTileClick(row, col);
                });
                
                // Create Leaflet marker
                const marker = L.marker(center, {
                    icon: L.divIcon({
                        html: labelDiv.outerHTML,
                        className: 'tile-label-icon',
                        iconSize: null,
                        iconAnchor: [0, 0]
                    }),
                    interactive: true
                });
                
                marker.addTo(this.map);
                this.labels.push(marker);
            }
        }
        
        console.log(`[TileLabelsPlugin] Rendered ${this.labels.length} labels`);
    }
    
    clearLabels() {
        this.labels.forEach(label => label.remove());
        this.labels = [];
    }
    
    onTileClick(row, col) {
        // Emit event that can be handled by the main app
        const event = new CustomEvent('tile-clicked', {
            detail: { row, col }
        });
        window.dispatchEvent(event);
        
        // For now, just center the map on the clicked tile
        const bounds = this.coordSystem.tileBounds(row, col);
        const center = [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2
        ];
        this.map.setView(center, this.map.getZoom());
    }
    
    setLabelColor(color) {
        this.labelColor = color;
        this.renderLabels();
    }
    
    setLabelSize(size) {
        this.labelSize = size;
        this.renderLabels();
    }
}
