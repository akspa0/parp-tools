import { OverlayPlugin } from '../core/OverlayPlugin.js';
/**
 * GridPlugin - Displays ADT tile grid overlay
 * Simple plugin with no data loading - just draws grid lines
 */
export class GridPlugin extends OverlayPlugin {
    constructor(map, coordSystem, options = {}) {
        super('grid', 'ADT Grid', map, coordSystem);
        
        this.gridColor = options.gridColor || '#00FF00';
        this.gridWeight = options.gridWeight || 1;
        this.showTileLabels = options.showTileLabels || false; // Default false - labels are for debugging only
        this.opacity = options.opacity || 1.0;
        this.zIndex = options.zIndex || 400;
        
        this.layers = [];
        this.gridLayer = null;
    }
    
    async onLoad(version, mapName) {
        console.log('[GridPlugin] Loaded');
    }
    
    onShow() {
        super.onShow();
        if (this.gridLayer) {
            this.gridLayer.addTo(this.map);
        }
    }
    
    onHide() {
        super.onHide();
        if (this.gridLayer) {
            this.gridLayer.remove();
        }
    }
    
    async loadVisibleData(bounds, zoom) {
        // Grid doesn't need to load data based on viewport
        // It's always the same 64x64 grid
    }
    
    renderGrid() {
        // Remove existing grid if any
        if (this.gridLayer) {
            this.gridLayer.remove();
        }
        
        // Create SVG overlay for the grid
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        svg.setAttribute('viewBox', '0 0 64 64');
        svg.style.width = '100%';
        svg.style.height = '100%';
        
        // Draw vertical lines (columns)
        for (let col = 0; col <= 64; col++) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', col);
            line.setAttribute('y1', 0);
            line.setAttribute('x2', col);
            line.setAttribute('y2', 64);
            line.setAttribute('stroke', this.gridColor);
            line.setAttribute('stroke-width', this.gridWeight / 64); // Scale to viewBox
            svg.appendChild(line);
        }
        
        // Draw horizontal lines (rows)
        for (let row = 0; row <= 64; row++) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', 0);
            line.setAttribute('y1', row);
            line.setAttribute('x2', 64);
            line.setAttribute('y2', row);
            line.setAttribute('stroke', this.gridColor);
            line.setAttribute('stroke-width', this.gridWeight / 64); // Scale to viewBox
            svg.appendChild(line);
        }
        
        // Add tile labels if enabled
        if (this.showTileLabels) {
            for (let svgRow = 0; svgRow < 64; svgRow++) {
                for (let svgCol = 0; svgCol < 64; svgCol++) {
                    // SVG position (svgRow, svgCol) corresponds to WoW tile (63-svgRow, svgCol)
                    // because tiles are Y-flipped in tileBounds()
                    const wowRow = 63 - svgRow;
                    const wowCol = svgCol;
                    
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', svgCol + 0.5);
                    text.setAttribute('y', svgRow + 0.5);
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('dominant-baseline', 'middle');
                    text.setAttribute('font-size', '0.3');
                    text.setAttribute('fill', '#888888');
                    text.setAttribute('class', 'tile-label');
                    text.textContent = `${wowRow}_${wowCol}`;
                    svg.appendChild(text);
                }
            }
        }
        
        // Create Leaflet SVG overlay
        const bounds = [[0, 0], [64, 64]];
        this.gridLayer = L.svgOverlay(svg, bounds, {
            opacity: this.opacity,
            interactive: false,
            pane: 'overlayPane'
        });
        
        this.layers.push(this.gridLayer);
        
        if (this.visible) {
            this.gridLayer.addTo(this.map);
        }
        
        console.log('[GridPlugin] Grid rendered');
    }
    
    setGridColor(color) {
        this.gridColor = color;
        this.renderGrid();
    }
    
    setGridWeight(weight) {
        this.gridWeight = weight;
        this.renderGrid();
    }
    
    toggleTileLabels(show) {
        this.showTileLabels = show;
        this.renderGrid();
    }
}
