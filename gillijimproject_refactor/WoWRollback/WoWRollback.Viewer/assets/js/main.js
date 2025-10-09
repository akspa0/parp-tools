import { CoordinateSystem } from './core/CoordinateSystem.js';
import { PluginManager } from './core/PluginManager.js';
import { GridPlugin } from './plugins/GridPlugin.js';
import { state } from './state.js';

let map;
let coordSystem;
let pluginManager;
let minimapLayer;

export async function init() {
    try {
        console.log('[Viewer] Starting initialization...');
        
        // Load config and index
        await state.loadConfig();
        await state.loadIndex();
        
        console.log('[Viewer] Config loaded:', state.config);
        console.log('[Viewer] Index loaded:', state.index);
        
        // Initialize coordinate system
        coordSystem = new CoordinateSystem(state.config);
        console.log('[Viewer] CoordinateSystem initialized');
        
        // Initialize map
        map = L.map('map', {
            crs: L.CRS.Simple,
            minZoom: 0,
            maxZoom: 12,
            zoom: 2
        });
        
        map.setView([32, 32], 2);
        console.log('[Viewer] Leaflet map initialized');
        
        // Add minimap tile layer
        minimapLayer = L.layerGroup();
        minimapLayer.addTo(map);
        console.log('[Viewer] Minimap layer added');
        
        // Initialize plugin manager
        pluginManager = new PluginManager(map, coordSystem);
        
        // Register plugins (only GridPlugin for now)
        const gridPlugin = new GridPlugin(map, coordSystem);
        pluginManager.register(gridPlugin);
        
        // Load all plugins
        await pluginManager.loadAll(state.selectedVersion, state.selectedMap);
        
        // Enable and show grid by default
        gridPlugin.onEnable();
        gridPlugin.onShow();
        
        // Setup UI
        setupUI();
        
        // Load initial minimap tiles
        loadMinimapTiles();
        
        // Reload tiles on map move
        map.on('moveend', () => {
            loadMinimapTiles();
        });
        
        console.log('[Viewer] Initialized successfully');
    } catch (error) {
        console.error('[Viewer] Initialization failed:', error);
        document.body.innerHTML = `
            <div style="padding: 20px; color: #F44336; background: #1a1a1a; font-family: monospace;">
                <h2>Failed to Load Viewer</h2>
                <p>Error: ${error.message}</p>
                <pre>${error.stack}</pre>
            </div>
        `;
    }
}

function setupUI() {
    // Grid toggle
    const gridToggle = document.getElementById('gridToggle');
    if (gridToggle) {
        gridToggle.addEventListener('change', (e) => {
            const plugin = pluginManager.get('grid');
            if (e.target.checked) {
                plugin.onEnable();
                plugin.onShow();
            } else {
                plugin.onHide();
                plugin.onDisable();
            }
            pluginManager.saveState();
        });
        gridToggle.checked = true; // Default enabled
    }
    
    console.log('[Viewer] UI setup complete');
}

function loadMinimapTiles() {
    const tiles = state.getTilesForMap(state.selectedMap);
    if (!tiles || tiles.length === 0) {
        console.log('[Viewer] No tiles available for map:', state.selectedMap);
        return;
    }
    
    const bounds = map.getBounds();
    const minRow = Math.max(0, Math.floor(bounds.getSouth()));
    const maxRow = Math.min(63, Math.ceil(bounds.getNorth()));
    const minCol = Math.max(0, Math.floor(bounds.getWest()));
    const maxCol = Math.min(63, Math.ceil(bounds.getEast()));
    
    console.log(`[Viewer] Loading tiles for rows ${minRow}-${maxRow}, cols ${minCol}-${maxCol}`);
    
    for (let row = minRow; row <= maxRow; row++) {
        for (let col = minCol; col <= maxCol; col++) {
            const tile = tiles.find(t => t.row === row && t.col === col);
            if (tile && tile.versions && tile.versions.includes(state.selectedVersion)) {
                const url = state.getMinimapPath(state.selectedMap, row, col, state.selectedVersion);
                const tileBounds = coordSystem.tileBounds(row, col);
                
                const overlay = L.imageOverlay(url, tileBounds, {
                    interactive: false,
                    opacity: 1.0
                });
                
                overlay.addTo(minimapLayer);
            }
        }
    }
}

// Auto-init
document.addEventListener('DOMContentLoaded', init);
