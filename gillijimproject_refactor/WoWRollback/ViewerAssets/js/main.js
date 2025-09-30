// Main entry point for index.html - Leaflet Map Viewer
import { state } from './state.js';
import { loadOverlay } from './overlayLoader.js';

let map;
let tileLayer;
let objectMarkers = L.layerGroup();
let showObjects = true;

export async function init() {
    try {
        console.log('Initializing viewer...');
        await state.loadIndex();
        console.log('Index loaded:', state.index);
        await state.loadConfig();
        console.log('Config loaded:', state.config);
        
        initializeMap();
        setupUI();
        state.subscribe(onStateChange);
        updateTileLayer();
        console.log('Viewer initialized successfully');
    } catch (error) {
        console.error('Failed to initialize viewer:', error);
        document.body.innerHTML = `
            <div style="padding: 20px; color: #F44336; background: #1a1a1a; font-family: monospace;">
                <h2>Failed to Load Viewer</h2>
                <p>Error: ${error.message}</p>
                <p style="color: #9E9E9E;">Make sure you're running this via a web server (not file:// protocol)</p>
                <p style="color: #9E9E9E;">See README.md for instructions</p>
            </div>
        `;
    }
}

function initializeMap() {
    // Initialize Leaflet map with CRS.Simple for game coordinates
    map = L.map('map', {
        crs: L.CRS.Simple,
        minZoom: -2,
        maxZoom: 2,
        zoomControl: true
    });

    objectMarkers.addTo(map);
    
    // Set initial view
    map.setView([32, 32], 0);
}

function setupUI() {
    const versionSelect = document.getElementById('versionSelect');
    const mapSelect = document.getElementById('mapSelect');
    const showObjectsCheck = document.getElementById('showObjects');
    
    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // Populate version select
    state.index.versions.forEach(version => {
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version;
        option.selected = version === state.selectedVersion;
        versionSelect.appendChild(option);
    });

    // Populate map select
    state.index.maps.forEach(mapData => {
        const option = document.createElement('option');
        option.value = mapData.map;
        option.textContent = mapData.map;
        option.selected = mapData.map === state.selectedMap;
        mapSelect.appendChild(option);
    });

    versionSelect.addEventListener('change', (e) => {
        state.setVersion(e.target.value);
    });

    mapSelect.addEventListener('change', (e) => {
        state.setMap(e.target.value);
    });

    showObjectsCheck.addEventListener('change', (e) => {
        showObjects = e.target.checked;
        if (showObjects) {
            objectMarkers.addTo(map);
        } else {
            objectMarkers.remove();
        }
    });

    renderComparisonInfo();
}

function onStateChange() {
    updateTileLayer();
    updateObjectMarkers();
}

function renderComparisonInfo() {
    const info = document.getElementById('comparisonInfo');
    const compKey = state.index.comparisonKey;
    const versionCount = state.index.versions.length;
    const mapCount = state.index.maps.length;
    
    info.innerHTML = `
        <strong>${compKey}</strong> | 
        ${versionCount} version(s) | 
        ${mapCount} map(s)
    `;
}

function updateTileLayer() {
    if (tileLayer) {
        map.removeLayer(tileLayer);
    }

    const tiles = state.getTilesForMap(state.selectedMap);
    if (tiles.length === 0) return;

    const tileBounds = calculateBounds(tiles);

    // Create custom tile layer with strict bounds
    tileLayer = L.tileLayer.wms('', {
        tileSize: 512,
        noWrap: true,
        bounds: tileBounds,
        minNativeZoom: 0,
        maxNativeZoom: 0
    });

    // Override getTileUrl to use our custom tile structure
    tileLayer.getTileUrl = function(coords) {
        const row = 63 - coords.y; // Flip Y coordinate
        const col = coords.x;
        
        // Validate coordinates are within 0-63 range
        if (row < 0 || row > 63 || col < 0 || col > 63) {
            console.warn(`Tile coords out of bounds: row=${row}, col=${col}`);
            return null; // Return null for invalid tiles
        }
        
        const tilePath = state.getMinimapPath(state.selectedMap, row, col, state.selectedVersion);
        return tilePath;
    };

    tileLayer.on('tileload', function(e) {
        addTileLabel(e.tile, e.coords);
    });

    tileLayer.on('tileloadstart', function(e) {
        e.tile.style.border = '1px solid #444';
    });

    tileLayer.on('tileerror', function(e) {
        const row = 63 - e.coords.y;
        const col = e.coords.x;
        e.tile.style.backgroundColor = '#2a2a2a';
        e.tile.alt = `${row}_${col} (unavailable)`;
    });

    tileLayer.addTo(map);
    
    // Add click handler for tiles
    map.on('click', handleMapClick);
    
    // Fit bounds and set max bounds to prevent scrolling outside map
    // Convert plain array bounds to LatLngBounds for padding
    const latLngBounds = L.latLngBounds(tileBounds);
    map.setMaxBounds(latLngBounds.pad(0.5));
    map.fitBounds(tileBounds);
    
    updateObjectMarkers();
}

function calculateBounds(tiles) {
    const rows = tiles.map(t => t.row);
    const cols = tiles.map(t => t.col);
    const minRow = Math.min(...rows);
    const maxRow = Math.max(...rows);
    const minCol = Math.min(...cols);
    const maxCol = Math.max(...cols);

    return [[minRow, minCol], [maxRow + 1, maxCol + 1]];
}

function addTileLabel(tileElement, coords) {
    const row = 63 - coords.y;
    const col = coords.x;
    
    const label = document.createElement('div');
    label.className = 'tile-label';
    label.textContent = `${row}_${col}`;
    label.style.position = 'absolute';
    label.style.bottom = '5px';
    label.style.right = '5px';
    
    tileElement.parentElement.style.position = 'relative';
    tileElement.parentElement.appendChild(label);
}

async function updateObjectMarkers() {
    objectMarkers.clearLayers();
    
    if (!showObjects) return;

    const tiles = state.getTilesForMap(state.selectedMap);
    
    for (const tile of tiles) {
        try {
            const overlayPath = state.getOverlayPath(state.selectedMap, tile.row, tile.col);
            const data = await loadOverlay(overlayPath);
            
            const versionData = data.layers?.find(l => l.version === state.selectedVersion);
            if (!versionData || !versionData.kinds) continue;

            const objects = versionData.kinds.flatMap(kind => kind.points || []);
            
            objects.forEach(obj => {
                if (!obj.pixel || !obj.world || (obj.world.x === 0 && obj.world.y === 0 && obj.world.z === 0)) return;
                
                // Convert pixel coords to map coords
                const y = tile.row + (obj.pixel.y / 512);
                const x = tile.col + (obj.pixel.x / 512);
                
                const marker = L.circleMarker([y, x], {
                    radius: 3,
                    fillColor: '#2196F3',
                    color: '#fff',
                    weight: 1,
                    fillOpacity: 0.8
                }).bindPopup(`
                    <strong>${obj.fileName || 'Unknown'}</strong><br>
                    UID: ${obj.uniqueId || 'N/A'}<br>
                    World: (${obj.world.x.toFixed(1)}, ${obj.world.y.toFixed(1)}, ${obj.world.z.toFixed(1)})
                `);
                
                objectMarkers.addLayer(marker);
            });
        } catch (e) {
            // Skip tiles without overlay data
        }
    }
}

function handleMapClick(e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;
    
    const row = Math.floor(lat);
    const col = Math.floor(lng);
    
    const tiles = state.getTilesForMap(state.selectedMap);
    const tile = tiles.find(t => t.row === row && t.col === col);
    
    if (tile && state.isTileAvailable(state.selectedMap, row, col, state.selectedVersion)) {
        openTileViewer(state.selectedMap, row, col);
    }
}

function openTileViewer(map, row, col) {
    const url = `tile.html?map=${encodeURIComponent(map)}&row=${row}&col=${col}&version=${encodeURIComponent(state.selectedVersion)}`;
    window.location.href = url;
}
