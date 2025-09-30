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
    // Bounds represent the full 64x64 tile grid (0-63 range)
    const bounds = [[0, 0], [64, 64]];
    
    map = L.map('map', {
        crs: L.CRS.Simple,
        minZoom: 0,
        maxZoom: 5,
        zoom: 0,
        maxBounds: bounds,
        maxBoundsViscosity: 1.0, // Prevent dragging outside bounds
        zoomControl: true
    });

    objectMarkers.addTo(map);
    
    // Set initial view to center of map at zoom level where tiles align
    map.setView([32, 32], 0);
    console.log('Map initialized with bounds:', bounds);
    console.log('Initial view: [32,32] at zoom 0');
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
    if (tiles.length === 0) {
        console.warn('No tiles found for map:', state.selectedMap);
        return;
    }

    console.log(`Loading ${tiles.length} tiles for ${state.selectedMap}, version ${state.selectedVersion}`);

    // Use simple tileLayer with URL template
    // At zoom 6, with tileSize 512, each Leaflet tile = 1 game tile (64x64 grid)
    // This is because at zoom 6: scale = 2^6 = 64, and 512/scale = 8 pixels per coordinate unit
    // But we want 512 pixels per game tile, so we use zoom where this works out
    const urlTemplate = `minimap/${state.selectedVersion}/${state.selectedMap}/${state.selectedMap}_{x}_{y}.png`;
    
    console.log(`Tile URL template: ${urlTemplate}`);

    // Create a custom tile layer that bypasses Leaflet's coordinate system
    const CustomTileLayer = L.GridLayer.extend({
        createTile: function(coords) {
            const tile = document.createElement('img');
            const row = coords.y;
            const col = coords.x;
            
            // At zoom 6, Leaflet uses native tile coordinates
            const url = state.getMinimapPath(state.selectedMap, row, col, state.selectedVersion);
            tile.src = url;
            tile.style.width = '512px';
            tile.style.height = '512px';
            
            tile.onerror = function() {
                tile.style.backgroundColor = '#1a1a1a';
            };
            
            return tile;
        }
    });

    tileLayer = new CustomTileLayer({
        tileSize: 512,
        noWrap: true,
        bounds: [[0, 0], [64, 64]],
        minZoom: 0,
        maxZoom: 5,
        minNativeZoom: 6,
        maxNativeZoom: 6
    });

    tileLayer.on('tileload', function(e) {
        const coords = e.coords;
        console.log(`✓ Loaded tile [${coords.y},${coords.x}]`);
        addTileLabel(e.tile, coords);
    });

    tileLayer.on('tileerror', function(e) {
        const coords = e.coords;
        console.warn(`✗ Failed tile [${coords.y},${coords.x}]`);
        e.tile.style.backgroundColor = '#2a2a2a';
        e.tile.alt = `${coords.y}_${coords.x}`;
    });

    tileLayer.on('tileloadstart', function(e) {
        const coords = e.coords;
        console.log(`→ Requesting tile [${coords.y},${coords.x}]: ${urlTemplate.replace('{x}', coords.x).replace('{y}', coords.y)}`);
    });

    tileLayer.addTo(map);
    
    // Add click handler for tiles
    map.off('click', handleMapClick);
    map.on('click', handleMapClick);
    
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
    const row = coords.y;
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
    
    // Clamp to 0-63 range
    const row = Math.max(0, Math.min(63, Math.floor(lat)));
    const col = Math.max(0, Math.min(63, Math.floor(lng)));
    
    // Update sidebar click info
    document.getElementById('clickedTile').textContent = `${row}_${col}`;
    document.getElementById('clickedCoord').textContent = `Tile [${row},${col}] | Leaflet (${lat.toFixed(2)}, ${lng.toFixed(2)})`;
    
    const tiles = state.getTilesForMap(state.selectedMap);
    const tile = tiles.find(t => t.row === row && t.col === col);
    
    if (tile && state.isTileAvailable(state.selectedMap, row, col, state.selectedVersion)) {
        console.log(`Opening tile detail for [${row},${col}]`);
        openTileViewer(state.selectedMap, row, col);
    } else {
        console.log(`Tile [${row},${col}] not available for version ${state.selectedVersion}`);
    }
}

function openTileViewer(map, row, col) {
    const url = `tile.html?map=${encodeURIComponent(map)}&row=${row}&col=${col}&version=${encodeURIComponent(state.selectedVersion)}`;
    window.location.href = url;
}
