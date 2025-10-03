// Main entry point for index.html - Leaflet Map Viewer
import { state } from './state.js';
import { clearCache } from './overlayLoader.js';
import { loadOverlay } from './overlayLoader.js';

let map;
let tileLayer; // no longer used, kept for minimal diff
let minimapLayer = L.layerGroup();
const minimapImages = new Map(); // key: "r{row}_c{col}" -> L.ImageOverlay
const overlayVariants = {
    combined: { label: 'All Objects (M2 + WMO)', color: '#2196F3', radius: 5 },
    m2: { label: 'M2 Models Only', color: '#2196F3', radius: 8 },
    wmo: { label: 'WMO Objects Only', color: '#FF9800', radius: 7 }
};

let objectMarkers = L.layerGroup();
let lastVersion = null;
let showObjects = true;
let overviewCanvas, overviewCtx;
let uidFilter = null; // { min: number, max: number } or null
// Drag state for overview PiP
let dragging = false;
let dragStart = null;
let dragCurrent = null;

export async function init() {
    try {
        console.log('Initializing viewer...');
        await state.loadIndex();
        console.log('Index loaded:', state.index);
        await state.loadConfig();
        console.log('Config loaded:', state.config);
        
        initializeMap();
        setupUI();
        setupOverview();
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
    // Use Leaflet CRS.Simple without custom flip; do all flips in our math to match wow.tools
    map = L.map('map', {
        crs: L.CRS.Simple,
        minZoom: 0,
        maxZoom: 12,  // Increased for 4x more zoom detail
        zoom: 2,
        zoomControl: true,
        zoomSnap: 0.1,
        zoomDelta: 0.5
    });

    objectMarkers.addTo(map);
    minimapLayer.addTo(map);
    
    // Re-render markers when zoom changes for dynamic scaling
    map.on('zoomend', () => {
        updateObjectMarkers();
    });
    
    // Set initial view to center
    // Start roughly center; for wow.tools mapping lat ≈ 31–32 is center too
    map.setView([32, 32], 2);
    console.log('Map initialized with WoW coordinate system (0,0 = NW)');
}

// Dynamic radius scaling based on zoom level
function getScaledRadius(baseRadius) {
    const zoom = map.getZoom();
    // Scale from 0.7x at zoom 0 to 2.5x at zoom 12
    const scale = 0.7 + (zoom / 12) * 1.8;
    return Math.max(2, baseRadius * scale);
}

function setupUI() {
    const versionSelect = document.getElementById('versionSelect');
    const mapSelect = document.getElementById('mapSelect');
    const showObjectsCheck = document.getElementById('showObjects');
    const layersSearch = document.getElementById('layersSearch');
    const overlayVariantSelect = document.getElementById('overlayVariantSelect');
    
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
        updateObjectMarkers();
    });

    showObjectsCheck.addEventListener('change', (e) => {
        showObjects = e.target.checked;
        if (showObjects) {
            objectMarkers.addTo(map);
        } else {
            objectMarkers.remove();
        }
        updateObjectMarkers();
    });

    // UniqueID range filter: supports "min-max" (e.g., 1000-2000)
    if (layersSearch) {
        layersSearch.addEventListener('input', () => {
            uidFilter = parseUidRange(layersSearch.value);
            updateObjectMarkers();
            drawOverview();
        });
    }

    if (overlayVariantSelect) {
        overlayVariantSelect.innerHTML = '';
        Object.entries(overlayVariants).forEach(([value, meta]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = meta.label;
            if (value === state.overlayVariant) option.selected = true;
            overlayVariantSelect.appendChild(option);
        });
        overlayVariantSelect.addEventListener('change', (e) => {
            state.setOverlayVariant(e.target.value);
        });
    }

    renderComparisonInfo();
}

function onStateChange() {
    const variantSelect = document.getElementById('overlayVariantSelect');
    if (variantSelect && variantSelect.value !== state.overlayVariant) {
        variantSelect.value = state.overlayVariant;
    }
    // If version changed, clear caches and force rebuild of minimap overlays
    if (lastVersion !== state.selectedVersion) {
        lastVersion = state.selectedVersion;
        clearCache();
        // Remove all existing minimap image overlays so URLs (with version) are refreshed
        for (const [, overlay] of minimapImages.entries()) {
            minimapLayer.removeLayer(overlay);
        }
        minimapImages.clear();
    }

    updateTileLayer();
    updateObjectMarkers();
    drawOverview();
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
    refreshMinimapTiles();

    // Reload objects when map moves (debounced)
    let updateTimeout;
    map.on('moveend zoomend', () => {
        clearTimeout(updateTimeout);
        updateTimeout = setTimeout(() => {
            refreshMinimapTiles();
            updateObjectMarkers();
            drawOverview();
        }, 300);
    });

    map.off('click', handleMapClick);
    map.on('click', handleMapClick);

    updateObjectMarkers();
}

function refreshMinimapTiles() {
    const tiles = state.getTilesForMap(state.selectedMap);
    if (!tiles || tiles.length === 0) return;

    const bounds = map.getBounds();
    // Convert lat/lng bounds to row/col bounds according to coordMode
    const rows = tiles.map(t => t.row);
    const cols = tiles.map(t => t.col);
    const minRowAll = Math.min(...rows), maxRowAll = Math.max(...rows);
    const minColAll = Math.min(...cols), maxColAll = Math.max(...cols);

    const latS = bounds.getSouth();
    const latN = bounds.getNorth();
    const west = bounds.getWest();
    const east = bounds.getEast();

    // For wowtools: row = 63 - lat
    const rowNorth = latToRow(latN);
    const rowSouth = latToRow(latS);
    let minRow = Math.max(minRowAll, Math.floor(Math.min(rowNorth, rowSouth)));
    let maxRow = Math.min(maxRowAll, Math.ceil(Math.max(rowNorth, rowSouth)));
    let minCol = Math.max(minColAll, Math.floor(west));
    let maxCol = Math.min(maxColAll, Math.ceil(east));

    // Pad by 1 tile around viewport to reduce pop-in
    minRow = Math.max(minRowAll, minRow - 1);
    maxRow = Math.min(maxRowAll, maxRow + 1);
    minCol = Math.max(minColAll, minCol - 1);
    maxCol = Math.min(maxColAll, maxCol + 1);

    const needed = new Set();
    for (let r = minRow; r <= maxRow; r++) {
        for (let c = minCol; c <= maxCol; c++) {
            // Require availability for selected version
            const available = tiles.find(t => t.row === r && t.col === c && t.versions && t.versions.includes(state.selectedVersion));
            if (!available) continue;
            const key = `r${r}_c${c}`;
            needed.add(key);
            if (!minimapImages.has(key)) {
                const url = state.getMinimapPath(state.selectedMap, r, c, state.selectedVersion);
                const b = tileBounds(r, c);
                const overlay = L.imageOverlay(url, b, { interactive: false, opacity: 1.0 });
                overlay.addTo(minimapLayer);
                minimapImages.set(key, overlay);
            }
        }
    }

    // Remove no-longer-needed overlays
    for (const [key, overlay] of minimapImages.entries()) {
        if (!needed.has(key)) {
            minimapLayer.removeLayer(overlay);
            minimapImages.delete(key);
        }
    }
}

function getBounds() {
    const tiles = state.getTilesForMap(state.selectedMap);
    if (tiles.length === 0) return [[0, 0], [64, 64]];
    
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
    
    // Only load overlays for visible tiles (max 8x8 grid)
    const bounds = map.getBounds();
    const latS = bounds.getSouth();
    const latN = bounds.getNorth();
    const west = bounds.getWest();
    const east = bounds.getEast();
    let minRow = Math.max(0, Math.floor(Math.min(latToRow(latN), latToRow(latS))));
    let maxRow = Math.min(63, Math.ceil(Math.max(latToRow(latN), latToRow(latS))));
    let minCol = Math.max(0, Math.floor(west));
    let maxCol = Math.min(63, Math.ceil(east));
    
    // Limit to 8x8 grid max
    if (maxRow - minRow > 7) {
        maxRow = minRow + 7;
    }
    if (maxCol - minCol > 7) {
        maxCol = minCol + 7;
    }
    
    const tileCount = (maxRow - minRow + 1) * (maxCol - minCol + 1);
    console.log(`Loading objects for ${tileCount} tiles: r${minRow}-${maxRow}, c${minCol}-${maxCol}`);
    
    const tiles = state.getTilesForMap(state.selectedMap);
    const currentVariant = state.overlayVariant || 'combined';
    const style = overlayVariants[currentVariant] ?? overlayVariants.combined;
    for (let row = minRow; row <= maxRow; row++) {
        for (let col = minCol; col <= maxCol; col++) {
            // Only fetch overlays for tiles listed in index.json and available for the selected version
            const available = tiles.find(t => t.row === row && t.col === col && t.versions && t.versions.includes(state.selectedVersion));
            if (!available) continue;
            try {
                const overlayPath = state.getOverlayPath(state.selectedMap, row, col, state.selectedVersion, currentVariant);
                const data = await loadOverlay(overlayPath);
                
                const versionData = data.layers?.find(l => l.version === state.selectedVersion);
                if (!versionData || !versionData.kinds) continue;

                const objects = versionData.kinds.flatMap(kind => {
                    const label = typeof kind.kind === 'string' ? kind.kind : (kind.label ?? 'unknown');
                    return (kind.points || []).map(point => ({
                        ...point,
                        __kind: label
                    }));
                });
                const tileW = (data.minimap && data.minimap.width) ? data.minimap.width : 512;
                const tileH = (data.minimap && data.minimap.height) ? data.minimap.height : 512;
                
                objects.forEach(obj => {
                    if (!obj.pixel || !obj.world || (obj.world.x === 0 && obj.world.y === 0 && obj.world.z === 0)) return;
                    if (!passesUidFilter(obj.uniqueId)) return;
                    
                    // Convert pixel coords to lat/lng using coordMode
                    const { lat, lng } = pixelToLatLng(row, col, obj.pixel.x, obj.pixel.y, tileW, tileH);
                    
                    const isCombined = currentVariant === 'combined';
                    const kindTag = (obj.__kind || '').toString().toLowerCase();
                    const isWmo = kindTag.includes('wmo');

                    const popupHtml = `
                        <strong>${obj.fileName || 'Unknown'}</strong><br>
                        UID: ${obj.uniqueId || 'N/A'}<br>
                        World: (${obj.world.x.toFixed(1)}, ${obj.world.y.toFixed(1)}, ${obj.world.z.toFixed(1)})
                    `;

                    if (isCombined)
                    {
                        if (isWmo)
                        {
                            const baseSize = 0.002;
                            const zoom = map.getZoom();
                            const sizeScale = 0.7 + (zoom / 12) * 1.8;
                            const squareHalf = baseSize * sizeScale;
                            const bounds = [
                                [lat - squareHalf, lng - squareHalf],
                                [lat + squareHalf, lng + squareHalf]
                            ];
                            const square = L.rectangle(bounds, {
                                color: '#000',
                                weight: 1,
                                fillColor: '#FF9800',
                                fillOpacity: 0.85
                            }).bindPopup(popupHtml);
                            objectMarkers.addLayer(square);
                        }
                        else
                        {
                            const circle = L.circleMarker([lat, lng], {
                                radius: getScaledRadius(4),
                                fillColor: '#2196F3',
                                color: '#000',
                                weight: 1,
                                fillOpacity: 0.9
                            }).bindPopup(popupHtml);
                            objectMarkers.addLayer(circle);
                        }
                    }
                    else
                    {
                        const marker = L.circleMarker([lat, lng], {
                            radius: getScaledRadius(style.radius),
                            fillColor: style.color,
                            color: '#000',
                            weight: 1,
                            fillOpacity: 0.85
                        }).bindPopup(popupHtml);
                        objectMarkers.addLayer(marker);
                    }
                });

                // Debug: draw overlay corner markers if enabled in config
                if (state.config && state.config.debugOverlayCorners) {
                    const corners = [
                        { x: 0, y: 0 },
                        { x: tileW, y: 0 },
                        { x: tileW, y: tileH },
                        { x: 0, y: tileH }
                    ];
                    corners.forEach(p => {
                        const ll = pixelToLatLng(row, col, p.x, p.y, tileW, tileH);
                        const m = L.circleMarker([ll.lat, ll.lng], {
                            radius: 2,
                            color: '#FFD54F',
                            weight: 1,
                            fillColor: '#FFD54F',
                            fillOpacity: 0.9
                        });
                        objectMarkers.addLayer(m);
                    });
                }
            } catch (e) {
                // Skip tiles without overlay data
            }
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

// --- Mini-map overview (PiP) ---
function setupOverview() {
    overviewCanvas = document.getElementById('overviewCanvas');
    if (!overviewCanvas) return;
    overviewCtx = overviewCanvas.getContext('2d');
    syncOverviewCanvasSize();

    // Click to center, drag to fit bounds

    overviewCanvas.addEventListener('mousedown', (e) => {
        dragging = true;
        const p = canvasPoint(e);
        dragStart = p; dragCurrent = p;
        drawOverview();
    });
    overviewCanvas.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        dragCurrent = canvasPoint(e);
        drawOverview();
    });
    overviewCanvas.addEventListener('mouseup', (e) => {
        const start = dragStart; const end = canvasPoint(e);
        dragging = false; dragStart = null; dragCurrent = null;
        // If small movement -> treat as click to center
        if (start && Math.hypot(end.x - start.x, end.y - start.y) < 4) {
            const target = canvasToRowCol(end);
            map.setView([target.row + 0.5, target.col + 0.5], map.getZoom());
        } else if (start) {
            const a = canvasToRowCol(start);
            const b = canvasToRowCol(end);
            const minRow = Math.max(0, Math.min(a.row, b.row));
            const maxRow = Math.min(64, Math.max(a.row, b.row) + 1);
            const minCol = Math.max(0, Math.min(a.col, b.col));
            const maxCol = Math.min(64, Math.max(a.col, b.col) + 1);
            map.fitBounds([[minRow, minCol], [maxRow, maxCol]]);
        }
        drawOverview();
    });
    window.addEventListener('resize', () => { syncOverviewCanvasSize(); drawOverview(); });
    drawOverview();

    function canvasPoint(evt) {
        const r = overviewCanvas.getBoundingClientRect();
        return { x: evt.clientX - r.left, y: evt.clientY - r.top };
    }
    function syncOverviewCanvasSize() {
        const r = overviewCanvas.getBoundingClientRect();
        overviewCanvas.width = Math.max(Math.floor(r.width), 1);
        overviewCanvas.height = Math.max(Math.floor(r.height), 1);
    }
    function canvasToRowCol(pt) {
        // Use same mapping as drawOverview
        const tiles = state.getTilesForMap(state.selectedMap);
        const rows = tiles.map(t => t.row);
        const cols = tiles.map(t => t.col);
        const minRow = Math.min(...rows); const maxRow = Math.max(...rows);
        const minCol = Math.min(...cols); const maxCol = Math.max(...cols);
        const gridW = (maxCol - minCol + 1); const gridH = (maxRow - minRow + 1);
        const W = overviewCanvas.width; const H = overviewCanvas.height;
        const margin = 8;
        const cellSize = Math.floor(Math.min((W - 2 * margin) / gridW, (H - 2 * margin) / gridH));
        const originX = Math.floor((W - cellSize * gridW) / 2);
        const originY = Math.floor((H - cellSize * gridH) / 2);
        const col = Math.floor((pt.x - originX) / cellSize) + minCol;
        const row = Math.floor((pt.y - originY) / cellSize) + minRow;
        return { row: clamp(row, minRow, maxRow), col: clamp(col, minCol, maxCol) };
    }
}

function drawOverview() {
    if (!overviewCtx) return;
    syncSize();
    const W = overviewCanvas.width;
    const H = overviewCanvas.height;
    overviewCtx.clearRect(0, 0, W, H);

    // Tile quilt bounds
    const tiles = state.getTilesForMap(state.selectedMap);
    if (!tiles || tiles.length === 0) return;
    const rows = tiles.map(t => t.row);
    const cols = tiles.map(t => t.col);
    const minRow = Math.min(...rows);
    const maxRow = Math.max(...rows);
    const minCol = Math.min(...cols);
    const maxCol = Math.max(...cols);
    const gridW = (maxCol - minCol + 1);
    const gridH = (maxRow - minRow + 1);

    // Compute scale to fit canvas
    const margin = 8;
    const cellSize = Math.floor(Math.min((W - 2 * margin) / gridW, (H - 2 * margin) / gridH));
    const originX = Math.floor((W - cellSize * gridW) / 2);
    const originY = Math.floor((H - cellSize * gridH) / 2);

    // Draw grid tiles
    tiles.forEach(t => {
        const x = originX + (t.col - minCol) * cellSize;
        const y = originY + (t.row - minRow) * cellSize;
        overviewCtx.fillStyle = '#2a2a2a';
        overviewCtx.fillRect(x, y, cellSize - 1, cellSize - 1);
        // Mark availability for current version
        const available = t.versions && t.versions.includes(state.selectedVersion);
        overviewCtx.fillStyle = available ? '#4CAF50' : '#555';
        overviewCtx.fillRect(x + 1, y + 1, cellSize - 3, cellSize - 3);
    });

    // Draw current viewport rectangle
    const bounds = map.getBounds();
    const latS = bounds.getSouth();
    const latN = bounds.getNorth();
    const viewMinRow = Math.max(minRow, Math.floor(Math.min(latToRow(latN), latToRow(latS))));
    const viewMaxRow = Math.min(maxRow + 1, Math.ceil(Math.max(latToRow(latN), latToRow(latS))));
    const viewMinCol = Math.max(minCol, Math.floor(bounds.getWest()));
    const viewMaxCol = Math.min(maxCol + 1, Math.ceil(bounds.getEast()));
    const rx = originX + (viewMinCol - minCol) * cellSize;
    const ry = originY + (viewMinRow - minRow) * cellSize;
    const rw = Math.max(2, (viewMaxCol - viewMinCol) * cellSize);
    const rh = Math.max(2, (viewMaxRow - viewMinRow) * cellSize);
    overviewCtx.strokeStyle = '#FFD54F';
    overviewCtx.lineWidth = 2;
    overviewCtx.strokeRect(rx, ry, rw, rh);

    // If dragging, draw selection rectangle
    if (typeof dragStart !== 'undefined' && dragStart && dragCurrent) {
        overviewCtx.strokeStyle = '#64B5F6';
        overviewCtx.setLineDash([4, 3]);
        overviewCtx.strokeRect(
            Math.min(dragStart.x, dragCurrent.x),
            Math.min(dragStart.y, dragCurrent.y),
            Math.abs(dragCurrent.x - dragStart.x),
            Math.abs(dragCurrent.y - dragStart.y)
        );
        overviewCtx.setLineDash([]);
    }

    function syncSize() {
        const r = overviewCanvas.getBoundingClientRect();
        if (overviewCanvas.width !== Math.floor(r.width) || overviewCanvas.height !== Math.floor(r.height)) {
            overviewCanvas.width = Math.max(Math.floor(r.width), 1);
            overviewCanvas.height = Math.max(Math.floor(r.height), 1);
        }
    }
    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
}

// --- Mapping helpers ---
function isWowTools() {
    return state.config && state.config.coordMode === 'wowtools';
}

function rowToLat(row) {
    return isWowTools() ? (63 - row) : row;
}

function latToRow(lat) {
    return isWowTools() ? (63 - lat) : lat;
}

function tileBounds(row, col) {
    if (isWowTools()) {
        const latTop = rowToLat(row);      // 63 - row
        const latBottom = rowToLat(row + 1); // 63 - (row+1)
        const north = Math.max(latTop, latBottom);
        const south = Math.min(latTop, latBottom);
        return [[south, col], [north, col + 1]];
    }
    return [[row, col], [row + 1, col + 1]];
}

function pixelToLatLng(row, col, px, py, w, h) {
    if (isWowTools()) {
        const lat = rowToLat(row) - (py / h);
        const lng = col + (px / w);
        return { lat, lng };
    }
    return { lat: row + (py / h), lng: col + (px / w) };
}

// --- UniqueID filter helpers ---
function parseUidRange(text) {
    if (!text) return null;
    const trimmed = text.trim();
    const match = trimmed.match(/^(\d+)\s*-\s*(\d+)$/);
    if (!match) return null;
    const min = parseInt(match[1], 10);
    const max = parseInt(match[2], 10);
    if (isNaN(min) || isNaN(max) || min > max) return null;
    return { min, max };
}

function passesUidFilter(uid) {
    if (!uidFilter) return true;
    if (!uid && uid !== 0) return false;
    const value = Number(uid);
    if (isNaN(value)) return false;
    return value >= uidFilter.min && value <= uidFilter.max;
}
