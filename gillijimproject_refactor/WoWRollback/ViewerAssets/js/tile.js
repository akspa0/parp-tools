// Tile detail viewer
import { state } from './state.js';
import { TileCanvas } from './tileCanvas.js';
import { loadOverlay, loadDiff } from './overlayLoader.js';
import { autoFit, computeLocalForTile, toPixels } from './fit.js';

let currentMap, currentRow, currentCol, currentVersion;
let tileCanvas;
let overlayData = null;
let diffData = null;
let diffEnabled = false;

// Interactive fit state (per tile)
let fit = {
    axis: 'auto',    // 'auto' | 'xy' | 'xz' | 'zy'
    flipX: false,
    flipY: false,
    rotate: 0,       // 0|90|180|270
    invertY: true
};

export async function init() {
    await state.loadIndex();
    await state.loadConfig();

    const params = new URLSearchParams(window.location.search);
    currentMap = params.get('map');
    currentRow = parseInt(params.get('row'));
    currentCol = parseInt(params.get('col'));
    currentVersion = params.get('version') || state.selectedVersion;

    if (!currentMap || isNaN(currentRow) || isNaN(currentCol)) {
        alert('Invalid tile parameters');
        window.location.href = 'index.html';
        return;
    }

function getActivePoints(objects) {
    const pts = [];
    for (const o of objects) {
        if (!o || !o.world) continue;
        pts.push({ world: { x: o.world.x, y: o.world.y, z: o.world.z }, diffType: o.diffType || null });
    }
    return pts;
}

function drawOnCanvas(objects) {
    tileCanvas.draw();
    const width = tileCanvas.canvas.width;
    const height = tileCanvas.canvas.height;

    // Auto-fit on demand
    if (fit.axis === 'auto') {
        doAutoFit(objects);
    }

    let inRange = 0, outRange = 0, edgeAccum = 0;
    const display = [];
    for (const o of objects) {
        if (!o.world) continue;
        const { inRange: ok, lx, ly } = computeLocalForTile(
            o.world, fit.axis === 'auto' ? 'xy' : fit.axis, fit.flipX, fit.flipY, fit.rotate, currentRow, currentCol
        );
        if (ok) inRange++; else outRange++;
        const px = toPixels(lx, ly, width, height, fit.invertY);
        const ex = Math.min(Math.abs(lx), Math.abs(1 - lx));
        const ey = Math.min(Math.abs(ly), Math.abs(1 - ly));
        edgeAccum += 0.5 * (ex + ey);
        display.push({ pixelX: px.x, pixelY: px.y, diffType: o.diffType || 'default' });
    }

    const colorMap = { default: '#4CAF50', added: '#4CAF50', removed: '#E53935', moved: '#FFB300', changed: '#8E24AA' };
    tileCanvas.drawOverlay(display, colorMap);
    const meanEdge = objects.length > 0 ? edgeAccum / objects.length : 0;
    updateDiagnostics(display, inRange, outRange, meanEdge);
}

function doAutoFit(objects) {
    const pts = getActivePoints(objects || []);
    if (pts.length === 0) return;
    const cfg = autoFit(pts, currentRow, currentCol, tileCanvas.canvas.width, tileCanvas.canvas.height);
    fit.axis = cfg.axis; fit.flipX = cfg.flipX; fit.flipY = cfg.flipY; fit.rotate = cfg.rotate; fit.invertY = cfg.invertY;
    // Reflect in UI
    const fitAxis = document.getElementById('fitAxis');
    const fitFlipX = document.getElementById('fitFlipX');
    const fitFlipY = document.getElementById('fitFlipY');
    const fitRotate = document.getElementById('fitRotate');
    const fitInvertY = document.getElementById('fitInvertY');
    if (fitAxis) fitAxis.value = fit.axis;
    if (fitFlipX) fitFlipX.checked = fit.flipX;
    if (fitFlipY) fitFlipY.checked = fit.flipY;
    if (fitRotate) fitRotate.value = String(fit.rotate);
    if (fitInvertY) fitInvertY.checked = fit.invertY;
}

function resetFit() {
    fit = { axis: 'auto', flipX: false, flipY: false, rotate: 0, invertY: true };
    const fitAxis = document.getElementById('fitAxis');
    const fitFlipX = document.getElementById('fitFlipX');
    const fitFlipY = document.getElementById('fitFlipY');
    const fitRotate = document.getElementById('fitRotate');
    const fitInvertY = document.getElementById('fitInvertY');
    if (fitAxis) fitAxis.value = 'auto';
    if (fitFlipX) fitFlipX.checked = false;
    if (fitFlipY) fitFlipY.checked = false;
    if (fitRotate) fitRotate.value = '0';
    if (fitInvertY) fitInvertY.checked = true;
    renderObjects();
}

function updateDiagnostics(displayPoints, inRange, outRange, meanEdge) {
    const el = document.getElementById('fitDiagnostics');
    if (!el) return;
    const desc = `axis=${fit.axis} flipX=${fit.flipX} flipY=${fit.flipY} rot=${fit.rotate} invY=${fit.invertY}`;
    el.innerHTML = `<div>${desc}</div><div>inRange: ${inRange}, outOfRange: ${outRange}, meanEdge: ${meanEdge.toFixed(3)}</div>`;
}

    setupUI();
    tileCanvas = new TileCanvas('tileCanvas', state.config);
    await loadTile();
}

function setupUI() {
    document.getElementById('tileTitle').textContent = 
        `${currentMap} - Tile ${currentRow}_${currentCol}`;

    const versionSelect = document.getElementById('versionSelect');
    state.index.versions.forEach(version => {
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version;
        option.selected = version === currentVersion;
        versionSelect.appendChild(option);
    });

    versionSelect.addEventListener('change', async (e) => {
        currentVersion = e.target.value;
        await loadTile();
    });

    // Diff controls
    const diffEnabledCheckbox = document.getElementById('diffEnabled');
    const diffOptions = document.getElementById('diffOptions');
    const diffBaseline = document.getElementById('diffBaseline');
    const diffComparison = document.getElementById('diffComparison');

    state.index.versions.forEach(version => {
        const opt1 = document.createElement('option');
        opt1.value = version;
        opt1.textContent = version;
        diffBaseline.appendChild(opt1);

        const opt2 = document.createElement('option');
        opt2.value = version;
        opt2.textContent = version;
        diffComparison.appendChild(opt2);
    });

    if (state.index.diff) {
        diffBaseline.value = state.index.diff.baseline;
        diffComparison.value = state.index.diff.comparison;
    }

    diffEnabledCheckbox.addEventListener('change', async (e) => {
        diffOptions.style.display = e.target.checked ? 'block' : 'none';
        document.querySelector('.diff-legend').style.display = e.target.checked ? 'block' : 'none';
        if (e.target.checked) {
            await loadDiffData();
        } else {
            diffData = null;
            renderObjects();
        }
    });

    diffBaseline.addEventListener('change', async () => {
        if (diffEnabledCheckbox.checked) await loadDiffData();
    });

    diffComparison.addEventListener('change', async () => {
        if (diffEnabledCheckbox.checked) await loadDiffData();
    });

    // Fit controls
    const fitAxis = document.getElementById('fitAxis');
    const fitFlipX = document.getElementById('fitFlipX');
    const fitFlipY = document.getElementById('fitFlipY');
    const fitRotate = document.getElementById('fitRotate');
    const fitInvertY = document.getElementById('fitInvertY');
    const fitAutoBtn = document.getElementById('fitAuto');
    const fitResetBtn = document.getElementById('fitReset');

    if (fitAxis) fitAxis.addEventListener('change', () => { fit.axis = fitAxis.value; renderObjects(); });
    if (fitFlipX) fitFlipX.addEventListener('change', () => { fit.flipX = fitFlipX.checked; renderObjects(); });
    if (fitFlipY) fitFlipY.addEventListener('change', () => { fit.flipY = fitFlipY.checked; renderObjects(); });
    if (fitRotate) fitRotate.addEventListener('change', () => { fit.rotate = parseInt(fitRotate.value, 10) || 0; renderObjects(); });
    if (fitInvertY) fitInvertY.addEventListener('change', () => { fit.invertY = fitInvertY.checked; renderObjects(); });
    if (fitAutoBtn) fitAutoBtn.addEventListener('click', () => { doAutoFit(); renderObjects(); });
    if (fitResetBtn) fitResetBtn.addEventListener('click', () => { resetFit(); });
}

async function loadTile() {
    const minimapPath = state.getMinimapPath(currentMap, currentRow, currentCol, currentVersion);
    
    try {
        await tileCanvas.loadImage(minimapPath);
        await loadOverlayData();
        renderObjects();
    } catch (error) {
        console.error('Failed to load tile:', error);
        tileCanvas.drawPlaceholder();
    }
}

async function loadOverlayData() {
    const overlayPath = state.getOverlayPath(currentMap, currentRow, currentCol);
    try {
        overlayData = await loadOverlay(overlayPath);
    } catch (error) {
        console.warn('No overlay data:', error);
        overlayData = null;
    }
}

async function loadDiffData() {
    const diffPath = state.getDiffPath(currentMap, currentRow, currentCol);
    try {
        diffData = await loadDiff(diffPath);
        renderObjects();
    } catch (error) {
        console.warn('No diff data:', error);
        diffData = null;
    }
}

function renderObjects() {
    const objectList = document.getElementById('objectList');
    const objectCount = document.getElementById('objectCount');

    if (!overlayData) {
        objectList.innerHTML = '<p style="color: #9E9E9E;">No object data available.</p>';
        objectCount.textContent = '0';
        return;
    }

    let objects = [];
    if (diffData && document.getElementById('diffEnabled').checked) {
        // Show diff results
        // Moved/changed have {from, to} structure, use 'to' for display
        objects = [
            ...(diffData.added || []).map(o => ({ ...o, diffType: 'added' })),
            ...(diffData.removed || []).map(o => ({ ...o, diffType: 'removed' })),
            ...(diffData.moved || []).map(m => ({ ...m.to, diffType: 'moved', distance: m.distance })),
            ...(diffData.changed || []).map(c => ({ ...c.to, diffType: 'changed' }))
        ];
    } else {
        // Show objects for current version
        const versionData = overlayData.layers?.find(l => l.version === currentVersion);
        
        // Flatten kinds[].points into a single objects array
        if (versionData && versionData.kinds) {
            objects = versionData.kinds.flatMap(kind => kind.points || []);
        } else {
            objects = [];
        }
    }

    objectCount.textContent = objects.length;

    if (objects.length === 0) {
        objectList.innerHTML = '<p style="color: #9E9E9E;">No objects in this tile.</p>';
        updateDiagnostics([], 0, 0, 0);
        tileCanvas.draw();
        return;
    }

    objectList.innerHTML = objects.map(obj => createObjectItem(obj)).join('');

    // Draw markers on canvas using current fit
    drawOnCanvas(objects);
}

function createObjectItem(obj) {
    const diffClass = obj.diffType ? `diff-${obj.diffType}` : '';
    const diffLabel = obj.diffType ? `[${obj.diffType.toUpperCase()}] ` : '';
    
    const worldPos = obj.world ? `(${obj.world.x.toFixed(1)}, ${obj.world.y.toFixed(1)}, ${obj.world.z.toFixed(1)})` : 'N/A';
    const pixelPos = obj.pixel ? `Pixel: (${obj.pixel.x.toFixed(0)}, ${obj.pixel.y.toFixed(0)})` : '';
    
    return `
        <div class="object-item ${diffClass}">
            <div><strong>${diffLabel}${obj.fileName || obj.assetPath || 'Unknown'}</strong></div>
            <div style="font-size: 0.85em; color: #9E9E9E;">
                UID: ${obj.uniqueId || 'N/A'}<br>
                World: ${worldPos}<br>
                ${pixelPos ? pixelPos + '<br>' : ''}
                ${obj.designKit ? `Kit: ${obj.designKit}` : ''}
            </div>
        </div>
    `;
}
