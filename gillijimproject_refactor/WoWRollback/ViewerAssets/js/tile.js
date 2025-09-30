// Tile detail viewer
import { state } from './state.js';
import { TileCanvas } from './tileCanvas.js';
import { loadOverlay, loadDiff } from './overlayLoader.js';

let currentMap, currentRow, currentCol, currentVersion;
let tileCanvas;
let overlayData = null;
let diffData = null;
let diffEnabled = false;

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
        return;
    }

    objectList.innerHTML = objects.map(obj => createObjectItem(obj)).join('');
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
