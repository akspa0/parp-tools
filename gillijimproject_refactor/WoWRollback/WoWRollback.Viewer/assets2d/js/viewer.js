import { OverlayManager, GridOverlay } from './overlayManager.js';
import { CoordsOverlay } from './plugins/coords.js';
import { M2Overlay } from './plugins/m2.js';
import { WMOOverlay } from './plugins/wmo.js';
import { initSidebar } from './ui/sidebar.js';
import { initInspector, updateInspector } from './ui/inspector.js';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const mapSelect = document.getElementById('mapSelect');
const btnReset = document.getElementById('btnReset');
const chkGrid = document.getElementById('chkGrid');
const fpsEl = document.getElementById('fps');

const overlayMgr = new OverlayManager();
overlayMgr.register(GridOverlay);
overlayMgr.register(CoordsOverlay);
overlayMgr.register(M2Overlay);
overlayMgr.register(WMOOverlay);

let state = {
  map: null,
  tileSize: 256,
  scale: 0.75,
  targetScale: 0.75,
  offsetX: 0,
  offsetY: 0,
  zoomAnchor: null, // {mx,my,wx,wy}
  showGrid: true,
  overlays: { manifest: null },
  selection: [], // array of { pluginId, feature }
  canvasWidth: 0,
  canvasHeight: 0
};

const tileCache = new Map(); // key: `${map}_${x}_${y}` -> Image

function resizeCanvas() {
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth;
  const h = vp.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  state.canvasWidth = canvas.width;
  state.canvasHeight = canvas.height;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

async function loadIndex() {
  // Viewer-pack index.json hosted at /data/index.json
  const res = await fetch('/data/index.json');
  const data = await res.json();
  (data.maps || []).forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.name; opt.textContent = m.name;
    mapSelect.appendChild(opt);
  });
  state.map = data.defaultMap || (data.maps && data.maps[0]?.name) || null;
  if (state.map) mapSelect.value = state.map;
  await loadOverlayManifest();
  initSidebar(overlayMgr, state);
  initInspector(state);
  resetView();
  loop();
}

mapSelect.addEventListener('change', async () => { state.map = mapSelect.value; tileCache.clear(); await loadOverlayManifest(); initSidebar(overlayMgr, state); initInspector(state); resetView(); });
btnReset.addEventListener('click', resetView);
chkGrid.addEventListener('change', () => state.showGrid = chkGrid.checked);

let isPanning = false; let lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', (e) => { isPanning = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => { isPanning = false; });
window.addEventListener('mousemove', (e) => {
  if (!isPanning) return;
  const dx = e.clientX - lastX; const dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  state.offsetX += dx; state.offsetY += dy;
});
canvas.addEventListener('click', (e) => {
  // Picking on left-click
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const hit = overlayMgr.hitTest(state, sx, sy);
  if (!hit) {
    if (!e.ctrlKey && !e.shiftKey) { state.selection = []; updateInspector(state); }
    return;
  }
  // Multi-select logic
  const key = (h) => `${h.pluginId}:${h.feature?.uid ?? h.feature?.UniqueID ?? ''}:${h.feature?.FileDataID ?? h.feature?.fileId ?? ''}`;
  const existsIdx = state.selection.findIndex(s => key(s) === key(hit));
  if (e.ctrlKey) {
    if (existsIdx >= 0) state.selection.splice(existsIdx, 1);
    else state.selection.push(hit);
  } else if (e.shiftKey) {
    // simple add without range semantics for now
    if (existsIdx < 0) state.selection.push(hit);
  } else {
    state.selection = [hit];
  }
  updateInspector(state);
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  // Normalize wheel delta across devices
  const delta = (e.deltaMode === 1 /*line*/ ? e.deltaY * 16 : e.deltaY);
  const raw = Math.exp(-delta * 0.0015);
  const zf = clamp(raw, 0.9, 1.1); // tame per-tick zoom
  const mx = e.offsetX, my = e.offsetY;
  const before = screenToWorld(mx, my);
  state.targetScale = clamp(state.targetScale * zf, 0.25, 8.0);
  state.zoomAnchor = { mx, my, wx: before.x, wy: before.y };
}, { passive: false });

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function resetView() {
  state.scale = 0.75;
  state.targetScale = state.scale;
  const mapPx = 64 * state.tileSize * state.scale;
  state.offsetX = (canvas.width - mapPx) / 2;
  state.offsetY = (canvas.height - mapPx) / 2;
}
function worldToScreen(wx, wy) {
  return { x: state.offsetX + wx * state.tileSize * state.scale, y: state.offsetY + wy * state.tileSize * state.scale };
}
function screenToWorld(sx, sy) {
  return { x: (sx - state.offsetX) / (state.tileSize * state.scale), y: (sy - state.offsetY) / (state.tileSize * state.scale) };
}

function clampOffsets() {
  const margin = 64; // px
  const mapPx = 64 * state.tileSize * state.scale;
  const minX = Math.min(margin, canvas.width - mapPx - margin);
  const maxX = Math.max(margin, canvas.width - mapPx - margin);
  // Ensure minX <= maxX
  const loX = Math.min(minX, maxX), hiX = Math.max(minX, maxX);
  state.offsetX = clamp(state.offsetX, loX, hiX);

  const minY = Math.min(margin, canvas.height - mapPx - margin);
  const maxY = Math.max(margin, canvas.height - mapPx - margin);
  const loY = Math.min(minY, maxY), hiY = Math.max(minY, maxY);
  state.offsetY = clamp(state.offsetY, loY, hiY);
}

async function loadOverlayManifest() {
  try {
    const res = await fetch(`/data/overlays/${encodeURIComponent(state.map)}/manifest.json`);
    if (res.ok) {
      state.overlays.manifest = await res.json();
    } else {
      state.overlays.manifest = null;
    }
  } catch { state.overlays.manifest = null; }
}

async function draw(ts) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.map) return;
  const vis = getVisibleTiles();
  await drawTiles(vis);
  overlayMgr.render(ctx, { ...state });
}

function getVisibleTiles() {
  const tl = screenToWorld(0, 0), br = screenToWorld(canvas.width, canvas.height);
  const x0 = clamp(Math.floor(tl.x), 0, 63), y0 = clamp(Math.floor(tl.y), 0, 63);
  const x1 = clamp(Math.ceil(br.x), 0, 63), y1 = clamp(Math.ceil(br.y), 0, 63);
  return { x0, y0, x1, y1 };
}

async function drawTiles({ x0, y0, x1, y1 }) {
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const key = `${state.map}_${x}_${y}`;
      let img = tileCache.get(key);
      if (!img) {
        img = new Image(); img.decoding = 'async'; img.crossOrigin = 'anonymous';
        img.src = `/data/tiles/${encodeURIComponent(state.map)}/${x}_${y}.webp`;
        img.onload = () => { if (!state.tileSize || state.tileSize <= 0) state.tileSize = img.width || 256; };
        tileCache.set(key, img);
      }
      const pos = worldToScreen(x, y), sz = state.tileSize * state.scale;
      if (img.complete && img.naturalWidth > 0) ctx.drawImage(img, pos.x, pos.y, sz, sz);
      else { ctx.fillStyle = '#1e1e1e'; ctx.fillRect(pos.x, pos.y, sz, sz); ctx.strokeStyle = '#2a2a2a'; ctx.strokeRect(pos.x, pos.y, sz, sz); }
    }
  }
}

let last = 0, frames = 0;
function loop(ts=0) {
  resizeCanvas();
  // Allow overlays to update based on viewport (fetch visible tile data, etc.)
  overlayMgr.onViewport(state);
  // Eased scale towards targetScale
  if (Math.abs(state.targetScale - state.scale) > 1e-3) {
    state.scale += (state.targetScale - state.scale) * 0.2;
    // Keep cursor world point anchored during zoom
    if (state.zoomAnchor) {
      const { mx, my, wx, wy } = state.zoomAnchor;
      state.offsetX = mx - wx * state.tileSize * state.scale;
      state.offsetY = my - wy * state.tileSize * state.scale;
    }
  } else {
    state.scale = state.targetScale;
    state.zoomAnchor = null;
  }

  // Apply pan inertia if desired (disabled by default)
  clampOffsets();

  draw(ts);
  frames++; if (ts - last > 1000) { if (fpsEl) fpsEl.textContent = `${frames} FPS`; frames = 0; last = ts; }
  requestAnimationFrame(loop);
}

loadIndex();
