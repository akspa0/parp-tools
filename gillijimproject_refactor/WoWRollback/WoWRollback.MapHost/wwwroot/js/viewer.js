import { OverlayManager, GridOverlay } from './overlayManager.js';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const mapSelect = document.getElementById('mapSelect');
const btnReset = document.getElementById('btnReset');
const chkGrid = document.getElementById('chkGrid');
const fpsEl = document.getElementById('fps');

const overlayMgr = new OverlayManager();
overlayMgr.register(GridOverlay);

let state = {
  map: null,
  tileSize: 256, // will be confirmed on first tile load
  // view parameters
  scale: 1.0, // pixels per tile (zoom factor multiplies tileSize)
  offsetX: 0,
  offsetY: 0,
  showGrid: true
};

// Image cache
const tileCache = new Map(); // key: `${map}_${x}_${y}` => HTMLImageElement

// Resize canvas to device size
function resizeCanvas() {
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth;
  const h = vp.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Fetch index and populate maps
async function init() {
  const res = await fetch('/api/index');
  const data = await res.json();
  (data.maps || []).forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.name; opt.textContent = m.name;
    mapSelect.appendChild(opt);
  });
  state.map = (data.maps && data.maps[0]?.name) || 'development';
  mapSelect.value = state.map;
  resetView();
  loop();
}

mapSelect.addEventListener('change', () => {
  state.map = mapSelect.value;
  tileCache.clear();
  resetView();
});

btnReset.addEventListener('click', () => resetView());
chkGrid.addEventListener('change', () => state.showGrid = chkGrid.checked);

// Basic pan/zoom
let isPanning = false;
let lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', (e) => {
  isPanning = true; lastX = e.clientX; lastY = e.clientY;
});
window.addEventListener('mouseup', () => { isPanning = false; });
window.addEventListener('mousemove', (e) => {
  if (!isPanning) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  state.offsetX += dx;
  state.offsetY += dy;
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const { offsetX, offsetY } = e;
  const zoomFactor = Math.exp(-e.deltaY * 0.0015);
  const worldBefore = screenToWorld(offsetX, offsetY);
  state.scale = clamp(state.scale * zoomFactor, 0.25, 8.0);
  const worldAfter = screenToWorld(offsetX, offsetY);
  const dx = (worldAfter.x - worldBefore.x) * state.tileSize;
  const dy = (worldAfter.y - worldBefore.y) * state.tileSize;
  state.offsetX += dx;
  state.offsetY += dy;
}, { passive: false });

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

function resetView() {
  // Center the 64x64 map, assume tileSize px per tile at scale=1, then pick an initial zoom
  state.scale = 0.75; // fits a big chunk of the map
  const mapPx = state.tileSize * 64 * state.scale;
  state.offsetX = (canvas.width - mapPx) / 2;
  state.offsetY = (canvas.height - mapPx) / 2;
}

function worldToScreen(wx, wy) {
  // world tile coords to screen px
  return {
    x: state.offsetX + wx * state.tileSize * state.scale,
    y: state.offsetY + wy * state.tileSize * state.scale
  };
}
function screenToWorld(sx, sy) {
  return {
    x: (sx - state.offsetX) / (state.tileSize * state.scale),
    y: (sy - state.offsetY) / (state.tileSize * state.scale)
  };
}

async function draw(time) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const visible = getVisibleTiles();
  await drawTiles(visible);

  // overlays
  overlayMgr.render(ctx, { ...state });
}

function getVisibleTiles() {
  const topLeft = screenToWorld(0, 0);
  const bottomRight = screenToWorld(canvas.width, canvas.height);
  const x0 = clamp(Math.floor(topLeft.x), 0, 63);
  const y0 = clamp(Math.floor(topLeft.y), 0, 63);
  const x1 = clamp(Math.ceil(bottomRight.x), 0, 63);
  const y1 = clamp(Math.ceil(bottomRight.y), 0, 63);
  return { x0, y0, x1, y1 };
}

async function drawTiles({ x0, y0, x1, y1 }) {
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const key = `${state.map}_${x}_${y}`;
      let img = tileCache.get(key);
      if (!img) {
        img = new Image();
        img.decoding = 'async';
        img.crossOrigin = 'anonymous';
        img.src = `/api/minimap/${encodeURIComponent(state.map)}/${x}/${y}.png`;
        img.onload = () => { if (!state.tileSize || state.tileSize <= 0) state.tileSize = img.width || 256; };
        tileCache.set(key, img);
      }
      const pos = worldToScreen(x, y);
      const sz = state.tileSize * state.scale;
      if (img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, pos.x, pos.y, sz, sz);
      } else {
        // placeholder
        ctx.fillStyle = '#1e1e1e';
        ctx.fillRect(pos.x, pos.y, sz, sz);
        ctx.strokeStyle = '#2a2a2a';
        ctx.strokeRect(pos.x, pos.y, sz, sz);
      }
    }
  }
}

let lastTime = 0, frames = 0;
function loop(ts=0) {
  resizeCanvas();
  draw(ts);
  frames++;
  if (ts - lastTime > 1000) {
    if (fpsEl) fpsEl.textContent = `${frames} FPS`;
    frames = 0; lastTime = ts;
  }
  requestAnimationFrame(loop);
}

init();
