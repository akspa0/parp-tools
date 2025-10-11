export const WMOOverlay = {
  id: 'wmo',
  title: 'WMO',
  enabledByDefault: false,
  _cache: new Map(),

  onViewport(view) {
    const { x0, y0, x1, y1 } = visibleTiles(view);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        this._ensureTile(view.map, x, y);
      }
    }
  },

  render(ctx, view) {
    const { x0, y0, x1, y1 } = visibleTiles(view);
    const size = Math.max(2, 4 * view.scale);
    ctx.save();
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const t = this._cache.get(key(view.map, x, y));
        if (!t || !t.worldRect || !Array.isArray(t.features)) continue;
        for (const f of t.features) {
          const pos = f.position || f.Position;
          if (!pos) continue;
          const p = worldToScreenTile(view, x, y, pos.x ?? pos.X, pos.y ?? pos.Y, t.worldRect);
          ctx.fillStyle = '#90caf9';
          ctx.beginPath();
          ctx.rect(p.x - size, p.y - size, size * 2, size * 2);
          ctx.fill();
        }
      }
    }
    ctx.restore();
  },

  hitTest(view, sx, sy) {
    const { x0, y0, x1, y1 } = visibleTiles(view);
    const maxR2 = (8 * 8);
    let best = null;
    let bestD2 = Infinity;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const t = this._cache.get(key(view.map, x, y));
        if (!t || !t.worldRect || !Array.isArray(t.features)) continue;
        for (const f of t.features) {
          const pos = f.position || f.Position;
          if (!pos) continue;
          const p = worldToScreenTile(view, x, y, pos.x ?? pos.X, pos.y ?? pos.Y, t.worldRect);
          const dx = p.x - sx, dy = p.y - sy;
          const d2 = dx*dx + dy*dy;
          if (d2 < bestD2 && d2 <= maxR2) { bestD2 = d2; best = { feature: f, dist2: d2 }; }
        }
      }
    }
    return best;
  },

  _ensureTile(map, x, y) {
    const k = key(map, x, y);
    if (this._cache.has(k)) return;
    this._cache.set(k, null);
    const coordsUrl = `/data/overlays/${encodeURIComponent(map)}/coords/${x}_${y}.json`;
    const dataUrl = `/data/overlays/${encodeURIComponent(map)}/wmo/${x}_${y}.json`;
    Promise.all([
      fetch(coordsUrl).then(r => r.ok ? r.json() : null),
      fetch(dataUrl).then(r => r.ok ? r.json() : null)
    ]).then(([c, d]) => {
      const worldRect = c?.worldRect || null;
      const features = Array.isArray(d?.features) ? d.features : [];
      this._cache.set(k, { worldRect, features });
    }).catch(() => {});
  }
};

function key(map, x, y) { return `${map}_${x}_${y}`; }

function visibleTiles(view) {
  const tl = screenToWorld(view, 0, 0);
  const br = screenToWorld(view, view.canvas?.width || 2048, view.canvas?.height || 2048);
  const x0 = clamp(Math.floor(tl.x), 0, 63), y0 = clamp(Math.floor(tl.y), 0, 63);
  const x1 = clamp(Math.ceil(br.x), 0, 63), y1 = clamp(Math.ceil(br.y), 0, 63);
  return { x0, y0, x1, y1 };
}
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function screenToWorld(view, sx, sy) {
  return { x: (sx - view.offsetX) / (view.tileSize * view.scale), y: (sy - view.offsetY) / (view.tileSize * view.scale) };
}
function worldToScreenTile(view, tileX, tileY, worldX, worldY, worldRect) {
  const px = (worldX - worldRect.minX) / Math.max(1e-6, (worldRect.maxX - worldRect.minX));
  const py = (worldY - worldRect.minY) / Math.max(1e-6, (worldRect.maxY - worldRect.minY));
  const tileLeft = view.offsetX + tileX * view.tileSize * view.scale;
  const tileTop  = view.offsetY + tileY * view.tileSize * view.scale;
  const size = view.tileSize * view.scale;
  return { x: tileLeft + px * size, y: tileTop + (1 - py) * size };
}
