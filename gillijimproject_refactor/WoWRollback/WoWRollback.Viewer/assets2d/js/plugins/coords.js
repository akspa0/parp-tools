export const CoordsOverlay = {
  id: 'coords',
  title: 'Game Coords',
  enabledByDefault: true,
  _cache: new Map(), // key: `${map}_${x}_${y}` -> { worldRect }

  onViewport(view) {
    // Load coords tiles for visible range
    const { x0, y0, x1, y1 } = visibleTiles(view);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        this._ensureTile(view.map, x, y);
      }
    }
  },

  render(ctx, view) {
    const { x0, y0, x1, y1 } = visibleTiles(view);
    ctx.save();
    ctx.strokeStyle = 'rgba(66, 165, 245, 0.35)';
    ctx.fillStyle = 'rgba(197, 225, 165, 0.9)';
    ctx.lineWidth = Math.max(1 / (window.devicePixelRatio || 1), 0.5);

    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const meta = this._cache.get(key(view.map, x, y));
        if (!meta || !meta.worldRect) continue;
        const { minX, maxX, minY, maxY } = meta.worldRect;
        // Draw tile worldRect outline and a couple of labels
        const tl = worldToScreenTile(view, x, y, minX, maxY, meta);
        const tr = worldToScreenTile(view, x, y, maxX, maxY, meta);
        const bl = worldToScreenTile(view, x, y, minX, minY, meta);
        const br = worldToScreenTile(view, x, y, maxX, minY, meta);

        ctx.beginPath();
        ctx.moveTo(tl.x, tl.y);
        ctx.lineTo(tr.x, tr.y);
        ctx.lineTo(br.x, br.y);
        ctx.lineTo(bl.x, bl.y);
        ctx.closePath();
        ctx.stroke();

        // Labels
        ctx.font = '11px system-ui, Arial, sans-serif';
        ctx.fillText(`${minX.toFixed(1)}, ${maxY.toFixed(1)}`, tl.x + 4, tl.y + 14);
        ctx.fillText(`${maxX.toFixed(1)}, ${minY.toFixed(1)}`, br.x - 120, br.y - 4);
      }
    }

    ctx.restore();
  },

  _ensureTile(map, x, y) {
    const k = key(map, x, y);
    if (this._cache.has(k)) return;
    const url = `/data/overlays/${encodeURIComponent(map)}/coords/${x}_${y}.json`;
    this._cache.set(k, null);
    fetch(url).then(r => r.ok ? r.json() : null).then(j => {
      if (!j) return;
      this._cache.set(k, { worldRect: j.worldRect || null, tile: { x, y } });
    }).catch(() => {});
  }
};

function key(map, x, y) { return `${map}_${x}_${y}`; }

function visibleTiles(view) {
  const tlx = Math.max(0, Math.floor((0 - view.offsetX) / (view.tileSize * view.scale)));
  const tly = Math.max(0, Math.floor((0 - view.offsetY) / (view.tileSize * view.scale)));
  const brx = Math.min(63, Math.ceil((view.canvasWidth || view.canvas?.width || 2048 - view.offsetX) / (view.tileSize * view.scale)));
  const bry = Math.min(63, Math.ceil((view.canvasHeight || view.canvas?.height || 2048 - view.offsetY) / (view.tileSize * view.scale)));
  return { x0: tlx, y0: tly, x1: brx, y1: bry };
}

// Convert world XY within this tile to screen XY, using worldRect and tile rect
function worldToScreenTile(view, tileX, tileY, worldX, worldY, meta) {
  const px = (worldX - meta.worldRect.minX) / Math.max(1e-6, (meta.worldRect.maxX - meta.worldRect.minX));
  const py = (worldY - meta.worldRect.minY) / Math.max(1e-6, (meta.worldRect.maxY - meta.worldRect.minY));
  const tileLeft = view.offsetX + tileX * view.tileSize * view.scale;
  const tileTop  = view.offsetY + tileY * view.tileSize * view.scale;
  const size = view.tileSize * view.scale;
  return { x: tileLeft + px * size, y: tileTop + (1 - py) * size };
}
