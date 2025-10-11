export class OverlayManager {
  constructor() {
    this.plugins = [];
    this.enabled = new Set();
    this.shared = {}; // cross-plugin shared cache (e.g., coords mapping)
  }
  has(id) { return this.plugins.some(p => p.id === id); }
  register(plugin) {
    if (!plugin?.id) return;
    if (this.has(plugin.id)) return; // avoid duplicates
    this.plugins.push(plugin);
    if (plugin.enabledByDefault !== false) this.enabled.add(plugin.id);
  }
  toggle(id, on) {
    if (on) this.enabled.add(id); else this.enabled.delete(id);
  }
  async onViewport(view) {
    for (const p of this.plugins) {
      if (!this.enabled.has(p.id)) continue;
      if (p.onViewport) await p.onViewport(view);
    }
  }
  render(ctx, view) {
    for (const p of this.plugins) {
      if (!this.enabled.has(p.id)) continue;
      if (p.render) p.render(ctx, view);
    }
  }
  /**
   * Ask enabled plugins if any feature is hit at screen point (sx, sy).
   * Returns the closest hit: { pluginId, feature } or null.
   */
  hitTest(view, sx, sy) {
    let best = null, bestDist2 = Infinity;
    for (const p of this.plugins) {
      if (!this.enabled.has(p.id)) continue;
      if (!p.hitTest) continue;
      const hit = p.hitTest(view, sx, sy, this.shared);
      if (!hit) continue;
      const hits = Array.isArray(hit) ? hit : [hit];
      for (const h of hits) {
        const d2 = h?.dist2 ?? 0;
        if (best === null || d2 < bestDist2) { best = { pluginId: p.id, feature: h.feature, dist2: d2 }; bestDist2 = d2; }
      }
    }
    return best;
  }
}

export const GridOverlay = {
  id: 'grid',
  title: 'Grid',
  enabledByDefault: true,
  onViewport(view) {},
  render(ctx, view) {
    if (!view.showGrid) return;
    const { width, height } = ctx.canvas;
    const dpr = (window.devicePixelRatio || 1);
    const ts = view.tileSize; // pixels per world unit (tile) at scale=1
    const s = view.scale;
    const ox = view.offsetX;
    const oy = view.offsetY;

    // World <-> Screen
    const wxToSx = (wx) => ox + wx * ts * s;
    const wyToSy = (wy) => oy + wy * ts * s;
    const sxToWx = (sx) => (sx - ox) / (ts * s);
    const syToWy = (sy) => (sy - oy) / (ts * s);

    // Visible world extents
    const tlx = Math.max(0, sxToWx(0));
    const tly = Math.max(0, syToWy(0));
    const brx = Math.min(64, sxToWx(width));
    const bry = Math.min(64, syToWy(height));

    // Styling
    const minorColor = 'rgba(255,255,255,0.08)';
    const majorColor = 'rgba(255,255,255,0.25)';
    const lwMinor = Math.max(1 / dpr, 0.5);
    const lwMajor = Math.max(2 / dpr, 1);

    ctx.save();
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';

    // Vertical lines at each 1/16 world units (chunk grid)
    const xStartChunk = Math.floor(tlx * 16);
    const xEndChunk = Math.ceil(brx * 16);
    for (let cx = xStartChunk; cx <= xEndChunk; cx++) {
      const isTileBoundary = (cx % 16) === 0;
      const wx = cx / 16;
      const x = wxToSx(wx);
      if (x < 0 || x > width) continue;
      ctx.beginPath();
      ctx.strokeStyle = isTileBoundary ? majorColor : minorColor;
      ctx.lineWidth = isTileBoundary ? lwMajor : lwMinor;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines at each 1/16 world units
    const yStartChunk = Math.floor(tly * 16);
    const yEndChunk = Math.ceil(bry * 16);
    for (let cy = yStartChunk; cy <= yEndChunk; cy++) {
      const isTileBoundary = (cy % 16) === 0;
      const wy = cy / 16;
      const y = wyToSy(wy);
      if (y < 0 || y > height) continue;
      ctx.beginPath();
      ctx.strokeStyle = isTileBoundary ? majorColor : minorColor;
      ctx.lineWidth = isTileBoundary ? lwMajor : lwMinor;
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    ctx.restore();
  }
};
