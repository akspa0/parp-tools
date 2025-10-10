export class OverlayManager {
  constructor() {
    this.plugins = [];
    this.enabled = new Set();
  }
  register(plugin) {
    this.plugins.push(plugin);
    this.enabled.add(plugin.id);
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
}

// Example built-in grid overlay
export const GridOverlay = {
  id: 'grid',
  title: 'Grid',
  init(ctx) {},
  onViewport(view) {},
  render(ctx, view) {
    if (!view.showGrid) return;
    const { width, height } = ctx.canvas;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    const step = view.scale * 16; // one ADT tile cell size hint
    if (step < 8) { ctx.restore(); return; }
    const startX = - (view.offsetX % step);
    const startY = - (view.offsetY % step);
    for (let x = startX; x < width; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = startY; y < height; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
    ctx.restore();
  }
};
