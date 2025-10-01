// Canvas rendering for tile minimap
export class TileCanvas {
    constructor(canvasId, config) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.config = config || {};
        this.image = null;
        this.width = this.config.minimap?.width || 512;
        this.height = this.config.minimap?.height || 512;
        // Zoom/Pan state
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this._panning = false;
        this._last = { x: 0, y: 0 };

        // Input handlers
        this._bindInteractions();
    }

    async loadImage(path) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.decoding = 'async';
            const src = path + (path.includes('?') ? '&' : '?') + 't=' + Date.now();
            img.onload = () => {
                this.image = img;
                this.canvas.width = img.width;
                this.canvas.height = img.height;
                this.draw();
                resolve();
            };
            img.onerror = () => {
                reject(new Error(`Failed to load image: ${path}`));
            };
            img.src = src;
        });
    }

    draw() {
        // Clear and apply transform
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw placeholder if no image
        if (!this.image) {
            this.drawPlaceholder();
            return;
        }

        this.ctx.imageSmoothingEnabled = false;
        this.ctx.setTransform(this.scale, 0, 0, this.scale, this.offsetX, this.offsetY);
        this.ctx.drawImage(this.image, 0, 0);
    }

    drawPlaceholder() {
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        this.ctx.strokeStyle = '#444';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(0, 0, this.width, this.height);
        
        this.ctx.fillStyle = '#9E9E9E';
        this.ctx.font = '20px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('No minimap available', this.width / 2, this.height / 2);
    }

    drawOverlay(objects, colorMap) {
        if (!objects || objects.length === 0) return;
        // Use current transform (same as image)
        objects.forEach(obj => {
            if (typeof obj.pixelX !== 'number' || typeof obj.pixelY !== 'number') return;

            const label = obj.label || obj.diffType || 'default';
            const color = colorMap[label] || colorMap.default || '#4CAF50';
            const radius = this._radiusFor(label);

            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(obj.pixelX, obj.pixelY, radius, 0, 2 * Math.PI);
            this.ctx.fill();

            // Outline for visibility
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        });
    }

    worldToPixel(worldX, worldY) {
        // Convert WoW world coordinates to pixel coordinates
        // Based on the coordinate system in viewer-diff-plan.md
        const tileSize = 533.33333;
        const localX = (32 - (worldX / tileSize)) - Math.floor(32 - (worldX / tileSize));
        const localY = (32 - (worldY / tileSize)) - Math.floor(32 - (worldY / tileSize));
        
        const pixelX = localX * this.width;
        const pixelY = (1 - localY) * this.height;
        
        return { x: pixelX, y: pixelY };
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}
