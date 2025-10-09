/**
 * Base class for all overlay plugins
 * Extend this to create new overlays
 */
export class OverlayPlugin {
    constructor(id, name, map, coordSystem) {
        this.id = id;
        this.name = name;
        this.map = map;
        this.coords = coordSystem; // Short alias
        this.coordSystem = coordSystem; // Full name for clarity
        
        this.enabled = false;
        this.visible = false;
        this.opacity = 1.0;
        this.zIndex = 400;
        
        this.cache = new Map();
        this.layers = [];
    }
    
    // Lifecycle hooks (override these)
    async onLoad(version, mapName) {
        // Load plugin data
    }
    
    onEnable() {
        // Plugin enabled
        this.enabled = true;
    }
    
    onDisable() {
        // Plugin disabled
        this.enabled = false;
        this.clearLayers();
    }
    
    onShow() {
        // Show layers
        this.visible = true;
        this.layers.forEach(layer => layer.addTo(this.map));
    }
    
    onHide() {
        // Hide layers
        this.visible = false;
        this.layers.forEach(layer => layer.remove());
    }
    
    onViewportChange(bounds, zoom) {
        // Viewport changed - load visible data
        if (!this.enabled || !this.visible) return;
        this.loadVisibleData(bounds, zoom);
    }
    
    onDestroy() {
        // Cleanup
        this.clearLayers();
        this.cache.clear();
    }
    
    // Helper methods
    clearLayers() {
        this.layers.forEach(layer => layer.remove());
        this.layers = [];
    }
    
    setOpacity(value) {
        this.opacity = value;
        this.layers.forEach(layer => {
            if (layer.setOpacity) layer.setOpacity(value);
            if (layer.setStyle) layer.setStyle({ opacity: value, fillOpacity: value });
        });
    }
    
    setZIndex(value) {
        this.zIndex = value;
        this.layers.forEach(layer => {
            if (layer.setZIndex) layer.setZIndex(value);
        });
    }
    
    // Must implement
    async loadVisibleData(bounds, zoom) {
        throw new Error('Plugin must implement loadVisibleData()');
    }
    
    getConfig() {
        return {
            enabled: this.enabled,
            visible: this.visible,
            opacity: this.opacity,
            zIndex: this.zIndex
        };
    }
    
    setConfig(config) {
        this.opacity = config.opacity ?? this.opacity;
        this.zIndex = config.zIndex ?? this.zIndex;
        if (config.enabled !== undefined) {
            config.enabled ? this.onEnable() : this.onDisable();
        }
        if (config.visible !== undefined) {
            config.visible ? this.onShow() : this.onHide();
        }
    }
}
