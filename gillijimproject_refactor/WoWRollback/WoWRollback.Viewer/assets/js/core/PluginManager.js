export class PluginManager {
    constructor(map, coordSystem) {
        this.map = map;
        this.coords = coordSystem;
        this.plugins = new Map();
        
        // Setup viewport change listener
        this.map.on('moveend zoomend', () => {
            const bounds = this.map.getBounds();
            const zoom = this.map.getZoom();
            this.notifyViewportChange(bounds, zoom);
        });
    }
    
    register(plugin) {
        this.plugins.set(plugin.id, plugin);
        console.log(`[PluginManager] Registered: ${plugin.name}`);
    }
    
    unregister(pluginId) {
        const plugin = this.plugins.get(pluginId);
        if (plugin) {
            plugin.onDestroy();
            this.plugins.delete(pluginId);
            console.log(`[PluginManager] Unregistered: ${plugin.name}`);
        }
    }
    
    get(pluginId) {
        return this.plugins.get(pluginId);
    }
    
    async loadAll(version, mapName) {
        const promises = Array.from(this.plugins.values()).map(plugin =>
            plugin.onLoad(version, mapName)
        );
        await Promise.all(promises);
    }
    
    notifyViewportChange(bounds, zoom) {
        this.plugins.forEach(plugin => {
            plugin.onViewportChange(bounds, zoom);
        });
    }
    
    saveState() {
        const state = {};
        this.plugins.forEach((plugin, id) => {
            state[id] = plugin.getConfig();
        });
        localStorage.setItem('pluginState', JSON.stringify(state));
    }
    
    loadState() {
        const state = JSON.parse(localStorage.getItem('pluginState') || '{}');
        Object.entries(state).forEach(([id, config]) => {
            const plugin = this.plugins.get(id);
            if (plugin) {
                plugin.setConfig(config);
            }
        });
    }
}
