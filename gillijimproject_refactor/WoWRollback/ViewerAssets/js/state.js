// Global state management for viewer
export class State {
    constructor() {
        this.index = null;
        this.config = null;
        this.selectedVersion = null;
        this.selectedMap = null;
        this.lastVersion = null;
        this.lastMap = null;
        this.listeners = [];
        this.cacheBust = 0;
        this.overlayVariant = 'combined';
    }

    async loadIndex() {
        try {
            const response = await fetch('index.json');
            if (!response.ok) {
                throw new Error(`Failed to load index.json: ${response.status} ${response.statusText}`);
            }
            this.index = await response.json();
            this.selectedVersion = this.index.defaultVersion || this.index.versions[0];
            if (this.index.maps.length > 0) {
                this.selectedMap = this.index.maps[0].map;
            }
            this.notify();
            return this.index;
        } catch (error) {
            console.error('Error loading index:', error);
            throw error;
        }
    }

    async loadConfig() {
        try {
            const response = await fetch('config.json');
            if (response.ok) {
                this.config = await response.json();
            } else {
                this.config = defaultConfig();
            }
        } catch (e) {
            this.config = defaultConfig();
        }
        return this.config;
    }

    setVersion(version) {
        if (this.index.versions.includes(version)) {
            this.lastVersion = this.selectedVersion;
            this.selectedVersion = version;
            this.cacheBust = Date.now();
            this.notify();
        }
    }

    setMap(map) {
        const mapExists = this.index.maps.some(m => m.map === map);
        if (mapExists) {
            this.lastMap = this.selectedMap;
            this.selectedMap = map;
            this.cacheBust = Date.now();
            this.notify();
        }
    }

    setOverlayVariant(variant) {
        this.overlayVariant = variant;
        this.notify();
    }

    getMapData(mapName) {
        return this.index.maps.find(m => m.map === mapName);
    }

    getTilesForMap(mapName) {
        const mapData = this.getMapData(mapName);
        return mapData ? mapData.tiles : [];
    }

    isTileAvailable(mapName, row, col, version) {
        const tiles = this.getTilesForMap(mapName);
        const tile = tiles.find(t => t.row === row && t.col === col);
        return tile && tile.versions.includes(version);
    }

    getMinimapPath(mapName, row, col, version) {
        // New per-version structure: minimap/{version}/{map}/{map}_{col}_{row}.webp
        // Using WebP for 50-70% memory savings vs PNG
        const t = this.cacheBust || 0;
        return `minimap/${version}/${mapName}/${mapName}_${col}_${row}.webp?t=${t}`;
    }

    getOverlayPath(mapName, row, col, version, variant) {
        const safeVersion = version || this.selectedVersion;
        const safeVariant = variant || this.overlayVariant || 'combined';
        return `overlays/${safeVersion}/${mapName}/${safeVariant}/tile_r${row}_c${col}.json`;
    }

    getDiffPath(mapName, row, col) {
        return `diffs/${mapName}/tile_r${row}_c${col}.json`;
    }

    subscribe(listener) {
        this.listeners.push(listener);
    }

    notify() {
        this.listeners.forEach(listener => listener(this));
    }
}

export const state = new State();

function defaultConfig() {
    // Sensible defaults that match wow.tools expectations
    return {
        coordMode: 'wowtools',
        minimap: { width: 512, height: 512 },
        debugOverlayCorners: false
    };
}
