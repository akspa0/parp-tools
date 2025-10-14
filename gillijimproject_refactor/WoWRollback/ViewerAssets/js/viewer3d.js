/**
 * WoWRollback 3D Viewer
 * Three.js-based 3D visualization of WoW map placements
 */

class Viewer3D {
    constructor(mapName, version) {
        this.mapName = mapName;
        this.version = version;
        
        // Three.js core
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Data
        this.markers = [];
        this.terrainMeshes = [];
        this.layers = new Map();
        this.manifest = null;
        
        // Raycasting for click detection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Performance tracking
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
        
        // UI elements
        this.canvas = document.getElementById('viewer3d');
        this.loading = document.getElementById('loading');
        this.statsDiv = document.getElementById('stats');
        
        this.init();
    }

    async init() {
        console.log('[Viewer3D] Initializing...');
        
        // Setup Three.js scene
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLights();
        this.setupControls();
        this.setupEventListeners();
        
        // Load data
        try {
            await this.loadLayers();
            await this.loadManifest();
            await this.loadPlacements();
            
            // Hide loading screen
            this.loading.classList.add('hidden');
            
            // Start render loop
            this.animate();
            
            console.log('[Viewer3D] Initialization complete');
        } catch (error) {
            console.error('[Viewer3D] Initialization failed:', error);
            this.loading.innerHTML = `<div style="color: #f44336;">Error loading viewer: ${error.message}</div>`;
        }
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x87CEEB); // Sky blue
        this.scene.fog = new THREE.Fog(0x87CEEB, 5000, 20000);
    }

    setupCamera() {
        const aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 50000);
        this.camera.position.set(0, 1000, 1000);
        this.camera.lookAt(0, 0, 0);
    }

    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: this.canvas,
            antialias: true 
        });
        this.renderer.setSize(window.innerWidth - 300, window.innerHeight); // Account for sidebar
        this.renderer.setPixelRatio(window.devicePixelRatio);
    }

    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1000, 2000, 1000);
        directionalLight.castShadow = false; // Disable shadows for performance
        this.scene.add(directionalLight);

        // Hemisphere light (sky/ground)
        const hemisphereLight = new THREE.HemisphereLight(0x87CEEB, 0x8B4513, 0.4);
        this.scene.add(hemisphereLight);
    }

    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 100;
        this.controls.maxDistance = 10000;
        this.controls.maxPolarAngle = Math.PI / 2; // Don't go below ground
    }

    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Click detection
        this.canvas.addEventListener('click', (e) => this.onClick(e));

        // UI controls
        document.getElementById('select-all-btn').addEventListener('click', () => this.selectAllLayers());
        document.getElementById('deselect-all-btn').addEventListener('click', () => this.deselectAllLayers());
        document.getElementById('show-terrain').addEventListener('change', (e) => this.toggleTerrain(e.target.checked));
        document.getElementById('show-stats').addEventListener('change', (e) => this.toggleStats(e.target.checked));
        document.getElementById('show-grid').addEventListener('change', (e) => this.toggleGrid(e.target.checked));
    }

    onWindowResize() {
        const width = window.innerWidth - 300; // Account for sidebar
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    async loadLayers() {
        console.log('[Viewer3D] Loading UniqueID layers...');
        
        const url = `cached_maps/analysis/${this.version}/${this.mapName}/csv/id_ranges_by_map.csv`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.warn('[Viewer3D] No layer CSV found, using default layer');
                this.layers.set('all', { min: 0, max: 999999, count: 0, enabled: true });
                return;
            }

            const text = await response.text();
            const lines = text.split('\n').slice(1); // Skip header

            lines.forEach(line => {
                const trimmed = line.trim();
                if (!trimmed) return;

                const [map, min, max, count] = trimmed.split(',');
                if (map === this.mapName) {
                    const key = `${min}-${max}`;
                    this.layers.set(key, {
                        min: parseInt(min),
                        max: parseInt(max),
                        count: parseInt(count),
                        enabled: true
                    });
                }
            });

            console.log(`[Viewer3D] Loaded ${this.layers.size} layers`);
            this.renderLayerUI();
        } catch (error) {
            console.error('[Viewer3D] Failed to load layers:', error);
            this.layers.set('all', { min: 0, max: 999999, count: 0, enabled: true });
        }
    }

    renderLayerUI() {
        const list = document.getElementById('layer-list');
        list.innerHTML = '';

        this.layers.forEach((layer, key) => {
            const item = document.createElement('div');
            item.className = 'layer-item';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = layer.enabled;
            checkbox.id = `layer-${key}`;
            checkbox.addEventListener('change', () => {
                layer.enabled = checkbox.checked;
                this.applyLayerFilters();
            });

            const label = document.createElement('label');
            label.htmlFor = `layer-${key}`;
            label.textContent = `Range ${key}`;

            const count = document.createElement('span');
            count.className = 'layer-count';
            count.textContent = `(${layer.count})`;

            item.appendChild(checkbox);
            item.appendChild(label);
            item.appendChild(count);
            list.appendChild(item);
        });
    }

    async loadManifest() {
        console.log('[Viewer3D] Loading mesh manifest...');
        
        const url = `overlays/${this.version}/${this.mapName}/mesh/mesh_manifest.json`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.warn('[Viewer3D] No mesh manifest found');
                return;
            }

            this.manifest = await response.json();
            console.log(`[Viewer3D] Manifest loaded: ${this.manifest.tile_count} tiles`);
        } catch (error) {
            console.warn('[Viewer3D] Failed to load manifest:', error);
        }
    }

    async loadPlacements() {
        console.log('[Viewer3D] Loading placements...');
        
        if (!this.manifest) {
            console.warn('[Viewer3D] No manifest, cannot load placements');
            return;
        }

        let totalPlacements = 0;

        for (const tile of this.manifest.tiles) {
            const url = `overlays/${this.version}/${this.mapName}/combined/tile_r${tile.y}_c${tile.x}.json`;
            
            try {
                const response = await fetch(url);
                if (!response.ok) continue;

                const data = await response.json();
                
                data.layers?.forEach(layer => {
                    layer.placements?.forEach(p => {
                        this.createMarker(p);
                        totalPlacements++;
                    });
                });
            } catch (error) {
                console.warn(`[Viewer3D] Failed to load tile ${tile.x}_${tile.y}:`, error);
            }
        }

        console.log(`[Viewer3D] Loaded ${totalPlacements} placements`);
        document.getElementById('marker-count').textContent = totalPlacements;
        this.applyLayerFilters();
    }

    createMarker(placement) {
        // Create sphere marker
        const geometry = new THREE.SphereGeometry(10, 8, 8);
        
        // Color based on type
        const color = placement.kind === 'M2' ? 0xFF0000 : 0x0000FF; // Red for M2, Blue for WMO
        const material = new THREE.MeshPhongMaterial({ 
            color: color,
            emissive: color,
            emissiveIntensity: 0.2
        });
        
        const marker = new THREE.Mesh(geometry, material);
        
        // Position in world space (WoW coords → Three.js coords)
        // WoW: X=East, Y=South, Z=Up
        // Three.js: X=Right, Y=Up, Z=Forward
        marker.position.set(
            placement.worldX,
            placement.worldZ,
            -placement.worldY
        );
        
        // Store placement data
        marker.userData = {
            uniqueId: placement.uniqueId,
            assetPath: placement.assetPath,
            kind: placement.kind,
            worldX: placement.worldX,
            worldY: placement.worldY,
            worldZ: placement.worldZ,
            tileRow: placement.tileRow,
            tileCol: placement.tileCol
        };
        
        this.scene.add(marker);
        this.markers.push(marker);
    }

    applyLayerFilters() {
        let visibleCount = 0;

        this.markers.forEach(marker => {
            const uid = marker.userData.uniqueId;
            let visible = false;

            this.layers.forEach(layer => {
                if (layer.enabled && uid >= layer.min && uid <= layer.max) {
                    visible = true;
                }
            });

            marker.visible = visible;
            if (visible) visibleCount++;
        });

        document.getElementById('visible-count').textContent = visibleCount;
        console.log(`[Viewer3D] Visible markers: ${visibleCount}/${this.markers.length}`);
    }

    onClick(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Check for intersections with visible markers
        const visibleMarkers = this.markers.filter(m => m.visible);
        const intersects = this.raycaster.intersectObjects(visibleMarkers);

        if (intersects.length > 0) {
            const marker = intersects[0].object;
            this.showDetails(marker.userData);
        } else {
            this.hideDetails();
        }
    }

    showDetails(placement) {
        const popup = document.getElementById('details-popup');
        const title = document.getElementById('popup-title');
        const content = document.getElementById('popup-content');

        title.textContent = `${placement.kind} Placement`;

        content.innerHTML = `
            <div class="detail-row">
                <span class="detail-label">Asset Path:</span>
                <span class="detail-value">${placement.assetPath}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">UniqueID:</span>
                <span class="detail-value">${placement.uniqueId}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">World X:</span>
                <span class="detail-value">${placement.worldX.toFixed(2)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">World Y:</span>
                <span class="detail-value">${placement.worldY.toFixed(2)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">World Z:</span>
                <span class="detail-value">${placement.worldZ.toFixed(2)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Tile:</span>
                <span class="detail-value">${placement.tileCol}_${placement.tileRow}</span>
            </div>
        `;

        popup.classList.remove('hidden');
    }

    hideDetails() {
        document.getElementById('details-popup').classList.add('hidden');
    }

    selectAllLayers() {
        this.layers.forEach(layer => layer.enabled = true);
        this.renderLayerUI();
        this.applyLayerFilters();
    }

    deselectAllLayers() {
        this.layers.forEach(layer => layer.enabled = false);
        this.renderLayerUI();
        this.applyLayerFilters();
    }

    toggleTerrain(enabled) {
        this.terrainMeshes.forEach(mesh => mesh.visible = enabled);
        console.log(`[Viewer3D] Terrain ${enabled ? 'shown' : 'hidden'}`);
    }

    toggleStats(enabled) {
        this.statsDiv.classList.toggle('hidden', !enabled);
    }

    toggleGrid(enabled) {
        if (enabled && !this.gridHelper) {
            this.gridHelper = new THREE.GridHelper(10000, 100, 0x444444, 0x222222);
            this.scene.add(this.gridHelper);
        } else if (!enabled && this.gridHelper) {
            this.scene.remove(this.gridHelper);
            this.gridHelper = null;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        this.controls.update();

        // Update FPS
        this.frameCount++;
        const now = performance.now();
        if (now >= this.lastTime + 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (now - this.lastTime));
            document.getElementById('fps').textContent = this.fps;
            this.frameCount = 0;
            this.lastTime = now;
        }

        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
}
