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
        this.markerData = []; // Store placement data separately
        this.terrainMeshes = [];
        this.layers = new Map();
        this.manifest = null;
        
        // Instanced rendering for performance
        this.instancedM2 = null;
        this.instancedWMO = null;
        
        // Dynamic terrain loading
        this.loadedTerrainTiles = new Map(); // tile key -> mesh
        this.terrainLoader = null;
        this.lastCameraPosition = new THREE.Vector3();
        
        // Dynamic placement loading
        this.loadedPlacementTiles = new Map(); // tile key -> markers array
        this.placementTileData = new Map(); // tile key -> tile info from manifest
        
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
            await this.loadTerrainMeshes(); // Load terrain GLB files
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
            antialias: false  // Disable antialiasing for better performance
        });
        this.renderer.setSize(window.innerWidth - 300, window.innerHeight); // Account for sidebar
        // Limit pixel ratio to 1 for better performance (no retina scaling)
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
    }

    setupLights() {
        // Using MeshBasicMaterial for markers, so lights don't affect them
        // Only add ambient light for potential future terrain meshes
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
        this.scene.add(ambientLight);
    }

    setupControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 100;
        this.controls.maxDistance = 10000;
        // Allow full 360° rotation - no angle restrictions
        // this.controls.maxPolarAngle = Math.PI / 2; // Removed restriction
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
                // Map name in CSV is like "Kalimdor_(21_30)", we want "Kalimdor"
                if (map && map.startsWith(this.mapName)) {
                    const key = `${min}-${max}`;
                    this.layers.set(key, {
                        min: parseInt(min),
                        max: parseInt(max),
                        count: parseInt(count),
                        enabled: true
                    });
                }
            });

            // If no layers were loaded, create a default "all" layer
            if (this.layers.size === 0) {
                console.warn('[Viewer3D] No layers matched, creating default layer');
                this.layers.set('all', { min: 0, max: 999999, count: 0, enabled: true });
            }

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

    async loadTerrainMeshes() {
        if (!this.manifest || !this.manifest.tiles) {
            console.warn('[Viewer3D] No manifest, skipping terrain meshes');
            return;
        }

        console.log('[Viewer3D] Dynamic terrain loading enabled');
        
        // Initialize loader
        this.terrainLoader = new window.GLTFLoader();
        
        // Start dynamic loading loop
        this.updateTerrainLOD();
    }

    updateTerrainLOD() {
        if (!this.manifest || !this.terrainLoader) return;
        
        // Check if camera moved significantly
        const cameraMoved = this.camera.position.distanceTo(this.lastCameraPosition) > 500;
        if (!cameraMoved && this.loadedTerrainTiles.size > 0) return;
        
        this.lastCameraPosition.copy(this.camera.position);
        
        // Find closest tiles to camera
        const maxLoadedTerrainTiles = 4;
        const maxLoadedPlacementTiles = 16; // 4x4 grid around camera
        const terrainLoadDistance = 5000;
        const placementLoadDistance = 5000; // Same distance for both
        
        // Calculate distances to all tiles using SAME method for both terrain and placements
        const tileDistances = this.manifest.tiles.map(tile => {
            // Use tile center from bounds for distance calculation
            const centerX = (tile.bounds.min_x + tile.bounds.max_x) / 2;
            const centerY = (tile.bounds.min_y + tile.bounds.max_y) / 2;
            const centerZ = (tile.bounds.min_z + tile.bounds.max_z) / 2;
            
            // Distance in 3D space
            const dx = this.camera.position.x - centerX;
            const dy = this.camera.position.y - centerZ; // Y is height in Three.js
            const dz = this.camera.position.z - (-centerY); // Z is depth
            const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            return { tile, distance, key: `${tile.x}_${tile.y}` };
        });
        
        // Sort by distance
        tileDistances.sort((a, b) => a.distance - b.distance);
        
        // Unload distant terrain tiles
        for (const [key, mesh] of this.loadedTerrainTiles.entries()) {
            const shouldKeep = tileDistances.slice(0, maxLoadedTerrainTiles).some(t => t.key === key);
            if (!shouldKeep) {
                this.scene.remove(mesh);
                this.loadedTerrainTiles.delete(key);
                console.log(`[Viewer3D] Unloaded terrain tile ${key}`);
            }
        }
        
        // DON'T unload placement tiles - they're all loaded upfront now
        // (Placement lazy loading is disabled)
        
        // Load closest terrain tiles
        const terrainTilesToLoad = tileDistances
            .slice(0, maxLoadedTerrainTiles)
            .filter(t => t.distance < terrainLoadDistance && !this.loadedTerrainTiles.has(t.key));
        
        for (const { tile, key } of terrainTilesToLoad) {
            this.loadTerrainTile(tile, key);
        }
    }

    async loadTerrainTile(tile, key) {
        const glbUrl = `overlays/${this.version}/${this.mapName}/mesh/${tile.glb}`;
        
        try {
            const gltf = await new Promise((resolve, reject) => {
                this.terrainLoader.load(glbUrl, resolve, undefined, reject);
            });
            
            // Apply material
            gltf.scene.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshBasicMaterial({ 
                        color: 0x808080,
                        wireframe: false
                    });
                }
            });
            
            // DON'T rotate or reposition the mesh!
            // The GLB is exported with world-space coordinates already baked in
            // Vertices are exported as (z, x, y) which matches Three.js (X, Y, Z)
            // No transformation needed - mesh is already in correct coordinate system
            
            this.scene.add(gltf.scene);
            this.loadedTerrainTiles.set(key, gltf.scene);
            
            const centerX = (tile.bounds.min_x + tile.bounds.max_x) / 2;
            const centerZ = (tile.bounds.min_z + tile.bounds.max_z) / 2;
            console.log(`[Viewer3D] Loaded tile ${key} | Bounds center: (${centerX.toFixed(0)}, ${centerZ.toFixed(0)}) | X[${tile.bounds.min_x.toFixed(0)}, ${tile.bounds.max_x.toFixed(0)}] Z[${tile.bounds.min_z.toFixed(0)}, ${tile.bounds.max_z.toFixed(0)}]`);
            
            // Add debug wireframe box around tile bounds
            const boxGeom = new THREE.BoxGeometry(
                tile.bounds.max_x - tile.bounds.min_x,
                100, // Small height
                tile.bounds.max_z - tile.bounds.min_z
            );
            const boxMat = new THREE.MeshBasicMaterial({ 
                color: 0x00ff00, 
                wireframe: true 
            });
            const box = new THREE.Mesh(boxGeom, boxMat);
            box.position.set(centerX, 50, centerZ);
            this.scene.add(box);
        } catch (error) {
            console.warn(`[Viewer3D] Failed to load tile ${key}`);
        }
    }

    async loadPlacements() {
        console.log('[Viewer3D] Loading ALL placements...');
        
        if (!this.manifest) {
            console.warn('[Viewer3D] No manifest, cannot load placements');
            return;
        }

        let totalMarkers = 0;
        let tilesLoaded = 0;
        
        // Load ALL tiles immediately
        for (const tile of this.manifest.tiles) {
            const key = `${tile.x}_${tile.y}`;
            const url = `overlays/${this.version}/${this.mapName}/combined/tile_r${tile.y}_c${tile.x}.json`;
            
            try {
                const response = await fetch(url);
                if (!response.ok) continue; // Skip tiles without placement data
                
                const data = await response.json();
                const tileMarkers = [];
                
                // Calculate tile world offset from bounds
                const tileOffsetX = (tile.bounds.min_x + tile.bounds.max_x) / 2;
                const tileOffsetZ = (tile.bounds.min_z + tile.bounds.max_z) / 2;
                
                data.layers?.forEach(layer => {
                    layer.kinds?.forEach(kind => {
                        kind.points?.forEach(p => {
                            const marker = this.createMarker(p, tileOffsetX, tileOffsetZ);
                            tileMarkers.push(marker);
                            totalMarkers++;
                        });
                    });
                });
                
                this.loadedPlacementTiles.set(key, tileMarkers);
                tilesLoaded++;
                
                // Update UI every 10 tiles
                if (tilesLoaded % 10 === 0) {
                    document.getElementById('marker-count').textContent = `${totalMarkers} (loading...)`;
                    await new Promise(resolve => setTimeout(resolve, 0)); // Yield to browser
                }
            } catch (error) {
                // Silently skip tiles that don't exist
            }
        }

        console.log(`[Viewer3D] Loaded ${totalMarkers} placements from ${tilesLoaded} tiles`);
        document.getElementById('marker-count').textContent = totalMarkers;
        this.applyLayerFilters();
    }

    async loadPlacementTile(tile, key) {
        // Don't reload if already loaded
        if (this.loadedPlacementTiles.has(key)) return;
        
        const url = `overlays/${this.version}/${this.mapName}/combined/tile_r${tile.y}_c${tile.x}.json`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) return;

            const data = await response.json();
            const tileMarkers = [];
            
            data.layers?.forEach(layer => {
                layer.kinds?.forEach(kind => {
                    kind.points?.forEach(p => {
                        const marker = this.createMarker(p);
                        tileMarkers.push(marker);
                    });
                });
            });
            
            this.loadedPlacementTiles.set(key, tileMarkers);
            console.log(`[Viewer3D] Loaded placement tile ${key} (${tileMarkers.length} markers)`);
            
            // Update counter
            const totalLoaded = Array.from(this.loadedPlacementTiles.values()).reduce((sum, arr) => sum + arr.length, 0);
            document.getElementById('marker-count').textContent = totalLoaded;
        } catch (error) {
            console.warn(`[Viewer3D] Failed to load placement tile ${key}`);
        }
    }

    unloadPlacementTile(key) {
        const markers = this.loadedPlacementTiles.get(key);
        if (!markers) return;
        
        // Remove markers from scene
        markers.forEach(marker => {
            this.scene.remove(marker);
            const index = this.markers.indexOf(marker);
            if (index > -1) this.markers.splice(index, 1);
        });
        
        this.loadedPlacementTiles.delete(key);
        console.log(`[Viewer3D] Unloaded placement tile ${key}`);
    }

    createMarker(placement, tileOffsetX = 0, tileOffsetZ = 0) {
        // Use shared geometries and materials for performance
        if (!this.sharedGeometry) {
            // Much larger spheres to be visible at terrain scale (16000+ units)
            this.sharedGeometry = new THREE.SphereGeometry(50, 8, 8);
        }
        
        if (!this.sharedMaterialM2) {
            this.sharedMaterialM2 = new THREE.MeshBasicMaterial({ 
                color: 0xFF0000,
                // Use MeshBasicMaterial instead of Phong - no lighting calculations
            });
        }
        
        if (!this.sharedMaterialWMO) {
            this.sharedMaterialWMO = new THREE.MeshBasicMaterial({ 
                color: 0x0000FF,
            });
        }
        
        // Color based on type
        const kind = placement.__kind || placement.kind;
        const material = kind === 'M2' ? this.sharedMaterialM2 : this.sharedMaterialWMO;
        
        const marker = new THREE.Mesh(this.sharedGeometry, material);
        
        // Use WORLD coordinates (already transformed by C# exporter)
        // The JSON has both placement (tile-local ADT coords) and world (transformed)
        const worldX = placement.world?.x ?? 0;
        const worldY = placement.world?.z ?? 0; // WoW Z = height
        const worldZ = placement.world?.y ?? 0; // WoW Y = north/south
        
        // Debug: Log first few markers to verify coordinates
        if (this.markers.length < 5) {
            console.log(`[Viewer3D] Marker ${this.markers.length}:`, { 
                worldX, worldY, worldZ,
                finalPos: { x: worldX, y: worldY, z: worldZ },
                placement: placement
            });
            
            // Add a HUGE debug sphere for first few markers
            const debugGeom = new THREE.SphereGeometry(200, 16, 16);
            const debugMat = new THREE.MeshBasicMaterial({ 
                color: 0xFFFF00,
                wireframe: false
            });
            const debugSphere = new THREE.Mesh(debugGeom, debugMat);
            debugSphere.position.copy(marker.position);
            this.scene.add(debugSphere);
            
            // Add text label
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 512;
            canvas.height = 128;
            ctx.fillStyle = '#ffff00';
            ctx.font = 'Bold 48px Arial';
            ctx.fillText(`M${this.markers.length}: (${worldX.toFixed(0)}, ${worldY.toFixed(0)}, ${worldZ.toFixed(0)})`, 10, 64);
            
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(1000, 250, 1);
            sprite.position.set(worldX, worldY + 300, worldZ);
            this.scene.add(sprite);
        }
        
        marker.position.set(
            worldX,      // World X
            worldY,      // World height
            worldZ       // World Z
        );
        
        // Store placement data
        marker.userData = {
            uniqueId: placement.uniqueId,
            assetPath: placement.assetPath,
            kind: kind,
            worldX: worldX,
            worldY: worldY,
            worldZ: worldZ,
            tileRow: placement.tileRow,
            tileCol: placement.tileCol
        };
        
        // Disable frustum culling for better performance with many objects
        marker.frustumCulled = true;
        
        this.scene.add(marker);
        this.markers.push(marker);
        
        return marker; // Return for lazy loading
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

        title.textContent = `${placement.kind || 'Unknown'} Placement`;

        content.innerHTML = `
            <div class="detail-row">
                <span class="detail-label">Asset Path:</span>
                <span class="detail-value">${placement.assetPath || 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">UniqueID:</span>
                <span class="detail-value">${placement.uniqueId || 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">World X:</span>
                <span class="detail-value">${placement.worldX != null ? placement.worldX.toFixed(2) : 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">World Y:</span>
                <span class="detail-value">${placement.worldY != null ? placement.worldY.toFixed(2) : 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">World Z:</span>
                <span class="detail-value">${placement.worldZ != null ? placement.worldZ.toFixed(2) : 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Tile:</span>
                <span class="detail-value">${placement.tileCol != null && placement.tileRow != null ? `${placement.tileCol}_${placement.tileRow}` : 'undefined_undefined'}</span>
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
    
    addAxisLabels() {
        // Create text sprites for axis labels
        const createTextSprite = (text, color) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 256;
            canvas.height = 128;
            ctx.fillStyle = color;
            ctx.font = 'Bold 48px Arial';
            ctx.fillText(text, 10, 64);
            
            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(material);
            sprite.scale.set(500, 250, 1);
            return sprite;
        };
        
        const xLabel = createTextSprite('+X (East)', '#ff0000');
        xLabel.position.set(5500, 0, 0);
        this.scene.add(xLabel);
        
        const yLabel = createTextSprite('+Y (Up)', '#00ff00');
        yLabel.position.set(0, 5500, 0);
        this.scene.add(yLabel);
        
        const zLabel = createTextSprite('+Z (North)', '#0000ff');
        zLabel.position.set(0, 0, 5500);
        this.scene.add(zLabel);
    }

    toggleGrid(enabled) {
        if (enabled) {
            // Add grid helper
            const gridHelper = new THREE.GridHelper(10000, 100);
            this.scene.add(gridHelper);
            
            // Add coordinate axes (RGB = XYZ)
            const axesHelper = new THREE.AxesHelper(5000);
            this.scene.add(axesHelper);
            
            // Add labeled axes at origin
            this.addAxisLabels();
        } else if (!enabled && this.gridHelper) {
            this.scene.remove(this.gridHelper);
            this.gridHelper = null;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        this.controls.update();

        // Update terrain LOD (dynamic loading/unloading)
        this.updateTerrainLOD();

        // Update marker LOD (distance-based culling for performance)
        // TEMPORARILY DISABLED to debug visibility
        // this.updateMarkerLOD();

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

    updateMarkerLOD() {
        // Only update every 10 frames for performance
        if (this.frameCount % 10 !== 0) return;
        
        const cameraPos = this.camera.position;
        const maxDistance = 3000; // Hide markers beyond 3000 units
        
        this.markers.forEach(marker => {
            if (!marker.userData) return;
            
            // Calculate distance to camera
            const dx = marker.position.x - cameraPos.x;
            const dy = marker.position.y - cameraPos.y;
            const dz = marker.position.z - cameraPos.z;
            const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            // Hide distant markers (but respect layer filtering)
            if (distance > maxDistance) {
                marker.visible = false;
            } else {
                // Restore visibility based on layer filtering
                const uid = marker.userData.uniqueId;
                let shouldBeVisible = false;
                this.layers.forEach(layer => {
                    if (layer.enabled && uid >= layer.min && uid <= layer.max) {
                        shouldBeVisible = true;
                    }
                });
                marker.visible = shouldBeVisible;
            }
        });
    }
}
