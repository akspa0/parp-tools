/**
 * Optimized Viewer with Canvas Rendering + Cluster LOD
 * 
 * Memory-efficient rendering using:
 * - Canvas for distant objects (no DOM!)
 * - Cluster-based LOD
 * - Viewport culling
 * - Lazy marker creation
 */

export class OptimizedObjectRenderer {
    constructor(map, clusters, placements) {
        this.map = map;
        this.clusters = clusters || [];
        this.placements = placements || [];
        
        // Canvas layer for distant rendering
        this.canvasLayer = L.canvas({ pane: 'overlayPane' });
        
        // Marker layer for close-up rendering
        this.markerLayer = L.layerGroup();
        this.markerLayer.addTo(map);
        
        // Viewport state
        this.currentZoom = map.getZoom();
        this.visibleTiles = new Set();
        this.loadedMarkers = new Map(); // tile -> markers
        
        // LOD thresholds
        this.LOD_CLUSTER_ONLY = 7;  // Zoom 0-7: Show only clusters
        this.LOD_MIXED = 10;        // Zoom 8-10: Clusters + nearby objects
        this.LOD_FULL = 13;         // Zoom 11+: All objects
        
        this.initCanvasRendering();
        this.initEventHandlers();
        
        console.log(`[OptimizedRenderer] Initialized with ${this.clusters.length} clusters, ${this.placements.length} objects`);
    }
    
    initCanvasRendering() {
        // Create canvas overlay using Leaflet's Canvas renderer
        const canvasPane = this.map.createPane('canvasObjects');
        canvasPane.style.zIndex = 450; // Above tiles, below UI
        
        this.canvas = L.canvas({ pane: canvasPane });
        this.canvasObjectsLayer = L.layerGroup();
        this.canvasObjectsLayer.addTo(this.map);
    }
    
    initEventHandlers() {
        this.map.on('zoomend', () => this.handleZoomChange());
        this.map.on('moveend', () => this.handleViewportChange());
        
        // Initial render
        this.render();
    }
    
    handleZoomChange() {
        const newZoom = this.map.getZoom();
        const oldZoom = this.currentZoom;
        this.currentZoom = newZoom;
        
        // Check if we crossed LOD boundaries
        if ((oldZoom <= this.LOD_CLUSTER_ONLY && newZoom > this.LOD_CLUSTER_ONLY) ||
            (oldZoom > this.LOD_CLUSTER_ONLY && newZoom <= this.LOD_CLUSTER_ONLY)) {
            this.render();
        } else if ((oldZoom <= this.LOD_MIXED && newZoom > this.LOD_MIXED) ||
                   (oldZoom > this.LOD_MIXED && newZoom <= this.LOD_MIXED)) {
            this.render();
        }
    }
    
    handleViewportChange() {
        const newVisibleTiles = this.getVisibleTiles();
        
        // Check if viewport changed significantly
        if (!this.tilesEqual(this.visibleTiles, newVisibleTiles)) {
            this.visibleTiles = newVisibleTiles;
            this.render();
        }
    }
    
    getVisibleTiles() {
        const bounds = this.map.getBounds();
        const zoom = this.map.getZoom();
        
        // Convert bounds to ADT tile coordinates
        // WoW coordinate system: 64x64 ADT tiles
        const tiles = new Set();
        
        // Add buffer tiles
        const buffer = 1;
        
        // Calculate tile range from viewport bounds
        const minTileX = Math.floor(bounds.getWest() / 533.33) - buffer;
        const maxTileX = Math.ceil(bounds.getEast() / 533.33) + buffer;
        const minTileY = Math.floor(bounds.getSouth() / 533.33) - buffer;
        const maxTileY = Math.ceil(bounds.getNorth() / 533.33) + buffer;
        
        for (let x = minTileX; x <= maxTileX; x++) {
            for (let y = minTileY; y <= maxTileY; y++) {
                if (x >= 0 && x < 64 && y >= 0 && y < 64) {
                    tiles.add(`${x}_${y}`);
                }
            }
        }
        
        return tiles;
    }
    
    tilesEqual(set1, set2) {
        if (set1.size !== set2.size) return false;
        for (let tile of set1) {
            if (!set2.has(tile)) return false;
        }
        return true;
    }
    
    render() {
        const zoom = this.currentZoom;
        
        if (zoom <= this.LOD_CLUSTER_ONLY) {
            this.renderClustersOnly();
        } else if (zoom <= this.LOD_MIXED) {
            this.renderMixed();
        } else {
            this.renderFull();
        }
    }
    
    renderClustersOnly() {
        console.log('[OptimizedRenderer] LOD: Clusters only');
        
        // Clear markers
        this.markerLayer.clearLayers();
        this.loadedMarkers.clear();
        
        // Render clusters as simple circles on canvas
        this.canvasObjectsLayer.clearLayers();
        
        for (const cluster of this.clusters) {
            // Skip clusters outside viewport
            if (!this.isClusterVisible(cluster)) continue;
            
            const latlng = this.coordsToLatLng(cluster.centroidX, cluster.centroidY);
            
            // Create circle marker for cluster
            const circle = L.circleMarker(latlng, {
                renderer: this.canvas,
                radius: Math.max(3, Math.log(cluster.objectCount) * 2),
                fillColor: cluster.isPlacementStamp ? '#FF6B35' : '#2196F3',
                color: '#000',
                weight: 1,
                fillOpacity: 0.7
            });
            
            // Simple popup with cluster info
            circle.bindPopup(`
                <b>Cluster ${cluster.clusterId}</b><br>
                Objects: ${cluster.objectCount}<br>
                ${cluster.isPlacementStamp ? '🖌️ Placement Stamp' : ''}
                <br><small>Zoom in to see objects</small>
            `);
            
            this.canvasObjectsLayer.addLayer(circle);
        }
        
        console.log(`[OptimizedRenderer] Rendered ${this.canvasObjectsLayer.getLayers().length} clusters`);
    }
    
    renderMixed() {
        console.log('[OptimizedRenderer] LOD: Mixed');
        
        // Show clusters as icons + nearby individual objects
        this.canvasObjectsLayer.clearLayers();
        this.markerLayer.clearLayers();
        this.loadedMarkers.clear();
        
        const visibleObjects = this.getVisibleObjects();
        const objectsToShow = Math.min(visibleObjects.length, 1000); // Limit to 1000
        
        for (let i = 0; i < objectsToShow; i++) {
            const obj = visibleObjects[i];
            this.createMarker(obj);
        }
        
        console.log(`[OptimizedRenderer] Rendered ${objectsToShow} objects (${visibleObjects.length} visible)`);
    }
    
    renderFull() {
        console.log('[OptimizedRenderer] LOD: Full detail');
        
        this.canvasObjectsLayer.clearLayers();
        
        const visibleObjects = this.getVisibleObjects();
        const objectsToShow = Math.min(visibleObjects.length, 5000); // Hard limit
        
        // Clear markers for tiles no longer visible
        for (const [tile, markers] of this.loadedMarkers) {
            if (!this.visibleTiles.has(tile)) {
                markers.forEach(m => this.markerLayer.removeLayer(m));
                this.loadedMarkers.delete(tile);
            }
        }
        
        // Create markers for new tiles
        for (const tile of this.visibleTiles) {
            if (!this.loadedMarkers.has(tile)) {
                this.loadTileMarkers(tile);
            }
        }
        
        console.log(`[OptimizedRenderer] Rendered ${this.markerLayer.getLayers().length} objects in ${this.visibleTiles.size} tiles`);
    }
    
    getVisibleObjects() {
        const bounds = this.map.getBounds();
        
        return this.placements.filter(obj => {
            const latlng = this.coordsToLatLng(obj.worldX, obj.worldY);
            return bounds.contains(latlng);
        });
    }
    
    isClusterVisible(cluster) {
        const latlng = this.coordsToLatLng(cluster.centroidX, cluster.centroidY);
        return this.map.getBounds().contains(latlng);
    }
    
    loadTileMarkers(tile) {
        const [x, y] = tile.split('_').map(Number);
        const tileObjects = this.placements.filter(obj => 
            this.getTileForObject(obj) === tile
        );
        
        const markers = [];
        for (const obj of tileObjects) {
            const marker = this.createMarker(obj);
            if (marker) markers.push(marker);
        }
        
        this.loadedMarkers.set(tile, markers);
    }
    
    createMarker(obj) {
        const latlng = this.coordsToLatLng(obj.worldX, obj.worldY);
        
        const marker = L.circleMarker(latlng, {
            radius: 3,
            fillColor: obj.type === 'M2' ? '#2196F3' : '#FF9800',
            color: '#000',
            weight: 1,
            fillOpacity: 0.8
        });
        
        marker.bindPopup(`
            <b>${obj.assetPath}</b><br>
            Type: ${obj.type}<br>
            UniqueID: ${obj.uniqueId}<br>
            Position: (${obj.worldX.toFixed(1)}, ${obj.worldY.toFixed(1)}, ${obj.worldZ.toFixed(1)})
        `);
        
        this.markerLayer.addLayer(marker);
        return marker;
    }
    
    getTileForObject(obj) {
        const x = Math.floor(obj.worldX / 533.33);
        const y = Math.floor(obj.worldY / 533.33);
        return `${x}_${y}`;
    }
    
    coordsToLatLng(x, y) {
        // Convert WoW coordinates to Leaflet lat/lng
        // This needs to match your coordinate system
        return L.latLng(y / 533.33, x / 533.33);
    }
}
