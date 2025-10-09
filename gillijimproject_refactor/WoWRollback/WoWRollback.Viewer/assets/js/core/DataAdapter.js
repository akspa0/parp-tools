/**
 * DataAdapter - Converts WoW Rollback pipeline output to viewer format
 * Handles chunk-aware loading and LOD optimization
 */
export class DataAdapter {
    constructor() {
        this.masterIndex = null;
        this.idRanges = null;
        this.basePath = '';
    }
    
    /**
     * Load master index and ID ranges for a map
     */
    async loadMap(version, mapName, basePath = 'parp_out') {
        this.basePath = basePath;
        
        try {
            // Load master index (full placement data)
            const masterUrl = `${basePath}/${version}/master/${mapName}_master_index.json`;
            const masterResponse = await fetch(masterUrl);
            if (!masterResponse.ok) {
                console.warn(`[DataAdapter] Master index not found: ${masterUrl}`);
                return false;
            }
            this.masterIndex = await masterResponse.json();
            console.log(`[DataAdapter] Loaded master index: ${this.masterIndex.tileCount} tiles`);
            
            // Load ID ranges (optimization index)
            const rangesUrl = `${basePath}/${version}/master/${mapName}_id_ranges_by_tile.json`;
            const rangesResponse = await fetch(rangesUrl);
            if (rangesResponse.ok) {
                this.idRanges = await rangesResponse.json();
                console.log(`[DataAdapter] Loaded ID ranges for optimization`);
            }
            
            return true;
        } catch (error) {
            console.error('[DataAdapter] Failed to load map data:', error);
            return false;
        }
    }
    
    /**
     * Get placements for a specific tile
     */
    getTilePlacements(tileX, tileY, kind = null) {
        if (!this.masterIndex) return [];
        
        const tile = this.masterIndex.tiles.find(t => t.tileX === tileX && t.tileY === tileY);
        if (!tile) return [];
        
        let placements = tile.placements || [];
        
        // Filter by kind if specified
        if (kind) {
            placements = placements.filter(p => {
                if (kind === 'M2') return p.kind === 'MdxOrM2';
                if (kind === 'WMO') return p.kind === 'Wmo';
                return true;
            });
        }
        
        return placements;
    }
    
    /**
     * Get placements for a specific chunk within a tile
     */
    getChunkPlacements(tileX, tileY, chunkX, chunkY, kind = null) {
        const tilePlacements = this.getTilePlacements(tileX, tileY, kind);
        return tilePlacements.filter(p => p.chunkX === chunkX && p.chunkY === chunkY);
    }
    
    /**
     * Get chunk density data for a tile (for heatmap visualization)
     */
    getTileChunkDensity(tileX, tileY) {
        if (!this.idRanges) return null;
        
        const tile = this.idRanges.tiles.find(t => t.tileX === tileX && t.tileY === tileY);
        if (!tile) return null;
        
        // Create 16x16 density grid
        const density = Array(16).fill(0).map(() => Array(16).fill(0));
        
        tile.chunks?.forEach(chunk => {
            let total = 0;
            chunk.kinds?.forEach(kindData => {
                total += kindData.count || 0;
            });
            density[chunk.chunkY][chunk.chunkX] = total;
        });
        
        return density;
    }
    
    /**
     * Get chunks with high object density (potential prefabs)
     */
    getHighDensityChunks(threshold = 10) {
        if (!this.idRanges) return [];
        
        const highDensity = [];
        
        this.idRanges.tiles?.forEach(tile => {
            tile.chunks?.forEach(chunk => {
                let total = 0;
                chunk.kinds?.forEach(kindData => {
                    total += kindData.count || 0;
                });
                
                if (total >= threshold) {
                    highDensity.push({
                        tileX: tile.tileX,
                        tileY: tile.tileY,
                        chunkX: chunk.chunkX,
                        chunkY: chunk.chunkY,
                        count: total,
                        kinds: chunk.kinds
                    });
                }
            });
        });
        
        return highDensity.sort((a, b) => b.count - a.count);
    }
    
    /**
     * Convert pipeline placement to viewer format
     * NOTE: Recalculates world coordinates from tile position because the C# pipeline has a bug
     * @param {Object} placement - The placement data from pipeline
     * @param {number} tileX - WoW tile X coordinate (optional, will try to get from placement or uniqueId)
     * @param {number} tileY - WoW tile Y coordinate (optional, will try to get from placement or uniqueId)
     */
    convertPlacement(placement, tileX = null, tileY = null) {
        // Get tile coordinates from parameters, placement data, or derive from uniqueId
        if (tileX === null || tileY === null) {
            // Try to find the tile this placement belongs to
            const tile = this.masterIndex?.tiles?.find(t => 
                t.placements?.some(p => p.uniqueId === placement.uniqueId)
            );
            if (tile) {
                tileX = tile.tileX;
                tileY = tile.tileY;
            }
        }
        
        // Recalculate world coordinates from tile + offset
        // Formula: worldCoord = (32 - tileCoord) * 533.33333 + tileOffset
        // NOTE: C# pipeline bug - tileOffsetWest is wrong (set to TILE_SIZE), use rawWest instead
        const TILE_SIZE = 533.33333;
        const worldWest = (32 - tileX) * TILE_SIZE + (placement.rawWest || 0);
        const worldNorth = (32 - tileY) * TILE_SIZE + (placement.tileOffsetNorth || 0);
        
        return {
            uniqueId: placement.uniqueId,
            modelPath: placement.assetPath,
            // Use recalculated world coordinates
            worldX: worldWest,
            worldY: worldNorth,
            worldZ: placement.worldUp,
            // Additional metadata
            rotation: [placement.rotationX || 0, placement.rotationY || 0, placement.rotationZ || 0],
            scale: placement.scale || 1,
            flags: placement.flags || 0,
            // Chunk info
            chunkX: placement.chunkX,
            chunkY: placement.chunkY,
            // Type-specific
            doodadSet: placement.doodadSet,
            nameSet: placement.nameSet
        };
    }
    
    /**
     * Get placements for multiple tiles (for viewport loading)
     */
    getPlacementsForTiles(tiles, kind = null) {
        const allPlacements = [];
        
        tiles.forEach(tile => {
            const placements = this.getTilePlacements(tile.tileX, tile.tileY, kind);
            placements.forEach(p => {
                allPlacements.push(this.convertPlacement(p));
            });
        });
        
        return allPlacements;
    }
    
    /**
     * Get object count statistics
     */
    getStatistics() {
        if (!this.masterIndex) return null;
        
        let totalM2 = 0;
        let totalWMO = 0;
        let tilesWithData = 0;
        
        this.masterIndex.tiles?.forEach(tile => {
            if (tile.placements && tile.placements.length > 0) {
                tilesWithData++;
                tile.placements.forEach(p => {
                    if (p.kind === 'MdxOrM2') totalM2++;
                    else if (p.kind === 'Wmo') totalWMO++;
                });
            }
        });
        
        return {
            totalM2,
            totalWMO,
            totalObjects: totalM2 + totalWMO,
            totalTiles: this.masterIndex.tileCount,
            tilesWithData,
            version: this.masterIndex.version,
            map: this.masterIndex.map,
            generatedAt: this.masterIndex.generatedAtUtc
        };
    }
    
    /**
     * Find potential prefab clusters (objects that frequently appear together)
     */
    analyzePrefabClusters(minClusterSize = 5) {
        if (!this.idRanges) return [];
        
        const clusters = [];
        
        this.idRanges.tiles?.forEach(tile => {
            tile.chunks?.forEach(chunk => {
                // Look for chunks with multiple object types and high counts
                if (chunk.kinds && chunk.kinds.length > 1) {
                    const totalCount = chunk.kinds.reduce((sum, k) => sum + k.count, 0);
                    
                    if (totalCount >= minClusterSize) {
                        clusters.push({
                            tileX: tile.tileX,
                            tileY: tile.tileY,
                            chunkX: chunk.chunkX,
                            chunkY: chunk.chunkY,
                            totalCount,
                            composition: chunk.kinds.map(k => ({
                                kind: k.kind,
                                count: k.count
                            }))
                        });
                    }
                }
            });
        });
        
        return clusters.sort((a, b) => b.totalCount - a.totalCount);
    }
}
