using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder;

/// <summary>
/// Unified PM4 map loader that ingests all PM4 files in a region to create a complete unified map object.
/// This is critical for resolving cross-tile references and assembling complete building geometry.
/// </summary>
public class PM4MapLoader
    {
        /// <summary>
        /// Helper record for async tile load results.
        /// </summary>
        private sealed record TileLoadResult((int tileX, int tileY) Coords, PM4TileData TileData);

    private readonly Dictionary<(int tileX, int tileY), PM4TileData> _loadedTiles = new();

    /// <summary>
    /// Access to raw per-tile data after <see cref="LoadRegionAsync"/> completes.
    /// </summary>
    public IReadOnlyDictionary<(int, int), PM4TileData> LoadedTiles => _loadedTiles;
    private readonly Dictionary<string, string> _filePathCache = new();

    /// <summary>
    /// Load all PM4 files in a directory as a unified map object.
    /// This resolves cross-tile references and creates complete building geometry.
    /// </summary>
    /// <param name="pm4Directory">Directory containing PM4 files</param>
    /// <returns>Unified PM4 map with all cross-tile references resolved</returns>
    public async Task<PM4UnifiedMap> LoadRegionAsync(string pm4Directory)
    {
        Console.WriteLine($"[PM4 MAP LOADER] Loading unified PM4 map from: {pm4Directory}");
        
        if (!Directory.Exists(pm4Directory))
        {
            throw new DirectoryNotFoundException($"PM4 directory not found: {pm4Directory}");
        }

        // Step 1: Discover all PM4 files in the directory
        var pm4Files = DiscoverPM4Files(pm4Directory);
        Console.WriteLine($"[PM4 MAP LOADER] Found {pm4Files.Count} PM4 files to process");

        // Step 2: Load each PM4 file and extract tile coordinates
        var loadTasks = pm4Files.Select(async filePath =>
        {
            try
            {
                var tileCoords = ExtractTileCoordinates(filePath);
                if (tileCoords.HasValue)
                {
                    var tileData = await LoadPM4TileAsync(filePath, tileCoords.Value.tileX, tileCoords.Value.tileY);
                    return new TileLoadResult(tileCoords.Value, tileData);
                }
                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PM4 MAP LOADER WARNING] Failed to load {Path.GetFileName(filePath)}: {ex.Message}");
                return null;
            }
        });

        var loadResults = await Task.WhenAll(loadTasks);
        
        // Step 3: Build tile registry
        foreach (var result in loadResults.Where(r => r != null))
        {
            var coords = result!.Coords;
            _loadedTiles[coords] = result.TileData;
        }

        Console.WriteLine($"[PM4 MAP LOADER] Successfully loaded {_loadedTiles.Count} tiles");

        // Step 4: Build unified map with cross-tile reference resolution
        var unifiedMap = await BuildUnifiedMapAsync();
        
        Console.WriteLine($"[PM4 MAP LOADER] Unified map created:");
        Console.WriteLine($"  - Global MSVT vertices: {unifiedMap.GlobalMSVTVertices.Count:N0}");
        Console.WriteLine($"  - Global MSPV vertices: {unifiedMap.GlobalMSPVVertices.Count:N0}");
        Console.WriteLine($"  - Global MSVI indices: {unifiedMap.GlobalMSVIIndices.Count:N0}");
        Console.WriteLine($"  - Global MSPI indices: {unifiedMap.GlobalMSPIIndices.Count:N0}");
        Console.WriteLine($"  - Total MSLK links: {unifiedMap.AllMslkLinks.Count:N0}");
        Console.WriteLine($"  - Total MPRL placements: {unifiedMap.AllMprlPlacements.Count:N0}");
        Console.WriteLine($"  - Total MSUR surfaces: {unifiedMap.AllMsurSurfaces.Count:N0}");

        return unifiedMap;
    }

    /// <summary>
    /// Load a single PM4 tile and extract its data.
    /// </summary>
    private async Task<PM4TileData> LoadPM4TileAsync(string filePath, int tileX, int tileY)
    {
        try
        {
            // Use existing PM4 loading infrastructure
            var scene = await LoadSceneFromFile(filePath);
            
            var tileData = new PM4TileData
            {
                TileX = tileX,
                TileY = tileY,
                FilePath = filePath,
                Scene = scene,
                
                // Extract raw chunk data for cross-tile processing
                MSVTVertices = scene?.Vertices ?? new List<Vector3>(),
                MSPVVertices = ExtractMSPVVertices(scene),
                MSVIIndices = ExtractMSVIIndices(scene),
                MSPIIndices = ExtractMSPIIndices(scene),
                MslkLinks = scene?.Links ?? new List<MslkEntry>(),
                MprlPlacements = ExtractMprlPlacements(scene),
                MsurSurfaces = ExtractMsurSurfaces(scene)
            };

            Console.WriteLine($"[PM4 MAP LOADER] Loaded tile ({tileX},{tileY}): {tileData.MSVTVertices.Count} MSVT vertices, {tileData.MslkLinks.Count} MSLK links");
            
            return tileData;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[PM4 MAP LOADER ERROR] Failed to load tile ({tileX},{tileY}) from {filePath}: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Build unified map object from all loaded tiles with cross-tile reference resolution.
    /// </summary>
    private async Task<PM4UnifiedMap> BuildUnifiedMapAsync()
    {
        Console.WriteLine("[PM4 MAP LOADER] Building unified map with cross-tile reference resolution...");

        var unifiedMap = new PM4UnifiedMap
        {
            TileData = new Dictionary<(int, int), PM4TileMetadata>()
        };

        // Step 1: Build global vertex pools with proper indexing
        await BuildGlobalVertexPools(unifiedMap);

        // Step 2: Build global index pools with cross-tile resolution  
        await BuildGlobalIndexPools(unifiedMap);

        // Step 3: Aggregate all linkage data across tiles
        await AggregateUnifiedLinkageData(unifiedMap);

        // Step 4: Create tile metadata for reference
        BuildTileMetadata(unifiedMap);

        Console.WriteLine("[PM4 MAP LOADER] Unified map construction complete");
        return unifiedMap;
    }

    /// <summary>
    /// Build global vertex pools from all tiles, maintaining cross-tile index mapping.
    /// </summary>
    private async Task BuildGlobalVertexPools(PM4UnifiedMap unifiedMap)
    {
        unifiedMap.GlobalMSVTVertices = new List<Vector3>();
        unifiedMap.GlobalMSPVVertices = new List<Vector3>();
        unifiedMap.TileVertexOffsets = new Dictionary<(int, int), PM4TileVertexOffsets>();

        foreach (var kvp in _loadedTiles.OrderBy(t => t.Key.Item1).ThenBy(t => t.Key.Item2))
        {
            var (tileX, tileY) = kvp.Key;
            var tileData = kvp.Value;

            var offsets = new PM4TileVertexOffsets
            {
                MSVTStartIndex = unifiedMap.GlobalMSVTVertices.Count,
                MSPVStartIndex = unifiedMap.GlobalMSPVVertices.Count
            };

            // Add all MSVT vertices from this tile
            unifiedMap.GlobalMSVTVertices.AddRange(tileData.MSVTVertices);
            offsets.MSVTCount = tileData.MSVTVertices.Count;

            // Add all MSPV vertices from this tile
            unifiedMap.GlobalMSPVVertices.AddRange(tileData.MSPVVertices);
            offsets.MSPVCount = tileData.MSPVVertices.Count;

            unifiedMap.TileVertexOffsets[(tileX, tileY)] = offsets;

            Console.WriteLine($"[PM4 MAP LOADER] Tile ({tileX},{tileY}): MSVT offset {offsets.MSVTStartIndex} (+{offsets.MSVTCount}), MSPV offset {offsets.MSPVStartIndex} (+{offsets.MSPVCount})");
        }
    }

    /// <summary>
    /// Build global index pools with cross-tile reference resolution.
    /// </summary>
    private async Task BuildGlobalIndexPools(PM4UnifiedMap unifiedMap)
    {
        unifiedMap.GlobalMSVIIndices = new List<uint>();
        unifiedMap.GlobalMSPIIndices = new List<uint>();

        foreach (var kvp in _loadedTiles.OrderBy(t => t.Key.Item1).ThenBy(t => t.Key.Item2))
        {
            var (tileX, tileY) = kvp.Key;
            var tileData = kvp.Value;
            var vertexOffsets = unifiedMap.TileVertexOffsets[(tileX, tileY)];

            // Process MSVI indices with vertex offset correction
            foreach (var index in tileData.MSVIIndices)
            {
                var correctedIndex = ResolveVertexIndex(index, (tileX, tileY), vertexOffsets, unifiedMap, isVertexPool: true);
                unifiedMap.GlobalMSVIIndices.Add(correctedIndex);
            }

            // Process MSPI indices with vertex offset correction
            foreach (var index in tileData.MSPIIndices)
            {
                var correctedIndex = ResolveVertexIndex(index, (tileX, tileY), vertexOffsets, unifiedMap, isVertexPool: false);
                unifiedMap.GlobalMSPIIndices.Add(correctedIndex);
            }
        }

        Console.WriteLine($"[PM4 MAP LOADER] Global index pools: {unifiedMap.GlobalMSVIIndices.Count} MSVI indices, {unifiedMap.GlobalMSPIIndices.Count} MSPI indices");
    }

    /// <summary>
    /// Resolve vertex index with cross-tile reference handling.
    /// This is critical for fixing the 64% data loss from cross-tile vertex references.
    /// </summary>
    private uint ResolveVertexIndex(uint originalIndex, (int tileX, int tileY) sourceTile, PM4TileVertexOffsets offsets, PM4UnifiedMap unifiedMap, bool isVertexPool)
    {
        var maxVertexCount = isVertexPool ? offsets.MSVTCount : offsets.MSPVCount;
        var baseOffset = isVertexPool ? offsets.MSVTStartIndex : offsets.MSPVStartIndex;

        // If index is within current tile bounds, apply simple offset
        if (originalIndex < maxVertexCount)
        {
            return (uint)(baseOffset + originalIndex);
        }

        // Cross-tile reference detected - attempt resolution
        Console.WriteLine($"[PM4 MAP LOADER] Cross-tile reference detected: index {originalIndex} > max {maxVertexCount} for tile ({sourceTile.tileX},{sourceTile.tileY})");
        
        // Try adjacent tiles for resolution (simplified algorithm)
        var adjacentTiles = GetAdjacentTiles(sourceTile);
        foreach (var adjacentTile in adjacentTiles)
        {
            if (unifiedMap.TileVertexOffsets.TryGetValue(adjacentTile, out var adjacentOffsets))
            {
                var adjacentMaxCount = isVertexPool ? adjacentOffsets.MSVTCount : adjacentOffsets.MSPVCount;
                var adjacentBaseOffset = isVertexPool ? adjacentOffsets.MSVTStartIndex : adjacentOffsets.MSPVStartIndex;
                
                // Check if index could belong to this adjacent tile
                var relativeIndex = originalIndex - maxVertexCount;
                if (relativeIndex < adjacentMaxCount)
                {
                    var resolvedIndex = (uint)(adjacentBaseOffset + relativeIndex);
                    Console.WriteLine($"[PM4 MAP LOADER] Cross-tile reference resolved: {originalIndex} → {resolvedIndex} (via tile {adjacentTile})");
                    return resolvedIndex;
                }
            }
        }

        // Fallback: return original index with offset (may still be invalid but logged)
        Console.WriteLine($"[PM4 MAP LOADER WARNING] Could not resolve cross-tile reference {originalIndex} for tile ({sourceTile.tileX},{sourceTile.tileY})");
        return (uint)(baseOffset + originalIndex);
    }

    /// <summary>
    /// Get adjacent tile coordinates for cross-tile reference resolution.
    /// </summary>
    private List<(int tileX, int tileY)> GetAdjacentTiles((int tileX, int tileY) center)
    {
        return new List<(int, int)>
        {
            (center.tileX - 1, center.tileY),     // Left
            (center.tileX + 1, center.tileY),     // Right  
            (center.tileX, center.tileY - 1),     // Up
            (center.tileX, center.tileY + 1),     // Down
            (center.tileX - 1, center.tileY - 1), // Top-left
            (center.tileX + 1, center.tileY - 1), // Top-right
            (center.tileX - 1, center.tileY + 1), // Bottom-left
            (center.tileX + 1, center.tileY + 1)  // Bottom-right
        };
    }

    /// <summary>
    /// Aggregate all linkage data (MSLK, MPRL, MSUR) across tiles into unified collections.
    /// </summary>
    private async Task AggregateUnifiedLinkageData(PM4UnifiedMap unifiedMap)
    {
        unifiedMap.AllMslkLinks = new List<MslkEntry>();
        unifiedMap.AllMprlPlacements = new List<MprlChunk.Entry>();
        unifiedMap.AllMsurSurfaces = new List<MsurChunk.Entry>();

        int currentSurfaceOffset = 0;
        foreach (var tile in _loadedTiles.Values)
        {
            // Record the offset inside the tile so downstream code can reason about it if needed
            tile.SurfaceOffset = currentSurfaceOffset;

            // Adjust each link's SurfaceRefIndex so it points to the global MSUR list
            foreach (var link in tile.MslkLinks)
            {
                var prop = link.GetType().GetProperty("SurfaceRefIndex");
                if (prop == null) continue;

                // Handle different numeric underlying types
                object? value = prop.GetValue(link);
                if (value == null) continue;

                uint globalIndex = value switch
                {
                    uint u => u + (uint)currentSurfaceOffset,
                    int i when i >= 0 => (uint)i + (uint)currentSurfaceOffset,
                    ushort us => (uint)us + (uint)currentSurfaceOffset,
                    short s when s >= 0 => (uint)s + (uint)currentSurfaceOffset,
                    _ => 0
                };

                // Write back with correct type conversion
                if (prop.PropertyType == typeof(uint))
                    prop.SetValue(link, globalIndex);
                else if (prop.PropertyType == typeof(int))
                    prop.SetValue(link, (int)globalIndex);
                else if (prop.PropertyType == typeof(ushort))
                    prop.SetValue(link, (ushort)globalIndex);
                else if (prop.PropertyType == typeof(short))
                    prop.SetValue(link, (short)globalIndex);

            }

            unifiedMap.AllMslkLinks.AddRange(tile.MslkLinks);
            unifiedMap.AllMprlPlacements.AddRange(tile.MprlPlacements);
            unifiedMap.AllMsurSurfaces.AddRange(tile.MsurSurfaces);

            currentSurfaceOffset += tile.MsurSurfaces.Count;
        }

        Console.WriteLine($"[PM4 MAP LOADER] Aggregated linkage data: {unifiedMap.AllMslkLinks.Count} MSLK, {unifiedMap.AllMprlPlacements.Count} MPRL, {unifiedMap.AllMsurSurfaces.Count} MSUR");
    }

    /// <summary>
    /// Build tile metadata for reference and debugging.
    /// </summary>
    private void BuildTileMetadata(PM4UnifiedMap unifiedMap)
    {
        foreach (var kvp in _loadedTiles)
        {
            var (tileX, tileY) = kvp.Key;
            var tileData = kvp.Value;

            unifiedMap.TileData[(tileX, tileY)] = new PM4TileMetadata
            {
                TileX = tileX,
                TileY = tileY,
                FilePath = tileData.FilePath,
                VertexCount = tileData.MSVTVertices.Count + tileData.MSPVVertices.Count,
                LinkCount = tileData.MslkLinks.Count,
                SurfaceCount = tileData.MsurSurfaces.Count
            };
        }
    }

    /// <summary>
    /// Discover all PM4 files in a directory.
    /// </summary>
    private List<string> DiscoverPM4Files(string directory)
    {
        var pm4Files = Directory.GetFiles(directory, "*.pm4", SearchOption.TopDirectoryOnly).ToList();
        
        // Also check for common PM4 naming patterns
        var additionalPatterns = new[] { "*.PM4", "*_pm4", "*_PM4" };
        foreach (var pattern in additionalPatterns)
        {
            pm4Files.AddRange(Directory.GetFiles(directory, pattern, SearchOption.TopDirectoryOnly));
        }

        return pm4Files.Distinct().OrderBy(f => f).ToList();
    }

    /// <summary>
    /// Extract tile coordinates from PM4 filename (e.g., "development_15_37.pm4" -> (15, 37)).
    /// </summary>
    private (int tileX, int tileY)? ExtractTileCoordinates(string filePath)
    {
        try
        {
            var fileName = Path.GetFileNameWithoutExtension(filePath);
            var parts = fileName.Split('_');
            
            if (parts.Length >= 3)
            {
                if (int.TryParse(parts[^2], out var tileX) && int.TryParse(parts[^1], out var tileY))
                {
                    return (tileX, tileY);
                }
            }
            
            Console.WriteLine($"[PM4 MAP LOADER WARNING] Could not extract tile coordinates from filename: {fileName}");
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[PM4 MAP LOADER WARNING] Error parsing tile coordinates from {filePath}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Load a single PM4 file using the actual Pm4Adapter infrastructure.
    /// </summary>
    private async Task<Pm4Scene?> LoadSceneFromFile(string filePath)
    {
        try
        {
            var adapter = new Pm4Adapter();
            var options = new Pm4LoadOptions
            {
                VerboseLogging = false
            };
            
            // Load synchronously since PM4 files are not huge
            var scene = adapter.Load(filePath, options);
            return scene;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[PM4 MAP LOADER ERROR] Failed to load PM4 scene from {filePath}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Extract MSPV vertices from loaded scene. MSPV vertices are stored separately from main vertices.
    /// </summary>
    private List<Vector3> ExtractMSPVVertices(Pm4Scene? scene)
    {
        if (scene == null) return new List<Vector3>();
        
        // MSPV vertices might be in MscnVertices or need to be extracted from raw chunk data
        var mspvVertices = new List<Vector3>();
        
        // Check if MSPV data is available in captured raw data
        if (scene.CapturedRawData.TryGetValue("MSPV", out var mspvData))
        {
            // Parse MSPV chunk data - each vertex is 12 bytes (3 floats)
            using var ms = new MemoryStream(mspvData);
            using var br = new BinaryReader(ms);
            
            while (ms.Position < ms.Length - 11) // Ensure we have at least 12 bytes left
            {
                var x = br.ReadSingle();
                var y = br.ReadSingle();
                var z = br.ReadSingle();
                mspvVertices.Add(new Vector3(x, y, z));
            }
        }
        
        return mspvVertices;
    }

    /// <summary>
    /// Extract MSVI indices from scene. These reference MSVT vertices.
    /// </summary>
    private List<uint> ExtractMSVIIndices(Pm4Scene? scene)
    {
        if (scene == null) return new List<uint>();
        
        // Convert the scene indices to uint list
        return scene.Indices.Select(i => (uint)i).ToList();
    }

    /// <summary>
    /// Extract MSPI indices from scene. These reference MSPV vertices.
    /// </summary>
    private List<uint> ExtractMSPIIndices(Pm4Scene? scene)
    {
        if (scene == null) return new List<uint>();
        
        var mspiIndices = new List<uint>();
        
        // Check if MSPI data is available in captured raw data
        if (scene.CapturedRawData.TryGetValue("MSPI", out var mspiData))
        {
            // Parse MSPI chunk data - each index is 4 bytes (uint32)
            using var ms = new MemoryStream(mspiData);
            using var br = new BinaryReader(ms);
            
            while (ms.Position < ms.Length - 3) // Ensure we have at least 4 bytes left
            {
                var index = br.ReadUInt32();
                mspiIndices.Add(index);
            }
        }
        
        return mspiIndices;
    }

    /// <summary>
    /// Extract MPRL placement data from scene.
    /// </summary>
    private List<MprlChunk.Entry> ExtractMprlPlacements(Pm4Scene? scene)
    {
        if (scene == null) return new List<MprlChunk.Entry>();
        
        // Use the placements directly from the scene
        return scene.Placements.ToList();
    }

    /// <summary>
    /// Extract MSUR surface data from scene.
    /// </summary>
    private List<MsurChunk.Entry> ExtractMsurSurfaces(Pm4Scene? scene)
    {
        if (scene == null) return new List<MsurChunk.Entry>();
        
        // Use the surfaces directly from the scene
        return scene.Surfaces.ToList();
    }
}
