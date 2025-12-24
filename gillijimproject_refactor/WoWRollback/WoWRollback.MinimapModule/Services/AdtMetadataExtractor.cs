using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using Warcraft.NET.Files.ADT.TerrainObject.Zero;
using WoWRollback.MinimapModule.Models;

namespace WoWRollback.MinimapModule.Services;

/// <summary>
/// Extracts complete metadata from ADT files for VLM training.
/// </summary>
public sealed class AdtMetadataExtractor
{
    private readonly ILogger<AdtMetadataExtractor> _logger;

    public AdtMetadataExtractor(ILogger<AdtMetadataExtractor> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Extracts VLM training metadata from a single ADT tile.
    /// </summary>
    public VlmTrainingSample? ExtractMetadata(string adtDirectory, string mapName, int tileX, int tileY)
    {
        var adtPath = Path.Combine(adtDirectory, $"{mapName}_{tileX}_{tileY}.adt");
        var obj0Path = Path.Combine(adtDirectory, $"{mapName}_{tileX}_{tileY}_obj0.adt");
        
        // Determine format
        bool isCataclysm = File.Exists(obj0Path);
        
        if (!isCataclysm && !File.Exists(adtPath))
        {
            _logger.LogWarning("ADT file not found: {AdtPath}", adtPath);
            return null;
        }

        try
        {
            var textures = new List<ChunkTextureInfo>();
            var objects = new List<ObjectPlacement>();
            float heightMin = float.MaxValue;
            float heightMax = float.MinValue;
            bool hasWater = false;

            // Read terrain data from main ADT
            if (File.Exists(adtPath))
            {
                var terrain = new Terrain(File.ReadAllBytes(adtPath));
                
                // Extract chunk-level texture info (16×16 = 256 chunks)
                if (terrain.Chunks != null)
                {
                    foreach (var chunk in terrain.Chunks)
                    {
                        if (chunk?.Header == null) continue;
                        
                        var chunkX = (int)chunk.Header.MapIndexX;
                        var chunkY = (int)chunk.Header.MapIndexY;
                        
                        // Extract texture layers
                        var layers = new List<string>();
                        if (chunk.TextureLayers?.Layers != null)
                        {
                            foreach (var layer in chunk.TextureLayers.Layers)
                            {
                                // Get texture path from MTEX table
                                if (terrain.Textures?.Filenames != null && 
                                    layer.TextureID < terrain.Textures.Filenames.Count)
                                {
                                    layers.Add(terrain.Textures.Filenames[(int)layer.TextureID]);
                                }
                            }
                        }
                        
                        textures.Add(new ChunkTextureInfo(
                            new[] { chunkX, chunkY },
                            layers.ToArray()));
                        
                        // Extract heightmap bounds
                        if (chunk.Heightmap?.Vertices != null)
                        {
                            foreach (var vertex in chunk.Heightmap.Vertices)
                            {
                                var absoluteHeight = chunk.Header.MapTilePosition.Z + vertex;
                                if (absoluteHeight < heightMin) heightMin = absoluteHeight;
                                if (absoluteHeight > heightMax) heightMax = absoluteHeight;
                            }
                        }
                        
                        // Check for water
                        if (chunk.Header.LiquidSize > 8)
                        {
                            hasWater = true;
                        }
                    }
                }
                
                // Extract objects from pre-Cataclysm monolithic ADT
                if (!isCataclysm)
                {
                    objects.AddRange(ExtractObjectsFromTerrain(terrain, mapName, tileX, tileY));
                }
            }
            
            // Extract objects from Cataclysm _obj0.adt
            if (isCataclysm && File.Exists(obj0Path))
            {
                var objFile = new TerrainObjectZero(File.ReadAllBytes(obj0Path));
                objects.AddRange(ExtractObjectsFromObj0(objFile, mapName, tileX, tileY));
            }

            // Handle case where no heights were found
            if (heightMin == float.MaxValue) heightMin = 0;
            if (heightMax == float.MinValue) heightMax = 0;

            return new VlmTrainingSample(
                $"{mapName}_{tileX}_{tileY}",
                textures,
                objects,
                new TerrainSummary(heightMin, heightMax, hasWater));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to extract metadata from {MapName}_{TileX}_{TileY}", mapName, tileX, tileY);
            return null;
        }
    }

    private List<ObjectPlacement> ExtractObjectsFromTerrain(Terrain terrain, string mapName, int tileX, int tileY)
    {
        var objects = new List<ObjectPlacement>();
        
        // Build M2 path lookup
        var m2Paths = new Dictionary<uint, string>();
        if (terrain.Models?.Filenames != null)
        {
            for (int i = 0; i < terrain.Models.Filenames.Count; i++)
            {
                m2Paths[(uint)i] = terrain.Models.Filenames[i];
            }
        }
        
        // Build WMO path lookup
        var wmoPaths = new Dictionary<uint, string>();
        if (terrain.WorldModelObjects?.Filenames != null)
        {
            for (int i = 0; i < terrain.WorldModelObjects.Filenames.Count; i++)
            {
                wmoPaths[(uint)i] = terrain.WorldModelObjects.Filenames[i];
            }
        }
        
        // Extract M2 placements
        if (terrain.ModelPlacementInfo?.MDDFEntries != null)
        {
            foreach (var m2 in terrain.ModelPlacementInfo.MDDFEntries)
            {
                var name = m2Paths.TryGetValue(m2.NameId, out var path) 
                    ? Path.GetFileNameWithoutExtension(path) 
                    : $"m2_{m2.NameId}";
                    
                objects.Add(new ObjectPlacement(
                    name,
                    m2.Position.X,
                    m2.Position.Y,
                    m2.Position.Z,
                    "m2"));
            }
        }
        
        // Extract WMO placements
        if (terrain.WorldModelObjectPlacementInfo?.MODFEntries != null)
        {
            foreach (var wmo in terrain.WorldModelObjectPlacementInfo.MODFEntries)
            {
                var name = wmoPaths.TryGetValue(wmo.NameId, out var path)
                    ? Path.GetFileNameWithoutExtension(path)
                    : $"wmo_{wmo.NameId}";
                    
                objects.Add(new ObjectPlacement(
                    name,
                    wmo.Position.X,
                    wmo.Position.Y,
                    wmo.Position.Z,
                    "wmo"));
            }
        }
        
        return objects;
    }

    private List<ObjectPlacement> ExtractObjectsFromObj0(TerrainObjectZero objFile, string mapName, int tileX, int tileY)
    {
        var objects = new List<ObjectPlacement>();
        
        // Build M2 path lookup
        var m2Paths = new Dictionary<uint, string>();
        if (objFile.Models?.Filenames != null)
        {
            for (int i = 0; i < objFile.Models.Filenames.Count; i++)
            {
                m2Paths[(uint)i] = objFile.Models.Filenames[i];
            }
        }
        
        // Build WMO path lookup
        var wmoPaths = new Dictionary<uint, string>();
        if (objFile.WorldModelObjects?.Filenames != null)
        {
            for (int i = 0; i < objFile.WorldModelObjects.Filenames.Count; i++)
            {
                wmoPaths[(uint)i] = objFile.WorldModelObjects.Filenames[i];
            }
        }
        
        // Extract M2 placements
        if (objFile.ModelPlacementInfo?.MDDFEntries != null)
        {
            foreach (var m2 in objFile.ModelPlacementInfo.MDDFEntries)
            {
                var name = m2Paths.TryGetValue(m2.NameId, out var path)
                    ? Path.GetFileNameWithoutExtension(path)
                    : $"m2_{m2.NameId}";
                    
                objects.Add(new ObjectPlacement(
                    name,
                    m2.Position.X,
                    m2.Position.Y,
                    m2.Position.Z,
                    "m2"));
            }
        }
        
        // Extract WMO placements
        if (objFile.WorldModelObjectPlacementInfo?.MODFEntries != null)
        {
            foreach (var wmo in objFile.WorldModelObjectPlacementInfo.MODFEntries)
            {
                var name = wmoPaths.TryGetValue(wmo.NameId, out var path)
                    ? Path.GetFileNameWithoutExtension(path)
                    : $"wmo_{wmo.NameId}";
                    
                objects.Add(new ObjectPlacement(
                    name,
                    wmo.Position.X,
                    wmo.Position.Y,
                    wmo.Position.Z,
                    "wmo"));
            }
        }
        
        return objects;
    }
}
