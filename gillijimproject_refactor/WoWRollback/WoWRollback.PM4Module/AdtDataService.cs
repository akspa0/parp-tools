using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace WoWRollback.PM4Module;

/// <summary>
/// Unified service for extracting all data from ADT tiles.
/// Provides a single-call interface for placements + terrain + UniqueIDs.
/// Uses MpqAdtExtractor internally for MPQ access.
/// </summary>
public sealed class AdtDataService : IDisposable
{
    private readonly MpqAdtExtractor? _mpqExtractor;
    private readonly string _mapName;
    private bool _disposed;
    
    /// <summary>
    /// Create a service for extracting ADT data from an MPQ archive.
    /// </summary>
    public AdtDataService(string mpqPath, string mapName)
    {
        _mpqExtractor = new MpqAdtExtractor(mpqPath);
        _mapName = mapName;
    }
    
    /// <summary>
    /// Extract all data for a tile in one call.
    /// Returns placements, terrain, and UniqueID summary.
    /// </summary>
    public AdtTileData? ExtractTile(int tileX, int tileY)
    {
        if (_mpqExtractor == null) return null;
        
        var placements = _mpqExtractor.ExtractPlacements(_mapName, tileX, tileY);
        var terrain = _mpqExtractor.ExtractTerrain(_mapName, tileX, tileY);
        
        if (placements == null && terrain == null)
            return null;
            
        return new AdtTileData
        {
            Map = _mapName,
            TileX = tileX,
            TileY = tileY,
            Placements = placements,
            Terrain = terrain,
            UniqueIds = ExtractUniqueIdSummary(placements)
        };
    }
    
    /// <summary>
    /// Extract all tiles for the map.
    /// </summary>
    public IEnumerable<AdtTileData> ExtractAllTiles()
    {
        if (_mpqExtractor == null) yield break;
        
        for (int x = 0; x < 64; x++)
        {
            for (int y = 0; y < 64; y++)
            {
                var tile = ExtractTile(x, y);
                if (tile != null)
                    yield return tile;
            }
        }
    }
    
    /// <summary>
    /// Export all tile data to JSON files in output directory.
    /// </summary>
    public int ExportToJson(string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        int count = 0;
        
        foreach (var tile in ExtractAllTiles())
        {
            var jsonPath = Path.Combine(outputDir, $"{_mapName}_{tile.TileX}_{tile.TileY}.json");
            var json = JsonSerializer.Serialize(tile, new JsonSerializerOptions 
            { 
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            File.WriteAllText(jsonPath, json);
            count++;
            
            if (count % 50 == 0)
            {
                Console.WriteLine($"[AdtDataService] Exported {count} tiles...");
            }
        }
        
        Console.WriteLine($"[AdtDataService] Export complete: {count} tiles to {outputDir}");
        return count;
    }
    
    private UniqueIdSummary? ExtractUniqueIdSummary(AdtPlacementData? placements)
    {
        if (placements == null) return null;
        
        var m2Ids = new List<uint>();
        var wmoIds = new List<uint>();
        
        foreach (var m2 in placements.M2Placements)
            m2Ids.Add(m2.UniqueId);
            
        foreach (var wmo in placements.WmoPlacements)
            wmoIds.Add(wmo.UniqueId);
            
        return new UniqueIdSummary
        {
            M2Count = m2Ids.Count,
            WmoCount = wmoIds.Count,
            M2UniqueIds = m2Ids,
            WmoUniqueIds = wmoIds
        };
    }
    
    public void Dispose()
    {
        if (!_disposed)
        {
            _mpqExtractor?.Dispose();
            _disposed = true;
        }
    }
}

#region Unified Data Models

/// <summary>Complete data for an ADT tile.</summary>
public class AdtTileData
{
    public string Map { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public AdtPlacementData? Placements { get; set; }
    public TileTerrainData? Terrain { get; set; }
    public UniqueIdSummary? UniqueIds { get; set; }
}

/// <summary>Summary of UniqueIDs in a tile.</summary>
public class UniqueIdSummary
{
    public int M2Count { get; set; }
    public int WmoCount { get; set; }
    public List<uint> M2UniqueIds { get; set; } = new();
    public List<uint> WmoUniqueIds { get; set; } = new();
}

#endregion
