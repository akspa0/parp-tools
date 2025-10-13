using System;
using System.Collections.Generic;
using System.Linq;
using WoWRollback.DbcModule;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Discovers all available maps from Map.dbc and analyzes their WDT files.
/// Provides self-guided exploration of MPQ data.
/// </summary>
public sealed class MapDiscoveryService
{
    private readonly string _dbdDir;

    public MapDiscoveryService(string dbdDir)
    {
        _dbdDir = dbdDir ?? throw new ArgumentNullException(nameof(dbdDir));
    }

    /// <summary>
    /// Discovers all maps from Map.dbc and analyzes their WDT files.
    /// </summary>
    /// <param name="src">MPQ archive source</param>
    /// <param name="buildVersion">Build version (e.g., "0.5.3", "3.3.5")</param>
    /// <param name="dbcDir">Directory containing Map.dbc (can be extracted from MPQ)</param>
    /// <returns>Discovery result with all map information</returns>
    public MapDiscoveryResult DiscoverMaps(IArchiveSource src, string buildVersion, string dbcDir)
    {
        try
        {
            // Read Map.dbc
            var mapReader = new MapDbcReader(_dbdDir);
            var mapResult = mapReader.ReadMaps(buildVersion, dbcDir);

            if (!mapResult.Success)
            {
                return new MapDiscoveryResult(
                    Success: false,
                    ErrorMessage: mapResult.ErrorMessage,
                    Maps: Array.Empty<DiscoveredMap>());
            }

            // Analyze WDT for each map
            var wdtAnalyzer = new WdtAnalyzer();
            var discoveredMaps = new List<DiscoveredMap>();

            foreach (var mapEntry in mapResult.Maps)
            {
                var wdtResult = wdtAnalyzer.Analyze(src, mapEntry.Folder);

                discoveredMaps.Add(new DiscoveredMap(
                    Id: mapEntry.Id,
                    Name: mapEntry.MapName,
                    Folder: mapEntry.Folder,
                    WdtExists: wdtResult.Success,
                    HasTerrain: wdtResult.HasTerrain,
                    IsWmoOnly: wdtResult.IsWmoOnly,
                    TileCount: wdtResult.TileCount,
                    WmoPlacement: wdtResult.WmoPlacement
                ));
            }

            return new MapDiscoveryResult(
                Success: true,
                ErrorMessage: null,
                Maps: discoveredMaps.ToArray());
        }
        catch (Exception ex)
        {
            return new MapDiscoveryResult(
                Success: false,
                ErrorMessage: $"Map discovery failed: {ex.Message}",
                Maps: Array.Empty<DiscoveredMap>());
        }
    }

    /// <summary>
    /// Extracts Map.dbc from MPQ to a temporary directory.
    /// </summary>
    public string? ExtractMapDbc(IArchiveSource src, string tempDir)
    {
        try
        {
            var mapDbcPath = "DBFilesClient/Map.dbc";
            if (!src.FileExists(mapDbcPath))
            {
                // Try alternate path
                mapDbcPath = "dbc/Map.dbc";
                if (!src.FileExists(mapDbcPath))
                {
                    return null;
                }
            }

            System.IO.Directory.CreateDirectory(tempDir);
            var outputPath = System.IO.Path.Combine(tempDir, "Map.dbc");

            using var stream = src.OpenFile(mapDbcPath);
            using var fileStream = System.IO.File.Create(outputPath);
            stream.CopyTo(fileStream);

            return outputPath;
        }
        catch
        {
            return null;
        }
    }
}

/// <summary>
/// Represents a discovered map with WDT analysis.
/// </summary>
public record DiscoveredMap(
    int Id,
    string Name,
    string Folder,
    bool WdtExists,
    bool HasTerrain,
    bool IsWmoOnly,
    int TileCount,
    WmoPlacementInfo? WmoPlacement
)
{
    /// <summary>
    /// Gets the map type description.
    /// </summary>
    public string MapType
    {
        get
        {
            if (!WdtExists) return "No WDT";
            if (IsWmoOnly && !HasTerrain) return "WMO-Only";
            if (IsWmoOnly && HasTerrain) return "Hybrid (WMO + Terrain)";
            if (HasTerrain) return "Terrain";
            return "Unknown";
        }
    }
}

/// <summary>
/// Result of map discovery.
/// </summary>
public record MapDiscoveryResult(
    bool Success,
    string? ErrorMessage,
    DiscoveredMap[] Maps
);
