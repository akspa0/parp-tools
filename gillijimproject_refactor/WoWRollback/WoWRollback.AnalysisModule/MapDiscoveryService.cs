using System;
using System.Collections.Generic;
using System.Linq;
using WoWRollback.DbcModule;
using WoWRollback.Core.Services.Archive;
using DBCD;
using DBCD.Providers;

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
    /// Discovers maps from CASC by reading Map.db2 directly via DBCD using a CASC-backed provider.
    /// </summary>
    public MapDiscoveryResult DiscoverMapsFromCasc(IArchiveSource src, string buildVersion)
    {
        try
        {
            var dbdProvider = new FilesystemDBDProvider(_dbdDir);
            var dbcProvider = new CascDbcProvider(src);
            var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

            IDBCDStorage storage;
            try { storage = dbcd.Load("Map", buildVersion, Locale.EnUS); }
            catch { storage = dbcd.Load("Map", buildVersion, Locale.None); }

            var entries = new List<WoWRollback.DbcModule.MapEntry>();
            foreach (var row in storage.Values)
            {
                try
                {
                    var id = SafeField<int>(row, "ID");
                    var mapName = FirstNonEmpty(
                        SafeField<string>(row, "MapName_lang"),
                        SafeField<string>(row, "MapName"),
                        SafeField<string>(row, "InternalName"),
                        string.Empty
                    );
                    var folder = FirstNonEmpty(
                        SafeField<string>(row, "Directory"),
                        SafeField<string>(row, "Folder"),
                        SafeField<string>(row, "FolderName"),
                        string.Empty
                    );
                    if (string.IsNullOrWhiteSpace(folder)) continue;
                    entries.Add(new WoWRollback.DbcModule.MapEntry(id, mapName ?? string.Empty, folder));
                }
                catch { }
            }

            // Analyze WDTs with the archive source
            var wdtAnalyzer = new WdtAnalyzer();
            var discoveredMaps = new List<DiscoveredMap>();
            foreach (var m in entries)
            {
                var w = wdtAnalyzer.Analyze(src, m.Folder);
                discoveredMaps.Add(new DiscoveredMap(
                    Id: m.Id,
                    Name: m.MapName,
                    Folder: m.Folder,
                    WdtExists: w.Success,
                    HasTerrain: w.HasTerrain,
                    IsWmoOnly: w.IsWmoOnly,
                    TileCount: w.TileCount,
                    WmoPlacement: w.WmoPlacement
                ));
            }

            return new MapDiscoveryResult(true, null, discoveredMaps.ToArray());
        }
        catch (Exception ex)
        {
            return new MapDiscoveryResult(false, $"Map.db2 discovery failed: {ex.Message}", Array.Empty<DiscoveredMap>());
        }

        static T? SafeField<T>(DBCDRow row, string name)
        {
            try
            {
                var v = row[name];
                if (v is T tv) return tv;
                if (typeof(T) == typeof(string)) return (T)(object)(v?.ToString() ?? "");
                if (v != null) return (T)Convert.ChangeType(v, typeof(T));
            }
            catch { }
            return default;
        }

        static string FirstNonEmpty(params string[] vals)
        {
            foreach (var v in vals) if (!string.IsNullOrWhiteSpace(v)) return v;
            return string.Empty;
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
