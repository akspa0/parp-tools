using System;
using System.Collections.Generic;
using System.Linq;
using DBCD;
using DBCD.IO;
using DBCD.Providers;

namespace WoWRollback.DbcModule;

/// <summary>
/// Reads Map.dbc to discover all available maps in a WoW client.
/// Extracts map folder names for self-guided exploration.
/// </summary>
public sealed class MapDbcReader
{
    private readonly string _dbdDir;
    private readonly string _locale;

    public MapDbcReader(string dbdDir, string locale = "enUS")
    {
        _dbdDir = dbdDir ?? throw new ArgumentNullException(nameof(dbdDir));
        _locale = locale;
    }

    /// <summary>
    /// Reads Map.dbc and extracts all map entries with their folder names.
    /// </summary>
    /// <param name="buildVersion">Build version string (e.g., "0.5.3", "3.3.5")</param>
    /// <param name="dbcDir">Directory containing Map.dbc</param>
    /// <returns>Result containing list of map entries or error message</returns>
    public MapDbcResult ReadMaps(string buildVersion, string dbcDir)
    {
        try
        {
            var mapDbcPath = System.IO.Path.Combine(dbcDir, "Map.dbc");
            if (!System.IO.File.Exists(mapDbcPath))
            {
                return new MapDbcResult(
                    Success: false,
                    ErrorMessage: $"Map.dbc not found in: {dbcDir}",
                    Maps: Array.Empty<MapEntry>());
            }

            var dbdProvider = new FilesystemDBDProvider(_dbdDir);
            var dbcProvider = new FilesystemDBCProvider(dbcDir, useCache: true);
            var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

            // Try with locale first, fall back to Locale.None if it fails
            IDBCDStorage storage;
            try
            {
                var locale = Enum.TryParse<Locale>(_locale, ignoreCase: true, out var loc) ? loc : Locale.EnUS;
                storage = dbcd.Load("Map", buildVersion, locale);
            }
            catch
            {
                storage = dbcd.Load("Map", buildVersion, Locale.None);
            }

            var maps = new List<MapEntry>();

            foreach (var row in storage.Values)
            {
                try
                {
                    var id = GetField<int>(row, "ID");
                    var mapName = GetField<string>(row, "MapName") ?? GetField<string>(row, "MapName_lang") ?? "";
                    
                    // Try different column names for folder (varies by version)
                    var folder = GetField<string>(row, "Directory") 
                                 ?? GetField<string>(row, "Folder") 
                                 ?? GetField<string>(row, "FolderName")
                                 ?? "";

                    // Skip entries without a folder name
                    if (string.IsNullOrWhiteSpace(folder))
                        continue;

                    maps.Add(new MapEntry(
                        Id: id,
                        MapName: mapName,
                        Folder: folder
                    ));
                }
                catch
                {
                    // Skip malformed entries
                }
            }

            return new MapDbcResult(
                Success: true,
                ErrorMessage: null,
                Maps: maps.ToArray());
        }
        catch (Exception ex)
        {
            return new MapDbcResult(
                Success: false,
                ErrorMessage: $"Failed to read Map.dbc: {ex.Message}",
                Maps: Array.Empty<MapEntry>());
        }
    }

    private static T? GetField<T>(DBCDRow row, string fieldName)
    {
        try
        {
            var value = row[fieldName];
            if (value is T typedValue)
                return typedValue;
            
            // Try conversion for common types
            if (typeof(T) == typeof(string))
                return (T)(object)(value?.ToString() ?? "");
            
            if (value != null)
                return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            // Field doesn't exist or conversion failed
        }
        
        return default;
    }
}

/// <summary>
/// Represents a single map entry from Map.dbc.
/// </summary>
public record MapEntry(
    int Id,
    string MapName,
    string Folder
);

/// <summary>
/// Result of reading Map.dbc.
/// </summary>
public record MapDbcResult(
    bool Success,
    string? ErrorMessage,
    MapEntry[] Maps
);
