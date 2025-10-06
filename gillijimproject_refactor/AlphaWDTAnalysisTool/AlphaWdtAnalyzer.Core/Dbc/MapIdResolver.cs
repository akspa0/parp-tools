using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AlphaWdtAnalyzer.Core.Dbc;

/// <summary>
/// Resolves map names to MapIDs using DBCTool.V2-generated maps.json metadata.
/// </summary>
public sealed class MapIdResolver
{
    public record MapEntry
    {
        [JsonPropertyName("id")]
        public int Id { get; init; }
        
        [JsonPropertyName("directory")]
        public string Directory { get; init; } = string.Empty;
        
        [JsonPropertyName("name")]
        public string Name { get; init; } = string.Empty;
        
        [JsonPropertyName("instanceType")]
        public int InstanceType { get; init; }
    }

    public record MapMetadata
    {
        [JsonPropertyName("version")]
        public string Version { get; init; } = string.Empty;
        
        [JsonPropertyName("build")]
        public string Build { get; init; } = string.Empty;
        
        [JsonPropertyName("maps")]
        public List<MapEntry> Maps { get; init; } = new();
    }

    private readonly Dictionary<string, int> _directoryToId = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<int, MapEntry> _idToEntry = new();
    public string Version { get; }

    public MapIdResolver(MapMetadata metadata)
    {
        Version = metadata.Version;
        foreach (var map in metadata.Maps)
        {
            if (!string.IsNullOrWhiteSpace(map.Directory))
            {
                _directoryToId[map.Directory] = map.Id;
            }
            _idToEntry[map.Id] = map;
        }
    }

    /// <summary>
    /// Load resolver from DBCTool.V2 output structure: dbctool_out/{version}/maps.json
    /// </summary>
    public static MapIdResolver? LoadFromDbcToolOutput(string dbctoolOutRoot, string version)
    {
        // Try base version (e.g., "0.5.3.3368" -> "0.5.3")
        var baseVersion = ExtractBaseVersion(version);
        var paths = new[]
        {
            Path.Combine(dbctoolOutRoot, baseVersion, "maps.json"),
            Path.Combine(dbctoolOutRoot, version, "maps.json")
        };

        foreach (var path in paths)
        {
            if (File.Exists(path))
            {
                try
                {
                    var json = File.ReadAllText(path);
                    var metadata = JsonSerializer.Deserialize<MapMetadata>(json);
                    if (metadata != null)
                    {
                        return new MapIdResolver(metadata);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MapIdResolver] Failed to load {path}: {ex.Message}");
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Get MapID by directory name (e.g., "Shadowfang" -> 33)
    /// </summary>
    public int? GetMapIdByDirectory(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory)) return null;
        return _directoryToId.TryGetValue(directory, out var id) ? id : null;
    }

    /// <summary>
    /// Get map entry by ID
    /// </summary>
    public MapEntry? GetMapById(int mapId)
    {
        return _idToEntry.TryGetValue(mapId, out var entry) ? entry : null;
    }

    /// <summary>
    /// Check if a map directory exists in this version
    /// </summary>
    public bool MapExists(string directory)
    {
        return !string.IsNullOrWhiteSpace(directory) && _directoryToId.ContainsKey(directory);
    }

    /// <summary>
    /// Get all valid map directories for this version
    /// </summary>
    public IEnumerable<string> GetAllDirectories()
    {
        return _directoryToId.Keys.OrderBy(k => k, StringComparer.OrdinalIgnoreCase);
    }

    private static string ExtractBaseVersion(string version)
    {
        // "0.5.3.3368" -> "0.5.3"
        if (string.IsNullOrWhiteSpace(version)) return version;
        var parts = version.Split('.');
        if (parts.Length >= 3)
        {
            return $"{parts[0]}.{parts[1]}.{parts[2]}";
        }
        return version;
    }
}
