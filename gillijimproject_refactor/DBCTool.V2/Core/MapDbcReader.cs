using DBCD;
using DBCD.Providers;
using DBCTool.V2.IO;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DBCTool.V2.Core;

/// <summary>
/// Reads Map.dbc to provide authoritative MapID → Directory mapping for a given WoW version.
/// </summary>
public sealed class MapDbcReader
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

    /// <summary>
    /// Read Map.dbc and return structured map metadata.
    /// </summary>
    public static MapMetadata ReadMapDbc(string dbcDir, string dbdDir, string canonicalBuild, string versionAlias, DBCD.Locale locale = DBCD.Locale.EnUS)
    {
        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var storage = DbdcHelper.LoadTable("Map", canonicalBuild, dbcDir, dbdProvider, locale);
        
        var idCol = DbdcHelper.DetectIdColumn(storage);
        var maps = new List<MapEntry>();

        foreach (var k in storage.Keys)
        {
            var row = storage[k];
            
            int mapId = !string.IsNullOrWhiteSpace(idCol) ? DbdcHelper.SafeField<int>(row, idCol) : k;
            string dirRaw = DbdcHelper.SafeField<string>(row, "Directory");
            string directory = DbdcHelper.DirToken(dirRaw);
            
            string name = DbdcHelper.FirstNonEmpty(
                DbdcHelper.SafeField<string>(row, "MapName_lang"),
                DbdcHelper.SafeField<string>(row, "MapName"),
                DbdcHelper.SafeField<string>(row, "InternalName"),
                directory
            );
            
            int instanceType = DbdcHelper.SafeField<int>(row, "InstanceType");

            maps.Add(new MapEntry
            {
                Id = mapId,
                Directory = directory ?? string.Empty,
                Name = name ?? string.Empty,
                InstanceType = instanceType
            });
        }

        return new MapMetadata
        {
            Version = versionAlias,
            Build = canonicalBuild,
            Maps = maps.OrderBy(m => m.Id).ToList()
        };
    }

    /// <summary>
    /// Write map metadata to JSON file.
    /// </summary>
    public static void WriteMapMetadata(MapMetadata metadata, string outputPath)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        var json = JsonSerializer.Serialize(metadata, options);
        File.WriteAllText(outputPath, json);
    }

    /// <summary>
    /// Load map metadata from JSON file.
    /// </summary>
    public static MapMetadata? LoadMapMetadata(string jsonPath)
    {
        if (!File.Exists(jsonPath)) return null;
        
        var json = File.ReadAllText(jsonPath);
        return JsonSerializer.Deserialize<MapMetadata>(json);
    }

    /// <summary>
    /// Get MapID by directory name.
    /// </summary>
    public static int? GetMapIdByDirectory(MapMetadata metadata, string directory)
    {
        var normalized = DbdcHelper.Norm(directory);
        var match = metadata.Maps.FirstOrDefault(m => 
            DbdcHelper.Norm(m.Directory) == normalized);
        return match?.Id;
    }
}
