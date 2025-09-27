using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace DBCTool.V2.Mapping;

public static class AreaHierarchyLoader
{
    private static readonly IDeserializer s_deserializer = new DeserializerBuilder()
        .WithNamingConvention(CamelCaseNamingConvention.Instance)
        .IgnoreUnmatchedProperties()
        .Build();

    public static AreaHierarchyGraph LoadFromFile(string yamlPath)
    {
        if (string.IsNullOrWhiteSpace(yamlPath)) throw new ArgumentException("Path cannot be null or empty", nameof(yamlPath));
        using var stream = File.OpenRead(yamlPath);
        using var reader = new StreamReader(stream);
        return LoadFromTextReader(reader);
    }

    public static AreaHierarchyGraph LoadFromTextReader(TextReader reader)
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        var root = s_deserializer.Deserialize<HierarchyRoot>(reader);
        if (root?.Maps is null)
        {
            return new AreaHierarchyGraph(Array.Empty<AreaHierarchyMap>());
        }

        var maps = root.Maps.Select(BuildMap).ToList();
        return new AreaHierarchyGraph(maps);
    }

    private static AreaHierarchyMap BuildMap(MapRecord mapRecord)
    {
        var zones = new List<AreaHierarchyNode>();
        if (mapRecord.Zones is not null)
        {
            foreach (var zoneRecord in mapRecord.Zones)
            {
                zones.Add(BuildNode(mapRecord.MapId, zoneRecord, parent: null));
            }
        }
        return new AreaHierarchyMap(mapRecord.MapId, mapRecord.MapName ?? string.Empty, zones);
    }

    private static AreaHierarchyNode BuildNode(int mapIdFallback, ZoneRecord record, AreaHierarchyNode? parent)
    {
        if (record is null) throw new ArgumentNullException(nameof(record));
        int mapId = record.MapId.HasValue && record.MapId.Value >= 0 ? record.MapId.Value : mapIdFallback;
        bool isZone = parent is null || record.ParentId == record.AreaId;
        var node = new AreaHierarchyNode(mapId, record.AreaId, record.Name ?? string.Empty, record.ParentId, isZone, parent);
        if (record.Children is not null)
        {
            foreach (var childRecord in record.Children)
            {
                var childNode = BuildNode(mapId, childRecord, node);
                node.AddChild(childNode);
            }
        }
        return node;
    }

    private sealed class HierarchyRoot
    {
        public List<MapRecord>? Maps { get; set; }
    }

    private sealed class MapRecord
    {
        public int MapId { get; set; }
        public string? MapName { get; set; }
        public List<ZoneRecord>? Zones { get; set; }
    }

    private sealed class ZoneRecord
    {
        public int AreaId { get; set; }
        public string? Name { get; set; }
        public int ParentId { get; set; }
        public int? MapId { get; set; }
        public List<ZoneRecord>? Children { get; set; }
    }
}
