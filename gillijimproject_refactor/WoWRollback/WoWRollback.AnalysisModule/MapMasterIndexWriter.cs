using System.Text.Json;
using System.Text.Json.Serialization;
using AlphaWdtAnalyzer.Core;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Serializes comprehensive per-map analysis data (placements, unique ID ranges, etc.)
/// into a single JSON master index for downstream consumption.
/// </summary>
public sealed class MapMasterIndexWriter
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true
    };

    public string Write(AnalysisIndex index, string map, string version, string outputDir)
    {
        ArgumentNullException.ThrowIfNull(index);
        if (string.IsNullOrWhiteSpace(map)) throw new ArgumentException("Map required", nameof(map));
        if (string.IsNullOrWhiteSpace(version)) throw new ArgumentException("Version required", nameof(version));

        Directory.CreateDirectory(outputDir);
        var outPath = Path.Combine(outputDir, $"{map}_master_index.json");
        var model = BuildModel(index, map, version);
        File.WriteAllText(outPath, JsonSerializer.Serialize(model, SerializerOptions));
        return outPath;
    }

    private static MapMasterIndex BuildModel(AnalysisIndex index, string map, string version)
    {
        var tiles = index.Placements
            .GroupBy(p => (p.TileX, p.TileY))
            .Select(g => new TileRecord
            {
                TileX = g.Key.TileX,
                TileY = g.Key.TileY,
                Placements = g.Select(p => new PlacementRecord
                {
                    Kind = p.Type.ToString(),
                    UniqueId = p.UniqueId.HasValue ? (uint?)p.UniqueId.Value : null,
                    AssetPath = p.AssetPath,
                    WorldX = p.WorldX,
                    WorldY = p.WorldY,
                    WorldZ = p.WorldZ,
                    RotationX = p.RotationX,
                    RotationY = p.RotationY,
                    RotationZ = p.RotationZ,
                    Scale = p.Scale,
                    Flags = p.Flags,
                    DoodadSet = p.DoodadSet,
                    NameSet = p.NameSet
                }).ToList()
            })
            .OrderBy(t => t.TileY)
            .ThenBy(t => t.TileX)
            .ToList();

        return new MapMasterIndex
        {
            Map = map,
            Version = version,
            GeneratedAtUtc = DateTime.UtcNow,
            TileCount = tiles.Count,
            Tiles = tiles
        };
    }

    private sealed record MapMasterIndex
    {
        public required string Map { get; init; }
        public required string Version { get; init; }
        public required DateTime GeneratedAtUtc { get; init; }
        public required int TileCount { get; init; }
        public required IReadOnlyList<TileRecord> Tiles { get; init; }
    }

    private sealed record TileRecord
    {
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required IReadOnlyList<PlacementRecord> Placements { get; init; }
    }

    private sealed record PlacementRecord
    {
        public required string Kind { get; init; }
        public uint? UniqueId { get; init; }
        public string? AssetPath { get; init; }
        public float WorldX { get; init; }
        public float WorldY { get; init; }
        public float WorldZ { get; init; }
        public float RotationX { get; init; }
        public float RotationY { get; init; }
        public float RotationZ { get; init; }
        public float Scale { get; init; }
        public ushort Flags { get; init; }
        public ushort DoodadSet { get; init; }
        public ushort NameSet { get; init; }
    }
}
