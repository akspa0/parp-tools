using System.Text.Json;
using System.Text.Json.Serialization;
using AlphaWdtAnalyzer.Core;
using PlacementRecordRaw = AlphaWdtAnalyzer.Core.PlacementRecord;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Serializes comprehensive per-map analysis data (placements, unique ID ranges, etc.)
/// into normalized JSON payloads for downstream consumption.
/// </summary>
public sealed class MapMasterIndexWriter
{
    private const float TileSize = 533.333333333f;
    private const float MapOffset = TileSize * 32f;
    private const int ChunksPerTile = 16;
    private const float ChunkSize = TileSize / ChunksPerTile;

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true
    };

    public MapMasterIndexWriteResult Write(AnalysisIndex index, string map, string version, string outputDir)
    {
        ArgumentNullException.ThrowIfNull(index);
        if (string.IsNullOrWhiteSpace(map)) throw new ArgumentException("Map required", nameof(map));
        if (string.IsNullOrWhiteSpace(version)) throw new ArgumentException("Version required", nameof(version));

        Directory.CreateDirectory(outputDir);

        var masterPath = Path.Combine(outputDir, $"{map}_master_index.json");
        var perTilePath = Path.Combine(outputDir, $"{map}_id_ranges_by_tile.json");

        var (masterModel, perTileModel) = BuildModels(index, map, version);

        File.WriteAllText(masterPath, JsonSerializer.Serialize(masterModel, SerializerOptions));
        File.WriteAllText(perTilePath, JsonSerializer.Serialize(perTileModel, SerializerOptions));

        return new MapMasterIndexWriteResult(masterPath, perTilePath);
    }

    private static (MapMasterIndex Master, TileRangeIndex TileRanges) BuildModels(AnalysisIndex index, string map, string version)
    {
        var generatedAt = DateTime.UtcNow;

        var tiles = index.Placements
            .GroupBy(p => (p.TileX, p.TileY))
            .OrderBy(g => g.Key.TileY)
            .ThenBy(g => g.Key.TileX)
            .Select(g => BuildTileRecord(map, version, generatedAt, g.Key.TileX, g.Key.TileY, g.ToList()))
            .ToList();

        var master = new MapMasterIndex
        {
            Map = map,
            Version = version,
            GeneratedAtUtc = generatedAt,
            TileCount = tiles.Count,
            Tiles = tiles.Select(t => t.Tile).ToList()
        };

        var tileRanges = new TileRangeIndex
        {
            Map = map,
            Version = version,
            GeneratedAtUtc = generatedAt,
            Tiles = tiles.Select(t => t.Range).Where(r => r.Chunks.Count > 0).ToList()
        };

        return (master, tileRanges);
    }

    private static TileBuildResult BuildTileRecord(string map, string version, DateTime generatedAt, int tileX, int tileY, IReadOnlyList<PlacementRecordRaw> placements)
    {
        var normalizedPlacements = placements
            .Select(p => NormalizePlacement(p))
            .OrderBy(p => p.ChunkY)
            .ThenBy(p => p.ChunkX)
            .ThenBy(p => p.Kind)
            .ThenBy(p => p.UniqueId ?? uint.MaxValue)
            .ToList();

        var tile = new TileRecord
        {
            TileX = tileX,
            TileY = tileY,
            Placements = normalizedPlacements
        };

        var chunkGroups = normalizedPlacements
            .Where(p => p.UniqueId.HasValue)
            .GroupBy(p => (p.ChunkX, p.ChunkY))
            .OrderBy(g => g.Key.ChunkY)
            .ThenBy(g => g.Key.ChunkX)
            .Select(g => new ChunkRangeRecord
            {
                ChunkX = g.Key.ChunkX,
                ChunkY = g.Key.ChunkY,
                Kinds = g
                    .GroupBy(p => p.Kind)
                    .OrderBy(kg => kg.Key, StringComparer.OrdinalIgnoreCase)
                    .Select(kg => new ChunkKindRecord
                    {
                        Kind = kg.Key,
                        Count = kg.Count(),
                        MinUniqueId = kg.Min(p => p.UniqueId!.Value),
                        MaxUniqueId = kg.Max(p => p.UniqueId!.Value),
                        UniqueIds = kg.Select(p => p.UniqueId!.Value).Distinct().OrderBy(id => id).ToList()
                    })
                    .Where(k => k.Count > 0)
                    .ToList()
            })
            .Where(c => c.Kinds.Count > 0)
            .ToList();

        var range = new TileRangeRecord
        {
            TileX = tileX,
            TileY = tileY,
            Chunks = chunkGroups
        };

        return new TileBuildResult(tile, range);
    }

    private static PlacementRecord NormalizePlacement(PlacementRecordRaw source)
    {
        float rawX = source.WorldX;
        float rawY = source.WorldY;
        float rawZ = source.WorldZ;

        // Apply MDDF/MODF -> world transformation from wowdev.wiki
        float worldNorthSouth = MapOffset - rawX;
        float worldUp = rawY;
        float worldWestEast = MapOffset - rawZ;

        float tileNorthMin = MapOffset - (source.TileX + 1) * TileSize;
        float tileWestMin = MapOffset - (source.TileY + 1) * TileSize;

        float localNorth = worldNorthSouth - tileNorthMin;
        float localWest = worldWestEast - tileWestMin;

        // Clamp local coordinates to tile bounds to avoid floating point spill-over
        localNorth = Math.Clamp(localNorth, 0f, TileSize);
        localWest = Math.Clamp(localWest, 0f, TileSize);

        int chunkX = Math.Clamp((int)Math.Floor(localNorth / ChunkSize), 0, ChunksPerTile - 1);
        int chunkY = Math.Clamp((int)Math.Floor(localWest / ChunkSize), 0, ChunksPerTile - 1);

        return new PlacementRecord
        {
            Kind = source.Type.ToString(),
            UniqueId = source.UniqueId.HasValue && source.UniqueId.Value >= 0
                ? (uint?)source.UniqueId.Value
                : null,
            AssetPath = source.AssetPath,
            RawNorth = rawX,
            RawUp = rawY,
            RawWest = rawZ,
            WorldNorth = worldNorthSouth,
            WorldWest = worldWestEast,
            WorldUp = worldUp,
            TileOffsetNorth = localNorth,
            TileOffsetWest = localWest,
            ChunkX = chunkX,
            ChunkY = chunkY,
            RotationX = source.RotationX,
            RotationY = source.RotationY,
            RotationZ = source.RotationZ,
            Scale = source.Scale,
            Flags = source.Flags,
            DoodadSet = source.DoodadSet,
            NameSet = source.NameSet
        };
    }

    private sealed record TileBuildResult(TileRecord Tile, TileRangeRecord Range);

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
        public float RawNorth { get; init; }
        public float RawUp { get; init; }
        public float RawWest { get; init; }
        public float WorldNorth { get; init; }
        public float WorldWest { get; init; }
        public float WorldUp { get; init; }
        public float TileOffsetNorth { get; init; }
        public float TileOffsetWest { get; init; }
        public int ChunkX { get; init; }
        public int ChunkY { get; init; }
        public float RotationX { get; init; }
        public float RotationY { get; init; }
        public float RotationZ { get; init; }
        public float Scale { get; init; }
        public ushort Flags { get; init; }
        public ushort DoodadSet { get; init; }
        public ushort NameSet { get; init; }
    }

    private sealed record TileRangeIndex
    {
        public required string Map { get; init; }
        public required string Version { get; init; }
        public required DateTime GeneratedAtUtc { get; init; }
        public required IReadOnlyList<TileRangeRecord> Tiles { get; init; }
    }

    private sealed record TileRangeRecord
    {
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required IReadOnlyList<ChunkRangeRecord> Chunks { get; init; }
    }

    private sealed record ChunkRangeRecord
    {
        public required int ChunkX { get; init; }
        public required int ChunkY { get; init; }
        public required IReadOnlyList<ChunkKindRecord> Kinds { get; init; }
    }

    private sealed record ChunkKindRecord
    {
        public required string Kind { get; init; }
        public required int Count { get; init; }
        public required uint MinUniqueId { get; init; }
        public required uint MaxUniqueId { get; init; }
        public required IReadOnlyList<uint> UniqueIds { get; init; }
    }
}

public sealed record MapMasterIndexWriteResult(string MasterIndexPath, string TileRangePath);
