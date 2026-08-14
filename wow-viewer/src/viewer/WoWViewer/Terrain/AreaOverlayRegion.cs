using System.Numerics;
using WowViewer.Core.World;

namespace WoWViewer.Terrain;

public enum AreaOverlayRegionKind
{
    Zone,
    Subzone,
}

public readonly record struct AreaOverlayFootprintCell(Vector3 BoundsMin, Vector3 BoundsMax);

public sealed class AreaOverlayRegion
{
    public string Key { get; }
    public AreaOverlayRegionKind Kind { get; }
    public int MapId { get; }
    public int CanonicalAreaId { get; }
    public int? ParentAreaId { get; }
    public string Name { get; }
    public Vector3 Color { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public Vector3 LabelPosition { get; }
    public IReadOnlyList<AreaOverlayFootprintCell> Cells { get; }
    public int ChunkCount => Cells.Count;

    internal AreaOverlayRegion(
        string key,
        AreaOverlayRegionKind kind,
        int mapId,
        int canonicalAreaId,
        int? parentAreaId,
        string name,
        Vector3 color,
        Vector3 boundsMin,
        Vector3 boundsMax,
        IReadOnlyList<AreaOverlayFootprintCell> cells)
    {
        Key = key;
        Kind = kind;
        MapId = mapId;
        CanonicalAreaId = canonicalAreaId;
        ParentAreaId = parentAreaId;
        Name = name;
        Color = color;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        LabelPosition = new Vector3(
            (boundsMin.X + boundsMax.X) * 0.5f,
            (boundsMin.Y + boundsMax.Y) * 0.5f,
            boundsMax.Z + 8f);
        Cells = cells;
    }
}

public sealed record AreaOverlayBuildResult(
    IReadOnlyList<AreaOverlayRegion> Regions,
    int ResidentChunkCount,
    int UnresolvedChunkCount);

/// <summary>
/// Builds an honest resident-area visualization from MCNK chunk metadata. The 0.5.3 client stores
/// area identity per CMapChunk, so chunk footprints are the available spatial evidence; this builder
/// deliberately does not infer unloaded or polygonal boundaries.
/// </summary>
public static class AreaOverlayRegionBuilder
{
    private sealed class Accumulator
    {
        public readonly string Key;
        public readonly AreaOverlayRegionKind Kind;
        public readonly int MapId;
        public readonly int CanonicalAreaId;
        public readonly int? ParentAreaId;
        public readonly string Name;
        public readonly Vector3 Color;
        public readonly List<AreaOverlayFootprintCell> Cells = new();
        public Vector3 BoundsMin = new(float.MaxValue);
        public Vector3 BoundsMax = new(float.MinValue);

        public Accumulator(
            string key,
            AreaOverlayRegionKind kind,
            int mapId,
            int canonicalAreaId,
            int? parentAreaId,
            string name)
        {
            Key = key;
            Kind = kind;
            MapId = mapId;
            CanonicalAreaId = canonicalAreaId;
            ParentAreaId = parentAreaId;
            Name = name;
            Color = GetColor(mapId, canonicalAreaId, kind);
        }

        public void Add(Vector3 boundsMin, Vector3 boundsMax)
        {
            Vector3 min = Vector3.Min(boundsMin, boundsMax);
            Vector3 max = Vector3.Max(boundsMin, boundsMax);
            Cells.Add(new AreaOverlayFootprintCell(min, max));
            BoundsMin = Vector3.Min(BoundsMin, min);
            BoundsMax = Vector3.Max(BoundsMax, max);
        }

        public AreaOverlayRegion Build()
        {
            return new AreaOverlayRegion(
                Key,
                Kind,
                MapId,
                CanonicalAreaId,
                ParentAreaId,
                Name,
                Color,
                BoundsMin,
                BoundsMax,
                Cells.ToArray());
        }
    }

    public static AreaOverlayBuildResult Build(
        IEnumerable<TerrainRenderer.TerrainChunkInfo> residentChunks,
        AreaTableService areaTableService,
        int mapId)
    {
        ArgumentNullException.ThrowIfNull(residentChunks);
        ArgumentNullException.ThrowIfNull(areaTableService);

        var groups = new Dictionary<string, Accumulator>(StringComparer.Ordinal);
        int residentChunkCount = 0;
        int unresolvedChunkCount = 0;

        foreach (TerrainRenderer.TerrainChunkInfo chunk in residentChunks)
        {
            residentChunkCount++;
            if (!IsFinite(chunk.BoundsMin) || !IsFinite(chunk.BoundsMax))
            {
                unresolvedChunkCount++;
                continue;
            }

            AreaLookupResult lookup = areaTableService.ResolveArea(chunk.AreaId, mapId);
            if (!lookup.IsResolved || lookup.CanonicalAreaId is not int canonicalAreaId)
            {
                unresolvedChunkCount++;
                continue;
            }

            string? zoneName = Clean(lookup.ZoneText);
            if (zoneName is null)
                zoneName = Clean(lookup.PrimaryText);

            int zoneId = lookup.ParentAreaId ?? canonicalAreaId;
            if (zoneName is not null)
            {
                string zoneKey = CreateKey(mapId, AreaOverlayRegionKind.Zone, zoneId);
                GetOrCreate(groups, zoneKey, AreaOverlayRegionKind.Zone, mapId, zoneId, null, zoneName)
                    .Add(chunk.BoundsMin, chunk.BoundsMax);
            }
            else
            {
                unresolvedChunkCount++;
            }

            string? subzoneName = Clean(lookup.SubzoneText);
            bool hasDistinctSubzone = lookup.ParentAreaId.HasValue
                && lookup.CanonicalAreaId.Value != lookup.ParentAreaId.Value
                && subzoneName is not null
                && !string.Equals(subzoneName, zoneName, StringComparison.OrdinalIgnoreCase);
            if (hasDistinctSubzone)
            {
                string subzoneKey = CreateKey(mapId, AreaOverlayRegionKind.Subzone, canonicalAreaId);
                GetOrCreate(groups, subzoneKey, AreaOverlayRegionKind.Subzone, mapId, canonicalAreaId,
                        lookup.ParentAreaId, subzoneName!)
                    .Add(chunk.BoundsMin, chunk.BoundsMax);
            }
        }

        IReadOnlyList<AreaOverlayRegion> regions = groups.Values
            .OrderBy(group => group.Kind)
            .ThenBy(group => group.Name, StringComparer.OrdinalIgnoreCase)
            .ThenBy(group => group.CanonicalAreaId)
            .Select(group => group.Build())
            .ToArray();

        return new AreaOverlayBuildResult(regions, residentChunkCount, unresolvedChunkCount);
    }

    private static Accumulator GetOrCreate(
        IDictionary<string, Accumulator> groups,
        string key,
        AreaOverlayRegionKind kind,
        int mapId,
        int canonicalAreaId,
        int? parentAreaId,
        string name)
    {
        if (groups.TryGetValue(key, out Accumulator? existing))
            return existing;

        var created = new Accumulator(key, kind, mapId, canonicalAreaId, parentAreaId, name);
        groups.Add(key, created);
        return created;
    }

    private static string CreateKey(int mapId, AreaOverlayRegionKind kind, int canonicalAreaId)
        => $"{mapId}:{kind}:{canonicalAreaId}";

    private static string? Clean(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim();

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);

    private static Vector3 GetColor(int mapId, int areaId, AreaOverlayRegionKind kind)
    {
        unchecked
        {
            uint hash = 2166136261u;
            hash = (hash ^ (uint)mapId) * 16777619u;
            hash = (hash ^ (uint)areaId) * 16777619u;
            hash = (hash ^ (uint)kind) * 16777619u;
            float hue = (hash % 360u) / 360f;
            if (kind == AreaOverlayRegionKind.Subzone)
                hue = (hue + 0.08f) % 1f;

            return HsvToRgb(hue, kind == AreaOverlayRegionKind.Zone ? 0.72f : 0.58f,
                kind == AreaOverlayRegionKind.Zone ? 0.96f : 0.90f);
        }
    }

    private static Vector3 HsvToRgb(float hue, float saturation, float value)
    {
        float scaled = hue * 6f;
        int sector = (int)MathF.Floor(scaled) % 6;
        float fraction = scaled - MathF.Floor(scaled);
        float p = value * (1f - saturation);
        float q = value * (1f - fraction * saturation);
        float t = value * (1f - (1f - fraction) * saturation);
        return sector switch
        {
            0 => new Vector3(value, t, p),
            1 => new Vector3(q, value, p),
            2 => new Vector3(p, value, t),
            3 => new Vector3(p, q, value),
            4 => new Vector3(t, p, value),
            _ => new Vector3(value, p, q),
        };
    }
}
