using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.Wmo;
using WoWViewer.Terrain;

namespace WoWViewer;

public sealed record Pm4WmoGroupMatchItem(
    int WmoGroupIndex,
    uint WmoGroupFlags,
    Vector3 WmoBoundsMin,
    Vector3 WmoBoundsMax,
    byte Pm4GroupKey,
    Vector3 Pm4BoundsMin,
    Vector3 Pm4BoundsMax,
    int Pm4SurfaceCount,
    float JaccardOverlap);

public sealed record Pm4WmoPlacementResult(
    int UniqueId,
    string ModelName,
    string ModelPath,
    Vector3 PlacementPosition,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax,
    int WmoGroupCount,
    Vector3 WmoBoundsMin,
    Vector3 WmoBoundsMax,
    IReadOnlyList<Pm4WmoGroupMatchItem> GroupMatches);

public sealed record Pm4WmoFallbackCandidate(
    string ModelPath,
    string ModelName,
    Vector3 WmoBoundsMin,
    Vector3 WmoBoundsMax,
    float VolumeRatio,
    float FootprintRatio,
    float SpanRatio,
    float CombinedScore);

public sealed record Pm4WmoMatchResult(
    bool HasAdtData,
    IReadOnlyList<Pm4WmoPlacementResult> Placements,
    IReadOnlyList<Pm4WmoFallbackCandidate> FallbackCandidates,
    string? ErrorMessage = null);

public static class Pm4WmoGroupMatchService
{
    private const float MapOrigin = 17066.666f;

    public static Pm4WmoMatchResult MatchFromPlacement(
        string clientRoot,
        string mapName,
        int tileX,
        int tileY,
        uint ck24,
        Vector3 pm4ObjectBoundsMin,
        Vector3 pm4ObjectBoundsMax,
        IReadOnlyList<Pm4SurfaceGroupCluster> surfaceClusters)
    {
        try
        {
            string mapDirectory = Path.Combine(clientRoot, "world", "maps", mapName);
            string adtObjPath = Path.Combine(mapDirectory, $"{tileX}_{tileY}_obj0.adt");
            string adtRootPath = Path.Combine(mapDirectory, $"{tileX}_{tileY}.adt");
            if (!File.Exists(adtObjPath) && !File.Exists(adtRootPath))
            {
                return new Pm4WmoMatchResult(
                    HasAdtData: false,
                    Placements: Array.Empty<Pm4WmoPlacementResult>(),
                    FallbackCandidates: Array.Empty<Pm4WmoFallbackCandidate>(),
                    ErrorMessage: $"No placement ADT found (tried '{Path.GetFileName(adtObjPath)}' and '{Path.GetFileName(adtRootPath)}') under: {mapDirectory}");
            }

            // Split-format corpora carry placements in _obj0.adt; monolithic LK ADTs (including the
            // hand-made Museum tiles) carry MDDF/MODF in the root file. Read whichever exist and
            // merge so both layouts feed the matcher identically.
            var wmoPlacements = new List<AdtWorldModelPlacement>();
            foreach (string adtPath in new[] { adtObjPath, adtRootPath })
            {
                if (!File.Exists(adtPath))
                    continue;

                AdtPlacementCatalog catalog = AdtPlacementReader.Read(adtPath);
                wmoPlacements.AddRange(catalog.WorldModelPlacements);
            }

            if (wmoPlacements.Count == 0)
            {
                return new Pm4WmoMatchResult(
                    HasAdtData: true,
                    Placements: Array.Empty<Pm4WmoPlacementResult>(),
                    FallbackCandidates: Array.Empty<Pm4WmoFallbackCandidate>(),
                    ErrorMessage: "No WMO placements found in the tile's placement ADT(s).");
            }

            var results = new List<Pm4WmoPlacementResult>();

            foreach (var placement in wmoPlacements)
            {
                if (!IsOverlappingBounds(pm4ObjectBoundsMin, pm4ObjectBoundsMax, placement.BoundsMin, placement.BoundsMax))
                    continue;

                string wmoRootPath = Path.Combine(clientRoot, placement.ModelPath.TrimStart('/','\\'));
                if (!File.Exists(wmoRootPath))
                    continue;

                WmoGroupInfoSummary wmoSummary;
                try
                {
                    wmoSummary = WmoGroupInfoSummaryReader.Read(wmoRootPath);
                }
                catch
                {
                    continue;
                }

                // Transform WMO group bounds from local to world space for comparison
                Vector3 wmoLocalSize = wmoSummary.BoundsMax - wmoSummary.BoundsMin;
                Vector3 placementWorldSize = placement.BoundsMax - placement.BoundsMin;
                Vector3 worldScale = wmoLocalSize.X > 0 && wmoLocalSize.Y > 0 && wmoLocalSize.Z > 0
                    ? new Vector3(
                        placementWorldSize.X / wmoLocalSize.X,
                        placementWorldSize.Y / wmoLocalSize.Y,
                        placementWorldSize.Z / wmoLocalSize.Z)
                    : Vector3.One;
                Vector3 wmoLocalCenter = (wmoSummary.BoundsMin + wmoSummary.BoundsMax) * 0.5f;
                Vector3 worldOffset = placement.Position - wmoLocalCenter;

                var groupMatches = new List<Pm4WmoGroupMatchItem>();
                if (wmoSummary.Entries.Count > 0 && surfaceClusters.Count > 0)
                {
                    foreach (var cluster in surfaceClusters)
                    {
                        for (int entryIdx = 0; entryIdx < wmoSummary.Entries.Count; entryIdx++)
                        {
                            var entry = wmoSummary.Entries[entryIdx];
                            var wmoWorldMin = (entry.BoundsMin - wmoLocalCenter) * worldScale + placement.Position;
                            var wmoWorldMax = (entry.BoundsMax - wmoLocalCenter) * worldScale + placement.Position;

                            float jaccard = ComputeJaccardOverlap(
                                cluster.BoundsMin, cluster.BoundsMax,
                                wmoWorldMin, wmoWorldMax);

                            groupMatches.Add(new Pm4WmoGroupMatchItem(
                                entryIdx,
                                entry.Flags,
                                wmoWorldMin,
                                wmoWorldMax,
                                cluster.GroupKey,
                                cluster.BoundsMin,
                                cluster.BoundsMax,
                                cluster.SurfaceCount,
                                jaccard));
                        }
                    }

                    groupMatches = groupMatches
                        .OrderByDescending(m => m.JaccardOverlap)
                        .ToList();
                }

                results.Add(new Pm4WmoPlacementResult(
                    placement.UniqueId,
                    placement.ModelPath.Contains('/')
                        ? placement.ModelPath[(placement.ModelPath.LastIndexOf('/') + 1)..]
                        : placement.ModelPath,
                    placement.ModelPath,
                    placement.Position,
                    placement.BoundsMin,
                    placement.BoundsMax,
                    wmoSummary.EntryCount,
                    wmoSummary.BoundsMin,
                    wmoSummary.BoundsMax,
                    groupMatches));
            }

            return new Pm4WmoMatchResult(
                HasAdtData: true,
                Placements: results.OrderBy(r => r.ModelName).ToList(),
                FallbackCandidates: Array.Empty<Pm4WmoFallbackCandidate>());
        }
        catch (Exception ex)
        {
            return new Pm4WmoMatchResult(
                HasAdtData: false,
                Placements: Array.Empty<Pm4WmoPlacementResult>(),
                FallbackCandidates: Array.Empty<Pm4WmoFallbackCandidate>(),
                ErrorMessage: ex.Message);
        }
    }

    public static IReadOnlyList<Pm4WmoFallbackCandidate> SearchWmoByShape(
        string clientRoot,
        Vector3 pm4BoundsMin,
        Vector3 pm4BoundsMax,
        int maxCandidates = 5)
    {
        string wmoDir = Path.Combine(clientRoot, "world", "wmo");
        if (!Directory.Exists(wmoDir))
            return Array.Empty<Pm4WmoFallbackCandidate>();

        Vector3 pm4Size = pm4BoundsMax - pm4BoundsMin;
        float pm4Volume = pm4Size.X * pm4Size.Y * pm4Size.Z;
        float pm4Footprint = pm4Size.X * pm4Size.Y;
        float pm4SpanSum = pm4Size.X + pm4Size.Y + pm4Size.Z;

        if (pm4Volume <= 0f || pm4Footprint <= 0f)
            return Array.Empty<Pm4WmoFallbackCandidate>();

        var candidates = new List<(Pm4WmoFallbackCandidate candidate, float score)>();

        try
        {
            foreach (string wmoFile in Directory.EnumerateFiles(wmoDir, "*.wmo", SearchOption.AllDirectories))
            {
                WmoGroupInfoSummary summary;
                try
                {
                    summary = WmoGroupInfoSummaryReader.Read(wmoFile);
                }
                catch
                {
                    continue;
                }

                Vector3 wmoSize = summary.BoundsMax - summary.BoundsMin;
                float wmoVolume = wmoSize.X * wmoSize.Y * wmoSize.Z;
                float wmoFootprint = wmoSize.X * wmoSize.Y;
                float wmoSpanSum = wmoSize.X + wmoSize.Y + wmoSize.Z;

                if (wmoVolume <= 0f || wmoFootprint <= 0f)
                    continue;

                float volRatio = Math.Min(pm4Volume, wmoVolume) / Math.Max(pm4Volume, wmoVolume);
                float fpRatio = Math.Min(pm4Footprint, wmoFootprint) / Math.Max(pm4Footprint, wmoFootprint);
                float spanRatio = Math.Min(pm4SpanSum, wmoSpanSum) / Math.Max(pm4SpanSum, wmoSpanSum);
                float combined = volRatio * 0.4f + fpRatio * 0.35f + spanRatio * 0.25f;

                if (combined < 0.1f)
                    continue;

                string fileName = Path.GetFileNameWithoutExtension(wmoFile);
                string relPath = GetRelativeWmoPath(clientRoot, wmoFile);

                candidates.Add((new Pm4WmoFallbackCandidate(
                    relPath, fileName,
                    summary.BoundsMin, summary.BoundsMax,
                    volRatio, fpRatio, spanRatio, combined), combined));
            }
        }
        catch
        {
            // If enumeration fails, return what we have
        }

        return candidates
            .OrderByDescending(c => c.score)
            .Take(maxCandidates)
            .Select(c => c.candidate)
            .ToList();
    }

    /// <summary>
    /// Scans every placement ADT of the map (split <c>_obj0.adt</c> and monolithic root LK files
    /// alike) for WMO placements whose world bounds match the selected PM4 object's shape. This
    /// mines the hand-made Museum corpus for likely candidates — sampled ground truth about what
    /// "sorta fits" — instead of relying only on raw client WMO geometry.
    /// </summary>
    public static IReadOnlyList<Pm4WmoFallbackCandidate> SearchAdtPlacementsByShape(
        string clientRoot,
        string mapName,
        Vector3 pm4BoundsMin,
        Vector3 pm4BoundsMax,
        int maxCandidates = 10)
    {
        string mapDirectory = Path.Combine(clientRoot, "world", "maps", mapName);
        if (!Directory.Exists(mapDirectory))
            return Array.Empty<Pm4WmoFallbackCandidate>();

        Vector3 pm4Size = pm4BoundsMax - pm4BoundsMin;
        float pm4Volume = pm4Size.X * pm4Size.Y * pm4Size.Z;
        float pm4Footprint = pm4Size.X * pm4Size.Y;
        float pm4SpanSum = pm4Size.X + pm4Size.Y + pm4Size.Z;
        if (pm4Volume <= 0f || pm4Footprint <= 0f)
            return Array.Empty<Pm4WmoFallbackCandidate>();

        var candidates = new List<(Pm4WmoFallbackCandidate candidate, float score)>();
        try
        {
            foreach (string adtFile in Directory.EnumerateFiles(mapDirectory, "*.adt", SearchOption.TopDirectoryOnly))
            {
                string fileName = Path.GetFileName(adtFile);
                if (fileName.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
                    || fileName.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
                    continue;

                AdtPlacementCatalog catalog;
                try
                {
                    catalog = AdtPlacementReader.Read(adtFile);
                }
                catch
                {
                    continue;
                }

                foreach (AdtWorldModelPlacement placement in catalog.WorldModelPlacements)
                {
                    Vector3 size = placement.BoundsMax - placement.BoundsMin;
                    float volume = size.X * size.Y * size.Z;
                    float footprint = size.X * size.Y;
                    float spanSum = size.X + size.Y + size.Z;
                    if (volume <= 0f || footprint <= 0f)
                        continue;

                    float volRatio = Math.Min(pm4Volume, volume) / Math.Max(pm4Volume, volume);
                    float fpRatio = Math.Min(pm4Footprint, footprint) / Math.Max(pm4Footprint, footprint);
                    float spanRatio = Math.Min(pm4SpanSum, spanSum) / Math.Max(pm4SpanSum, spanSum);
                    float combined = volRatio * 0.4f + fpRatio * 0.35f + spanRatio * 0.25f;
                    if (combined < 0.1f)
                        continue;

                    candidates.Add((new Pm4WmoFallbackCandidate(
                        placement.ModelPath,
                        Path.GetFileNameWithoutExtension(placement.ModelPath),
                        placement.BoundsMin,
                        placement.BoundsMax,
                        volRatio, fpRatio, spanRatio, combined), combined));
                }
            }
        }
        catch
        {
            // If enumeration fails, return what we have
        }

        return candidates
            .OrderByDescending(c => c.score)
            .Take(maxCandidates)
            .Select(c => c.candidate)
            .ToList();
    }

    public static float ComputeJaccardOverlap(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB)
    {
        Vector3 overlapMin = Vector3.Max(minA, minB);
        Vector3 overlapMax = Vector3.Min(maxA, maxB);

        float overlapX = Math.Max(0, overlapMax.X - overlapMin.X);
        float overlapY = Math.Max(0, overlapMax.Y - overlapMin.Y);
        float overlapZ = Math.Max(0, overlapMax.Z - overlapMin.Z);
        float overlapVolume = overlapX * overlapY * overlapZ;

        if (overlapVolume <= 0f)
            return 0f;

        Vector3 sizeA = maxA - minA;
        Vector3 sizeB = maxB - minB;
        float volumeA = sizeA.X * sizeA.Y * sizeA.Z;
        float volumeB = sizeB.X * sizeB.Y * sizeB.Z;
        float unionVolume = volumeA + volumeB - overlapVolume;

        return unionVolume > 0f ? overlapVolume / unionVolume : 0f;
    }

    private static bool IsOverlappingBounds(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB)
    {
        return minA.X <= maxB.X && maxA.X >= minB.X
            && minA.Y <= maxB.Y && maxA.Y >= minB.Y
            && minA.Z <= maxB.Z && maxA.Z >= minB.Z;
    }

    private static string GetRelativeWmoPath(string clientRoot, string fullPath)
    {
        string normalizedRoot = Path.GetFullPath(clientRoot).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string normalizedPath = Path.GetFullPath(fullPath);
        if (normalizedPath.StartsWith(normalizedRoot, StringComparison.OrdinalIgnoreCase))
            return normalizedPath[normalizedRoot.Length..].Replace('\\', '/');
        return fullPath;
    }

    public static string GetMatchKey(string mapName, int tileX, int tileY, uint ck24)
    {
        return $"{mapName.ToLowerInvariant()}|{tileX}|{tileY}|{ck24:X6}";
    }
}

public sealed class Pm4WmoMatchStore
{
    private readonly string _storePath;

    public Pm4WmoMatchStore(string baseOutputDir)
    {
        string outputDir = Path.Combine(baseOutputDir, "..", "..", "..", "..", "output");
        _storePath = Path.GetFullPath(Path.Combine(outputDir, "pm4_wmo_matches.json"));
        Directory.CreateDirectory(Path.GetDirectoryName(_storePath)!);
    }

    public Dictionary<string, Pm4WmoMatchEntry> Load()
    {
        try
        {
            if (!File.Exists(_storePath))
                return new Dictionary<string, Pm4WmoMatchEntry>(StringComparer.OrdinalIgnoreCase);

            string json = File.ReadAllText(_storePath);
            var entries = JsonSerializer.Deserialize<Dictionary<string, Pm4WmoMatchEntry>>(json);
            return entries ?? new Dictionary<string, Pm4WmoMatchEntry>(StringComparer.OrdinalIgnoreCase);
        }
        catch
        {
            return new Dictionary<string, Pm4WmoMatchEntry>(StringComparer.OrdinalIgnoreCase);
        }
    }

    public void Save(Dictionary<string, Pm4WmoMatchEntry> entries)
    {
        try
        {
            string json = JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(_storePath, json);
        }
        catch
        {
        }
    }

    public string StorePath => _storePath;
}

public sealed class Pm4WmoMatchEntry
{
    public string MapName { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public uint Ck24 { get; set; }
    public string WmoPath { get; set; } = "";
    public string ModelName { get; set; } = "";
    public string Source { get; set; } = "manual";
}
