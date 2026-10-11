using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WoWViewer;

public sealed record PmM2PlacementResult(
    int UniqueId,
    string ModelName,
    string ModelPath,
    Vector3 PlacementPosition,
    Vector3 PlacementRotation,
    float PlacementScale,
    float DistanceToPm4Center,
    float MatchScore);

public sealed record PmM2FallbackCandidate(
    string ModelPath,
    string ModelName,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    float VolumeRatio,
    float FootprintRatio,
    float SpanRatio,
    float CombinedScore);

public sealed record PmM2MatchResult(
    bool HasAdtData,
    IReadOnlyList<PmM2PlacementResult> Placements,
    IReadOnlyList<PmM2FallbackCandidate> FallbackCandidates,
    string? ErrorMessage = null);

public static class PmM2GroupMatchService
{
    public static PmM2MatchResult MatchFromPlacement(
        string clientRoot,
        string mapName,
        int tileX,
        int tileY,
        int objectPartId,
        Vector3 pm4BoundsMin,
        Vector3 pm4BoundsMax,
        Vector3 pm4Centroid)
    {
        try
        {
            string mapDirectory = Path.Combine(clientRoot, "world", "maps", mapName);
            string adtObjPath = Path.Combine(mapDirectory, $"{tileX}_{tileY}_obj0.adt");
            string adtRootPath = Path.Combine(mapDirectory, $"{tileX}_{tileY}.adt");
            if (!File.Exists(adtObjPath) && !File.Exists(adtRootPath))
            {
                return new PmM2MatchResult(
                    HasAdtData: false,
                    Placements: Array.Empty<PmM2PlacementResult>(),
                    FallbackCandidates: Array.Empty<PmM2FallbackCandidate>(),
                    ErrorMessage: $"No placement ADT found for tile ({tileX}, {tileY})");
            }

            var m2Placements = new List<AdtModelPlacement>();
            foreach (string adtPath in new[] { adtObjPath, adtRootPath })
            {
                if (!File.Exists(adtPath))
                    continue;

                AdtPlacementCatalog catalog = AdtPlacementReader.Read(adtPath);
                m2Placements.AddRange(catalog.ModelPlacements.Where(m => !m.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase)));
            }

            // Also search 8-neighbor tiles in case the doodad origin or span crosses tile borders
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = tileX + dx;
                    int ny = tileY + dy;
                    string nObj = Path.Combine(mapDirectory, $"{nx}_{ny}_obj0.adt");
                    string nRoot = Path.Combine(mapDirectory, $"{nx}_{ny}.adt");
                    string? companion = File.Exists(nObj) ? nObj : (File.Exists(nRoot) ? nRoot : null);
                    if (companion != null)
                    {
                        try
                        {
                            var cat = AdtPlacementReader.Read(companion);
                            m2Placements.AddRange(cat.ModelPlacements.Where(m => !m.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase)));
                        }
                        catch { }
                    }
                }
            }

            if (m2Placements.Count == 0)
            {
                return new PmM2MatchResult(
                    HasAdtData: true,
                    Placements: Array.Empty<PmM2PlacementResult>(),
                    FallbackCandidates: Array.Empty<PmM2FallbackCandidate>(),
                    ErrorMessage: "No M2 placements found in the tile's placement ADT(s).");
            }

            Vector2 planarCentroid = new(pm4Centroid.X, pm4Centroid.Y);
            var results = new List<PmM2PlacementResult>();

            foreach (var placement in m2Placements)
            {
                Vector2 planarPlacement = new(placement.Position.X, placement.Position.Y);
                float dist2D = Vector2.Distance(planarPlacement, planarCentroid);
                float dist3D = Vector3.Distance(placement.Position, pm4Centroid);

                // Candidates within 80 yards of the PM4 doodad collision
                if (dist2D <= 80f)
                {
                    float score = 1f / (1f + dist3D / 12f);
                    string modelName = placement.ModelPath.Contains('/') || placement.ModelPath.Contains('\\')
                        ? Path.GetFileName(placement.ModelPath)
                        : placement.ModelPath;

                    results.Add(new PmM2PlacementResult(
                        placement.UniqueId,
                        modelName,
                        placement.ModelPath,
                        placement.Position,
                        placement.Rotation,
                        placement.Scale,
                        dist3D,
                        score));
                }
            }

            results = results
                .GroupBy(r => r.UniqueId)
                .Select(g => g.First())
                .OrderByDescending(r => r.MatchScore)
                .ToList();

            return new PmM2MatchResult(
                HasAdtData: true,
                Placements: results,
                FallbackCandidates: Array.Empty<PmM2FallbackCandidate>());
        }
        catch (Exception ex)
        {
            return new PmM2MatchResult(
                HasAdtData: false,
                Placements: Array.Empty<PmM2PlacementResult>(),
                FallbackCandidates: Array.Empty<PmM2FallbackCandidate>(),
                ErrorMessage: ex.Message);
        }
    }

    public static IReadOnlyList<PmM2FallbackCandidate> SearchAdtPlacementsByShape(
        string clientRoot,
        string mapName,
        Vector3 pm4BoundsMin,
        Vector3 pm4BoundsMax,
        int maxCandidates = 10)
    {
        string mapDirectory = Path.Combine(clientRoot, "world", "maps", mapName);
        if (!Directory.Exists(mapDirectory))
            return Array.Empty<PmM2FallbackCandidate>();

        Vector3 pm4Size = pm4BoundsMax - pm4BoundsMin;
        float pm4Volume = MathF.Abs(pm4Size.X * pm4Size.Y * pm4Size.Z);
        float pm4Footprint = MathF.Abs(pm4Size.X * pm4Size.Y);
        float pm4SpanSum = MathF.Abs(pm4Size.X) + MathF.Abs(pm4Size.Y) + MathF.Abs(pm4Size.Z);
        if (pm4Volume <= 0f || pm4Footprint <= 0f)
            return Array.Empty<PmM2FallbackCandidate>();

        var candidates = new List<(PmM2FallbackCandidate candidate, float score)>();
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

                foreach (AdtModelPlacement placement in catalog.ModelPlacements)
                {
                    if (placement.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
                        continue;

                    string modelName = Path.GetFileNameWithoutExtension(placement.ModelPath);
                    float scale = placement.Scale > 0 ? placement.Scale : 1f;
                    float dummyVol = pm4Volume * scale;
                    float volRatio = MathF.Min(pm4Volume, dummyVol) / MathF.Max(pm4Volume, dummyVol);
                    float score = 0.5f + volRatio * 0.5f;

                    candidates.Add((new PmM2FallbackCandidate(
                        placement.ModelPath,
                        modelName,
                        pm4BoundsMin,
                        pm4BoundsMax,
                        volRatio,
                        1f,
                        1f,
                        score), score));
                }
            }
        }
        catch
        {
        }

        return candidates
            .GroupBy(c => c.candidate.ModelPath, StringComparer.OrdinalIgnoreCase)
            .Select(g => g.First().candidate)
            .Take(maxCandidates)
            .ToList();
    }

    public static string GetMatchKey(string mapName, int tileX, int tileY, int objectPartId)
    {
        return $"{mapName.ToLowerInvariant()}|{tileX}|{tileY}|m2_{objectPartId}";
    }
}

public sealed class PmM2MatchStore
{
    private readonly string _storePath;

    public PmM2MatchStore(string baseOutputDir)
    {
        string outputDir = Path.Combine(baseOutputDir, "..", "..", "..", "..", "output");
        _storePath = Path.GetFullPath(Path.Combine(outputDir, "pm4_m2_matches.json"));
        Directory.CreateDirectory(Path.GetDirectoryName(_storePath)!);
    }

    public Dictionary<string, PmM2MatchEntry> Load()
    {
        try
        {
            if (!File.Exists(_storePath))
                return new Dictionary<string, PmM2MatchEntry>(StringComparer.OrdinalIgnoreCase);

            string json = File.ReadAllText(_storePath);
            var entries = JsonSerializer.Deserialize<Dictionary<string, PmM2MatchEntry>>(json);
            return entries ?? new Dictionary<string, PmM2MatchEntry>(StringComparer.OrdinalIgnoreCase);
        }
        catch
        {
            return new Dictionary<string, PmM2MatchEntry>(StringComparer.OrdinalIgnoreCase);
        }
    }

    public void Save(Dictionary<string, PmM2MatchEntry> entries)
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

public sealed class PmM2MatchEntry
{
    public string MapName { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public int ObjectPartId { get; set; }
    public string M2Path { get; set; } = "";
    public string ModelName { get; set; } = "";
    public string Source { get; set; } = "manual";
}
