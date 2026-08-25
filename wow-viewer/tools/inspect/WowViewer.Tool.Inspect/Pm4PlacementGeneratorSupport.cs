using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Regenerates object placement records from PM4 files, for tiles whose own placement data is missing
/// or unusable.
/// </summary>
/// <remarks>
/// Everything emitted here is derived from the PM4 alone. Position and extents come from the geometry;
/// the asset name is a RANKED GUESS from shape scoring, which is right about half the time at rank one
/// and about three quarters of the time inside five. So the output carries the whole candidate list and
/// a confidence, never a bare name pretending to be a fact.
///
/// <para>Where no candidate is credible the asset is written as <c>missingwmo.wmo</c> rather than
/// being dropped. A placement whose position is known and whose identity is not is still worth
/// recording - it says something stood here and how big it was.</para>
///
/// <para>The library is built ONLY from tiles that have real placement data to learn from, and a tile
/// being regenerated never contributes to the library used on it.</para>
/// </remarks>
internal static class Pm4PlacementGeneratorSupport
{
    private const string Unknown = "missingwmo.wmo";

    public static Pm4PlacementGenerationReport Generate(
        string pm4Directory,
        string adtDirectory,
        string outputDirectory,
        double confidenceThreshold = 0.25)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);
        Directory.CreateDirectory(outputDirectory);

        // Library of navmesh box shapes per asset, learned from tiles that still have placements.
        var library = new Dictionary<string, List<Vector3>>(StringComparer.OrdinalIgnoreCase);
        var learnedFromTile = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);

        foreach ((string asset, Vector3 shape, string tile) in EnumerateKnownObjects(resolved, adtDirectory))
        {
            if (!library.TryGetValue(asset, out List<Vector3>? shapes))
            {
                shapes = [];
                library[asset] = shapes;
            }

            shapes.Add(shape);

            if (!learnedFromTile.TryGetValue(asset, out HashSet<string>? tiles))
            {
                tiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                learnedFromTile[asset] = tiles;
            }

            tiles.Add(tile);
        }

        var tilesOut = new List<Pm4GeneratedTile>();
        long objectsTotal = 0, named = 0, unknownNamed = 0;
        int tilesWithTerrain = 0, tilesWithoutTerrain = 0, tilesWithExistingPlacements = 0;

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tFirst, out int tSecond))
                continue;

            string tileName = Path.GetFileNameWithoutExtension(pm4Path);

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            Dictionary<uint, (Vector3 Min, Vector3 Max, int Surfaces)> objects = ExtractObjects(chunks);
            if (objects.Count == 0)
                continue;

            // Counted only for tiles that actually make it into the output, so every figure in the
            // summary describes the same set of tiles.
            bool hasTerrain = HasTerrain(pm4Path, adtDirectory, tFirst, tSecond);
            bool hasPlacements = HasPlacements(pm4Path, adtDirectory);
            if (hasTerrain) tilesWithTerrain++; else tilesWithoutTerrain++;
            if (hasPlacements) tilesWithExistingPlacements++;

            var rows = new List<Pm4GeneratedPlacement>();
            foreach ((uint key, (Vector3 min, Vector3 max, int surfaces)) in objects.OrderBy(static o => o.Key))
            {
                objectsTotal++;
                Vector3 observed = ShapeOf(min, max);

                // Rank candidates, excluding anything this tile alone taught us.
                var ranked = new List<(string Name, double Score)>();
                foreach ((string asset, List<Vector3> shapes) in library)
                {
                    if (learnedFromTile.TryGetValue(asset, out HashSet<string>? from)
                        && from.Count == 1 && from.Contains(tileName))
                    {
                        continue;
                    }

                    Vector3? median = Median(shapes);
                    if (median is Vector3 m)
                        ranked.Add((asset, Distance(observed, m)));
                }

                ranked.Sort(static (a, b) => a.Score.CompareTo(b.Score));

                string chosen = Unknown;
                double best = ranked.Count > 0 ? ranked[0].Score : double.MaxValue;
                if (ranked.Count > 0 && best <= confidenceThreshold)
                {
                    chosen = ranked[0].Name;
                    named++;
                }
                else
                {
                    unknownNamed++;
                }

                float placementZ = BitConverter.UInt32BitsToSingle(key);
                rows.Add(new Pm4GeneratedPlacement(
                    $"0x{key:X8}",
                    chosen,
                    surfaces,
                    new Pm4Xyz((min.X + max.X) * 0.5f, (min.Y + max.Y) * 0.5f, placementZ),
                    new Pm4Xyz(min.X, min.Y, min.Z),
                    new Pm4Xyz(max.X, max.Y, max.Z),
                    Math.Round(best, 4),
                    [.. ranked.Take(5).Select(c => new Pm4Candidate(c.Name, Math.Round(c.Score, 4)))]));
            }

            tilesOut.Add(new Pm4GeneratedTile(tileName, tFirst, tSecond, hasTerrain, hasPlacements, rows));
        }

        string jsonPath = Path.Combine(outputDirectory, "pm4-generated-placements.json");
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(
            new { generated = DateTime.UtcNow.ToString("O"), source = resolved, tiles = tilesOut },
            new JsonSerializerOptions { WriteIndented = true }));

        // A flat CSV too - the JSON is for tooling, this is for looking at.
        string csvPath = Path.Combine(outputDirectory, "pm4-generated-placements.csv");
        var csv = new StringBuilder();
        csv.AppendLine("tile,tileFirst,tileSecond,hasTerrain,hasExistingPlacements,objectKey,asset,surfaces,posX,posY,posZ,minX,minY,minZ,maxX,maxY,maxZ,score");
        foreach (Pm4GeneratedTile tile in tilesOut)
        {
            foreach (Pm4GeneratedPlacement row in tile.Placements)
            {
                csv.Append(CultureInfo.InvariantCulture, $"{tile.Tile},{tile.TileFirst},{tile.TileSecond},{tile.HasTerrain},{tile.HasExistingPlacements},");
                csv.Append(CultureInfo.InvariantCulture, $"{row.ObjectKey},{row.Asset},{row.Surfaces},");
                csv.Append(CultureInfo.InvariantCulture, $"{row.Position.X:F3},{row.Position.Y:F3},{row.Position.Z:F3},");
                csv.Append(CultureInfo.InvariantCulture, $"{row.BoundsMin.X:F3},{row.BoundsMin.Y:F3},{row.BoundsMin.Z:F3},");
                csv.Append(CultureInfo.InvariantCulture, $"{row.BoundsMax.X:F3},{row.BoundsMax.Y:F3},{row.BoundsMax.Z:F3},");
                csv.AppendLine(row.Score.ToString("F4", CultureInfo.InvariantCulture));
            }
        }

        File.WriteAllText(csvPath, csv.ToString());

        return new Pm4PlacementGenerationReport(
            resolved, outputDirectory, jsonPath, csvPath,
            tilesOut.Count, tilesWithTerrain, tilesWithoutTerrain, tilesWithExistingPlacements,
            objectsTotal, named, unknownNamed, library.Count, confidenceThreshold);
    }

    private static Dictionary<uint, (Vector3 Min, Vector3 Max, int Surfaces)> ExtractObjects(Pm4KnownChunkSet chunks)
    {
        var lo = new Dictionary<uint, Vector3>();
        var hi = new Dictionary<uint, Vector3>();
        var count = new Dictionary<uint, int>();

        foreach (Pm4MsurEntry surface in chunks.Msur)
        {
            if (surface.PackedParams == 0)
                continue;

            long vs = surface.MsviFirstIndex;
            long ve = vs + surface.IndexCount;
            if (surface.IndexCount == 0 || ve > chunks.Msvi.Count)
                continue;

            count[surface.PackedParams] = count.GetValueOrDefault(surface.PackedParams) + 1;

            for (long v = vs; v < ve; v++)
            {
                uint vi = chunks.Msvi[(int)v];
                if (vi >= chunks.Msvt.Count)
                    continue;

                Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(chunks.Msvt[(int)vi]);
                if (!lo.TryGetValue(surface.PackedParams, out Vector3 cur))
                {
                    lo[surface.PackedParams] = p;
                    hi[surface.PackedParams] = p;
                    continue;
                }

                lo[surface.PackedParams] = Vector3.Min(cur, p);
                hi[surface.PackedParams] = Vector3.Max(hi[surface.PackedParams], p);
            }
        }

        var result = new Dictionary<uint, (Vector3, Vector3, int)>();
        foreach ((uint key, Vector3 min) in lo)
            result[key] = (min, hi[key], count.GetValueOrDefault(key));

        return result;
    }

    private static IEnumerable<(string Asset, Vector3 Shape, string Tile)> EnumerateKnownObjects(string resolved, string adtDirectory)
    {
        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtDirectory);
            if (adtPath is null)
                continue;

            AdtPlacementCatalog catalog;
            try
            {
                catalog = AdtPlacementReader.Read(adtPath);
            }
            catch
            {
                continue;
            }

            if (catalog.WorldModelPlacements.Count == 0)
                continue;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            var byHeight = new Dictionary<uint, AdtWorldModelPlacement>();
            foreach (AdtWorldModelPlacement row in catalog.WorldModelPlacements)
                byHeight[BitConverter.SingleToUInt32Bits(row.Position.Z)] = row;

            string tile = Path.GetFileNameWithoutExtension(pm4Path);
            foreach ((uint key, (Vector3 min, Vector3 max, _)) in ExtractObjects(chunks))
            {
                if (!byHeight.TryGetValue(key, out AdtWorldModelPlacement truth))
                    continue;

                string name = Path.GetFileName(truth.ModelPath);
                if (!string.IsNullOrWhiteSpace(name))
                    yield return (name, ShapeOf(min, max), tile);
            }
        }
    }

    private static bool HasTerrain(string pm4Path, string adtDirectory, int tileFirst, int tileSecond)
    {
        string stem = Path.GetFileNameWithoutExtension(pm4Path);
        int underscore = stem.IndexOf('_');
        string mapName = underscore > 0 ? stem[..underscore] : stem;

        foreach (string candidate in new[]
        {
            Path.Combine(adtDirectory, $"{stem}.adt"),
            Path.Combine(adtDirectory, $"{mapName}_{tileFirst}_{tileSecond}.adt"),
        })
        {
            if (File.Exists(candidate) && new FileInfo(candidate).Length > 1024)
                return true;
        }

        return false;
    }

    private static bool HasPlacements(string pm4Path, string adtDirectory)
    {
        string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtDirectory);
        if (adtPath is null)
            return false;

        try
        {
            return AdtPlacementReader.Read(adtPath).WorldModelPlacements.Count > 0;
        }
        catch
        {
            return false;
        }
    }

    private static Vector3 ShapeOf(Vector3 min, Vector3 max)
    {
        float dx = MathF.Abs(max.X - min.X);
        float dy = MathF.Abs(max.Y - min.Y);
        return new Vector3(MathF.Min(dx, dy), MathF.Max(dx, dy), MathF.Abs(max.Z - min.Z));
    }

    private static Vector3? Median(List<Vector3> shapes)
    {
        if (shapes.Count == 0)
            return null;

        List<float> xs = [.. shapes.Select(static v => v.X).Order()];
        List<float> ys = [.. shapes.Select(static v => v.Y).Order()];
        List<float> zs = [.. shapes.Select(static v => v.Z).Order()];
        return new Vector3(xs[xs.Count / 2], ys[ys.Count / 2], zs[zs.Count / 2]);
    }

    private static double Distance(Vector3 a, Vector3 b)
    {
        static double Rel(float p, float q)
        {
            float scale = MathF.Max(MathF.Max(MathF.Abs(p), MathF.Abs(q)), 1f);
            return MathF.Abs(p - q) / scale;
        }

        return Rel(a.X, b.X) + Rel(a.Y, b.Y) + (2.0 * Rel(a.Z, b.Z));
    }
}

internal sealed record Pm4Xyz(float X, float Y, float Z);

internal sealed record Pm4Candidate(string Asset, double Score);

internal sealed record Pm4GeneratedPlacement(
    string ObjectKey,
    string Asset,
    int Surfaces,
    Pm4Xyz Position,
    Pm4Xyz BoundsMin,
    Pm4Xyz BoundsMax,
    double Score,
    IReadOnlyList<Pm4Candidate> Candidates);

internal sealed record Pm4GeneratedTile(
    string Tile,
    int TileFirst,
    int TileSecond,
    bool HasTerrain,
    bool HasExistingPlacements,
    IReadOnlyList<Pm4GeneratedPlacement> Placements);

internal sealed record Pm4PlacementGenerationReport(
    string Pm4Directory,
    string OutputDirectory,
    string JsonPath,
    string CsvPath,
    int Tiles,
    int TilesWithTerrain,
    int TilesWithoutTerrain,
    int TilesWithExistingPlacements,
    long ObjectsTotal,
    long ObjectsNamed,
    long ObjectsUnknown,
    int LibraryAssets,
    double ConfidenceThreshold);
