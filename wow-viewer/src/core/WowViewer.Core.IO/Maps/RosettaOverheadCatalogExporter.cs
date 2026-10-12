using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// A 2D overhead bounding box projected into minimap tile coordinates.
/// </summary>
public sealed record RosettaOverheadPlacementBox(
    int TileX,
    int TileY,
    [property: JsonPropertyName("pos")] Vector3 WorldPosition,
    float CellU,
    float CellV,
    float CellSize,
    float NormalizedMinU,
    float NormalizedMinV,
    float NormalizedMaxU,
    float NormalizedMaxV,
    int PixelMinX,
    int PixelMinY,
    int PixelMaxX,
    int PixelMaxY,
    int UniqueId);

/// <summary>
/// An exhibit record in the Rosetta overhead vision catalog.
/// </summary>
public sealed record RosettaOverheadExhibit(
    string AssetId,
    string AssetPath,
    string Kind,
    float ExtentX,
    float ExtentY,
    float ExtentZ,
    float FootprintArea,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    IReadOnlyList<RosettaOverheadPlacementBox> Placements);

/// <summary>
/// Root catalog container for Rosetta overhead objects.
/// </summary>
public sealed record RosettaOverheadCatalog(
    string MapName,
    int TotalExhibits,
    int TotalPlacements,
    IReadOnlyList<RosettaOverheadExhibit> Exhibits);

/// <summary>
/// Exports 2D top-down silhouettes, bounding extents, and orthographic placement
/// footprints from Rosetta calibration maps and datastores for vision model prompting (Spec 262).
/// </summary>
public static class RosettaOverheadCatalogExporter
{
    private const float AdtTileSize = 533.33333f; // ADT tile width/height in world meters

    public static RosettaOverheadCatalog ExportCatalog(RosettaCorpusData corpus, int minimapResolution = 256)
    {
        ArgumentNullException.ThrowIfNull(corpus);

        var exhibits = new List<RosettaOverheadExhibit>();

        // Group placements by unique asset
        var placementsByAsset = corpus.Placements
            .GroupBy(p => p.AssetId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.OrdinalIgnoreCase);

        foreach (var (assetPath, entry) in corpus.UniqueAssets)
        {
            string assetId = RosettaDatastoreWriter.ComputeAssetId(assetPath);

            Vector3 min = entry.BoundsMin;
            Vector3 max = entry.BoundsMax;
            float extentX = Math.Abs(max.X - min.X);
            float extentY = Math.Abs(max.Y - min.Y);
            float extentZ = Math.Abs(max.Z - min.Z);
            float footprintArea = extentX * extentY;

            var placementBoxes = new List<RosettaOverheadPlacementBox>();
            if (placementsByAsset.TryGetValue(assetId, out var placements))
            {
                foreach (RosettaDecodedPlacement p in placements)
                {
                    // Compute normalized UV bounding extent
                    float halfNormU = (extentX * 0.5f) / AdtTileSize;
                    float halfNormV = (extentY * 0.5f) / AdtTileSize;

                    float centerU = p.CellU;
                    float centerV = p.CellV;

                    float minU = Math.Clamp(centerU - halfNormU, 0f, 1f);
                    float maxU = Math.Clamp(centerU + halfNormU, 0f, 1f);
                    float minV = Math.Clamp(centerV - halfNormV, 0f, 1f);
                    float maxV = Math.Clamp(centerV + halfNormV, 0f, 1f);

                    int pixMinX = (int)Math.Floor(minU * minimapResolution);
                    int pixMaxX = (int)Math.Ceiling(maxU * minimapResolution);
                    int pixMinY = (int)Math.Floor(minV * minimapResolution);
                    int pixMaxY = (int)Math.Ceiling(maxV * minimapResolution);

                    placementBoxes.Add(new RosettaOverheadPlacementBox(
                        p.TileX,
                        p.TileY,
                        p.RawPosition,
                        p.CellU,
                        p.CellV,
                        p.CellSize,
                        minU,
                        minV,
                        maxU,
                        maxV,
                        pixMinX,
                        pixMinY,
                        pixMaxX,
                        pixMaxY,
                        p.UniqueId));
                }
            }

            exhibits.Add(new RosettaOverheadExhibit(
                assetId,
                assetPath,
                entry.Kind.ToString(),
                extentX,
                extentY,
                extentZ,
                footprintArea,
                min,
                max,
                placementBoxes));
        }

        return new RosettaOverheadCatalog(
            corpus.MapName,
            exhibits.Count,
            corpus.Placements.Count,
            exhibits);
    }

    public static string ExportToJson(RosettaCorpusData corpus, int minimapResolution = 256)
    {
        RosettaOverheadCatalog catalog = ExportCatalog(corpus, minimapResolution);
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
        return JsonSerializer.Serialize(catalog, options);
    }

    public static void ExportToFile(RosettaCorpusData corpus, string outputPath, int minimapResolution = 256)
    {
        string json = ExportToJson(corpus, minimapResolution);
        string? dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }
        File.WriteAllText(outputPath, json);
    }
}
