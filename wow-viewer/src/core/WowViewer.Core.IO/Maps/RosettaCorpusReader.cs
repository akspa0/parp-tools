using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// A decoded placement record loaded from a synthetic Rosetta map or manifest.
/// </summary>
public sealed record RosettaDecodedPlacement(
    string AssetId,
    string AssetPath,
    string NormalizedPath,
    RosettaAssetKind Kind,
    int TileX,
    int TileY,
    Vector3 RawPosition,
    Vector3 RendererPosition,
    float CellU,
    float CellV,
    float CellSize,
    float ObjectBandSize,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    string LabelText,
    int UniqueId);

/// <summary>
/// Structured data decoded from a Rosetta Calibration map corpus.
/// </summary>
public sealed record RosettaCorpusData(
    string MapName,
    string? BuildId,
    string? GeneratedUtc,
    IReadOnlyList<RosettaDecodedPlacement> Placements,
    IReadOnlyList<string> Tiles,
    IReadOnlyDictionary<string, RosettaAssetEntry> UniqueAssets);

/// <summary>
/// Reads and decodes synthetic Rosetta maps, manifests, and datastores back into structured placement
/// and geometry records for reference library generation (Spec 190 US2 / T010).
/// </summary>
public static class RosettaCorpusReader
{
    /// <summary>
    /// Reads a Rosetta corpus from a directory by automatically locating its manifest, index, or datastore.
    /// </summary>
    public static RosettaCorpusData ReadFromDirectory(string directoryPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryPath);
        if (!Directory.Exists(directoryPath))
            throw new DirectoryNotFoundException($"Rosetta corpus directory not found: {directoryPath}");

        // 1. Look for rosetta-manifest.json in the directory or immediate subdirectories
        string directManifest = Path.Combine(directoryPath, "rosetta-manifest.json");
        if (File.Exists(directManifest))
            return ReadFromManifestFile(directManifest);

        string[] subManifests = Directory.GetFiles(directoryPath, "rosetta-manifest.json", SearchOption.AllDirectories);
        if (subManifests.Length > 0)
            return ReadFromManifestFile(subManifests[0]);

        // 2. Look for global_assets/catalog.parquet (Zarr datastore)
        string catalogParquet = Path.Combine(directoryPath, "global_assets", "catalog.parquet");
        if (File.Exists(catalogParquet))
        {
            var library = RosettaObjectLibrary.Open(directoryPath);
            return ReadFromDatastore(library);
        }

        throw new InvalidDataException($"Could not find a valid rosetta-manifest.json or Zarr datastore in: {directoryPath}");
    }

    /// <summary>
    /// Reads a Rosetta corpus directly from a `rosetta-manifest.json` file.
    /// </summary>
    public static RosettaCorpusData ReadFromManifestFile(string manifestJsonPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestJsonPath);
        if (!File.Exists(manifestJsonPath))
            throw new FileNotFoundException($"Rosetta manifest not found: {manifestJsonPath}");

        string json = File.ReadAllText(manifestJsonPath, Encoding.UTF8);
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;

        string mapName = root.TryGetProperty("map", out var mapProp) ? mapProp.GetString() ?? "Rosetta" : "Rosetta";
        string? generatedUtc = root.TryGetProperty("generatedUtc", out var genProp) ? genProp.GetString() : null;
        string? clientRoot = root.TryGetProperty("clientRoot", out var clientProp) ? clientProp.GetString() : null;

        var placements = new List<RosettaDecodedPlacement>();
        var tiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var uniqueAssets = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase);

        if (root.TryGetProperty("placements", out var placementsArray) && placementsArray.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement elem in placementsArray.EnumerateArray())
            {
                string assetPath = elem.GetProperty("asset").GetString() ?? string.Empty;
                string normalized = assetPath.Replace('\\', '/').Trim().ToLowerInvariant();
                string kindStr = elem.TryGetProperty("kind", out var kProp) ? kProp.GetString() ?? "Model" : "Model";
                RosettaAssetKind kind = string.Equals(kindStr, "WorldModel", StringComparison.OrdinalIgnoreCase)
                    ? RosettaAssetKind.WorldModel
                    : RosettaAssetKind.Model;

                int tileX = elem.GetProperty("tileX").GetInt32();
                int tileY = elem.GetProperty("tileY").GetInt32();
                tiles.Add($"{tileX}_{tileY}");

                Vector3 rawPos = ReadVector3(elem, "rawPosition");
                Vector3 renderPos = ReadVector3(elem, "rendererPosition");

                float cellSize = elem.TryGetProperty("cellSize", out var cs) ? cs.GetSingle() : 266.66666f;
                float objBandSize = elem.TryGetProperty("objectBandSize", out var obs) ? obs.GetSingle() : cellSize * 0.75f;
                float cellU = elem.TryGetProperty("cellU", out var cu) ? cu.GetSingle() : 0f;
                float cellV = elem.TryGetProperty("cellV", out var cv) ? cv.GetSingle() : 0f;
                string label = elem.TryGetProperty("label", out var lbl) ? lbl.GetString() ?? string.Empty : string.Empty;
                int uniqueId = elem.TryGetProperty("uniqueId", out var uid) ? uid.GetInt32() : 0;

                string assetId = RosettaDatastoreWriter.ComputeAssetId(assetPath);

                // Reconstruct estimated or standard model-local bounding extents from placement footprint
                float halfObj = objBandSize * 0.5f;
                Vector3 boundsMin = new(-halfObj, -halfObj, 0f);
                Vector3 boundsMax = new(halfObj, halfObj, halfObj * 1.5f);

                var decoded = new RosettaDecodedPlacement(
                    assetId, assetPath, normalized, kind,
                    tileX, tileY, rawPos, renderPos,
                    cellU, cellV, cellSize, objBandSize,
                    boundsMin, boundsMax, label, uniqueId);

                placements.Add(decoded);

                if (!uniqueAssets.ContainsKey(normalized))
                {
                    uniqueAssets[normalized] = new RosettaAssetEntry(assetPath, kind, boundsMin, boundsMax);
                }
            }
        }

        return new RosettaCorpusData(
            mapName,
            clientRoot,
            generatedUtc,
            placements,
            tiles.OrderBy(static t => t).ToList(),
            uniqueAssets);
    }

    /// <summary>
    /// Reads and converts an in-memory generation result into decoded corpus data.
    /// </summary>
    public static RosettaCorpusData ReadFromGenerationResult(RosettaGenerationResult result)
    {
        ArgumentNullException.ThrowIfNull(result);

        var placements = new List<RosettaDecodedPlacement>();
        var tiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var uniqueAssets = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase);

        foreach (RosettaPlacementRecord p in result.Placements)
        {
            string normalized = p.Asset.AssetPath.Replace('\\', '/').Trim().ToLowerInvariant();
            string assetId = RosettaDatastoreWriter.ComputeAssetId(p.Asset.AssetPath);

            tiles.Add($"{p.TileX}_{p.TileY}");

            var decoded = new RosettaDecodedPlacement(
                assetId,
                p.Asset.AssetPath,
                normalized,
                p.Asset.Kind,
                p.TileX,
                p.TileY,
                p.RawPosition,
                p.RendererPosition,
                p.CellU,
                p.CellV,
                p.CellSize,
                p.ObjectBandSize,
                p.Asset.BoundsMin,
                p.Asset.BoundsMax,
                p.LabelText,
                p.UniqueId);

            placements.Add(decoded);

            if (!uniqueAssets.ContainsKey(normalized))
            {
                uniqueAssets[normalized] = p.Asset;
            }
        }

        return new RosettaCorpusData(
            result.MapName,
            BuildId: null,
            DateTime.UtcNow.ToString("o"),
            placements,
            tiles.OrderBy(static t => t).ToList(),
            uniqueAssets);
    }

    /// <summary>
    /// Reads asset data from a Unified Multi-Version Zarr Datastore.
    /// </summary>
    public static RosettaCorpusData ReadFromDatastore(RosettaObjectLibrary objectLibrary, string? buildId = null)
    {
        ArgumentNullException.ThrowIfNull(objectLibrary);

        string targetBuild = buildId ?? (objectLibrary.Builds.Count > 0 ? objectLibrary.Builds[0] : "default");
        var placements = new List<RosettaDecodedPlacement>();
        var tiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var uniqueAssets = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase);

        foreach (RosettaGlobalAssetRecord rec in objectLibrary.GetAllAssets())
        {
            RosettaAssetKind kind = string.Equals(rec.Kind, "WorldModel", StringComparison.OrdinalIgnoreCase) || string.Equals(rec.Kind, "wmo", StringComparison.OrdinalIgnoreCase)
                ? RosettaAssetKind.WorldModel
                : RosettaAssetKind.Model;

            Vector3 min = new(rec.BoundsMinX, rec.BoundsMinY, rec.BoundsMinZ);
            Vector3 max = new(rec.BoundsMaxX, rec.BoundsMaxY, rec.BoundsMaxZ);

            var entry = new RosettaAssetEntry(rec.OriginalPath, kind, min, max);
            uniqueAssets[rec.NormalizedPath] = entry;

            var decoded = new RosettaDecodedPlacement(
                rec.AssetId,
                rec.OriginalPath,
                rec.NormalizedPath,
                kind,
                TileX: 24,
                TileY: 24,
                Vector3.Zero,
                Vector3.Zero,
                CellU: 0f,
                CellV: 0f,
                CellSize: 266.66666f,
                ObjectBandSize: 200f,
                min,
                max,
                Path.GetFileName(rec.OriginalPath),
                UniqueId: 0);

            placements.Add(decoded);
        }

        return new RosettaCorpusData(
            "RosettaDatastore",
            targetBuild,
            DateTime.UtcNow.ToString("o"),
            placements,
            ["24_24"],
            uniqueAssets);
    }

    /// <summary>
    /// Builds a <see cref="RosettaReferenceLibrary"/> by extracting signatures from the decoded corpus.
    /// </summary>
    public static RosettaReferenceLibrary BuildReferenceLibrary(
        RosettaCorpusData corpus,
        string? buildLabel = null)
    {
        ArgumentNullException.ThrowIfNull(corpus);

        var referenceAssets = new List<RosettaReferenceAsset>();
        var tileCoordsByAsset = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);

        foreach (RosettaDecodedPlacement p in corpus.Placements)
        {
            if (!tileCoordsByAsset.TryGetValue(p.NormalizedPath, out var coords))
            {
                coords = [];
                tileCoordsByAsset[p.NormalizedPath] = coords;
            }
            string coord = $"{p.TileX}_{p.TileY}";
            if (!coords.Contains(coord))
                coords.Add(coord);
        }

        foreach (var (normPath, asset) in corpus.UniqueAssets)
        {
            string assetId = RosettaDatastoreWriter.ComputeAssetId(asset.AssetPath);
            string kind = asset.Kind == RosettaAssetKind.WorldModel ? "wmo" : "m2";
            IReadOnlyList<string> tiles = tileCoordsByAsset.GetValueOrDefault(normPath, []);

            Vector3 span = asset.BoundsMax - asset.BoundsMin;
            Vector3 center = (asset.BoundsMin + asset.BoundsMax) * 0.5f;
            float diagonalXY = MathF.Sqrt(span.X * span.X + span.Y * span.Y);
            float volume = MathF.Max(0.001f, span.X) * MathF.Max(0.001f, span.Y) * MathF.Max(0.001f, span.Z);
            float footprintArea = MathF.Max(0.001f, span.X) * MathF.Max(0.001f, span.Y);
            float aspectXY = span.X / MathF.Max(0.001f, span.Y);
            float aspectZMaxXY = span.Z / MathF.Max(0.001f, MathF.Max(span.X, span.Y));

            Vector2[] hull =
            [
                new Vector2(asset.BoundsMin.X, asset.BoundsMin.Y),
                new Vector2(asset.BoundsMax.X, asset.BoundsMin.Y),
                new Vector2(asset.BoundsMax.X, asset.BoundsMax.Y),
                new Vector2(asset.BoundsMin.X, asset.BoundsMax.Y)
            ];

            var bounds = new Pm4Bounds3(asset.BoundsMin, asset.BoundsMax);

            var refAsset = new RosettaReferenceAsset(
                assetId,
                asset.AssetPath,
                normPath,
                kind,
                corpus.BuildId,
                tiles,
                bounds,
                center,
                span,
                diagonalXY,
                volume,
                footprintArea,
                hull,
                aspectXY,
                aspectZMaxXY,
                SubPartBounds: null,
                Signals: new Dictionary<string, double>(StringComparer.Ordinal)
                {
                    ["boundsSpanX"] = span.X,
                    ["boundsSpanY"] = span.Y,
                    ["boundsSpanZ"] = span.Z,
                    ["boundsVolume"] = volume,
                    ["footprintDiagonalXY"] = diagonalXY,
                    ["footprintArea"] = footprintArea,
                    ["aspectRatioXY"] = aspectXY,
                    ["aspectRatioZMaxXY"] = aspectZMaxXY,
                },
                ValidationTags: ["rosetta-calibration", "ground-truth"]);

            referenceAssets.Add(refAsset);
        }

        string libId = RosettaReferenceLibrary.ComputeLibraryId(buildLabel ?? corpus.BuildId ?? "rosetta", referenceAssets);

        return new RosettaReferenceLibrary(
            libId,
            buildLabel ?? corpus.BuildId,
            referenceAssets);
    }

    private static Vector3 ReadVector3(JsonElement elem, string propName)
    {
        if (elem.TryGetProperty(propName, out var array) && array.ValueKind == JsonValueKind.Array && array.GetArrayLength() >= 3)
        {
            return new Vector3(
                array[0].GetSingle(),
                array[1].GetSingle(),
                array[2].GetSingle());
        }

        return Vector3.Zero;
    }
}
