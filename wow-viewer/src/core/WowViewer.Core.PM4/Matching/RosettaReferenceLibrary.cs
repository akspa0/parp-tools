using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// A single ground-truth asset entry in the Rosetta Reference Library.
/// Contains complete bounding, footprint, aspect, and geometric feature signals.
/// </summary>
public sealed record RosettaReferenceAsset(
    string AssetId,
    string AssetPath,
    string NormalizedPath,
    string AssetKind,
    string? ClientBuild,
    IReadOnlyList<string> TileCoordinates,
    Pm4Bounds3 Bounds,
    Vector3 Center,
    Vector3 Span,
    float DiagonalXY,
    float Volume,
    float FootprintArea,
    IReadOnlyList<Vector2> FootprintHull,
    float AspectRatioXY,
    float AspectRatioZMaxXY,
    IReadOnlyList<Pm4Bounds3>? SubPartBounds = null,
    IReadOnlyDictionary<string, double>? Signals = null,
    IReadOnlyList<string>? ValidationTags = null)
{
    /// <summary>
    /// Converts this reference asset into an interoperable <see cref="Pm4AssetReferenceSignalRecord"/>
    /// for consumption by <see cref="Pm4AssetMatchScorer"/> and reconciliation services.
    /// </summary>
    public Pm4AssetReferenceSignalRecord ToAssetReferenceSignalRecord(string? signalVersion = null)
    {
        var surfaceHist = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            [$"assetKind:{AssetKind}"] = 1,
            ["geometry:resolved"] = 1,
        };

        var renderSignals = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["boundsSpanX"] = Span.X,
            ["boundsSpanY"] = Span.Y,
            ["boundsSpanZ"] = Span.Z,
            ["boundsVolume"] = Volume,
            ["footprintDiagonalXY"] = DiagonalXY,
            ["footprintArea"] = FootprintArea,
            ["aspectRatioXY"] = AspectRatioXY,
            ["aspectRatioZMaxXY"] = AspectRatioZMaxXY,
        };

        if (Signals is not null)
        {
            foreach (var (k, v) in Signals)
                renderSignals[k] = v;
        }

        string? subPartStoreRow = null;
        if (SubPartBounds is { Count: > 0 })
        {
            var subParts = SubPartBounds.Select(b => new SubPartBoundsDto(
                b.Min.X, b.Min.Y, b.Min.Z,
                b.Max.X, b.Max.Y, b.Max.Z)).ToList();
            subPartStoreRow = $"subPartBounds:{JsonSerializer.Serialize(subParts)}";
        }

        return new Pm4AssetReferenceSignalRecord(
            AssetId,
            AssetPath,
            AssetKind,
            ClientBuild,
            TileCoordinates,
            Bounds,
            Center,
            FootprintHull,
            FootprintArea,
            ReferencePosition: null,
            ReferenceRotation: null,
            ReferenceScale: null,
            surfaceHist,
            renderSignals,
            signalVersion ?? Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
            subPartStoreRow,
            ValidationTags ?? ["rosetta-calibration", "ground-truth"]);
    }

    private sealed record SubPartBoundsDto(
        float MinX, float MinY, float MinZ,
        float MaxX, float MaxY, float MaxZ);
}

/// <summary>
/// Scored candidate evaluation against a reference library asset.
/// </summary>
public sealed record RosettaCandidateMatch(
    RosettaReferenceAsset Asset,
    double OverallScore,
    double SpanScore,
    double VolumeScore,
    double AspectRatioScore,
    double FootprintScore,
    string Rationale);

/// <summary>
/// A versioned, labelled reference library built from the Rosetta Calibration Corpus.
/// Serves as the ground-truth geometric knowledge base for deterministic PM4 object identification.
/// </summary>
public sealed class RosettaReferenceLibrary
{
    public const string CurrentVersion = "rosetta-reference-library-v1";

    public string Version { get; init; } = CurrentVersion;
    public string LibraryId { get; init; } = string.Empty;
    public string? BuildLabel { get; init; }
    public string GeneratedUtc { get; init; } = DateTime.UtcNow.ToString("o");
    public int TotalAssets => Assets.Count;
    public int ModelCount { get; init; }
    public int WorldModelCount { get; init; }
    public IReadOnlyList<RosettaReferenceAsset> Assets { get; init; } = [];

    [JsonIgnore]
    private readonly Dictionary<string, RosettaReferenceAsset> _assetsById = new(StringComparer.OrdinalIgnoreCase);

    [JsonIgnore]
    private readonly Dictionary<string, RosettaReferenceAsset> _assetsByNormalizedPath = new(StringComparer.OrdinalIgnoreCase);

    [JsonIgnore]
    private readonly List<RosettaReferenceAsset> _models = [];

    [JsonIgnore]
    private readonly List<RosettaReferenceAsset> _worldModels = [];

    public RosettaReferenceLibrary()
    {
    }

    public RosettaReferenceLibrary(
        string libraryId,
        string? buildLabel,
        IReadOnlyList<RosettaReferenceAsset> assets,
        string? version = null,
        string? generatedUtc = null)
    {
        ArgumentNullException.ThrowIfNull(assets);

        Version = version ?? CurrentVersion;
        LibraryId = libraryId ?? string.Empty;
        BuildLabel = buildLabel;
        GeneratedUtc = generatedUtc ?? DateTime.UtcNow.ToString("o");
        Assets = assets;

        ModelCount = assets.Count(static a => string.Equals(a.AssetKind, "m2", StringComparison.OrdinalIgnoreCase) || string.Equals(a.AssetKind, "model", StringComparison.OrdinalIgnoreCase));
        WorldModelCount = assets.Count(static a => string.Equals(a.AssetKind, "wmo", StringComparison.OrdinalIgnoreCase) || string.Equals(a.AssetKind, "worldmodel", StringComparison.OrdinalIgnoreCase));

        IndexAssets();
    }

    /// <summary>
    /// Re-indexes cached lookup tables after deserialization or asset modifications.
    /// </summary>
    public void IndexAssets()
    {
        _assetsById.Clear();
        _assetsByNormalizedPath.Clear();
        _models.Clear();
        _worldModels.Clear();

        foreach (RosettaReferenceAsset asset in Assets)
        {
            if (!string.IsNullOrWhiteSpace(asset.AssetId))
                _assetsById[asset.AssetId] = asset;

            if (!string.IsNullOrWhiteSpace(asset.NormalizedPath))
                _assetsByNormalizedPath[asset.NormalizedPath] = asset;

            bool isWmo = string.Equals(asset.AssetKind, "wmo", StringComparison.OrdinalIgnoreCase)
                || string.Equals(asset.AssetKind, "worldmodel", StringComparison.OrdinalIgnoreCase);

            if (isWmo)
                _worldModels.Add(asset);
            else
                _models.Add(asset);
        }
    }

    public bool TryGetAssetById(string assetId, out RosettaReferenceAsset? asset)
    {
        if (_assetsById.Count == 0 && Assets.Count > 0)
            IndexAssets();

        return _assetsById.TryGetValue(assetId, out asset);
    }

    public bool TryGetAssetByPath(string path, out RosettaReferenceAsset? asset)
    {
        if (_assetsByNormalizedPath.Count == 0 && Assets.Count > 0)
            IndexAssets();

        string normalized = (path ?? string.Empty).Replace('\\', '/').Trim().ToLowerInvariant();
        return _assetsByNormalizedPath.TryGetValue(normalized, out asset);
    }

    /// <summary>
    /// Converts all assets in this library to <see cref="Pm4AssetReferenceSignalRecord"/> instances
    /// for interoperability with legacy scorers.
    /// </summary>
    public IReadOnlyList<Pm4AssetReferenceSignalRecord> ToAssetReferenceSignalRecords()
    {
        return Assets.Select(static a => a.ToAssetReferenceSignalRecord()).ToList();
    }

    /// <summary>
    /// Evaluates candidate matches against a query bounding box.
    /// </summary>
    public IReadOnlyList<RosettaCandidateMatch> FindCandidatesByBounds(
        Pm4Bounds3 queryBounds,
        string? assetKind = null,
        float tolerance = 0.35f,
        int maxCandidates = 10)
    {
        if (_assetsById.Count == 0 && Assets.Count > 0)
            IndexAssets();

        IEnumerable<RosettaReferenceAsset> pool = Assets;
        if (!string.IsNullOrWhiteSpace(assetKind))
        {
            bool isWmo = string.Equals(assetKind, "wmo", StringComparison.OrdinalIgnoreCase)
                || string.Equals(assetKind, "worldmodel", StringComparison.OrdinalIgnoreCase);
            pool = isWmo ? _worldModels : _models;
        }

        Vector3 querySpan = queryBounds.Span;
        float queryVolume = MathF.Max(0.01f, querySpan.X) * MathF.Max(0.01f, querySpan.Y) * MathF.Max(0.01f, querySpan.Z);
        float queryDiagonal = MathF.Sqrt(querySpan.X * querySpan.X + querySpan.Y * querySpan.Y);
        float queryFootprint = MathF.Max(0.01f, querySpan.X) * MathF.Max(0.01f, querySpan.Y);
        float queryAspectXY = querySpan.X / MathF.Max(0.001f, querySpan.Y);
        float queryAspectZMax = querySpan.Z / MathF.Max(0.001f, MathF.Max(querySpan.X, querySpan.Y));

        var matches = new List<RosettaCandidateMatch>();

        foreach (RosettaReferenceAsset asset in pool)
        {
            // Compute component scores
            double spanScoreX = ComputeRatioScore(asset.Span.X, querySpan.X);
            double spanScoreY = ComputeRatioScore(asset.Span.Y, querySpan.Y);
            double spanScoreZ = ComputeRatioScore(asset.Span.Z, querySpan.Z);
            double spanScore = (spanScoreX + spanScoreY + spanScoreZ) / 3.0;

            double volumeScore = ComputeRatioScore(asset.Volume, queryVolume);
            double aspectScoreXY = ComputeRatioScore(asset.AspectRatioXY, queryAspectXY);
            double aspectScoreZ = ComputeRatioScore(asset.AspectRatioZMaxXY, queryAspectZMax);
            double aspectScore = (aspectScoreXY + aspectScoreZ) / 2.0;

            double footprintScore = ComputeRatioScore(asset.FootprintArea, queryFootprint);

            // Weighted overall score: Span (40%), Volume (25%), Aspect Ratio (20%), Footprint (15%)
            double overallScore = (spanScore * 0.40) + (volumeScore * 0.25) + (aspectScore * 0.20) + (footprintScore * 0.15);

            if (overallScore >= (1.0 - tolerance))
            {
                string rationale = $"SpanScore={spanScore:F3}, VolScore={volumeScore:F3}, AspectScore={aspectScore:F3}, FootprintScore={footprintScore:F3}";
                matches.Add(new RosettaCandidateMatch(asset, overallScore, spanScore, volumeScore, aspectScore, footprintScore, rationale));
            }
        }

        return matches
            .OrderByDescending(static m => m.OverallScore)
            .ThenByDescending(static m => m.SpanScore)
            .ThenByDescending(static m => m.VolumeScore)
            .Take(Math.Max(1, maxCandidates))
            .ToList();
    }

    /// <summary>
    /// Evaluates candidate matches against a query PM4 segment signal record.
    /// </summary>
    public IReadOnlyList<RosettaCandidateMatch> FindCandidatesBySignal(
        Pm4SegmentSignalRecord signal,
        string? assetKind = null,
        float tolerance = 0.35f,
        int maxCandidates = 10)
    {
        if (signal.Bounds is null)
            return [];

        return FindCandidatesByBounds(signal.Bounds, assetKind, tolerance, maxCandidates);
    }

    private static double ComputeRatioScore(float valA, float valB)
    {
        float a = MathF.Abs(valA);
        float b = MathF.Abs(valB);
        if (a < 0.0001f && b < 0.0001f)
            return 1.0;
        if (a < 0.0001f || b < 0.0001f)
            return 0.0;

        float min = MathF.Min(a, b);
        float max = MathF.Max(a, b);
        return Math.Clamp((double)(min / max), 0.0, 1.0);
    }

    public static JsonSerializerOptions GetJsonSerializerOptions()
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            PropertyNameCaseInsensitive = true
        };
        options.Converters.Add(new Vector3JsonConverter());
        options.Converters.Add(new Vector2JsonConverter());
        return options;
    }

    /// <summary>
    /// Saves the reference library to a JSON file.
    /// </summary>
    public void SaveToJson(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        var options = GetJsonSerializerOptions();
        string json = JsonSerializer.Serialize(this, options);
        File.WriteAllText(filePath, json, Encoding.UTF8);
    }

    /// <summary>
    /// Loads a reference library from a JSON file.
    /// </summary>
    public static RosettaReferenceLibrary LoadFromJson(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Rosetta Reference Library file not found: {filePath}");

        string json = File.ReadAllText(filePath, Encoding.UTF8);
        return LoadFromJsonString(json);
    }

    /// <summary>
    /// Loads a reference library from a JSON string.
    /// </summary>
    public static RosettaReferenceLibrary LoadFromJsonString(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        var options = GetJsonSerializerOptions();

        RosettaReferenceLibrary library = JsonSerializer.Deserialize<RosettaReferenceLibrary>(json, options)
            ?? throw new InvalidDataException("Failed to deserialize RosettaReferenceLibrary from JSON.");

        library.IndexAssets();
        return library;
    }

    /// <summary>
    /// Computes a deterministic library hash ID from the asset contents.
    /// </summary>
    public static string ComputeLibraryId(string buildLabel, IEnumerable<RosettaReferenceAsset> assets)
    {
        using var sha = SHA256.Create();
        var sb = new StringBuilder();
        sb.Append(buildLabel ?? string.Empty).Append('|');
        foreach (var asset in assets.OrderBy(static a => a.AssetPath, StringComparer.OrdinalIgnoreCase))
        {
            sb.Append(asset.AssetPath).Append(':')
              .Append(asset.Bounds.Min.X.ToString("F3")).Append(',')
              .Append(asset.Bounds.Min.Y.ToString("F3")).Append(',')
              .Append(asset.Bounds.Min.Z.ToString("F3")).Append('|')
              .Append(asset.Bounds.Max.X.ToString("F3")).Append(',')
              .Append(asset.Bounds.Max.Y.ToString("F3")).Append(',')
              .Append(asset.Bounds.Max.Z.ToString("F3")).Append(';');
        }

        byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
        return $"rosetta_lib_{Convert.ToHexString(hash)[..16].ToLowerInvariant()}";
    }
}

public sealed class Vector3JsonConverter : JsonConverter<Vector3>
{
    public override Vector3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.StartArray)
        {
            reader.Read();
            float x = reader.GetSingle();
            reader.Read();
            float y = reader.GetSingle();
            reader.Read();
            float z = reader.GetSingle();
            reader.Read(); // EndArray
            return new Vector3(x, y, z);
        }

        if (reader.TokenType == JsonTokenType.StartObject)
        {
            float x = 0f, y = 0f, z = 0f;
            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string prop = reader.GetString()!;
                    reader.Read();
                    if (string.Equals(prop, "X", StringComparison.OrdinalIgnoreCase))
                        x = reader.GetSingle();
                    else if (string.Equals(prop, "Y", StringComparison.OrdinalIgnoreCase))
                        y = reader.GetSingle();
                    else if (string.Equals(prop, "Z", StringComparison.OrdinalIgnoreCase))
                        z = reader.GetSingle();
                }
            }
            return new Vector3(x, y, z);
        }

        return Vector3.Zero;
    }

    public override void Write(Utf8JsonWriter writer, Vector3 value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteNumber("x", value.X);
        writer.WriteNumber("y", value.Y);
        writer.WriteNumber("z", value.Z);
        writer.WriteEndObject();
    }
}

public sealed class Vector2JsonConverter : JsonConverter<Vector2>
{
    public override Vector2 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.StartArray)
        {
            reader.Read();
            float x = reader.GetSingle();
            reader.Read();
            float y = reader.GetSingle();
            reader.Read(); // EndArray
            return new Vector2(x, y);
        }

        if (reader.TokenType == JsonTokenType.StartObject)
        {
            float x = 0f, y = 0f;
            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string prop = reader.GetString()!;
                    reader.Read();
                    if (string.Equals(prop, "X", StringComparison.OrdinalIgnoreCase))
                        x = reader.GetSingle();
                    else if (string.Equals(prop, "Y", StringComparison.OrdinalIgnoreCase))
                        y = reader.GetSingle();
                }
            }
            return new Vector2(x, y);
        }

        return Vector2.Zero;
    }

    public override void Write(Utf8JsonWriter writer, Vector2 value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteNumber("x", value.X);
        writer.WriteNumber("y", value.Y);
        writer.WriteEndObject();
    }
}
