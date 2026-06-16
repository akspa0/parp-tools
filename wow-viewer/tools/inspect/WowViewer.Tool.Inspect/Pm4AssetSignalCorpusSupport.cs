using System.Buffers.Binary;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;

internal sealed record Pm4AssetSignalCorpusManifest(
    string RunId,
    string ArchiveRoot,
    string ClientBuild,
    int AssetCount,
    IReadOnlyList<Pm4AssetReferenceSignalRecord> Assets,
    IReadOnlyList<string> Warnings);

internal sealed record SubPartBounds(
    int Flags,
    double MinX, double MinY, double MinZ,
    double MaxX, double MaxY, double MaxZ);

internal static class Pm4AssetSignalCorpusSupport
{
    public static Pm4AssetSignalCorpusManifest BuildFromArchive(
        string archiveRoot,
        ArchiveCatalogBootstrapOptions archiveBootstrapOptions,
        string? assetKindFilter,
        string? pathFilter,
        int? limit,
        string? seedPlacementsPath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(archiveRoot);

        string fullArchiveRoot = Path.GetFullPath(archiveRoot);
        string buildLabel = Path.GetFileName(fullArchiveRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [fullArchiveRoot], archiveBootstrapOptions);
        IReadOnlyList<string> candidateFiles = !string.IsNullOrWhiteSpace(seedPlacementsPath)
            ? LoadSeedPlacementAssetPaths(seedPlacementsPath)
            : bootstrap.AllFiles.Count > 0
                ? bootstrap.AllFiles
                : LoadFallbackListfileEntries(archiveBootstrapOptions);

        IEnumerable<string> candidates = candidateFiles
            .Where(IsSupportedAssetPath)
            .Where(path => MatchesKindFilter(path, assetKindFilter))
            .Where(path => string.IsNullOrWhiteSpace(pathFilter) || path.Contains(pathFilter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(static path => path, StringComparer.OrdinalIgnoreCase);

        if (limit is > 0)
            candidates = candidates.Take(limit.Value);

        List<Pm4AssetReferenceSignalRecord> assets = [];
        List<string> warnings = [];
        HashSet<string> seenAssetIds = new(StringComparer.OrdinalIgnoreCase);
        foreach (string assetPath in candidates)
        {
            try
            {
                Pm4AssetReferenceSignalRecord asset = BuildAssetRecord(assetPath, fullArchiveRoot, buildLabel, archiveBootstrapOptions);
                if (seenAssetIds.Add(asset.AssetId))
                    assets.Add(asset);
            }
            catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
            {
                warnings.Add($"Asset '{assetPath}' could not be summarized: {ex.Message}");
            }
        }

        string runId = BuildRunId(buildLabel, assets);
        return new Pm4AssetSignalCorpusManifest(
            runId,
            fullArchiveRoot,
            buildLabel,
            assets.Count,
            assets,
            warnings);
    }

    public static Pm4AssetReferenceBuildResult LoadFromManifest(string manifestPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);

		string fullPath = Path.GetFullPath(manifestPath);
		string json = File.ReadAllText(fullPath);
		Pm4AssetSignalCorpusManifest? manifest = JsonSerializer.Deserialize<Pm4AssetSignalCorpusManifest>(json, Pm4MatchSupport.CreateJsonOptions());
        if (manifest is null)
            throw new InvalidDataException($"Asset signal corpus '{fullPath}' could not be deserialized.");

        return new Pm4AssetReferenceBuildResult(
            manifest.Assets,
            manifest.Warnings);
    }

    private static Pm4AssetReferenceSignalRecord BuildAssetRecord(
        string assetPath,
        string archiveRoot,
        string buildLabel,
        ArchiveCatalogBootstrapOptions archiveBootstrapOptions)
    {
        string normalizedVirtualPath = NormalizeVirtualPath(assetPath);
        byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(normalizedVirtualPath, [archiveRoot], archiveBootstrapOptions);
        using MemoryStream stream = new(bytes, writable: false);

        if (IsWmoRootPath(assetPath))
        {
            WmoSummary summary = WmoSummaryReader.Read(stream, assetPath);
            return BuildWmoAssetReference(assetPath, buildLabel, summary, bytes);
        }

        MdxSummary mdxSummary = MdxSummaryReader.Read(stream, assetPath);
        return BuildM2AssetReference(assetPath, buildLabel, mdxSummary);
    }

    private static Pm4AssetReferenceSignalRecord BuildWmoAssetReference(
        string assetPath,
        string buildLabel,
        WmoSummary summary,
        byte[] rawFileBytes)
    {
        Vector3 boundsMin = summary.BoundsMin;
        Vector3 boundsMax = summary.BoundsMax;
        ValidateFiniteBounds(assetPath, boundsMin, boundsMax);
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        Vector2[] footprintHull = BuildAabbFootprintHull(boundsMin, boundsMax);
        float footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
        Vector3 span = boundsMax - boundsMin;
        float diagonalXY = MathF.Sqrt(span.X * span.X + span.Y * span.Y);
        float volume = MathF.Max(0f, span.X) * MathF.Max(0f, span.Y) * MathF.Max(0f, span.Z);
        ValidateFiniteSignal(assetPath, "footprintArea", footprintArea);
        ValidateFiniteSignal(assetPath, "diagonalXY", diagonalXY);
        ValidateFiniteSignal(assetPath, "volume", volume);

        IReadOnlyList<SubPartBounds>? groupBounds = ExtractWmoGroupBoundsViaReader(rawFileBytes, assetPath);
        string? groupBoundsJson = groupBounds is { Count: > 0 }
            ? JsonSerializer.Serialize(groupBounds)
            : null;

        return new Pm4AssetReferenceSignalRecord(
            BuildAssetId("wmo", buildLabel, assetPath),
            assetPath,
            "wmo",
            buildLabel,
            Array.Empty<string>(),
            new Pm4Bounds3(boundsMin, boundsMax),
            center,
            footprintHull,
            footprintArea,
            null,
            null,
            null,
            new Dictionary<string, int>(StringComparer.Ordinal)
            {
                ["assetKind:wmo"] = 1,
                ["wmo:hasSkybox"] = summary.HasSkybox ? 1 : 0,
                ["wmoGroupCount"] = summary.GroupInfoCount,
            },
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["boundsSpanX"] = RequireFiniteSignal(assetPath, "boundsSpanX", span.X),
                ["boundsSpanY"] = RequireFiniteSignal(assetPath, "boundsSpanY", span.Y),
                ["boundsSpanZ"] = RequireFiniteSignal(assetPath, "boundsSpanZ", span.Z),
                ["boundsVolume"] = RequireFiniteSignal(assetPath, "boundsVolume", volume),
                ["footprintDiagonalXY"] = RequireFiniteSignal(assetPath, "footprintDiagonalXY", diagonalXY),
                ["wmoMaterialCount"] = summary.MaterialEntryCount,
                ["wmoDoodadPlacementCount"] = summary.DoodadPlacementEntryCount,
                },
            Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
            groupBoundsJson is not null ? $"subPartBounds:{groupBoundsJson}" : null,
            ["durable-asset-corpus"]);
    }

    private static Pm4AssetReferenceSignalRecord BuildM2AssetReference(
        string assetPath,
        string buildLabel,
        MdxSummary summary)
    {
        Vector3 boundsMin = summary.Collision?.BoundsMin ?? summary.BoundsMin
            ?? throw new InvalidDataException($"Model '{assetPath}' does not expose usable bounds.");
        Vector3 boundsMax = summary.Collision?.BoundsMax ?? summary.BoundsMax
            ?? throw new InvalidDataException($"Model '{assetPath}' does not expose usable bounds.");
        ValidateFiniteBounds(assetPath, boundsMin, boundsMax);

        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        Vector2[] footprintHull = BuildAabbFootprintHull(boundsMin, boundsMax);
        float footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
        Vector3 span = boundsMax - boundsMin;
        float diagonalXY = MathF.Sqrt(span.X * span.X + span.Y * span.Y);
        float volume = MathF.Max(0f, span.X) * MathF.Max(0f, span.Y) * MathF.Max(0f, span.Z);
        ValidateFiniteSignal(assetPath, "footprintArea", footprintArea);
        ValidateFiniteSignal(assetPath, "diagonalXY", diagonalXY);
        ValidateFiniteSignal(assetPath, "volume", volume);

        IReadOnlyList<SubPartBounds> geosetBounds = ExtractM2GeosetBounds(summary);
        string? geosetBoundsJson = geosetBounds.Count > 0
            ? JsonSerializer.Serialize(geosetBounds)
            : null;

        return new Pm4AssetReferenceSignalRecord(
            BuildAssetId("m2", buildLabel, assetPath),
            assetPath,
            "m2",
            buildLabel,
            Array.Empty<string>(),
            new Pm4Bounds3(boundsMin, boundsMax),
            center,
            footprintHull,
            footprintArea,
            null,
            null,
            null,
            new Dictionary<string, int>(StringComparer.Ordinal)
            {
                ["assetKind:m2"] = 1,
                ["m2:hasCollision"] = summary.HasCollision ? 1 : 0,
            },
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["boundsSpanX"] = RequireFiniteSignal(assetPath, "boundsSpanX", span.X),
                ["boundsSpanY"] = RequireFiniteSignal(assetPath, "boundsSpanY", span.Y),
                ["boundsSpanZ"] = RequireFiniteSignal(assetPath, "boundsSpanZ", span.Z),
                ["boundsVolume"] = RequireFiniteSignal(assetPath, "boundsVolume", volume),
                ["footprintDiagonalXY"] = RequireFiniteSignal(assetPath, "footprintDiagonalXY", diagonalXY),
                ["m2GeosetCount"] = summary.GeosetCount,
                ["m2BoneCount"] = summary.BoneCount,
                ["m2MaterialLayerCount"] = summary.MaterialLayerCount,
                ["m2TextureCount"] = summary.TextureCount,
            },
            Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
            geosetBoundsJson is not null ? $"subPartBounds:{geosetBoundsJson}" : null,
            ["durable-asset-corpus"]);
    }

    private static string BuildAssetId(string assetKind, string buildLabel, string assetPath)
    {
        return $"{assetKind}:{buildLabel}:{NormalizeVirtualPath(assetPath)}";
    }

    private static void ValidateFiniteBounds(string assetPath, Vector3 boundsMin, Vector3 boundsMax)
    {
        if (!IsFinite(boundsMin) || !IsFinite(boundsMax))
            throw new InvalidDataException($"Asset '{assetPath}' exposed non-finite bounds.");
    }

    private static double RequireFiniteSignal(string assetPath, string signalName, float value)
    {
        ValidateFiniteSignal(assetPath, signalName, value);
        return value;
    }

    private static void ValidateFiniteSignal(string assetPath, string signalName, float value)
    {
        if (!float.IsFinite(value))
            throw new InvalidDataException($"Asset '{assetPath}' exposed non-finite signal '{signalName}'.");
    }

    private static IReadOnlyList<string> LoadFallbackListfileEntries(ArchiveCatalogBootstrapOptions archiveBootstrapOptions)
    {
        if (string.IsNullOrWhiteSpace(archiveBootstrapOptions.ExternalListfilePath) || !File.Exists(archiveBootstrapOptions.ExternalListfilePath))
            return Array.Empty<string>();

        return ArchiveCatalogBootstrapper.ParseExternalListfileLines(File.ReadLines(archiveBootstrapOptions.ExternalListfilePath))
            .OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    internal static IReadOnlyList<string> LoadSeedPlacementAssetPaths(string seedPlacementsPath)
    {
        string fullSeedPath = Path.GetFullPath(seedPlacementsPath);
        if (File.Exists(fullSeedPath))
            return BuildSeedPlacementAssetPaths([fullSeedPath]);

        if (Directory.Exists(fullSeedPath))
        {
            IReadOnlyList<string> files = Directory.EnumerateFiles(fullSeedPath, "*_obj0.adt", SearchOption.TopDirectoryOnly)
                .OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            return BuildSeedPlacementAssetPaths(files);
        }

        throw new DirectoryNotFoundException($"Seed placements path '{fullSeedPath}' does not exist.");
    }

    private static IReadOnlyList<string> BuildSeedPlacementAssetPaths(IReadOnlyList<string> placementFiles)
    {
        HashSet<string> assetPaths = new(StringComparer.OrdinalIgnoreCase);

        foreach (string placementFile in placementFiles)
        {
            AdtPlacementCatalog placements = AdtPlacementReader.Read(placementFile);
            foreach (AdtWorldModelPlacement placement in placements.WorldModelPlacements)
            {
                if (!string.IsNullOrWhiteSpace(placement.ModelPath))
                    assetPaths.Add(placement.ModelPath);
            }

            foreach (AdtModelPlacement placement in placements.ModelPlacements)
            {
                if (!string.IsNullOrWhiteSpace(placement.ModelPath))
                    assetPaths.Add(placement.ModelPath);
            }
        }

        return assetPaths
            .OrderBy(static path => BuildPriority(path))
            .ThenBy(static path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static string BuildRunId(string buildLabel, IReadOnlyList<Pm4AssetReferenceSignalRecord> assets)
    {
        StringBuilder builder = new();
        builder.Append(buildLabel);
        builder.Append('|');
        foreach (Pm4AssetReferenceSignalRecord asset in assets)
        {
            builder.Append(asset.AssetId);
            builder.Append('\n');
        }

        byte[] bytes = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
        string digest = Convert.ToHexString(bytes[..8]).ToLowerInvariant();
        return $"pm4-asset-corpus-{buildLabel}-{digest}";
    }

    private static bool IsSupportedAssetPath(string assetPath)
    {
        if (string.IsNullOrWhiteSpace(assetPath))
            return false;

        return IsWmoRootPath(assetPath)
            || assetPath.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)
            || assetPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)
            || assetPath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase);
    }

    private static bool MatchesKindFilter(string assetPath, string? assetKindFilter)
    {
        if (string.IsNullOrWhiteSpace(assetKindFilter) || string.Equals(assetKindFilter, "all", StringComparison.OrdinalIgnoreCase))
            return true;

        if (string.Equals(assetKindFilter, "wmo", StringComparison.OrdinalIgnoreCase))
            return IsWmoRootPath(assetPath);

        if (string.Equals(assetKindFilter, "m2", StringComparison.OrdinalIgnoreCase))
            return !IsWmoRootPath(assetPath);

        throw new ArgumentOutOfRangeException(nameof(assetKindFilter), assetKindFilter, "Asset kind filter must be one of: all, wmo, m2.");
    }

    private static bool IsWmoRootPath(string assetPath)
    {
        if (!assetPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
            return false;

        string fileName = Path.GetFileName(assetPath);
        int underscoreIndex = fileName.LastIndexOf('_');
        if (underscoreIndex < 0)
            return true;

        int digitsStart = underscoreIndex + 1;
        int digitsLength = fileName.Length - digitsStart - 4;
        if (digitsLength != 3)
            return true;

        for (int index = digitsStart; index < digitsStart + digitsLength; index++)
        {
            if (!char.IsDigit(fileName[index]))
                return true;
        }

        return false;
    }

    private static string NormalizeVirtualPath(string assetPath)
    {
        return assetPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }

    private static int BuildPriority(string assetPath)
    {
        string normalized = NormalizeVirtualPath(assetPath);
        if (normalized.StartsWith("world/wmo/", StringComparison.Ordinal))
            return 0;
        if (normalized.StartsWith("world/", StringComparison.Ordinal))
            return 1;
        if (normalized.StartsWith("doodads/", StringComparison.Ordinal))
            return 2;
        if (normalized.StartsWith("generic/", StringComparison.Ordinal))
            return 3;

        return 4;
    }

    private static IReadOnlyList<SubPartBounds> ExtractWmoGroupBoundsViaReader(byte[] rawFileBytes, string assetPath)
    {
        try
        {
            using MemoryStream stream = new(rawFileBytes, writable: false);
            WmoGroupInfoSummary groupInfo = WmoGroupInfoSummaryReader.Read(stream, assetPath);
            if (groupInfo.EntryCount == 0)
                return [];

            List<SubPartBounds> bounds = new(groupInfo.EntryCount);
            foreach (WmoGroupInfoEntry entry in groupInfo.Entries)
            {
                if (!float.IsFinite(entry.BoundsMin.X) || !float.IsFinite(entry.BoundsMin.Y) ||
                    !float.IsFinite(entry.BoundsMin.Z) || !float.IsFinite(entry.BoundsMax.X) ||
                    !float.IsFinite(entry.BoundsMax.Y) || !float.IsFinite(entry.BoundsMax.Z))
                    continue;

                bounds.Add(new SubPartBounds(
                    (int)entry.Flags,
                    entry.BoundsMin.X, entry.BoundsMin.Y, entry.BoundsMin.Z,
                    entry.BoundsMax.X, entry.BoundsMax.Y, entry.BoundsMax.Z));
            }
            return bounds;
        }
        catch
        {
            return [];
        }
    }

    private static IReadOnlyList<SubPartBounds> ExtractM2GeosetBounds(MdxSummary summary)
    {
        if (summary.GeosetCount == 0)
            return [];

        List<SubPartBounds> bounds = new(summary.GeosetCount);
        foreach (MdxGeosetSummary geoset in summary.Geosets)
        {
            if (!geoset.BoundsMin.HasValue || !geoset.BoundsMax.HasValue)
                continue;

            Vector3 min = geoset.BoundsMin.Value;
            Vector3 max = geoset.BoundsMax.Value;
            if (!float.IsFinite(min.X) || !float.IsFinite(min.Y) || !float.IsFinite(min.Z) ||
                !float.IsFinite(max.X) || !float.IsFinite(max.Y) || !float.IsFinite(max.Z))
                continue;

            bounds.Add(new SubPartBounds(
                (int)geoset.Flags,
                min.X, min.Y, min.Z,
                max.X, max.Y, max.Z));
        }

        return bounds;
    }

    private static int FindChunkOffset(byte[] data, string fourCc)
    {
        if (data.Length < 8)
            return -1;

        uint target = BinaryPrimitives.ReadUInt32LittleEndian(
            Encoding.ASCII.GetBytes(fourCc).AsSpan());

        for (int i = 0; i <= data.Length - 8; i++)
        {
            uint sig = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(i, 4));
            if (sig == target)
                return i;
        }

        return -1;
    }

    private static bool IsFinite(Vector3 value)
    {
        return float.IsFinite(value.X)
            && float.IsFinite(value.Y)
            && float.IsFinite(value.Z);
    }

    private static Vector2[] BuildAabbFootprintHull(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];
    }
}
