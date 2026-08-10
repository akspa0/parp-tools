using System.Security.Cryptography;

namespace WowViewer.Core.Maps;

/// <summary>Dataset root formats that the viewer can identify without guessing.</summary>
public enum DatasetSourceKind
{
    VlmProject,
    ZarrStore,
    RealTileObservation,
}

/// <summary>Provenance class for real imagery supplied to reconstruction.</summary>
public enum RealTileObservationKind
{
    ClientHarvest,
    AuthoredMinimap,
    MediaReference,
    Unknown,
}

/// <summary>
/// Viewer-facing metadata for one dataset version. This is deliberately a summary contract;
/// signal names describe what is present, not whether every tile has populated coverage.
/// </summary>
public sealed record DatasetVersionCatalogEntry(
    string Id,
    string DisplayName,
    string RootPath,
    DatasetSourceKind SourceKind,
    string? MapName,
    int TileCount,
    IReadOnlyList<string> Signals,
    bool Renderable,
    string? Diagnostic);

/// <summary>
/// Persistable selection state kept separate from client-source and secondary-overlay state.
/// </summary>
public sealed record DatasetVersionSelection(
    string CatalogRoot,
    string? SelectedRoot,
    string? ActiveRoot,
    bool ClientSourceUnchanged = true,
    bool SecondaryOverlayUnchanged = true);

/// <summary>
/// One real image observation. It is input evidence, not terrain ground truth or a renderable
/// dataset. Low-resolution references remain at their source resolution until a versioned
/// normalization/resampling step records how they entered a model lane.
/// </summary>
public sealed record RealTileObservation(
    string Id,
    string FilePath,
    string DisplayName,
    RealTileObservationKind Kind,
    string? MapHint,
    int? TileXHint,
    int? TileYHint,
    int? Width,
    int? Height,
    long FileSizeBytes,
    string? SourceSha256,
    IReadOnlyList<string> AvailableSignals,
    bool UsableAsModelInput,
    bool UsableAsTarget,
    string? Diagnostic);

/// <summary>
/// Bounded discovery for viewer dataset selection. The walker stops at recognized roots so it does
/// not descend through Zarr chunk directories or scan control/run artifacts as if they were maps.
/// </summary>
public static class DatasetVersionCatalog
{
    private const int MaxDiscoveryDepth = 4;

    public static IReadOnlyList<DatasetVersionCatalogEntry> Discover(string catalogRoot)
    {
        if (string.IsNullOrWhiteSpace(catalogRoot))
            return Array.Empty<DatasetVersionCatalogEntry>();

        string normalizedRoot;
        try
        {
            normalizedRoot = Path.GetFullPath(catalogRoot);
        }
        catch
        {
            return Array.Empty<DatasetVersionCatalogEntry>();
        }

        if (!Directory.Exists(normalizedRoot))
            return Array.Empty<DatasetVersionCatalogEntry>();

        var entries = new Dictionary<string, DatasetVersionCatalogEntry>(StringComparer.OrdinalIgnoreCase);
        Visit(normalizedRoot, depth: 0, entries);
        return entries.Values
            .OrderBy(entry => entry.DisplayName, StringComparer.OrdinalIgnoreCase)
            .ThenBy(entry => entry.RootPath, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    /// <summary>
    /// Finds explicitly named real-observation folders without classifying visual-review atlases,
    /// model runs, or arbitrary PNG output as source evidence.
    /// </summary>
    public static IReadOnlyList<RealTileObservation> DiscoverRealTileObservations(string observationRoot)
    {
        if (string.IsNullOrWhiteSpace(observationRoot))
            return Array.Empty<RealTileObservation>();

        string normalizedRoot;
        try
        {
            normalizedRoot = Path.GetFullPath(observationRoot);
        }
        catch
        {
            return Array.Empty<RealTileObservation>();
        }

        if (!Directory.Exists(normalizedRoot))
            return Array.Empty<RealTileObservation>();

        var observations = new List<RealTileObservation>();
        foreach (string path in EnumerateImageFiles(normalizedRoot))
        {
            FileInfo info;
            try
            {
                info = new FileInfo(path);
            }
            catch
            {
                continue;
            }

            observations.Add(new RealTileObservation(
                Id: Path.GetFullPath(path),
                FilePath: Path.GetFullPath(path),
                DisplayName: Path.GetFileName(path),
                Kind: InferObservationKind(normalizedRoot),
                MapHint: null,
                TileXHint: null,
                TileYHint: null,
                Width: null,
                Height: null,
                FileSizeBytes: info.Length,
                SourceSha256: TryComputeSha256(path),
                AvailableSignals: new[] { "real_rgb" },
                UsableAsModelInput: true,
                UsableAsTarget: false,
                Diagnostic: "Reference imagery requires provenance, albedo normalization, and a resolution-aware input contract."));
        }

        return observations
            .OrderBy(observation => observation.FilePath, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static void Visit(
        string directory,
        int depth,
        IDictionary<string, DatasetVersionCatalogEntry> entries)
    {
        if (TryDescribeVlmProject(directory, out DatasetVersionCatalogEntry? vlmEntry))
        {
            entries[vlmEntry!.Id] = vlmEntry;
            return;
        }

        if (TryDescribeZarrStore(directory, out DatasetVersionCatalogEntry? zarrEntry))
        {
            entries[zarrEntry!.Id] = zarrEntry;
            return;
        }

        if (TryDescribeRealTileSet(directory, out DatasetVersionCatalogEntry? realEntry))
        {
            entries[realEntry!.Id] = realEntry;
            return;
        }

        if (depth >= MaxDiscoveryDepth)
            return;

        IEnumerable<string> children;
        try
        {
            children = Directory.EnumerateDirectories(directory).ToArray();
        }
        catch
        {
            return;
        }

        foreach (string child in children.OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            Visit(child, depth + 1, entries);
    }

    private static bool TryDescribeRealTileSet(
        string directory,
        out DatasetVersionCatalogEntry? entry)
    {
        entry = null;
        string directoryName = Path.GetFileName(directory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (!IsExplicitRealObservationDirectory(directoryName)
            && !File.Exists(Path.Combine(directory, "real_tile_manifest.json")))
        {
            return false;
        }

        string[] imageFiles = EnumerateImageFiles(directory).ToArray();
        if (imageFiles.Length == 0)
            return false;

        string rootPath = Path.GetFullPath(directory);
        entry = new DatasetVersionCatalogEntry(
            Id: rootPath,
            DisplayName: $"{directoryName} (real observations)",
            RootPath: rootPath,
            SourceKind: DatasetSourceKind.RealTileObservation,
            MapName: null,
            TileCount: imageFiles.Length,
            Signals: new[] { "real_rgb", "reference_resolution", "provenance_required" },
            Renderable: false,
            Diagnostic: "Reference-only real imagery; preserve the source and normalize/resample it through a versioned observation pipeline before reconstruction.");
        return true;
    }

    private static bool TryDescribeVlmProject(
        string directory,
        out DatasetVersionCatalogEntry? entry)
    {
        entry = null;
        string datasetDirectory = Path.Combine(directory, "dataset");
        if (!Directory.Exists(datasetDirectory))
            return false;

        string[] tileFiles;
        try
        {
            tileFiles = Directory.EnumerateFiles(datasetDirectory, "*.json", SearchOption.TopDirectoryOnly)
                .Where(path => !string.Equals(Path.GetFileName(path), "texture_database.json", StringComparison.OrdinalIgnoreCase))
                .ToArray();
        }
        catch
        {
            return false;
        }

        if (tileFiles.Length == 0)
            return false;

        string? mapName = InferMapName(tileFiles);
        var signals = new List<string> { "terrain_json" };
        AddDirectorySignal(directory, signals, "images", "minimap_images");
        AddDirectorySignal(directory, signals, "shadows", "terrain_shadows");
        AddDirectorySignal(directory, signals, "masks", "alpha_or_object_masks");
        AddDirectorySignal(directory, signals, "semantic", "semantic_maps");
        AddDirectorySignal(directory, signals, "textures", "texture_assets");
        AddDirectorySignal(directory, signals, "tilesets", "tileset_assets");
        AddDirectorySignal(directory, signals, "liquids", "liquid_maps");

        string rootPath = Path.GetFullPath(directory);
        entry = new DatasetVersionCatalogEntry(
            Id: rootPath,
            DisplayName: BuildDisplayName(directory, mapName),
            RootPath: rootPath,
            SourceKind: DatasetSourceKind.VlmProject,
            MapName: mapName,
            TileCount: tileFiles.Length,
            Signals: signals,
            Renderable: true,
            Diagnostic: null);
        return true;
    }

    private static bool TryDescribeZarrStore(
        string directory,
        out DatasetVersionCatalogEntry? entry)
    {
        entry = null;
        string metadataPath = Path.Combine(directory, "zarr.json");
        if (!File.Exists(metadataPath))
            return false;

        var arrays = new List<string>();
        try
        {
            foreach (string child in Directory.EnumerateDirectories(directory))
            {
                if (File.Exists(Path.Combine(child, "zarr.json")))
                    arrays.Add(Path.GetFileName(child));
            }
        }
        catch
        {
            // Keep the root discoverable so the selector can show the diagnostic rather than
            // silently hiding a store with restricted or partially copied contents.
        }

        string rootPath = Path.GetFullPath(directory);
        string storeName = Path.GetFileName(rootPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        string displayName = storeName.EndsWith(".zarr", StringComparison.OrdinalIgnoreCase)
            ? storeName[..^5]
            : storeName;

        entry = new DatasetVersionCatalogEntry(
            Id: rootPath,
            DisplayName: $"{displayName} (Zarr summary)",
            RootPath: rootPath,
            SourceKind: DatasetSourceKind.ZarrStore,
            MapName: displayName,
            TileCount: 0,
            Signals: arrays.OrderBy(value => value, StringComparer.Ordinal).ToArray(),
            Renderable: false,
            Diagnostic: "Zarr tile decoding and TerrainTileTensorPack rehydration are not implemented in the viewer.");
        return true;
    }

    private static void AddDirectorySignal(
        string root,
        ICollection<string> signals,
        string directoryName,
        string signalName)
    {
        if (Directory.Exists(Path.Combine(root, directoryName)))
            signals.Add(signalName);
    }

    private static IEnumerable<string> EnumerateImageFiles(string root)
    {
        IEnumerable<string> files;
        try
        {
            files = Directory.EnumerateFiles(root, "*.*", SearchOption.AllDirectories)
                .Where(IsSupportedImagePath)
                .ToArray();
        }
        catch
        {
            yield break;
        }

        foreach (string file in files)
            yield return file;
    }

    private static string? TryComputeSha256(string path)
    {
        try
        {
            using FileStream stream = File.OpenRead(path);
            return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }
        catch
        {
            return null;
        }
    }

    private static bool IsSupportedImagePath(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".jpg", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".jpeg", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".png", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".webp", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".bmp", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsExplicitRealObservationDirectory(string directoryName)
    {
        string normalized = directoryName.Replace('_', '-').ToLowerInvariant();
        return normalized.Contains("real-tile", StringComparison.Ordinal)
            || normalized.Contains("real-observation", StringComparison.Ordinal)
            || normalized.Contains("reference-image", StringComparison.Ordinal)
            || normalized.Contains("media-reference", StringComparison.Ordinal)
            || normalized.Equals("leaked", StringComparison.Ordinal);
    }

    private static RealTileObservationKind InferObservationKind(string root)
    {
        string normalized = Path.GetFileName(root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
            .Replace('_', '-')
            .ToLowerInvariant();
        if (normalized.Contains("client", StringComparison.Ordinal)
            || normalized.Contains("harvest", StringComparison.Ordinal))
        {
            return RealTileObservationKind.ClientHarvest;
        }

        if (normalized.Contains("media", StringComparison.Ordinal)
            || normalized.Contains("reference", StringComparison.Ordinal)
            || normalized.Equals("leaked", StringComparison.Ordinal))
        {
            return RealTileObservationKind.MediaReference;
        }

        if (normalized.Contains("authored", StringComparison.Ordinal)
            || normalized.Contains("minimap", StringComparison.Ordinal))
        {
            return RealTileObservationKind.AuthoredMinimap;
        }

        return RealTileObservationKind.Unknown;
    }

    private static string? InferMapName(IEnumerable<string> tileFiles)
    {
        string? fileName = tileFiles
            .Select(Path.GetFileNameWithoutExtension)
            .FirstOrDefault(value => !string.IsNullOrWhiteSpace(value));
        if (string.IsNullOrWhiteSpace(fileName))
            return null;

        string[] parts = fileName.Split('_');
        if (parts.Length < 3
            || !int.TryParse(parts[^2], out _)
            || !int.TryParse(parts[^1], out _))
        {
            return null;
        }

        return string.Join('_', parts[..^2]);
    }

    private static string BuildDisplayName(string directory, string? mapName)
    {
        string versionName = Path.GetFileName(directory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        return string.IsNullOrWhiteSpace(mapName)
            ? versionName
            : $"{versionName} / {mapName}";
    }
}
