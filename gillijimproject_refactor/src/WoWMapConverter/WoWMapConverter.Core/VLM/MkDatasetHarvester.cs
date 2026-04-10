using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;

namespace WoWMapConverter.Core.VLM;

public sealed record MkDatasetHarvestOptions(
    string DatasetRoot,
    string? ManifestOutputPath = null,
    bool GenerateReferenceMinimaps = false,
    bool ForceRegenerateReferenceMinimaps = false,
    bool ApplyShadows = true,
    float ShadowIntensity = 0.5f,
    bool InvertAlpha = true,
    string? ReferenceMinimapDirectory = null);

public sealed record MkDatasetHarvestResult(
    string ManifestPath,
    int TilesProcessed,
    int SourceMinimapsFound,
    int LocalHeightmapsFound,
    int GlobalHeightmapsFound,
    int TilesWithAlphaMasks,
    int ReferenceMinimapsGenerated,
    string ReferenceMinimapDirectory);

public sealed class MkDatasetManifest
{
    public string SchemaVersion { get; set; } = "mk-dataset-manifest.v1";
    public DateTime HarvestedAtUtc { get; set; }
    public string DatasetRoot { get; set; } = string.Empty;
    public string DatasetName { get; set; } = string.Empty;
    public string SourceFormat { get; set; } = "legacy-vlm-json";
    public string TileJsonDirectory { get; set; } = "dataset";
    public string ReferenceMinimapDirectory { get; set; } = string.Empty;
    public MkDatasetCoverageSummary Coverage { get; set; } = new();
    public List<MkDatasetTileManifest> Tiles { get; set; } = new();
}

public sealed class MkDatasetCoverageSummary
{
    public int TilesProcessed { get; set; }
    public int TilesWithSourceMinimap { get; set; }
    public int TilesWithLocalHeightmap { get; set; }
    public int TilesWithGlobalHeightmap { get; set; }
    public int TilesWithAnyAlphaMask { get; set; }
    public int DeclaredAlphaMaskImages { get; set; }
    public int ExistingAlphaMaskImages { get; set; }
    public int TilesWithObjects { get; set; }
    public int TilesWithChunkLayerMetadata { get; set; }
    public int TilesWithReferenceMinimap { get; set; }
    public int ReferenceMinimapsGenerated { get; set; }
}

public sealed class MkDatasetTileManifest
{
    public string TileName { get; set; } = string.Empty;
    public string MapName { get; set; } = string.Empty;
    public string TileJsonPath { get; set; } = string.Empty;
    public string? SourceMinimapPath { get; set; }
    public bool SourceMinimapExists { get; set; }
    public string? HeightmapLocalPath { get; set; }
    public bool HeightmapLocalExists { get; set; }
    public string? HeightmapGlobalPath { get; set; }
    public bool HeightmapGlobalExists { get; set; }
    public string? NormalMapPath { get; set; }
    public bool NormalMapExists { get; set; }
    public string? MccvMapPath { get; set; }
    public bool MccvMapExists { get; set; }
    public int DeclaredAlphaMaskCount { get; set; }
    public int ExistingAlphaMaskCount { get; set; }
    public List<string> AlphaMaskPaths { get; set; } = new();
    public int ObjectCount { get; set; }
    public int ChunkLayerCount { get; set; }
    public string CompletenessClass { get; set; } = "partial";
    public string? ReferenceMinimapPath { get; set; }
    public bool ReferenceMinimapExists { get; set; }
    public bool ReferenceMinimapGenerated { get; set; }
}

public sealed class MkDatasetHarvester
{
    private readonly JsonSerializerOptions _datasetJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    private readonly JsonSerializerOptions _manifestJsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public async Task<MkDatasetHarvestResult> HarvestAsync(MkDatasetHarvestOptions options, IProgress<string>? progress = null)
    {
        string datasetRoot = Path.GetFullPath(options.DatasetRoot);
        string datasetDirectory = Path.Combine(datasetRoot, "dataset");
        if (!Directory.Exists(datasetDirectory))
            throw new DirectoryNotFoundException($"ML dataset directory not found: {datasetDirectory}");

        string[] datasetFiles = Directory.GetFiles(datasetDirectory, "*.json")
            .Where(path => !string.Equals(Path.GetFileName(path), "texture_database.json", StringComparison.OrdinalIgnoreCase))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (datasetFiles.Length == 0)
            throw new FileNotFoundException($"No tile JSON files found in {datasetDirectory}");

        string referenceDirectory = Path.GetFullPath(options.ReferenceMinimapDirectory ?? Path.Combine(datasetRoot, "reference_minimaps"));
        if (options.GenerateReferenceMinimaps)
            Directory.CreateDirectory(referenceDirectory);

        var baker = options.GenerateReferenceMinimaps
            ? new MinimapBakeService(datasetRoot)
            {
                ShadowIntensity = Math.Clamp(options.ShadowIntensity, 0f, 1f),
                InvertAlpha = options.InvertAlpha
            }
            : null;

        var manifest = new MkDatasetManifest
        {
            HarvestedAtUtc = DateTime.UtcNow,
            DatasetRoot = datasetRoot,
            TileJsonDirectory = RelativizePath(datasetRoot, datasetDirectory),
            ReferenceMinimapDirectory = RelativizePath(datasetRoot, referenceDirectory)
        };

        foreach (string datasetFile in datasetFiles)
        {
            progress?.Report($"Harvesting ML dataset tile {Path.GetFileName(datasetFile)}...");

            string json = await File.ReadAllTextAsync(datasetFile).ConfigureAwait(false);
            VlmTrainingSample? sample = JsonSerializer.Deserialize<VlmTrainingSample>(json, _datasetJsonOptions);
            string tileName = sample?.TerrainData?.AdtTile ?? Path.GetFileNameWithoutExtension(datasetFile);
            string mapName = ExtractMapName(tileName);
            if (string.IsNullOrWhiteSpace(manifest.DatasetName))
                manifest.DatasetName = mapName;

            string? sourceMinimapPath = sample?.ImagePath;
            bool sourceMinimapExists = TryResolveDatasetPath(datasetRoot, sourceMinimapPath);

            string? heightmapLocalPath = sample?.TerrainData?.HeightmapLocalPath ?? sample?.TerrainData?.HeightmapPath;
            bool heightmapLocalExists = TryResolveDatasetPath(datasetRoot, heightmapLocalPath);

            string? heightmapGlobalPath = sample?.TerrainData?.HeightmapGlobalPath;
            bool heightmapGlobalExists = TryResolveDatasetPath(datasetRoot, heightmapGlobalPath);

            string? normalMapPath = sample?.TerrainData?.NormalmapPath;
            bool normalMapExists = TryResolveDatasetPath(datasetRoot, normalMapPath);

            string? mccvMapPath = sample?.TerrainData?.MccvMapPath;
            bool mccvMapExists = TryResolveDatasetPath(datasetRoot, mccvMapPath);

            List<string> alphaMaskPaths = sample?.TerrainData?.AlphaMasks?
                .Where(path => !string.IsNullOrWhiteSpace(path))
                .Select(path => path.Replace('\\', '/'))
                .ToList()
                ?? new List<string>();
            int existingAlphaMaskCount = alphaMaskPaths.Count(path => TryResolveDatasetPath(datasetRoot, path));

            string referenceMinimapPath = Path.Combine(referenceDirectory, $"{tileName}_reference_minimap.png");
            bool referenceMinimapExists = File.Exists(referenceMinimapPath);
            bool referenceMinimapGenerated = false;
            if (options.GenerateReferenceMinimaps && (options.ForceRegenerateReferenceMinimaps || !referenceMinimapExists))
            {
                if (baker == null)
                    throw new InvalidOperationException("ML dataset reference minimap baker was not initialized.");

                using var bakedImage = options.ApplyShadows
                    ? await baker.BakeTileWithShadowsAsync(datasetFile, applyShadows: true).ConfigureAwait(false)
                    : await baker.BakeTileAsync(datasetFile).ConfigureAwait(false);
                await bakedImage.SaveAsPngAsync(referenceMinimapPath).ConfigureAwait(false);
                referenceMinimapExists = true;
                referenceMinimapGenerated = true;
            }

            manifest.Tiles.Add(new MkDatasetTileManifest
            {
                TileName = tileName,
                MapName = mapName,
                TileJsonPath = RelativizePath(datasetRoot, datasetFile),
                SourceMinimapPath = NormalizeDatasetRelativePath(sourceMinimapPath),
                SourceMinimapExists = sourceMinimapExists,
                HeightmapLocalPath = NormalizeDatasetRelativePath(heightmapLocalPath),
                HeightmapLocalExists = heightmapLocalExists,
                HeightmapGlobalPath = NormalizeDatasetRelativePath(heightmapGlobalPath),
                HeightmapGlobalExists = heightmapGlobalExists,
                NormalMapPath = NormalizeDatasetRelativePath(normalMapPath),
                NormalMapExists = normalMapExists,
                MccvMapPath = NormalizeDatasetRelativePath(mccvMapPath),
                MccvMapExists = mccvMapExists,
                DeclaredAlphaMaskCount = alphaMaskPaths.Count,
                ExistingAlphaMaskCount = existingAlphaMaskCount,
                AlphaMaskPaths = alphaMaskPaths,
                ObjectCount = sample?.TerrainData?.Objects?.Count ?? 0,
                ChunkLayerCount = sample?.TerrainData?.ChunkLayers?.Length ?? 0,
                CompletenessClass = BuildCompletenessClass(
                    sourceMinimapExists,
                    heightmapLocalExists,
                    heightmapGlobalExists,
                    existingAlphaMaskCount,
                    sample?.TerrainData?.ChunkLayers?.Length ?? 0),
                ReferenceMinimapPath = RelativizePath(datasetRoot, referenceMinimapPath),
                ReferenceMinimapExists = referenceMinimapExists,
                ReferenceMinimapGenerated = referenceMinimapGenerated
            });
        }

        manifest.Coverage = new MkDatasetCoverageSummary
        {
            TilesProcessed = manifest.Tiles.Count,
            TilesWithSourceMinimap = manifest.Tiles.Count(tile => tile.SourceMinimapExists),
            TilesWithLocalHeightmap = manifest.Tiles.Count(tile => tile.HeightmapLocalExists),
            TilesWithGlobalHeightmap = manifest.Tiles.Count(tile => tile.HeightmapGlobalExists),
            TilesWithAnyAlphaMask = manifest.Tiles.Count(tile => tile.ExistingAlphaMaskCount > 0),
            DeclaredAlphaMaskImages = manifest.Tiles.Sum(tile => tile.DeclaredAlphaMaskCount),
            ExistingAlphaMaskImages = manifest.Tiles.Sum(tile => tile.ExistingAlphaMaskCount),
            TilesWithObjects = manifest.Tiles.Count(tile => tile.ObjectCount > 0),
            TilesWithChunkLayerMetadata = manifest.Tiles.Count(tile => tile.ChunkLayerCount > 0),
            TilesWithReferenceMinimap = manifest.Tiles.Count(tile => tile.ReferenceMinimapExists),
            ReferenceMinimapsGenerated = manifest.Tiles.Count(tile => tile.ReferenceMinimapGenerated)
        };

        string manifestPath = Path.GetFullPath(options.ManifestOutputPath ?? Path.Combine(datasetRoot, "ml_dataset_manifest.json"));
        Directory.CreateDirectory(Path.GetDirectoryName(manifestPath) ?? datasetRoot);
        await File.WriteAllTextAsync(manifestPath, JsonSerializer.Serialize(manifest, _manifestJsonOptions)).ConfigureAwait(false);

        return new MkDatasetHarvestResult(
            ManifestPath: manifestPath,
            TilesProcessed: manifest.Coverage.TilesProcessed,
            SourceMinimapsFound: manifest.Coverage.TilesWithSourceMinimap,
            LocalHeightmapsFound: manifest.Coverage.TilesWithLocalHeightmap,
            GlobalHeightmapsFound: manifest.Coverage.TilesWithGlobalHeightmap,
            TilesWithAlphaMasks: manifest.Coverage.TilesWithAnyAlphaMask,
            ReferenceMinimapsGenerated: manifest.Coverage.ReferenceMinimapsGenerated,
            ReferenceMinimapDirectory: referenceDirectory);
    }

    private static string BuildCompletenessClass(bool sourceMinimapExists, bool heightmapLocalExists, bool heightmapGlobalExists, int existingAlphaMaskCount, int chunkLayerCount)
    {
        if (sourceMinimapExists && heightmapLocalExists && heightmapGlobalExists && existingAlphaMaskCount > 0 && chunkLayerCount > 0)
            return "core-terrain-ready";
        if (heightmapLocalExists && existingAlphaMaskCount > 0)
            return "terrain-ready-partial";
        if (heightmapLocalExists || sourceMinimapExists)
            return "partial";

        return "metadata-only";
    }

    private static string ExtractMapName(string tileName)
    {
        int lastSeparator = tileName.LastIndexOf('_');
        if (lastSeparator <= 0)
            return tileName;

        int secondLastSeparator = tileName.LastIndexOf('_', lastSeparator - 1);
        if (secondLastSeparator <= 0)
            return tileName;

        return tileName[..secondLastSeparator];
    }

    private static bool TryResolveDatasetPath(string datasetRoot, string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;

        string normalizedPath = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        string candidate = Path.IsPathRooted(normalizedPath)
            ? normalizedPath
            : Path.Combine(datasetRoot, normalizedPath);
        return File.Exists(candidate);
    }

    private static string? NormalizeDatasetRelativePath(string? path)
        => string.IsNullOrWhiteSpace(path)
            ? null
            : path.Replace('\\', '/');

    private static string RelativizePath(string root, string path)
    {
        try
        {
            string relative = Path.GetRelativePath(root, path).Replace('\\', '/');
            return relative.StartsWith("../", StringComparison.Ordinal) ? Path.GetFullPath(path) : relative;
        }
        catch
        {
            return Path.GetFullPath(path);
        }
    }
}