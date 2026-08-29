using System.Numerics;
using System.Text.Json;
using Parquet;
using Parquet.Data;
using Parquet.Schema;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Metadata for an ingested build inside the Rosetta Zarr datastore.
/// </summary>
public sealed record RosettaBuildMetadata(
    string BuildId,
    string ClientRoot,
    string MapName,
    string IngestTimestampUtc,
    int TotalPlacements,
    int TotalUniqueAssets,
    int NewAssetsAdded,
    int DeduplicatedAssetsReused,
    int TilesWritten);

public enum RosettaDiffClassification
{
    Added,
    Removed,
    FormatMigrated,
    GeometryModified,
    Identical
}

/// <summary>
/// Record of an individual asset's comparison between two builds.
/// </summary>
public sealed record RosettaAssetDiffRecord(
    string AssetPathOrStem,
    RosettaDiffClassification Classification,
    string? BaseAssetPath,
    string? TargetAssetPath,
    Vector3? BaseBoundsSize,
    Vector3? TargetBoundsSize,
    string? Details);

/// <summary>
/// Summary report comparing two client builds in the datastore.
/// </summary>
public sealed record RosettaBuildDiff(
    string BaseBuildId,
    string TargetBuildId,
    int TotalBaseAssets,
    int TotalTargetAssets,
    int AddedCount,
    int RemovedCount,
    int FormatMigratedCount,
    int GeometryModifiedCount,
    int IdenticalCount,
    IReadOnlyList<RosettaAssetDiffRecord> Records);

/// <summary>
/// High-performance C# query engine for the Unified Multi-Version Zarr Datastore.
/// Provides instant O(1) asset catalog queries, spatial bounding lookups, and on-demand tensor chunk loading.
/// </summary>
public sealed class RosettaObjectLibrary
{
    private readonly string _datastorePath;
    private readonly Dictionary<string, RosettaGlobalAssetRecord> _catalogByAssetId;
    private readonly Dictionary<string, RosettaGlobalAssetRecord> _catalogByNormalizedPath;
    private readonly List<string> _builds;

    public string DatastorePath => _datastorePath;
    public IReadOnlyList<string> Builds => _builds;
    public int TotalUniqueAssets => _catalogByAssetId.Count;

    private RosettaObjectLibrary(
        string datastorePath,
        Dictionary<string, RosettaGlobalAssetRecord> catalogByAssetId,
        Dictionary<string, RosettaGlobalAssetRecord> catalogByNormalizedPath,
        List<string> builds)
    {
        _datastorePath = datastorePath;
        _catalogByAssetId = catalogByAssetId;
        _catalogByNormalizedPath = catalogByNormalizedPath;
        _builds = builds;
    }

    /// <summary>
    /// Opens a Unified Multi-Version Zarr Datastore from the specified directory path.
    /// </summary>
    public static RosettaObjectLibrary Open(string datastorePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(datastorePath);
        if (!Directory.Exists(datastorePath))
            throw new DirectoryNotFoundException($"Datastore directory not found at: {datastorePath}");

        string globalAssetsDir = Path.Combine(datastorePath, "global_assets");
        string catalogParquet = Path.Combine(globalAssetsDir, "catalog.parquet");

        var byId = new Dictionary<string, RosettaGlobalAssetRecord>(StringComparer.OrdinalIgnoreCase);
        var byPath = new Dictionary<string, RosettaGlobalAssetRecord>(StringComparer.OrdinalIgnoreCase);

        if (File.Exists(catalogParquet))
        {
            using FileStream fs = File.OpenRead(catalogParquet);
            using ParquetReader reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();

            for (int rg = 0; rg < reader.RowGroupCount; rg++)
            {
                using ParquetRowGroupReader group = reader.OpenRowGroupReader(rg);
                DataField[] fields = reader.Schema.GetDataFields();

                string[] assetIds = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "asset_id")).GetAwaiter().GetResult().Data;
                string[] normPaths = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "normalized_path")).GetAwaiter().GetResult().Data;
                string[] origPaths = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "original_path")).GetAwaiter().GetResult().Data;
                string[] kinds = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "kind")).GetAwaiter().GetResult().Data;
                float[] minX = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "bounds_min_x")).GetAwaiter().GetResult().Data;
                float[] minY = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "bounds_min_y")).GetAwaiter().GetResult().Data;
                float[] minZ = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "bounds_min_z")).GetAwaiter().GetResult().Data;
                float[] maxX = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "bounds_max_x")).GetAwaiter().GetResult().Data;
                float[] maxY = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "bounds_max_y")).GetAwaiter().GetResult().Data;
                float[] maxZ = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "bounds_max_z")).GetAwaiter().GetResult().Data;
                string[] firstBuilds = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "first_seen_build")).GetAwaiter().GetResult().Data;
                int[] refCounts = (int[])group.ReadColumnAsync(fields.First(f => f.Name == "ref_count")).GetAwaiter().GetResult().Data;

                for (int i = 0; i < assetIds.Length; i++)
                {
                    var record = new RosettaGlobalAssetRecord(
                        AssetId: assetIds[i],
                        NormalizedPath: normPaths[i],
                        OriginalPath: origPaths[i],
                        Kind: kinds[i],
                        BoundsMinX: minX[i],
                        BoundsMinY: minY[i],
                        BoundsMinZ: minZ[i],
                        BoundsMaxX: maxX[i],
                        BoundsMaxY: maxY[i],
                        BoundsMaxZ: maxZ[i],
                        FirstSeenBuild: firstBuilds[i],
                        RefCount: refCounts[i]);

                    byId[record.AssetId] = record;
                    byPath[record.NormalizedPath] = record;
                }
            }
        }

        var builds = new List<string>();
        string versionsDir = Path.Combine(datastorePath, "versions");
        if (Directory.Exists(versionsDir))
        {
            builds.AddRange(Directory.EnumerateDirectories(versionsDir).Select(Path.GetFileName).Where(b => !string.IsNullOrEmpty(b))!);
        }

        return new RosettaObjectLibrary(datastorePath, byId, byPath, builds);
    }

    /// <summary>
    /// Lists all map names available under the given build ID.
    /// </summary>
    public IReadOnlyList<string> GetMaps(string buildId)
    {
        string mapsDir = Path.Combine(_datastorePath, "versions", buildId, "maps");
        if (!Directory.Exists(mapsDir))
            return Array.Empty<string>();

        return Directory.EnumerateDirectories(mapsDir).Select(Path.GetFileName).Where(m => !string.IsNullOrEmpty(m)).ToList()!;
    }

    /// <summary>
    /// Enumerates all deduplicated assets in the global catalog.
    /// </summary>
    public IReadOnlyCollection<RosettaGlobalAssetRecord> GetAllAssets() => _catalogByAssetId.Values;

    /// <summary>
    /// Looks up a globally deduplicated asset by asset ID (e.g. <c>objlib_xxx</c>) or by normalized asset path.
    /// </summary>
    public RosettaGlobalAssetRecord? LookupAsset(string assetPathOrId)
    {
        if (string.IsNullOrWhiteSpace(assetPathOrId))
            return null;

        if (_catalogByAssetId.TryGetValue(assetPathOrId, out RosettaGlobalAssetRecord? byId))
            return byId;

        string normalized = assetPathOrId.Replace('\\', '/').Trim().ToLowerInvariant();
        if (_catalogByNormalizedPath.TryGetValue(normalized, out RosettaGlobalAssetRecord? byPath))
            return byPath;

        string computedId = RosettaDatastoreWriter.ComputeAssetId(assetPathOrId);
        if (_catalogByAssetId.TryGetValue(computedId, out RosettaGlobalAssetRecord? byComputed))
            return byComputed;

        return null;
    }

    /// <summary>
    /// Reads all placement records for a specific build and map.
    /// </summary>
    public IReadOnlyList<RosettaPlacementRecord> GetPlacements(string buildId, string mapName)
    {
        string parquetPath = Path.Combine(_datastorePath, "versions", buildId, "maps", mapName, "placements.parquet");
        if (!File.Exists(parquetPath))
            return Array.Empty<RosettaPlacementRecord>();

        var results = new List<RosettaPlacementRecord>();
        using FileStream fs = File.OpenRead(parquetPath);
        using ParquetReader reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();

        for (int rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using ParquetRowGroupReader group = reader.OpenRowGroupReader(rg);
            DataField[] fields = reader.Schema.GetDataFields();

            string[] assetIds = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "asset_id")).GetAwaiter().GetResult().Data;
            int[] tileX = (int[])group.ReadColumnAsync(fields.First(f => f.Name == "tile_x")).GetAwaiter().GetResult().Data;
            int[] tileY = (int[])group.ReadColumnAsync(fields.First(f => f.Name == "tile_y")).GetAwaiter().GetResult().Data;
            float[] rawX = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "raw_x")).GetAwaiter().GetResult().Data;
            float[] rawY = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "raw_y")).GetAwaiter().GetResult().Data;
            float[] rawZ = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "raw_z")).GetAwaiter().GetResult().Data;
            float[] renderX = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "render_x")).GetAwaiter().GetResult().Data;
            float[] renderY = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "render_y")).GetAwaiter().GetResult().Data;
            float[] renderZ = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "render_z")).GetAwaiter().GetResult().Data;
            float[] cellU = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "cell_u")).GetAwaiter().GetResult().Data;
            float[] cellV = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "cell_v")).GetAwaiter().GetResult().Data;
            float[] cellSize = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "cell_size")).GetAwaiter().GetResult().Data;
            float[] objBand = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "object_band_size")).GetAwaiter().GetResult().Data;
            string[] labelText = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "label_text")).GetAwaiter().GetResult().Data;
            string[] labelLinesJson = (string[])group.ReadColumnAsync(fields.First(f => f.Name == "label_lines_json")).GetAwaiter().GetResult().Data;
            float[] labelPixel = (float[])group.ReadColumnAsync(fields.First(f => f.Name == "label_pixel_meters")).GetAwaiter().GetResult().Data;
            int[] uniqueIds = (int[])group.ReadColumnAsync(fields.First(f => f.Name == "unique_id")).GetAwaiter().GetResult().Data;

            for (int i = 0; i < assetIds.Length; i++)
            {
                RosettaGlobalAssetRecord? assetRec = LookupAsset(assetIds[i]);
                RosettaAssetKind kind = assetRec?.Kind.Equals("world_model", StringComparison.OrdinalIgnoreCase) == true
                    ? RosettaAssetKind.WorldModel
                    : RosettaAssetKind.Model;

                Vector3 bMin = assetRec != null ? new Vector3(assetRec.BoundsMinX, assetRec.BoundsMinY, assetRec.BoundsMinZ) : Vector3.Zero;
                Vector3 bMax = assetRec != null ? new Vector3(assetRec.BoundsMaxX, assetRec.BoundsMaxY, assetRec.BoundsMaxZ) : Vector3.One;
                string assetPath = assetRec?.OriginalPath ?? assetIds[i];

                var assetEntry = new RosettaAssetEntry(assetPath, kind, bMin, bMax);
                var labelLines = JsonSerializer.Deserialize<List<string>>(labelLinesJson[i]) ?? [];

                results.Add(new RosettaPlacementRecord(
                    Asset: assetEntry,
                    TileX: tileX[i],
                    TileY: tileY[i],
                    RawPosition: new Vector3(rawX[i], rawY[i], rawZ[i]),
                    RendererPosition: new Vector3(renderX[i], renderY[i], renderZ[i]),
                    CellU: cellU[i],
                    CellV: cellV[i],
                    CellSize: cellSize[i],
                    ObjectBandSize: objBand[i],
                    LabelText: labelText[i],
                    LabelLines: labelLines,
                    LabelPixelMeters: labelPixel[i],
                    UniqueId: uniqueIds[i]));
            }
        }

        return results;
    }

    /// <summary>
    /// Performs a spatial bounding box query on placements within a map.
    /// </summary>
    public IReadOnlyList<RosettaPlacementRecord> QuerySpatial(string buildId, string mapName, Vector3 min, Vector3 max)
    {
        IReadOnlyList<RosettaPlacementRecord> all = GetPlacements(buildId, mapName);
        return all.Where(p =>
            p.RendererPosition.X >= min.X && p.RendererPosition.X <= max.X &&
            p.RendererPosition.Y >= min.Y && p.RendererPosition.Y <= max.Y &&
            p.RendererPosition.Z >= min.Z && p.RendererPosition.Z <= max.Z).ToList();
    }

    /// <summary>
    /// Retrieves a tile's float32 heights array chunk (145 vertices) from the Zarr array.
    /// </summary>
    public float[]? GetTileHeights(string buildId, string mapName, int tileIndex)
    {
        string chunkFile = Path.Combine(_datastorePath, "versions", buildId, "maps", mapName, "heights", "c", tileIndex.ToString(), "0");
        if (!File.Exists(chunkFile))
            return null;

        byte[] bytes = File.ReadAllBytes(chunkFile);
        float[] floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    /// <summary>
    /// Retrieves a tile's 1024x1024 MCAL Layer 2 alpha canvas from the Zarr array.
    /// </summary>
    public byte[]? GetTileAlpha(string buildId, string mapName, int tileIndex)
    {
        string chunkFile = Path.Combine(_datastorePath, "versions", buildId, "maps", mapName, "mcal_alpha", "c", tileIndex.ToString(), "0", "0");
        return File.Exists(chunkFile) ? File.ReadAllBytes(chunkFile) : null;
    }

    /// <summary>
    /// Retrieves a tile's 1024x1024 checkers alpha canvas from the Zarr array.
    /// </summary>
    public byte[]? GetTileCheckers(string buildId, string mapName, int tileIndex)
    {
        string chunkFile = Path.Combine(_datastorePath, "versions", buildId, "maps", mapName, "checkers_alpha", "c", tileIndex.ToString(), "0", "0");
        return File.Exists(chunkFile) ? File.ReadAllBytes(chunkFile) : null;
    }

    /// <summary>
    /// Retrieves a tile's 256x256x3 RGB minimap image array from the Zarr array.
    /// </summary>
    public byte[]? GetTileMinimap(string buildId, string mapName, int tileIndex)
    {
        string chunkFile = Path.Combine(_datastorePath, "versions", buildId, "maps", mapName, "minimap_rgb_256", "c", tileIndex.ToString(), "0", "0", "0");
        return File.Exists(chunkFile) ? File.ReadAllBytes(chunkFile) : null;
    }

    /// <summary>
    /// Gets metadata for a specific build.
    /// </summary>
    public RosettaBuildMetadata? GetBuildMetadata(string buildId)
    {
        string metaFile = Path.Combine(_datastorePath, "versions", buildId, "build_metadata.json");
        if (File.Exists(metaFile))
        {
            try
            {
                string json = File.ReadAllText(metaFile);
                return JsonSerializer.Deserialize<RosettaBuildMetadata>(json);
            }
            catch
            {
                // Fallback to basic record if JSON fails
            }
        }

        var maps = GetMaps(buildId);
        string mapName = maps.Count > 0 ? maps[0] : "";
        var placements = mapName.Length > 0 ? GetPlacements(buildId, mapName) : Array.Empty<RosettaPlacementRecord>();
        return new RosettaBuildMetadata(
            BuildId: buildId,
            ClientRoot: "",
            MapName: mapName,
            IngestTimestampUtc: "",
            TotalPlacements: placements.Count,
            TotalUniqueAssets: placements.Select(p => p.Asset.AssetPath).Distinct(StringComparer.OrdinalIgnoreCase).Count(),
            NewAssetsAdded: 0,
            DeduplicatedAssetsReused: 0,
            TilesWritten: 0);
    }

    /// <summary>
    /// Computes a structural and cross-era asset diff between two builds in the datastore.
    /// </summary>
    public RosettaBuildDiff ComputeBuildDiff(string baseBuildId, string targetBuildId)
    {
        var baseMaps = GetMaps(baseBuildId);
        var targetMaps = GetMaps(targetBuildId);

        var basePlacements = baseMaps.Count > 0 ? GetPlacements(baseBuildId, baseMaps[0]) : Array.Empty<RosettaPlacementRecord>();
        var targetPlacements = targetMaps.Count > 0 ? GetPlacements(targetBuildId, targetMaps[0]) : Array.Empty<RosettaPlacementRecord>();

        var baseAssets = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase);
        foreach (var p in basePlacements)
            baseAssets[p.Asset.AssetPath] = p.Asset;

        var targetAssets = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase);
        foreach (var p in targetPlacements)
            targetAssets[p.Asset.AssetPath] = p.Asset;

        var records = new List<RosettaAssetDiffRecord>();
        int added = 0, removed = 0, migrated = 0, modified = 0, identical = 0;

        // Group by stem (path without extension) to detect cross-era format migrations (.mdx <-> .m2)
        static string GetStem(string path) => Path.ChangeExtension(path.Replace('\\', '/').Trim().ToLowerInvariant(), null);

        var baseByStem = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase);
        foreach (var (path, asset) in baseAssets)
            baseByStem[GetStem(path)] = asset;

        var targetByStem = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase);
        foreach (var (path, asset) in targetAssets)
            targetByStem[GetStem(path)] = asset;

        var allStems = new HashSet<string>(baseByStem.Keys, StringComparer.OrdinalIgnoreCase);
        allStems.UnionWith(targetByStem.Keys);

        foreach (string stem in allStems)
        {
            bool inBase = baseByStem.TryGetValue(stem, out var baseEntry);
            bool inTarget = targetByStem.TryGetValue(stem, out var targetEntry);

            if (inBase && !inTarget)
            {
                removed++;
                Vector3 bSize = (baseEntry!.BoundsMax - baseEntry.BoundsMin);
                records.Add(new RosettaAssetDiffRecord(
                    AssetPathOrStem: baseEntry.AssetPath,
                    Classification: RosettaDiffClassification.Removed,
                    BaseAssetPath: baseEntry.AssetPath,
                    TargetAssetPath: null,
                    BaseBoundsSize: bSize,
                    TargetBoundsSize: null,
                    Details: "Removed in target build"));
            }
            else if (!inBase && inTarget)
            {
                added++;
                Vector3 tSize = (targetEntry!.BoundsMax - targetEntry.BoundsMin);
                records.Add(new RosettaAssetDiffRecord(
                    AssetPathOrStem: targetEntry.AssetPath,
                    Classification: RosettaDiffClassification.Added,
                    BaseAssetPath: null,
                    TargetAssetPath: targetEntry.AssetPath,
                    BaseBoundsSize: null,
                    TargetBoundsSize: tSize,
                    Details: "New in target build"));
            }
            else
            {
                Vector3 bSize = (baseEntry!.BoundsMax - baseEntry.BoundsMin);
                Vector3 tSize = (targetEntry!.BoundsMax - targetEntry.BoundsMin);
                string bExt = Path.GetExtension(baseEntry.AssetPath);
                string tExt = Path.GetExtension(targetEntry.AssetPath);

                if (!bExt.Equals(tExt, StringComparison.OrdinalIgnoreCase))
                {
                    migrated++;
                    records.Add(new RosettaAssetDiffRecord(
                        AssetPathOrStem: stem,
                        Classification: RosettaDiffClassification.FormatMigrated,
                        BaseAssetPath: baseEntry.AssetPath,
                        TargetAssetPath: targetEntry.AssetPath,
                        BaseBoundsSize: bSize,
                        TargetBoundsSize: tSize,
                        Details: $"Format shifted from {bExt} to {tExt}"));
                }
                else if (Vector3.Distance(bSize, tSize) > 0.05f)
                {
                    modified++;
                    records.Add(new RosettaAssetDiffRecord(
                        AssetPathOrStem: baseEntry.AssetPath,
                        Classification: RosettaDiffClassification.GeometryModified,
                        BaseAssetPath: baseEntry.AssetPath,
                        TargetAssetPath: targetEntry.AssetPath,
                        BaseBoundsSize: bSize,
                        TargetBoundsSize: tSize,
                        Details: $"Bounds changed: {bSize:F2} -> {tSize:F2}"));
                }
                else
                {
                    identical++;
                    records.Add(new RosettaAssetDiffRecord(
                        AssetPathOrStem: baseEntry.AssetPath,
                        Classification: RosettaDiffClassification.Identical,
                        BaseAssetPath: baseEntry.AssetPath,
                        TargetAssetPath: targetEntry.AssetPath,
                        BaseBoundsSize: bSize,
                        TargetBoundsSize: tSize,
                        Details: "Identical path and bounds"));
                }
            }
        }

        return new RosettaBuildDiff(
            BaseBuildId: baseBuildId,
            TargetBuildId: targetBuildId,
            TotalBaseAssets: baseAssets.Count,
            TotalTargetAssets: targetAssets.Count,
            AddedCount: added,
            RemovedCount: removed,
            FormatMigratedCount: migrated,
            GeometryModifiedCount: modified,
            IdenticalCount: identical,
            Records: records);
    }
}
