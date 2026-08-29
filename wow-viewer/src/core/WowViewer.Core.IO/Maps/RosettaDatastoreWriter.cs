using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Parquet;
using Parquet.Data;
using Parquet.Schema;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Result report from ingesting a map into the unified multi-version Zarr datastore.
/// </summary>
public sealed record RosettaDatastoreIngestResult(
    string DatastorePath,
    string BuildId,
    string MapName,
    int TotalPlacements,
    int TotalUniqueAssetsInMap,
    int NewAssetsAdded,
    int DeduplicatedAssetsReused,
    int TilesWritten);

/// <summary>
/// Record representing an asset stored in the global deduplicated catalog.
/// </summary>
public sealed record RosettaGlobalAssetRecord(
    string AssetId,
    string NormalizedPath,
    string OriginalPath,
    string Kind,
    float BoundsMinX,
    float BoundsMinY,
    float BoundsMinZ,
    float BoundsMaxX,
    float BoundsMaxY,
    float BoundsMaxZ,
    string FirstSeenBuild,
    int RefCount);

/// <summary>
/// Writes and appends Rosetta calibration outputs into a standardized, versioned Zarr v3 datastore
/// with cross-version asset deduplication, Parquet placement tables, and chunked tensor arrays.
/// </summary>
public static class RosettaDatastoreWriter
{
    private static readonly ParquetSchema AssetCatalogSchema = new(
        new DataField<string>("asset_id"),
        new DataField<string>("normalized_path"),
        new DataField<string>("original_path"),
        new DataField<string>("kind"),
        new DataField<float>("bounds_min_x"),
        new DataField<float>("bounds_min_y"),
        new DataField<float>("bounds_min_z"),
        new DataField<float>("bounds_max_x"),
        new DataField<float>("bounds_max_y"),
        new DataField<float>("bounds_max_z"),
        new DataField<string>("first_seen_build"),
        new DataField<int>("ref_count"));

    private static readonly ParquetSchema PlacementsSchema = new(
        new DataField<string>("asset_id"),
        new DataField<int>("tile_x"),
        new DataField<int>("tile_y"),
        new DataField<float>("raw_x"),
        new DataField<float>("raw_y"),
        new DataField<float>("raw_z"),
        new DataField<float>("render_x"),
        new DataField<float>("render_y"),
        new DataField<float>("render_z"),
        new DataField<float>("cell_u"),
        new DataField<float>("cell_v"),
        new DataField<float>("cell_size"),
        new DataField<float>("object_band_size"),
        new DataField<string>("label_text"),
        new DataField<string>("label_lines_json"),
        new DataField<float>("label_pixel_meters"),
        new DataField<int>("unique_id"));

    /// <summary>
    /// Computes a deterministic asset identifier from its normalized path.
    /// </summary>
    public static string ComputeAssetId(string assetPath)
    {
        string normalized = (assetPath ?? string.Empty).Replace('\\', '/').Trim().ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(normalized))
            return string.Empty;

        Span<byte> hash = stackalloc byte[20];
        SHA1.HashData(Encoding.UTF8.GetBytes(normalized), hash);
        string hex = Convert.ToHexString(hash).ToLowerInvariant();
        return $"objlib_{hex[..14]}";
    }

    /// <summary>
    /// Ingests a planned Rosetta map into the unified multi-version Zarr datastore.
    /// Streams each tile's tensor data on demand to keep RAM usage minimal and prevent OOM.
    /// </summary>
    public static RosettaDatastoreIngestResult IngestMap(
        string datastorePath,
        string buildId,
        string clientRoot,
        RosettaMapPlan mapPlan,
        Func<int, int, (float[] Heights, byte[] AlphaCanvas, byte[] CheckersCanvas, byte[] MinimapRgb)> tileSignalProvider,
        bool overwriteMap = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(datastorePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(buildId);
        ArgumentNullException.ThrowIfNull(mapPlan);
        ArgumentNullException.ThrowIfNull(tileSignalProvider);

        Directory.CreateDirectory(datastorePath);

        // 1. Root group: zarr.json
        WriteRootGroupJson(datastorePath);

        // 2. Global deduplicated asset catalog
        string globalAssetsDir = Path.Combine(datastorePath, "global_assets");
        Directory.CreateDirectory(globalAssetsDir);
        WriteGroupJson(globalAssetsDir, new Dictionary<string, object>
        {
            ["description"] = "Global content-addressed deduplicated asset library across all builds",
        });

        Dictionary<string, RosettaGlobalAssetRecord> catalog = LoadGlobalCatalog(globalAssetsDir);
        int newAssetsAdded = 0;
        int deduplicatedReused = 0;

        // Group placements by asset to update references and catalog
        var uniqueAssetsInMap = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (RosettaPlacementRecord placement in mapPlan.Placements)
        {
            string assetId = ComputeAssetId(placement.Asset.AssetPath);
            uniqueAssetsInMap.Add(assetId);

            if (catalog.TryGetValue(assetId, out RosettaGlobalAssetRecord? existing))
            {
                catalog[assetId] = existing with { RefCount = existing.RefCount + 1 };
                deduplicatedReused++;
            }
            else
            {
                var record = new RosettaGlobalAssetRecord(
                    AssetId: assetId,
                    NormalizedPath: placement.Asset.AssetPath.Replace('\\', '/').Trim().ToLowerInvariant(),
                    OriginalPath: placement.Asset.AssetPath,
                    Kind: placement.Asset.Kind.ToString().ToLowerInvariant(),
                    BoundsMinX: placement.Asset.BoundsMin.X,
                    BoundsMinY: placement.Asset.BoundsMin.Y,
                    BoundsMinZ: placement.Asset.BoundsMin.Z,
                    BoundsMaxX: placement.Asset.BoundsMax.X,
                    BoundsMaxY: placement.Asset.BoundsMax.Y,
                    BoundsMaxZ: placement.Asset.BoundsMax.Z,
                    FirstSeenBuild: buildId,
                    RefCount: 1);
                catalog[assetId] = record;
                newAssetsAdded++;
            }
        }

        SaveGlobalCatalog(globalAssetsDir, catalog.Values.ToList());

        // 3. Version & Map hierarchy: versions/<buildId>/maps/<mapName>/
        string versionDir = Path.Combine(datastorePath, "versions", buildId);
        Directory.CreateDirectory(versionDir);
        WriteGroupJson(versionDir, new Dictionary<string, object>
        {
            ["build_id"] = buildId,
            ["client_root"] = clientRoot ?? string.Empty,
            ["updated_utc"] = DateTime.UtcNow.ToString("o"),
        });

        string mapDir = Path.Combine(versionDir, "maps", mapPlan.MapName);
        if (Directory.Exists(mapDir))
        {
            if (!overwriteMap)
                throw new InvalidOperationException($"Map '{mapPlan.MapName}' already exists under build '{buildId}'.");
            Directory.Delete(mapDir, recursive: true);
        }
        Directory.CreateDirectory(mapDir);

        WriteGroupJson(mapDir, new Dictionary<string, object>
        {
            ["map_name"] = mapPlan.MapName,
            ["tile_count"] = mapPlan.Tiles.Count,
            ["placement_count"] = mapPlan.Placements.Count,
            ["designkit_count"] = mapPlan.Designkits.Count,
            ["block_origin_x"] = mapPlan.BlockOriginX,
            ["block_origin_y"] = mapPlan.BlockOriginY,
            ["block_side"] = mapPlan.BlockSide,
            ["ground_texture"] = mapPlan.GroundTexture,
            ["ink_texture"] = mapPlan.InkTexture,
            ["checkers_texture"] = mapPlan.CheckersTexture,
        });

        // 4. Write placements.parquet
        WritePlacementsParquet(Path.Combine(mapDir, "placements.parquet"), mapPlan.Placements);

        // 4b. Write build_metadata.json
        var buildMeta = new RosettaBuildMetadata(
            BuildId: buildId,
            ClientRoot: clientRoot ?? string.Empty,
            MapName: mapPlan.MapName,
            IngestTimestampUtc: DateTime.UtcNow.ToString("o"),
            TotalPlacements: mapPlan.Placements.Count,
            TotalUniqueAssets: uniqueAssetsInMap.Count,
            NewAssetsAdded: newAssetsAdded,
            DeduplicatedAssetsReused: deduplicatedReused,
            TilesWritten: mapPlan.Tiles.Count);
        string metaJson = JsonSerializer.Serialize(buildMeta, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(versionDir, "build_metadata.json"), metaJson);

        // 5. Initialize Zarr v3 tensor arrays
        int tileCount = mapPlan.Tiles.Count;
        string heightsDir = Path.Combine(mapDir, "heights");
        string mcalDir = Path.Combine(mapDir, "mcal_alpha");
        string checkersDir = Path.Combine(mapDir, "checkers_alpha");
        string minimapDir = Path.Combine(mapDir, "minimap_rgb_256");

        WriteArrayJson(heightsDir, [tileCount, 145], [1, 145], "float32");
        WriteArrayJson(mcalDir, [tileCount, 1024, 1024], [1, 1024, 1024], "uint8");
        WriteArrayJson(checkersDir, [tileCount, 1024, 1024], [1, 1024, 1024], "uint8");
        WriteArrayJson(minimapDir, [tileCount, 256, 256, 3], [1, 256, 256, 3], "uint8");

        // 6. Stream tile tensor chunks directly to disk
        string heightsChunksDir = Path.Combine(heightsDir, "c");
        string mcalChunksDir = Path.Combine(mcalDir, "c");
        string checkersChunksDir = Path.Combine(checkersDir, "c");
        string minimapChunksDir = Path.Combine(minimapDir, "c");

        Directory.CreateDirectory(heightsChunksDir);
        Directory.CreateDirectory(mcalChunksDir);
        Directory.CreateDirectory(checkersChunksDir);
        Directory.CreateDirectory(minimapChunksDir);

        for (int i = 0; i < mapPlan.Tiles.Count; i++)
        {
            RosettaTilePlan tile = mapPlan.Tiles[i];
            (float[] heights, byte[] alphaCanvas, byte[] checkersCanvas, byte[] minimapRgb) =
                tileSignalProvider(tile.TileX, tile.TileY);

            // Heights chunk: heights/c/{i}/0
            string hTileDir = Path.Combine(heightsChunksDir, i.ToString());
            Directory.CreateDirectory(hTileDir);
            WriteFloatArrayChunk(Path.Combine(hTileDir, "0"), heights);

            // MCAL chunk: mcal_alpha/c/{i}/0/0
            string mcalTileDir = Path.Combine(mcalChunksDir, i.ToString(), "0");
            Directory.CreateDirectory(mcalTileDir);
            File.WriteAllBytes(Path.Combine(mcalTileDir, "0"), alphaCanvas);

            // Checkers chunk: checkers_alpha/c/{i}/0/0
            string chkTileDir = Path.Combine(checkersChunksDir, i.ToString(), "0");
            Directory.CreateDirectory(chkTileDir);
            File.WriteAllBytes(Path.Combine(chkTileDir, "0"), checkersCanvas);

            // Minimap chunk: minimap_rgb_256/c/{i}/0/0/0
            string miniTileDir = Path.Combine(minimapChunksDir, i.ToString(), "0", "0");
            Directory.CreateDirectory(miniTileDir);
            File.WriteAllBytes(Path.Combine(miniTileDir, "0"), minimapRgb);
        }

        return new RosettaDatastoreIngestResult(
            DatastorePath: datastorePath,
            BuildId: buildId,
            MapName: mapPlan.MapName,
            TotalPlacements: mapPlan.Placements.Count,
            TotalUniqueAssetsInMap: uniqueAssetsInMap.Count,
            NewAssetsAdded: newAssetsAdded,
            DeduplicatedAssetsReused: deduplicatedReused,
            TilesWritten: tileCount);
    }

    private static void WriteRootGroupJson(string rootDir)
    {
        string zarrJson = Path.Combine(rootDir, "zarr.json");
        var doc = new
        {
            zarr_format = 3,
            node_type = "group",
            attributes = new
            {
                datastore_type = "rosetta_multiversion_datastore",
                schema_version = 1,
                created_utc = DateTime.UtcNow.ToString("o"),
                description = "Universal multi-version calibration datastore and object interchange format",
            }
        };
        File.WriteAllText(zarrJson, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void WriteGroupJson(string dir, IReadOnlyDictionary<string, object> attributes)
    {
        string zarrJson = Path.Combine(dir, "zarr.json");
        var doc = new
        {
            zarr_format = 3,
            node_type = "group",
            attributes = attributes
        };
        File.WriteAllText(zarrJson, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void WriteArrayJson(
        string arrayDir, int[] shape, int[] chunkShape, string dataType)
    {
        Directory.CreateDirectory(arrayDir);
        string zarrJson = Path.Combine(arrayDir, "zarr.json");
        var doc = new
        {
            zarr_format = 3,
            node_type = "array",
            shape = shape,
            data_type = dataType,
            chunk_grid = new
            {
                name = "regular",
                configuration = new
                {
                    chunk_shape = chunkShape
                }
            },
            chunk_key_encoding = new
            {
                name = "default",
                configuration = new
                {
                    separator = "/"
                }
            },
            codecs = new object[]
            {
                new
                {
                    name = "bytes",
                    configuration = new
                    {
                        endian = "little"
                    }
                }
            },
            attributes = new { }
        };
        File.WriteAllText(zarrJson, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void WriteFloatArrayChunk(string chunkPath, float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(chunkPath, bytes);
    }

    private static Dictionary<string, RosettaGlobalAssetRecord> LoadGlobalCatalog(string globalAssetsDir)
    {
        string parquetPath = Path.Combine(globalAssetsDir, "catalog.parquet");
        var catalog = new Dictionary<string, RosettaGlobalAssetRecord>(StringComparer.OrdinalIgnoreCase);
        if (!File.Exists(parquetPath))
            return catalog;

        using FileStream fs = File.OpenRead(parquetPath);
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
                catalog[record.AssetId] = record;
            }
        }

        return catalog;
    }

    private static void SaveGlobalCatalog(string globalAssetsDir, IReadOnlyList<RosettaGlobalAssetRecord> records)
    {
        string parquetPath = Path.Combine(globalAssetsDir, "catalog.parquet");
        using FileStream fs = File.Create(parquetPath);
        using ParquetWriter writer = ParquetWriter.CreateAsync(AssetCatalogSchema, fs).GetAwaiter().GetResult();
        using ParquetRowGroupWriter group = writer.CreateRowGroup();

        WriteColumn(group, AssetCatalogSchema, "asset_id", records.Select(r => r.AssetId).ToArray());
        WriteColumn(group, AssetCatalogSchema, "normalized_path", records.Select(r => r.NormalizedPath).ToArray());
        WriteColumn(group, AssetCatalogSchema, "original_path", records.Select(r => r.OriginalPath).ToArray());
        WriteColumn(group, AssetCatalogSchema, "kind", records.Select(r => r.Kind).ToArray());
        WriteColumn(group, AssetCatalogSchema, "bounds_min_x", records.Select(r => r.BoundsMinX).ToArray());
        WriteColumn(group, AssetCatalogSchema, "bounds_min_y", records.Select(r => r.BoundsMinY).ToArray());
        WriteColumn(group, AssetCatalogSchema, "bounds_min_z", records.Select(r => r.BoundsMinZ).ToArray());
        WriteColumn(group, AssetCatalogSchema, "bounds_max_x", records.Select(r => r.BoundsMaxX).ToArray());
        WriteColumn(group, AssetCatalogSchema, "bounds_max_y", records.Select(r => r.BoundsMaxY).ToArray());
        WriteColumn(group, AssetCatalogSchema, "bounds_max_z", records.Select(r => r.BoundsMaxZ).ToArray());
        WriteColumn(group, AssetCatalogSchema, "first_seen_build", records.Select(r => r.FirstSeenBuild).ToArray());
        WriteColumn(group, AssetCatalogSchema, "ref_count", records.Select(r => r.RefCount).ToArray());
    }

    private static void WritePlacementsParquet(string parquetPath, IReadOnlyList<RosettaPlacementRecord> placements)
    {
        using FileStream fs = File.Create(parquetPath);
        using ParquetWriter writer = ParquetWriter.CreateAsync(PlacementsSchema, fs).GetAwaiter().GetResult();
        using ParquetRowGroupWriter group = writer.CreateRowGroup();

        WriteColumn(group, PlacementsSchema, "asset_id", placements.Select(p => ComputeAssetId(p.Asset.AssetPath)).ToArray());
        WriteColumn(group, PlacementsSchema, "tile_x", placements.Select(p => p.TileX).ToArray());
        WriteColumn(group, PlacementsSchema, "tile_y", placements.Select(p => p.TileY).ToArray());
        WriteColumn(group, PlacementsSchema, "raw_x", placements.Select(p => p.RawPosition.X).ToArray());
        WriteColumn(group, PlacementsSchema, "raw_y", placements.Select(p => p.RawPosition.Y).ToArray());
        WriteColumn(group, PlacementsSchema, "raw_z", placements.Select(p => p.RawPosition.Z).ToArray());
        WriteColumn(group, PlacementsSchema, "render_x", placements.Select(p => p.RendererPosition.X).ToArray());
        WriteColumn(group, PlacementsSchema, "render_y", placements.Select(p => p.RendererPosition.Y).ToArray());
        WriteColumn(group, PlacementsSchema, "render_z", placements.Select(p => p.RendererPosition.Z).ToArray());
        WriteColumn(group, PlacementsSchema, "cell_u", placements.Select(p => p.CellU).ToArray());
        WriteColumn(group, PlacementsSchema, "cell_v", placements.Select(p => p.CellV).ToArray());
        WriteColumn(group, PlacementsSchema, "cell_size", placements.Select(p => p.CellSize).ToArray());
        WriteColumn(group, PlacementsSchema, "object_band_size", placements.Select(p => p.ObjectBandSize).ToArray());
        WriteColumn(group, PlacementsSchema, "label_text", placements.Select(p => p.LabelText).ToArray());
        WriteColumn(group, PlacementsSchema, "label_lines_json", placements.Select(p => JsonSerializer.Serialize(p.LabelLines)).ToArray());
        WriteColumn(group, PlacementsSchema, "label_pixel_meters", placements.Select(p => p.LabelPixelMeters).ToArray());
        WriteColumn(group, PlacementsSchema, "unique_id", placements.Select(p => p.UniqueId).ToArray());
    }

    private static void WriteColumn(ParquetRowGroupWriter group, ParquetSchema schema, string fieldName, Array data)
    {
        DataField field = schema.GetDataFields().First(f => f.Name == fieldName);
        group.WriteColumnAsync(new DataColumn(field, data)).GetAwaiter().GetResult();
    }
}
