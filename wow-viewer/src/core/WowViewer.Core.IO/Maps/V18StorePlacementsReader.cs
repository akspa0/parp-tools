using Parquet;
using Parquet.Data;
using Parquet.Schema;
using System.Text.Json;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Inventory of unique asset paths found in a V18 store's placements.parquet.
/// </summary>
public sealed record EnrichmentAssetInventory(
    IReadOnlyList<string> UniqueM2Paths,
    IReadOnlyList<string> UniqueWmoPaths,
    IReadOnlyList<string> UniqueBlpPaths);

/// <summary>
/// Reads a V18 store's placements.parquet and returns an inventory of unique asset paths.
/// </summary>
public static class V18StorePlacementsReader
{
    /// <summary>
    /// Read placements.parquet from a V18 Zarr store directory and return the unique asset paths.
    /// </summary>
    public static EnrichmentAssetInventory ReadPlacements(string v18StorePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(v18StorePath);

        string parquetPath = Path.Combine(v18StorePath, "placements.parquet");
        if (!File.Exists(parquetPath))
            return new EnrichmentAssetInventory([], [], []);

        var m2Paths = new HashSet<string>(StringComparer.InvariantCultureIgnoreCase);
        var wmoPaths = new HashSet<string>(StringComparer.InvariantCultureIgnoreCase);
        var blpPaths = new HashSet<string>(StringComparer.InvariantCultureIgnoreCase);

        using var fs = File.OpenRead(parquetPath);
        using var reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();

        for (int rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using var rgReader = reader.OpenRowGroupReader(rg);
            DataField[] fields = reader.Schema.GetDataFields();

            // Find instance_type and asset_path data fields
            DataField? instanceTypeField = fields.FirstOrDefault(f => f.Name == "instance_type");
            DataField? assetPathField = fields.FirstOrDefault(f => f.Name == "asset_path");
            if (instanceTypeField is null || assetPathField is null)
                continue;

            var typesColumn = rgReader.ReadColumnAsync(instanceTypeField).GetAwaiter().GetResult();
            var pathsColumn = rgReader.ReadColumnAsync(assetPathField).GetAwaiter().GetResult();
            var types = (string[])typesColumn.Data;
            var paths = (string[])pathsColumn.Data;

            for (int i = 0; i < types.Length && i < paths.Length; i++)
            {
                string type = types[i] ?? "";
                string path = paths[i] ?? "";
                if (string.IsNullOrWhiteSpace(path))
                    continue;

                path = path.Replace('\\', '/').Trim();

                if (type.Equals("mddf", StringComparison.OrdinalIgnoreCase)
                    && path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                {
                    m2Paths.Add(path);
                }
                else if (type.Equals("modf", StringComparison.OrdinalIgnoreCase)
                    && path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                {
                    wmoPaths.Add(path);
                }
            }
        }

        string decodedMetadataPath = Path.Combine(v18StorePath, "decoded_metadata.parquet");
        if (File.Exists(decodedMetadataPath))
        {
            using var metadataStream = File.OpenRead(decodedMetadataPath);
            using var metadataReader = ParquetReader.CreateAsync(metadataStream).GetAwaiter().GetResult();

            for (int rg = 0; rg < metadataReader.RowGroupCount; rg++)
            {
                using var rgReader = metadataReader.OpenRowGroupReader(rg);
                DataField[] fields = metadataReader.Schema.GetDataFields();
                DataField? metadataJsonField = fields.FirstOrDefault(f => f.Name == "decoded_metadata_json");
                if (metadataJsonField is null)
                    continue;

                var metadataColumn = rgReader.ReadColumnAsync(metadataJsonField).GetAwaiter().GetResult();
                var metadataRows = (string[])metadataColumn.Data;

                foreach (string metadataJson in metadataRows)
                {
                    if (string.IsNullOrWhiteSpace(metadataJson))
                        continue;

                    try
                    {
                        using JsonDocument document = JsonDocument.Parse(metadataJson);
                        if (!document.RootElement.TryGetProperty("mcly_texture_names", out JsonElement textureNames)
                            || textureNames.ValueKind != JsonValueKind.Array)
                        {
                            continue;
                        }

                        foreach (JsonElement textureName in textureNames.EnumerateArray())
                        {
                            if (textureName.ValueKind != JsonValueKind.String)
                                continue;

                            string? path = textureName.GetString();
                            if (string.IsNullOrWhiteSpace(path))
                                continue;

                            string normalized = path.Replace('\\', '/').Trim();
                            if (normalized.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                                blpPaths.Add(normalized);
                        }
                    }
                    catch (JsonException)
                    {
                    }
                }
            }
        }

        return new EnrichmentAssetInventory(
            [.. m2Paths],
            [.. wmoPaths],
            [.. blpPaths]);
    }

    /// <summary>
    /// Try to read BLP texture paths from a tile's metadata.
    /// </summary>
    public static HashSet<string> ReadBlpPathsFromMetadata(IReadOnlyList<string> mtexTexturePaths)
    {
        var paths = new HashSet<string>(StringComparer.InvariantCultureIgnoreCase);
        if (mtexTexturePaths is null)
            return paths;

        foreach (string p in mtexTexturePaths)
        {
            string norm = p.Replace('\\', '/').Trim();
            if (!string.IsNullOrWhiteSpace(norm) && norm.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                paths.Add(norm);
        }

        return paths;
    }
}
