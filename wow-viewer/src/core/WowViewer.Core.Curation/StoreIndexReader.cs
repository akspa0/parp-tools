using Parquet;
using Parquet.Data;
using Parquet.Schema;

namespace WowViewer.Core.Curation;

/// <summary>One row of a v50 store's own <c>index.parquet</c> (tile identity only -- no signal
/// arrays; those live in the store's Zarr array data, which this library does not read).</summary>
public sealed record StoreIndexRow(long TileId, string Build, string Map, int TileX, int TileY);

/// <summary>
/// Reads a v50 store's <c>index.parquet</c> row identity list, read-only (FR-014) -- the tile
/// coordinates it returns are what <c>curate</c> re-derives a fresh
/// <see cref="WowViewer.Core.Maps.TerrainTileTensorPack"/> for, since this codebase has no C#
/// Zarr array reader; the store's actual signal arrays are never opened by this library.
/// </summary>
public static class StoreIndexReader
{
    public static IReadOnlyList<StoreIndexRow> Read(string storePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(storePath);
        string indexPath = Path.Combine(storePath, "index.parquet");
        if (!File.Exists(indexPath))
            throw new FileNotFoundException($"No index.parquet found under store '{storePath}'.", indexPath);

        var rows = new List<StoreIndexRow>();
        using FileStream stream = File.OpenRead(indexPath);
        using ParquetReader reader = ParquetReader.CreateAsync(stream).GetAwaiter().GetResult();
        DataField[] fields = reader.Schema.GetDataFields();

        DataField tileIdField = fields.First(f => f.Name == "tile_id");
        DataField buildField = fields.First(f => f.Name == "build");
        DataField mapField = fields.First(f => f.Name == "map");
        DataField tileXField = fields.First(f => f.Name == "tile_x");
        DataField tileYField = fields.First(f => f.Name == "tile_y");

        for (int rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using ParquetRowGroupReader group = reader.OpenRowGroupReader(rg);
            long[] tileIds = ReadNonNullableInt64(group, tileIdField, indexPath, "tile_id");
            var builds = (string[])group.ReadColumnAsync(buildField).GetAwaiter().GetResult().Data;
            var maps = (string[])group.ReadColumnAsync(mapField).GetAwaiter().GetResult().Data;
            long[] tileXs = ReadNonNullableInt64(group, tileXField, indexPath, "tile_x");
            long[] tileYs = ReadNonNullableInt64(group, tileYField, indexPath, "tile_y");

            for (int i = 0; i < tileIds.Length; i++)
                rows.Add(new StoreIndexRow(tileIds[i], builds[i], maps[i], (int)tileXs[i], (int)tileYs[i]));
        }

        return rows;
    }

    /// <summary>
    /// PyArrow can write an int64 column as physically nullable even when the Python producer
    /// never wrote a null into it (observed on the real v50 <c>index.parquet</c>: pyarrow's schema
    /// print shows a plain non-nullable <c>int64</c>, but Parquet.Net still returns
    /// <c>long?[]</c>). Reads either representation and fails loudly on a genuine null rather than
    /// silently coercing it to 0 -- a null tile coordinate would be real store corruption worth
    /// surfacing, not masking.
    /// </summary>
    private static long[] ReadNonNullableInt64(ParquetRowGroupReader group, DataField field, string sourcePath, string columnName)
    {
        Array data = group.ReadColumnAsync(field).GetAwaiter().GetResult().Data;
        if (data is long[] plain)
            return plain;

        if (data is long?[] nullable)
        {
            var result = new long[nullable.Length];
            for (int i = 0; i < nullable.Length; i++)
            {
                result[i] = nullable[i] ?? throw new InvalidDataException(
                    $"Column '{columnName}' in '{sourcePath}' contains a null value at row {i} -- expected every row to have a real tile coordinate.");
            }
            return result;
        }

        throw new InvalidDataException($"Column '{columnName}' in '{sourcePath}' has an unexpected CLR type '{data.GetType()}'.");
    }
}
