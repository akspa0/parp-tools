using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using ICSharpCode.SharpZipLib.Zip;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Serializes a <see cref="TerrainTileTensorPack"/> to a NumPy-compatible .npz file.
/// Each signal becomes one .npy entry inside the ZIP archive.
/// </summary>
public static class NpzTileSerializer
{
    private static ReadOnlySpan<byte> NumpyMagic => [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];
    private const ushort NumpyVersion = 0x0001; // version 1.0

    public static void Serialize(TerrainTileTensorPack pack, string outputPath)
    {
        ArgumentNullException.ThrowIfNull(pack);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using FileStream fs = File.Create(outputPath);
        using ZipOutputStream zip = new(fs);
        zip.SetLevel(3); // balance speed vs compression

        // Write each present signal as a .npy entry
        WriteArray(zip, "height_257", pack.Height257, "<f4");
        WriteArray(zip, "height_65", pack.Height65, "<f4");
        WriteArray(zip, "height_17", pack.Height17, "<f4");
        WriteArray(zip, "mcly_texture_ids", pack.MclyTextureIds, "<i4");
        WriteArray(zip, "mcly_layer_mask", pack.MclyLayerMask, "|b1");
        WriteArray(zip, "mcal_alpha_pack_256", pack.McalAlphaPack256, "<f4");
        WriteArray(zip, "mccv_rgb", pack.MccvRgb, "<f4");
        WriteArray(zip, "mcnr_normal_xyz", pack.McnrNormalXyz, "<f4");
        WriteArray(zip, "mh2o_surface_height", pack.Mh2oSurfaceHeight, "<f4");
        WriteArray(zip, "mh2o_depth", pack.Mh2oDepth, "<f4");
        WriteArray(zip, "mh2o_type_mask", pack.Mh2oTypeMask, "<i4");
        WriteArray(zip, "mclq_surface_height", pack.MclqSurfaceHeight, "<f4");
        WriteArray(zip, "mclq_type_mask", pack.MclqTypeMask, "<i4");
        WriteArray(zip, "wl_liquid_mask", pack.WlLiquidMask, "<f4");
        WriteArray(zip, "wl_liquid_height", pack.WlLiquidHeight, "<f4");
        WriteArray(zip, "unified_liquid_mask", pack.UnifiedLiquidMask, "<f4");
        WriteArray(zip, "unified_liquid_height", pack.UnifiedLiquidHeight, "<f4");
        WriteArray(zip, "object_mask_257", pack.ObjectMask257, "<f4");
        WriteArray(zip, "object_precise_mask_257", pack.ObjectPreciseMask257, "<f4");
        WriteArray(zip, "pm4_path_mask", pack.Pm4PathMask, "<f4");
        WriteArray(zip, "pm4_building_footprint_mask", pack.Pm4BuildingFootprintMask, "<f4");
        WriteArray(zip, "pm4_mprl_mask", pack.Pm4MprlMask, "<f4");
        WriteArray(zip, "mcsh_shadow_mask_256", pack.McshShadowMask256, "<f4");
        WriteArray(zip, "shadow_residual_mask_256", pack.ShadowResidualMask256, "<f4");
        WriteArray(zip, "minimap_rgb_256", pack.MinimapRgb256, "|u1");
        WriteArray(zip, "hole_mask_16", pack.HoleMask16, "|b1");
        WriteArray(zip, "mtxf_animated_mask", pack.MtxfAnimatedMask, "<i4");
        WriteArray(zip, "mtxf_transform_id", pack.MtxfTransformId, "<i4");

        // Write metadata JSON
        WriteMetadata(zip, pack);

        zip.Finish();
    }

    private static void WriteArray(ZipOutputStream zip, string name, Array? array, string dtype)
    {
        if (array is null)
            return;

        string entryName = $"{name}.npy";
        ZipEntry entry = new(entryName);
        zip.PutNextEntry(entry);

        byte[] header = BuildNpyHeader(dtype, array);
        zip.Write(header, 0, header.Length);

        byte[] data = ArrayToByteArray(array);
        zip.Write(data, 0, data.Length);

        zip.CloseEntry();
    }

    private static void WriteMetadata(ZipOutputStream zip, TerrainTileTensorPack pack)
    {
        (int? tileX, int? tileY) = TryParseTileCoordinates(pack.TileName);
        TileMetrics metrics = BuildTileMetrics(pack);
        string json = JsonSerializer.Serialize(new
        {
            tile_name = pack.TileName,
            tile_x = tileX,
            tile_y = tileY,
            map_name = pack.MapName,
            build_key = pack.BuildKey,
            source_adt_path = pack.SourceAdtPath,
            available_signals = pack.AvailableSignals.OrderBy(static signal => signal, StringComparer.OrdinalIgnoreCase),
            mcly_texture_names = pack.MclyTextureNames,
            minimap_source_tag = pack.MinimapSourceTag,
            // Kept at top-level for direct metadata consumers.
            mcnk_count = metrics.McnkCount,
            total_layer_count = metrics.TotalLayerCount,
            max_layer_count = metrics.MaxLayerCount,
            chunks_with_mcvt = metrics.ChunksWithMcvt,
            chunks_with_holes = metrics.ChunksWithHoles,
            chunks_with_liquid_flags = metrics.ChunksWithLiquidFlags,
            unique_texture_ids = metrics.UniqueTextureIds,
            global_min_height = metrics.GlobalMinHeight,
            global_max_height = metrics.GlobalMaxHeight,
            height_range = metrics.HeightRange,
            minimap_variance = metrics.MinimapVariance,
            minimap_gradient = metrics.MinimapGradient,
            object_mask_coverage = metrics.ObjectMaskCoverage,
            object_precise_mask_coverage = metrics.ObjectPreciseMaskCoverage,
            tile_metrics = new
            {
                metrics.McnkCount,
                metrics.TotalLayerCount,
                metrics.MaxLayerCount,
                metrics.ChunksWithMcvt,
                metrics.ChunksWithHoles,
                metrics.ChunksWithLiquidFlags,
                metrics.UniqueTextureIds,
                metrics.GlobalMinHeight,
                metrics.GlobalMaxHeight,
                metrics.HeightRange,
                metrics.MinimapVariance,
                metrics.MinimapGradient,
                metrics.ObjectMaskCoverage,
                metrics.ObjectPreciseMaskCoverage,
            },
        }, new JsonSerializerOptions
        {
            WriteIndented = true,
        });

        ZipEntry entry = new("metadata.json");
        zip.PutNextEntry(entry);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        zip.Write(bytes, 0, bytes.Length);
        zip.CloseEntry();
    }

    private static TileMetrics BuildTileMetrics(TerrainTileTensorPack pack)
    {
        int mcnkCount = 0;
        int chunksWithMcvt = 0;
        float globalMinHeight = 0f;
        float globalMaxHeight = 0f;
        float heightRange = 0f;
        if (pack.Height257 is not null)
        {
            mcnkCount = 256;
            chunksWithMcvt = 256;
            (globalMinHeight, globalMaxHeight) = ComputeMinMax(pack.Height257);
            heightRange = globalMaxHeight - globalMinHeight;
        }

        int chunksWithHoles = 0;
        if (pack.HoleMask16 is not null)
        {
            for (int y = 0; y < pack.HoleMask16.GetLength(0); y++)
            {
                for (int x = 0; x < pack.HoleMask16.GetLength(1); x++)
                {
                    if (pack.HoleMask16[y, x])
                        chunksWithHoles++;
                }
            }
        }

        HashSet<int> uniqueTextureIds = new();
        int totalLayerCount = 0;
        int maxLayerCount = 0;
        if (pack.MclyTextureIds is not null)
        {
            int chunkYLength = pack.MclyTextureIds.GetLength(0);
            int chunkXLength = pack.MclyTextureIds.GetLength(1);
            int layerLength = pack.MclyTextureIds.GetLength(2);
            mcnkCount = Math.Max(mcnkCount, chunkYLength * chunkXLength);
            for (int chunkY = 0; chunkY < chunkYLength; chunkY++)
            {
                for (int chunkX = 0; chunkX < chunkXLength; chunkX++)
                {
                    int chunkLayers = 0;
                    for (int layer = 0; layer < layerLength; layer++)
                    {
                        int textureId = pack.MclyTextureIds[chunkY, chunkX, layer];
                        if (textureId < 0)
                            continue;

                        chunkLayers++;
                        uniqueTextureIds.Add(textureId);
                    }

                    totalLayerCount += chunkLayers;
                    if (chunkLayers > maxLayerCount)
                        maxLayerCount = chunkLayers;
                }
            }
        }

        int chunksWithLiquidFlags = CountChunksWithLiquid(pack.UnifiedLiquidMask)
            + CountChunksWithLiquid(pack.Mh2oTypeMask)
            + CountChunksWithLiquid(pack.MclqTypeMask)
            + CountChunksWithLiquid(pack.WlLiquidMask);
        // Avoid double-counting across sources above by clamping to the chunk budget.
        chunksWithLiquidFlags = Math.Min(chunksWithLiquidFlags, 256);

        float minimapVariance = pack.MinimapRgb256 is null ? 0f : ComputeMinimapVariance(pack.MinimapRgb256);
        float minimapGradient = pack.MinimapRgb256 is null ? 0f : ComputeMinimapGradient(pack.MinimapRgb256);
        float objectMaskCoverage = pack.ObjectMask257 is null ? 0f : ComputeNonZeroCoverage(pack.ObjectMask257);
        float objectPreciseCoverage = pack.ObjectPreciseMask257 is null ? 0f : ComputeNonZeroCoverage(pack.ObjectPreciseMask257);

        return new TileMetrics(
            mcnkCount,
            totalLayerCount,
            maxLayerCount,
            chunksWithMcvt,
            chunksWithHoles,
            chunksWithLiquidFlags,
            uniqueTextureIds.OrderBy(static value => value).ToArray(),
            globalMinHeight,
            globalMaxHeight,
            heightRange,
            minimapVariance,
            minimapGradient,
            objectMaskCoverage,
            objectPreciseCoverage);
    }

    private static int CountChunksWithLiquid(float[,]? liquidMask257)
    {
        if (liquidMask257 is null)
            return 0;

        int width = liquidMask257.GetLength(1);
        int height = liquidMask257.GetLength(0);
        int chunkCount = 0;
        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                if (ChunkHasAnyValue(liquidMask257, chunkX, chunkY, width, height))
                    chunkCount++;
            }
        }

        return chunkCount;
    }

    private static int CountChunksWithLiquid(int[,]? liquidTypeMask257)
    {
        if (liquidTypeMask257 is null)
            return 0;

        int width = liquidTypeMask257.GetLength(1);
        int height = liquidTypeMask257.GetLength(0);
        int chunkCount = 0;
        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                if (ChunkHasAnyValue(liquidTypeMask257, chunkX, chunkY, width, height))
                    chunkCount++;
            }
        }

        return chunkCount;
    }

    private static bool ChunkHasAnyValue(float[,] grid, int chunkX, int chunkY, int width, int height)
    {
        int startX = chunkX * 16;
        int startY = chunkY * 16;
        int endX = Math.Min(startX + 16, width - 1);
        int endY = Math.Min(startY + 16, height - 1);
        for (int y = startY; y <= endY; y++)
        {
            for (int x = startX; x <= endX; x++)
            {
                if (grid[y, x] > 0f)
                    return true;
            }
        }

        return false;
    }

    private static bool ChunkHasAnyValue(int[,] grid, int chunkX, int chunkY, int width, int height)
    {
        int startX = chunkX * 16;
        int startY = chunkY * 16;
        int endX = Math.Min(startX + 16, width - 1);
        int endY = Math.Min(startY + 16, height - 1);
        for (int y = startY; y <= endY; y++)
        {
            for (int x = startX; x <= endX; x++)
            {
                if (grid[y, x] != 0)
                    return true;
            }
        }

        return false;
    }

    private static (float Min, float Max) ComputeMinMax(float[,] grid)
    {
        float min = float.MaxValue;
        float max = float.MinValue;
        for (int y = 0; y < grid.GetLength(0); y++)
        {
            for (int x = 0; x < grid.GetLength(1); x++)
            {
                float value = grid[y, x];
                if (value < min)
                    min = value;
                if (value > max)
                    max = value;
            }
        }

        return min == float.MaxValue ? (0f, 0f) : (min, max);
    }

    private static float ComputeMinimapVariance(byte[,,] minimap)
    {
        if (minimap.Length == 0)
            return 0f;

        double sum = 0d;
        double sumSquares = 0d;
        int samples = 0;
        for (int y = 0; y < minimap.GetLength(0); y++)
        {
            for (int x = 0; x < minimap.GetLength(1); x++)
            {
                double r = minimap[y, x, 0] / 255d;
                double g = minimap[y, x, 1] / 255d;
                double b = minimap[y, x, 2] / 255d;
                double luma = (0.2126d * r) + (0.7152d * g) + (0.0722d * b);
                sum += luma;
                sumSquares += luma * luma;
                samples++;
            }
        }

        if (samples == 0)
            return 0f;

        double mean = sum / samples;
        double variance = Math.Max(0d, (sumSquares / samples) - (mean * mean));
        return (float)variance;
    }

    private static float ComputeMinimapGradient(byte[,,] minimap)
    {
        int height = minimap.GetLength(0);
        int width = minimap.GetLength(1);
        if (height < 2 || width < 2)
            return 0f;

        double gradientSum = 0d;
        int sampleCount = 0;
        for (int y = 1; y < height; y++)
        {
            for (int x = 1; x < width; x++)
            {
                double current = ComputeLuma(minimap, y, x);
                double left = ComputeLuma(minimap, y, x - 1);
                double up = ComputeLuma(minimap, y - 1, x);
                double dx = current - left;
                double dy = current - up;
                gradientSum += Math.Sqrt((dx * dx) + (dy * dy));
                sampleCount++;
            }
        }

        return sampleCount == 0 ? 0f : (float)(gradientSum / sampleCount);
    }

    private static double ComputeLuma(byte[,,] minimap, int y, int x)
    {
        double r = minimap[y, x, 0] / 255d;
        double g = minimap[y, x, 1] / 255d;
        double b = minimap[y, x, 2] / 255d;
        return (0.2126d * r) + (0.7152d * g) + (0.0722d * b);
    }

    private static float ComputeNonZeroCoverage(float[,] mask)
    {
        int total = mask.GetLength(0) * mask.GetLength(1);
        if (total == 0)
            return 0f;

        int active = 0;
        for (int y = 0; y < mask.GetLength(0); y++)
        {
            for (int x = 0; x < mask.GetLength(1); x++)
            {
                if (mask[y, x] > 0f)
                    active++;
            }
        }

        return (float)active / total;
    }

    private static (int? X, int? Y) TryParseTileCoordinates(string tileName)
    {
        if (string.IsNullOrWhiteSpace(tileName))
            return (null, null);

        string[] segments = tileName.Split('_');
        if (segments.Length < 3)
            return (null, null);

        return int.TryParse(segments[^2], out int x) && int.TryParse(segments[^1], out int y)
            ? (x, y)
            : (null, null);
    }

    private sealed record TileMetrics(
        int McnkCount,
        int TotalLayerCount,
        int MaxLayerCount,
        int ChunksWithMcvt,
        int ChunksWithHoles,
        int ChunksWithLiquidFlags,
        int[] UniqueTextureIds,
        float GlobalMinHeight,
        float GlobalMaxHeight,
        float HeightRange,
        float MinimapVariance,
        float MinimapGradient,
        float ObjectMaskCoverage,
        float ObjectPreciseMaskCoverage);

    private static byte[] BuildNpyHeader(string dtype, Array array)
    {
        string shapeStr = string.Join(", ", Enumerable.Range(0, array.Rank).Select(r => array.GetLength(r).ToString()));
        string header = $"{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({shapeStr},)}}";

        // NumPy v1 headers must end with a newline, and the full preamble plus
        // header must align to a 64-byte boundary.
        int prefixSize = NumpyMagic.Length + 2 + 2; // magic + version + header_len
        int totalSize = prefixSize + header.Length + 1;
        int paddingNeeded = (64 - (totalSize % 64)) % 64;
        header += new string(' ', paddingNeeded) + '\n';

        int headerLen = header.Length;

        byte[] result = new byte[prefixSize + headerLen];
        int offset = 0;

        NumpyMagic.CopyTo(result.AsSpan(offset, NumpyMagic.Length));
        offset += NumpyMagic.Length;

        BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(offset, 2), NumpyVersion);
        offset += 2;

        BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(offset, 2), (ushort)headerLen);
        offset += 2;

        Encoding.ASCII.GetBytes(header).CopyTo(result, offset);

        return result;
    }

    private static byte[] ArrayToByteArray(Array array)
    {
        Type elementType = array.GetType().GetElementType()!;

        if (elementType == typeof(float))
            return FlattenFloatArray((Array)array);
        if (elementType == typeof(int))
            return FlattenIntArray((Array)array);
        if (elementType == typeof(bool))
            return FlattenBoolArray((Array)array);
        if (elementType == typeof(byte))
            return FlattenByteArray((Array)array);

        throw new NotSupportedException($"NPZ serialization does not yet support element type {elementType.Name}");
    }

    private static byte[] FlattenFloatArray(Array array)
    {
        int totalElements = array.GetType().GetElementType()!.IsArray
            ? Enumerable.Range(0, array.Rank).Aggregate(1, (acc, r) => acc * array.GetLength(r))
            : array.Length;

        byte[] result = new byte[totalElements * sizeof(float)];
        int index = 0;

        switch (array.Rank)
        {
            case 2:
                var f2 = (float[,])array;
                for (int y = 0; y < f2.GetLength(0); y++)
                    for (int x = 0; x < f2.GetLength(1); x++)
                        BitConverter.GetBytes(f2[y, x]).CopyTo(result, (index++) * sizeof(float));
                break;
            case 3:
                var f3 = (float[,,])array;
                for (int z = 0; z < f3.GetLength(0); z++)
                    for (int y = 0; y < f3.GetLength(1); y++)
                        for (int x = 0; x < f3.GetLength(2); x++)
                            BitConverter.GetBytes(f3[z, y, x]).CopyTo(result, (index++) * sizeof(float));
                break;
            default:
                throw new NotSupportedException($"Cannot flatten float array with rank {array.Rank}");
        }

        return result;
    }

    private static byte[] FlattenIntArray(Array array)
    {
        int totalElements = Enumerable.Range(0, array.Rank).Aggregate(1, (acc, r) => acc * array.GetLength(r));
        byte[] result = new byte[totalElements * sizeof(int)];
        int index = 0;

        switch (array.Rank)
        {
            case 2:
                var i2 = (int[,])array;
                for (int y = 0; y < i2.GetLength(0); y++)
                    for (int x = 0; x < i2.GetLength(1); x++)
                        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan((index++) * sizeof(int), sizeof(int)), i2[y, x]);
                break;
            case 3:
                var i3 = (int[,,])array;
                for (int z = 0; z < i3.GetLength(0); z++)
                    for (int y = 0; y < i3.GetLength(1); y++)
                        for (int x = 0; x < i3.GetLength(2); x++)
                            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan((index++) * sizeof(int), sizeof(int)), i3[z, y, x]);
                break;
            default:
                throw new NotSupportedException($"Cannot flatten int array with rank {array.Rank}");
        }

        return result;
    }

    private static byte[] FlattenBoolArray(Array array)
    {
        int totalElements = Enumerable.Range(0, array.Rank).Aggregate(1, (acc, r) => acc * array.GetLength(r));
        byte[] result = new byte[totalElements];
        int index = 0;

        switch (array.Rank)
        {
            case 2:
                var b2 = (bool[,])array;
                for (int y = 0; y < b2.GetLength(0); y++)
                    for (int x = 0; x < b2.GetLength(1); x++)
                        result[index++] = b2[y, x] ? (byte)1 : (byte)0;
                break;
            case 3:
                var b3 = (bool[,,])array;
                for (int z = 0; z < b3.GetLength(0); z++)
                    for (int y = 0; y < b3.GetLength(1); y++)
                        for (int x = 0; x < b3.GetLength(2); x++)
                            result[index++] = b3[z, y, x] ? (byte)1 : (byte)0;
                break;
            default:
                throw new NotSupportedException($"Cannot flatten bool array with rank {array.Rank}");
        }

        return result;
    }

    private static byte[] FlattenByteArray(Array array)
    {
        int totalElements = Enumerable.Range(0, array.Rank).Aggregate(1, (acc, r) => acc * array.GetLength(r));
        byte[] result = new byte[totalElements];
        int index = 0;

        switch (array.Rank)
        {
            case 2:
                var b2 = (byte[,])array;
                for (int y = 0; y < b2.GetLength(0); y++)
                    for (int x = 0; x < b2.GetLength(1); x++)
                        result[index++] = b2[y, x];
                break;
            case 3:
                var b3 = (byte[,,])array;
                for (int z = 0; z < b3.GetLength(0); z++)
                    for (int y = 0; y < b3.GetLength(1); y++)
                        for (int x = 0; x < b3.GetLength(2); x++)
                            result[index++] = b3[z, y, x];
                break;
            default:
                throw new NotSupportedException($"Cannot flatten byte array with rank {array.Rank}");
        }

        return result;
    }
}
