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
        WriteArray(zip, "object_mask_257", pack.ObjectMask257, "<f4");
        WriteArray(zip, "object_precise_mask_257", pack.ObjectPreciseMask257, "<f4");
        WriteArray(zip, "pm4_path_mask", pack.Pm4PathMask, "<f4");
        WriteArray(zip, "pm4_building_footprint_mask", pack.Pm4BuildingFootprintMask, "<f4");
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
        string json = JsonSerializer.Serialize(new
        {
            tile_name = pack.TileName,
            map_name = pack.MapName,
            build_key = pack.BuildKey,
            source_adt_path = pack.SourceAdtPath,
            available_signals = pack.AvailableSignals.OrderBy(static signal => signal, StringComparer.OrdinalIgnoreCase),
            mcly_texture_names = pack.MclyTextureNames,
            minimap_source_tag = pack.MinimapSourceTag,
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
