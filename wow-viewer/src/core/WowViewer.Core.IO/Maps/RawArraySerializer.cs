using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Serializes a <see cref="TerrainTileTensorPack"/> to a raw binary stream.
/// No compression. Each array is written as: [name][ndim][shape][dtype][byte_length][raw_bytes].
/// Format: "ARRY" magic + metadata JSON + arrays + "ENDS" sentinel.
/// </summary>
public static class RawArraySerializer
{
    public enum StreamProfile
    {
        Full,
        V16,
        V22
    }

    private static ReadOnlySpan<byte> ArrayMagic => "ARRY"u8;
    private static ReadOnlySpan<byte> EndsMagic => "ENDS"u8;

    public static void Serialize(TerrainTileTensorPack pack, Stream outputStream)
        => Serialize(pack, outputStream, StreamProfile.Full);

    public static void Serialize(TerrainTileTensorPack pack, Stream outputStream, StreamProfile profile)
    {
        ArgumentNullException.ThrowIfNull(pack);
        ArgumentNullException.ThrowIfNull(outputStream);

        // Magic
        outputStream.Write(ArrayMagic);

        // Metadata JSON
        string metadata = BuildMetadataJson(pack);
        byte[] metaBytes = Encoding.UTF8.GetBytes(metadata);
        byte[] metaLen = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(metaLen, metaBytes.Length);
        outputStream.Write(metaLen);
        outputStream.Write(metaBytes);

        if (profile == StreamProfile.V16)
        {
            WriteV16Arrays(pack, outputStream);
        }
        else if (profile == StreamProfile.V22)
        {
            WriteV22Arrays(pack, outputStream);
        }
        else
        {
            WriteFullArrays(pack, outputStream);
        }

        // Ends sentinel
        outputStream.Write(EndsMagic);
    }

    private static void WriteV16Arrays(TerrainTileTensorPack pack, Stream outputStream)
    {
        WriteArray(outputStream, "height_257", pack.Height257);
        WriteArray(outputStream, "mcly_texture_ids", pack.MclyTextureIds);
        WriteArray(outputStream, "mcly_layer_mask", pack.MclyLayerMask);
        WriteArray(outputStream, "mcal_alpha_pack_256", pack.McalAlphaPack256);
        WriteArray(outputStream, "mcnr_normal_xyz", pack.McnrNormalXyz);
        WriteArray(outputStream, "mcnr_mask_257", pack.McnrMask257);
        WriteArray(outputStream, "mcnk_flags_16", pack.McnkFlags16);
        WriteArray(outputStream, "mh2o_surface_height", pack.Mh2oSurfaceHeight);
        WriteArray(outputStream, "mh2o_depth", pack.Mh2oDepth);
        WriteArray(outputStream, "mh2o_type_mask", pack.Mh2oTypeMask);
        WriteArray(outputStream, "mh2o_presence_mask", pack.Mh2oPresenceMask);
        WriteArray(outputStream, "mclq_surface_height", pack.MclqSurfaceHeight);
        WriteArray(outputStream, "mclq_type_mask", pack.MclqTypeMask);
        WriteArray(outputStream, "mclq_presence_mask", pack.MclqPresenceMask);
        WriteArray(outputStream, "wl_liquid_mask", pack.WlLiquidMask);
        WriteArray(outputStream, "wl_liquid_height", pack.WlLiquidHeight);
        WriteArray(outputStream, "unified_liquid_mask", pack.UnifiedLiquidMask);
        WriteArray(outputStream, "unified_liquid_height", pack.UnifiedLiquidHeight);
        WriteArray(outputStream, "liquid_basic_type_257", pack.LiquidBasicType257);
        WriteArray(outputStream, "object_mask_257", pack.ObjectMask257);
        WriteArray(outputStream, "object_precise_mask_257", pack.ObjectPreciseMask257);
        WriteArray(outputStream, "object_instance_mask_257", pack.ObjectInstanceMask257);
        WriteArray(outputStream, "mddf_mask_257", pack.MddfMask257);
        WriteArray(outputStream, "modf_mask_257", pack.ModfMask257);
        WriteArray(outputStream, "object_filtered_mask_257", pack.ObjectFilteredMask257);
        WriteArray(outputStream, "mcsh_shadow_mask_256", pack.McshShadowMask256);
        WriteArray(outputStream, "object_roof_mask_256", pack.ObjectRoofMask256);
        WriteArray(outputStream, "object_roof_confidence_256", pack.ObjectRoofConfidence256);
        WriteArray(outputStream, "minimap_rgb_256", pack.MinimapRgb256);
        WriteArray(outputStream, "hole_mask_16", pack.HoleMask16);
        WriteArray(outputStream, "placement_mddf_data", pack.PlacementMddfData);
        WriteArray(outputStream, "placement_modf_data", pack.PlacementModfData);
    }

    private static void WriteFullArrays(TerrainTileTensorPack pack, Stream outputStream)
    {
        WriteArray(outputStream, "height_257", pack.Height257);
        WriteArray(outputStream, "height_65", pack.Height65);
        WriteArray(outputStream, "height_17", pack.Height17);
        WriteArray(outputStream, "mcly_texture_ids", pack.MclyTextureIds);
        WriteArray(outputStream, "mcly_layer_mask", pack.MclyLayerMask);
        WriteArray(outputStream, "mcmt_material_ids", pack.McmtMaterialIds);
        WriteArray(outputStream, "mamp_value", pack.MampValue);
        WriteArray(outputStream, "mcal_alpha_pack", pack.McalAlphaPack);
        WriteArray(outputStream, "mcal_alpha_pack_256", pack.McalAlphaPack256);
        WriteArray(outputStream, "mccv_rgb", pack.MccvRgb);
        WriteArray(outputStream, "mclv_lighting_bytes", pack.MclvLightingBytes);
        WriteArray(outputStream, "mcnr_normal_xyz", pack.McnrNormalXyz);
        WriteArray(outputStream, "mcnr_mask_257", pack.McnrMask257);
        WriteArray(outputStream, "mfbo_flight_bounds", pack.MfboFlightBounds);
        WriteArray(outputStream, "mcnk_flags_16", pack.McnkFlags16);
        WriteArray(outputStream, "mh2o_surface_height", pack.Mh2oSurfaceHeight);
        WriteArray(outputStream, "mh2o_depth", pack.Mh2oDepth);
        WriteArray(outputStream, "mh2o_type_mask", pack.Mh2oTypeMask);
        WriteArray(outputStream, "mh2o_presence_mask", pack.Mh2oPresenceMask);
        WriteArray(outputStream, "mclq_surface_height", pack.MclqSurfaceHeight);
        WriteArray(outputStream, "mclq_type_mask", pack.MclqTypeMask);
        WriteArray(outputStream, "mclq_presence_mask", pack.MclqPresenceMask);
        WriteArray(outputStream, "wl_liquid_mask", pack.WlLiquidMask);
        WriteArray(outputStream, "wl_liquid_height", pack.WlLiquidHeight);
        WriteArray(outputStream, "unified_liquid_mask", pack.UnifiedLiquidMask);
        WriteArray(outputStream, "unified_liquid_height", pack.UnifiedLiquidHeight);
        WriteArray(outputStream, "liquid_basic_type_257", pack.LiquidBasicType257);
        WriteArray(outputStream, "object_mask_257", pack.ObjectMask257);
        WriteArray(outputStream, "object_precise_mask_257", pack.ObjectPreciseMask257);
        WriteArray(outputStream, "object_instance_mask_257", pack.ObjectInstanceMask257);
        WriteArray(outputStream, "mddf_mask_257", pack.MddfMask257);
        WriteArray(outputStream, "modf_mask_257", pack.ModfMask257);
        WriteArray(outputStream, "object_filtered_mask_257", pack.ObjectFilteredMask257);
        WriteArray(outputStream, "pm4_path_mask", pack.Pm4PathMask);
        WriteArray(outputStream, "pm4_building_footprint_mask", pack.Pm4BuildingFootprintMask);
        WriteArray(outputStream, "pm4_mprl_mask", pack.Pm4MprlMask);
        WriteArray(outputStream, "mcsh_shadow_mask_256", pack.McshShadowMask256);
        WriteArray(outputStream, "shadow_residual_mask_256", pack.ShadowResidualMask256);
        WriteArray(outputStream, "object_roof_mask_256", pack.ObjectRoofMask256);
        WriteArray(outputStream, "object_roof_confidence_256", pack.ObjectRoofConfidence256);
        WriteArray(outputStream, "minimap_rgb_256", pack.MinimapRgb256);
        WriteArray(outputStream, "hole_mask_16", pack.HoleMask16);
        WriteArray(outputStream, "mtxf_animated_mask", pack.MtxfAnimatedMask);
        WriteArray(outputStream, "mtxf_transform_id", pack.MtxfTransformId);
        WriteArray(outputStream, "mcse_emitter_counts_16", pack.McseEmitterCounts16);
        WriteArray(outputStream, "mcse_entry_ids", pack.McseEntryIds);
        WriteArray(outputStream, "mcse_position_xyz", pack.McsePositionXyz);
        WriteArray(outputStream, "mcse_entry_bytes", pack.McseEntryBytes);
        WriteArray(outputStream, "mcrf_doodad_ref_counts_16", pack.McrfDoodadRefCounts16);
        WriteArray(outputStream, "mcrf_doodad_ref_indices", pack.McrfDoodadRefIndices);
        WriteArray(outputStream, "mcrf_wmo_ref_counts_16", pack.McrfWmoRefCounts16);
        WriteArray(outputStream, "mcrf_wmo_ref_indices", pack.McrfWmoRefIndices);
        WriteArray(outputStream, "mcrd_ref_counts_16", pack.McrdRefCounts16);
        WriteArray(outputStream, "mcrd_ref_indices", pack.McrdRefIndices);
        WriteArray(outputStream, "mcrw_ref_counts_16", pack.McrwRefCounts16);
        WriteArray(outputStream, "mcrw_ref_indices", pack.McrwRefIndices);
        WriteArray(outputStream, "placement_mddf_data", pack.PlacementMddfData);
        WriteArray(outputStream, "placement_modf_data", pack.PlacementModfData);

        if (pack.MclyTexturePixels is { Count: > 0 } pixels)
        {
            for (int i = 0; i < pixels.Count; i++)
                WriteArray(outputStream, $"mcly_texture_pixels_{i}", pixels[i]);
        }

        if (pack.RawChunks is { Count: > 0 })
        {
            foreach (var rawChunk in pack.RawChunks)
            {
                if (rawChunk.Data.Length > 0 && !string.IsNullOrWhiteSpace(rawChunk.EntryName))
                    WriteArray(outputStream, rawChunk.EntryName, rawChunk.Data);
            }
        }
    }

    private static void WriteV22Arrays(TerrainTileTensorPack pack, Stream outputStream)
    {
        WriteArray(outputStream, "height_257", pack.Height257);
        WriteArray(outputStream, "normal_xyz", pack.McnrNormalXyz);
        WriteArray(outputStream, "normal_mask", pack.McnrMask257 ?? BuildNormalMask(pack.McnrNormalXyz));
        WriteArray(outputStream, "alpha_256", pack.McalAlphaPack256);
        WriteArray(outputStream, "holes_16", pack.HoleMask16);
        WriteArray(outputStream, "liquid_mask", Crop257To256(pack.UnifiedLiquidMask));
        WriteArray(outputStream, "liquid_height", Crop257To256(pack.UnifiedLiquidHeight));
        WriteArray(outputStream, "object_mask", pack.ObjectMask257);
        WriteArray(outputStream, "object_precise_mask", pack.ObjectPreciseMask257);
        WriteArray(outputStream, "object_instance_mask", pack.ObjectInstanceMask257);
        WriteArray(outputStream, "mcnk_flags_16", pack.McnkFlags16);
        WriteArray(outputStream, "mddf_mask", pack.MddfMask257);
        WriteArray(outputStream, "modf_mask", pack.ModfMask257);
        WriteArray(outputStream, "object_filtered_mask", pack.ObjectFilteredMask257);
        WriteArray(outputStream, "object_roof_mask", pack.ObjectRoofMask256);
        WriteArray(outputStream, "object_roof_confidence", pack.ObjectRoofConfidence256);
        WriteArray(outputStream, "minimap_rgb", pack.MinimapRgb256);
        WriteArray(outputStream, "shadow_mask", pack.McshShadowMask256);
        WriteArray(outputStream, "mcly_texture_ids", pack.MclyTextureIds);
        WriteArray(outputStream, "mcly_layer_mask", BoolMaskToFloat(pack.MclyLayerMask));
        WriteArray(outputStream, "mcnr_mask_257", pack.McnrMask257);
        WriteArray(outputStream, "liquid_type_256", BuildLiquidType256(pack.LiquidBasicType257));
        WriteArray(outputStream, "ground_intent_height_257", BuildGroundIntentHeight257(pack.Height257, pack.ObjectPreciseMask257));
        WriteArray(outputStream, "mddf_placement_data", pack.PlacementMddfData);
        WriteArray(outputStream, "modf_placement_data", ConvertModfPlacementDataToV22(pack.PlacementModfData));
        WriteArray(outputStream, "mddf_unique_ids", ExtractPlacementColumnAsInt(pack.PlacementMddfData, 1));
        WriteArray(outputStream, "modf_unique_ids", ExtractPlacementColumnAsInt(pack.PlacementModfData, 1));
        WriteArray(outputStream, "mddf_model_ids", ExtractPlacementColumnAsInt(pack.PlacementMddfData, 0));
        WriteArray(outputStream, "modf_model_ids", ExtractPlacementColumnAsInt(pack.PlacementModfData, 0));
        WriteArray(outputStream, "mddf_count", new[] { pack.PlacementMddfCount });
        WriteArray(outputStream, "modf_count", new[] { pack.PlacementModfCount });

        if (pack.MclyTexturePixels is { Count: > 0 } pixels)
        {
            for (int i = 0; i < pixels.Count; i++)
                WriteArray(outputStream, $"tileset_texture_rgb_{i}", pixels[i]);
        }
    }

    private static void WriteArray(Stream stream, string name, Array? array)
    {
        if (array is null)
            return;

        // Name
        byte[] nameBytes = Encoding.UTF8.GetBytes(name);
        byte[] nameLen = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(nameLen, nameBytes.Length);
        stream.Write(nameLen);
        stream.Write(nameBytes);

        // Ndim
        byte[] ndimBuf = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(ndimBuf, array.Rank);
        stream.Write(ndimBuf);

        // Shape
        byte[] shapeBuf = new byte[array.Rank * 4];
        for (int r = 0; r < array.Rank; r++)
            BinaryPrimitives.WriteInt32LittleEndian(shapeBuf.AsSpan(r * 4), array.GetLength(r));
        stream.Write(shapeBuf);

        // Dtype (8 bytes ASCII, null-padded)
        byte[] dtype = new byte[8];
        string dtypeStr = GetDtypeString(array);
        Encoding.ASCII.GetBytes(dtypeStr, 0, Math.Min(dtypeStr.Length, 8), dtype, 0);
        stream.Write(dtype);

        // Raw data
        byte[] data = array is byte[] ba ? ba : FlattenArray(array);
        byte[] dataLen = new byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(dataLen, data.Length);
        stream.Write(dataLen);
        stream.Write(data);
    }

    private static string GetDtypeString(Array array)
    {
        Type t = array.GetType().GetElementType()!;
        if (t == typeof(float)) return "<f4";
        if (t == typeof(double)) return "<f8";
        if (t == typeof(int)) return "<i4";
        if (t == typeof(uint)) return "<u4";
        if (t == typeof(short)) return "<i2";
        if (t == typeof(ushort)) return "<u2";
        if (t == typeof(byte)) return "|u1";
        if (t == typeof(sbyte)) return "|i1";
        if (t == typeof(bool)) return "|b1";
        if (t == typeof(char)) return "|U1";
        return "|u1";
    }

    private static byte[] FlattenArray(Array array)
    {
        Type et = array.GetType().GetElementType()!;
        int total = Enumerable.Range(0, array.Rank).Aggregate(1, (a, r) => a * array.GetLength(r));

        if (et == typeof(float))
        {
            var result = new byte[total * 4];
            int idx = 0;
            foreach (float v in array)
                BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(idx++ * 4), BitConverter.SingleToInt32Bits(v));
            return result;
        }
        if (et == typeof(int))
        {
            var result = new byte[total * 4];
            int idx = 0;
            foreach (int v in array)
                BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(idx++ * 4), v);
            return result;
        }
        if (et == typeof(bool))
        {
            var result = new byte[total];
            int idx = 0;
            foreach (bool v in array)
                result[idx++] = v ? (byte)1 : (byte)0;
            return result;
        }
        if (et == typeof(byte))
        {
            // For multidimensional byte arrays, use Buffer.BlockCopy
            var result = new byte[total];
            Buffer.BlockCopy(array, 0, result, 0, total);
            return result;
        }

        // Fallback: convert to bytes via BitConverter
        var fallback = new byte[total * 8];
        int fi = 0;
        foreach (object v in array)
        {
            if (v is float f) BitConverter.GetBytes(f).CopyTo(fallback, fi * 4);
            else if (v is int i) BitConverter.GetBytes(i).CopyTo(fallback, fi * 4);
            else if (v is double d) BitConverter.GetBytes(d).CopyTo(fallback, fi * 8);
            fi++;
        }
        return fallback;
    }

    private static bool[,]? BuildNormalMask(float[,,]? normals)
    {
        if (normals is null)
            return null;

        int height = normals.GetLength(0);
        int width = normals.GetLength(1);
        int channels = normals.GetLength(2);
        if (channels < 3)
            return null;

        bool[,] mask = new bool[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                mask[y, x] = normals[y, x, 0] != 0f || normals[y, x, 1] != 0f || normals[y, x, 2] != 0f;
            }
        }

        return mask;
    }

    private static float[,]? Crop257To256(float[,]? source)
    {
        if (source is null)
            return null;

        int height = Math.Min(256, source.GetLength(0));
        int width = Math.Min(256, source.GetLength(1));
        float[,] result = new float[height, width];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                result[y, x] = source[y, x];
        return result;
    }

    private static byte[,]? BuildLiquidType256(byte[,]? liquidBasicType257)
    {
        if (liquidBasicType257 is null)
            return null;

        int height = Math.Min(256, liquidBasicType257.GetLength(0));
        int width = Math.Min(256, liquidBasicType257.GetLength(1));
        byte[,] result = new byte[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte value = liquidBasicType257[y, x];
                result[y, x] = value == 0xFF ? (byte)0 : (byte)(value + 1);
            }
        }

        return result;
    }

    private static float[,,]? BoolMaskToFloat(bool[,,]? source)
    {
        if (source is null)
            return null;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        float[,,] result = new float[dim0, dim1, dim2];
        for (int i = 0; i < dim0; i++)
            for (int j = 0; j < dim1; j++)
                for (int k = 0; k < dim2; k++)
                    result[i, j, k] = source[i, j, k] ? 1f : 0f;
        return result;
    }

    private static float[,]? BuildGroundIntentHeight257(float[,]? height257, float[,]? objectPreciseMask257)
    {
        if (height257 is null)
            return null;

        int height = height257.GetLength(0);
        int width = height257.GetLength(1);
        float[,] result = new float[height, width];
        bool[,] unresolved = new bool[height, width];
        int unresolvedCount = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                result[y, x] = height257[y, x];
                bool masked = objectPreciseMask257 is not null
                    && y < objectPreciseMask257.GetLength(0)
                    && x < objectPreciseMask257.GetLength(1)
                    && objectPreciseMask257[y, x] >= 0.05f;
                if (masked)
                {
                    unresolved[y, x] = true;
                    unresolvedCount++;
                }
            }
        }

        if (unresolvedCount == 0)
            return result;

        float[,] next = new float[height, width];
        bool[,] nextUnresolved = new bool[height, width];
        int maxIterations = height + width;
        for (int iteration = 0; iteration < maxIterations && unresolvedCount > 0; iteration++)
        {
            Array.Copy(result, next, result.Length);
            Array.Clear(nextUnresolved);
            int nextUnresolvedCount = 0;
            bool madeProgress = false;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (!unresolved[y, x])
                        continue;

                    float sum = 0f;
                    int count = 0;
                    AddResolvedNeighbor(x - 1, y, result, unresolved, ref sum, ref count);
                    AddResolvedNeighbor(x + 1, y, result, unresolved, ref sum, ref count);
                    AddResolvedNeighbor(x, y - 1, result, unresolved, ref sum, ref count);
                    AddResolvedNeighbor(x, y + 1, result, unresolved, ref sum, ref count);

                    if (count > 0)
                    {
                        next[y, x] = sum / count;
                        madeProgress = true;
                    }
                    else
                    {
                        nextUnresolved[y, x] = true;
                        nextUnresolvedCount++;
                    }
                }
            }

            (result, next) = (next, result);
            (unresolved, nextUnresolved) = (nextUnresolved, unresolved);
            unresolvedCount = nextUnresolvedCount;

            if (!madeProgress)
                break;
        }

        return result;
    }

    private static void AddResolvedNeighbor(int x, int y, float[,] values, bool[,] unresolved, ref float sum, ref int count)
    {
        if ((uint)y >= (uint)values.GetLength(0) || (uint)x >= (uint)values.GetLength(1) || unresolved[y, x])
            return;

        sum += values[y, x];
        count++;
    }

    private static float[,]? ConvertModfPlacementDataToV22(float[,]? source)
    {
        if (source is null)
            return null;

        int count = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] result = new float[count, 17];
        for (int i = 0; i < count; i++)
        {
            result[i, 0] = GetPlacementValue(source, i, columns, 0);
            result[i, 1] = GetPlacementValue(source, i, columns, 1);
            result[i, 2] = GetPlacementValue(source, i, columns, 2);
            result[i, 3] = GetPlacementValue(source, i, columns, 3);
            result[i, 4] = GetPlacementValue(source, i, columns, 4);
            result[i, 5] = GetPlacementValue(source, i, columns, 5);
            result[i, 6] = GetPlacementValue(source, i, columns, 6);
            result[i, 7] = GetPlacementValue(source, i, columns, 7);
            result[i, 8] = columns > 14 ? GetPlacementValue(source, i, columns, 8) : 0f;
            result[i, 9] = columns > 15 ? GetPlacementValue(source, i, columns, 9) : 0f;
            result[i, 10] = columns > 16 ? GetPlacementValue(source, i, columns, 10) : 0f;

            int boundsOffset = columns >= 17 ? 11 : 8;
            result[i, 11] = GetPlacementValue(source, i, columns, boundsOffset);
            result[i, 12] = GetPlacementValue(source, i, columns, boundsOffset + 1);
            result[i, 13] = GetPlacementValue(source, i, columns, boundsOffset + 2);
            result[i, 14] = GetPlacementValue(source, i, columns, boundsOffset + 3);
            result[i, 15] = GetPlacementValue(source, i, columns, boundsOffset + 4);
            result[i, 16] = GetPlacementValue(source, i, columns, boundsOffset + 5);
        }

        return result;
    }

    private static int[]? ExtractPlacementColumnAsInt(float[,]? source, int column)
    {
        if (source is null || source.GetLength(1) <= column)
            return null;

        int count = source.GetLength(0);
        int[] result = new int[count];
        for (int i = 0; i < count; i++)
            result[i] = (int)source[i, column];
        return result;
    }

    private static float GetPlacementValue(float[,] source, int row, int columns, int column)
        => column >= 0 && column < columns ? source[row, column] : 0f;

    private static string BuildMetadataJson(TerrainTileTensorPack pack)
    {
        var sb = new StringBuilder();
        sb.Append('{');
        sb.Append($"\"tile_name\":\"{Escape(pack.TileName)}\",");
        sb.Append($"\"map_name\":\"{Escape(pack.MapName)}\",");
        sb.Append($"\"tile_x\":{pack.TileX},");
        sb.Append($"\"tile_y\":{pack.TileY},");
        sb.Append($"\"build_key\":\"{Escape(pack.BuildKey)}\",");
        sb.Append($"\"source_adt_path\":\"{Escape(pack.SourceAdtPath)}\",");

        // available_signals
        sb.Append("\"available_signals\":[");
        bool first = true;
        foreach (string s in pack.AvailableSignals.OrderBy(static x => x, StringComparer.OrdinalIgnoreCase))
        {
            if (!first) sb.Append(',');
            sb.Append($"\"{Escape(s)}\"");
            first = false;
        }
        sb.Append("],");

        // mcly_texture_names
        sb.Append("\"mcly_texture_names\":[");
        first = true;
        foreach (string n in pack.MclyTextureNames)
        {
            if (!first) sb.Append(',');
            sb.Append($"\"{Escape(n)}\"");
            first = false;
        }
        sb.Append("],");

        sb.Append("\"mtex_texture_paths\":[");
        first = true;
        foreach (string n in pack.MclyTextureNames)
        {
            if (!first) sb.Append(',');
            sb.Append($"\"{Escape(n)}\"");
            first = false;
        }
        sb.Append("],");

        // placement names
        sb.Append("\"placement_mddf_names\":[");
        first = true;
        foreach (string n in pack.PlacementMddfNames)
        {
            if (!first) sb.Append(',');
            sb.Append($"\"{Escape(n)}\"");
            first = false;
        }
        sb.Append("],");
        sb.Append("\"placement_modf_names\":[");
        first = true;
        foreach (string n in pack.PlacementModfNames)
        {
            if (!first) sb.Append(',');
            sb.Append($"\"{Escape(n)}\"");
            first = false;
        }
        sb.Append("],");

        AppendPlacementAssetPaths(sb, "placement_mddf_asset_paths", pack.PlacementMddfData, pack.PlacementMddfNames);
        sb.Append(',');
        AppendPlacementAssetPaths(sb, "placement_modf_asset_paths", pack.PlacementModfData, pack.PlacementModfNames);
        sb.Append(',');

        sb.Append($"\"object_roof_mask_source\":\"{Escape(pack.ObjectRoofMaskSource)}\",");

        sb.Append($"\"placement_mddf_count\":{pack.PlacementMddfCount},");
        sb.Append($"\"placement_modf_count\":{pack.PlacementModfCount}");

        sb.Append('}');
        return sb.ToString();
    }

    private static void AppendPlacementAssetPaths(StringBuilder sb, string propertyName, float[,]? placementData, IReadOnlyList<string> names)
    {
        sb.Append('"').Append(propertyName).Append("\":[");
        if (placementData is not null)
        {
            for (int i = 0; i < placementData.GetLength(0); i++)
            {
                if (i > 0)
                    sb.Append(',');

                int nameId = placementData.GetLength(1) > 0 ? (int)placementData[i, 0] : -1;
                string path = nameId >= 0 && nameId < names.Count ? names[nameId] : string.Empty;
                sb.Append('"').Append(Escape(path)).Append('"');
            }
        }
        sb.Append(']');
    }

    private static string Escape(string? s) => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
}
