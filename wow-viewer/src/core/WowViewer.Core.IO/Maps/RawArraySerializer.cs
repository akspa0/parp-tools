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
        V16
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

        sb.Append($"\"object_roof_mask_source\":\"{Escape(pack.ObjectRoofMaskSource)}\",");

        sb.Append($"\"placement_mddf_count\":{pack.PlacementMddfCount},");
        sb.Append($"\"placement_modf_count\":{pack.PlacementModfCount}");

        sb.Append('}');
        return sb.ToString();
    }

    private static string Escape(string? s) => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
}
