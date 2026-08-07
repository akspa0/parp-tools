using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
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
        // All three profiles now serialize the strict object-geometry union targets (Spec 118 added
        // them to V22), so validate for every profile.
        ValidateStrictUnionTargetSerialization(pack, writesStrictUnionArrays: true);

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

    /// <summary>
    /// Write one generic inner blob (same "ARRY" + metadata JSON + named arrays + "ENDS" format
    /// as <see cref="Serialize(TerrainTileTensorPack,Stream,StreamProfile)"/>) for callers with no
    /// <see cref="TerrainTileTensorPack"/> -- e.g. Spec 118's per-object capture records
    /// (<c>image_rgb</c>/<c>mask</c> arrays keyed by asset metadata rather than a tile). Reuses the
    /// same <see cref="WriteArray"/> writer, so every consumer of the harvest-stream wire format
    /// (``harvester.raw_reader.read_tile_blob``) decodes this with zero format-specific changes.
    /// </summary>
    public static void SerializeGeneric(
        IReadOnlyDictionary<string, object?> metadata,
        IEnumerable<(string Name, Array Array)> arrays,
        Stream outputStream)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(arrays);
        ArgumentNullException.ThrowIfNull(outputStream);

        outputStream.Write(ArrayMagic);

        string metadataJson = JsonSerializer.Serialize(metadata);
        byte[] metaBytes = Encoding.UTF8.GetBytes(metadataJson);
        byte[] metaLen = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(metaLen, metaBytes.Length);
        outputStream.Write(metaLen);
        outputStream.Write(metaBytes);

        foreach ((string name, Array array) in arrays)
            WriteArray(outputStream, name, array);

        outputStream.Write(EndsMagic);
    }

    /// <summary>
    /// Strict union labels are meaningful only when their provenance says the
    /// target was materialized. Do not inspect fragment-trace sidecars here:
    /// incomplete geometry may legitimately retain those audit records.
    /// </summary>
    private static void ValidateStrictUnionTargetSerialization(
        TerrainTileTensorPack pack,
        bool writesStrictUnionArrays)
    {
        if (!writesStrictUnionArrays)
            return;

        // A fragment trace is permitted for an incomplete target, but when it
        // is present it must still be structurally and cryptographically
        // coherent before we write any bytes.
        ValidateStrictFragmentTrace(pack);

        bool hasMask = pack.ObjectGeometryVisibleMask257 is not null;
        bool hasTopElevation = pack.ObjectGeometryVisibleTopElevation257 is not null;
        bool hasTerrainElevation = pack.ObjectGeometryVisibleTerrainElevation257 is not null;
        bool hasSource = pack.ObjectGeometryVisibleSource257 is not null;
        bool hasAnyUnionArray = hasMask || hasTopElevation || hasTerrainElevation || hasSource;
        ObjectGeometryTargetProvenance? provenance = pack.ObjectGeometryTargetProvenance;

        if (!hasAnyUnionArray)
        {
            if (provenance?.IsMaterialized == true)
            {
                throw new InvalidDataException(
                    "A materialized strict object target must include all four union target arrays.");
            }

            return;
        }

        if (provenance is null)
        {
            throw new InvalidDataException(
                "Strict object union target arrays require explicit materialized provenance.");
        }

        if (!provenance.IsMaterialized)
        {
            throw new InvalidDataException(
                $"Strict object union target arrays cannot be serialized for nonmaterialized status {provenance.Status}. "+
                "Retain only provenance and optional fragment-trace sidecars.");
        }

        if (!hasMask || !hasTopElevation || !hasTerrainElevation || !hasSource)
        {
            throw new InvalidDataException(
                "A materialized strict object target must include mask, top elevation, terrain elevation, and source arrays together.");
        }

    }

    private static void ValidateStrictFragmentTrace(TerrainTileTensorPack pack)
    {
        ObjectGeometryTargetProvenance? provenance = pack.ObjectGeometryTargetProvenance;
        ObjectGeometryFragmentTrace? trace = pack.ObjectGeometryFragmentTrace;
        if (trace is null)
        {
            if (provenance?.IsMaterialized == true)
            {
                throw new InvalidDataException(
                    "A materialized strict object target requires its v3 uncollapsed fragment trace.");
            }

            return;
        }
        if (provenance is null)
            throw new InvalidDataException("A strict object fragment trace requires strict target provenance.");

        int count = trace.Count;
        ValidateTraceShape(trace.RasterXy, count, 2, nameof(trace.RasterXy));
        ValidateTraceShape(trace.ObjectWorldXyzElevation, count, 4, nameof(trace.ObjectWorldXyzElevation));
        ValidateTraceShape(trace.SourceIds, count, 3, nameof(trace.SourceIds));
        ValidateTraceShape(trace.SourceClassification, count, 2, nameof(trace.SourceClassification));
        ValidateTraceShape(trace.TerrainVertexDenseXy, count, 6, nameof(trace.TerrainVertexDenseXy));
        ValidateTraceShape(trace.TerrainVertexZ, count, 3, nameof(trace.TerrainVertexZ));
        ValidateTraceShape(trace.TerrainVertexPresent, count, 3, nameof(trace.TerrainVertexPresent));
        ValidateTraceShape(trace.TerrainWeights, count, 3, nameof(trace.TerrainWeights));
        ValidateTraceShape(trace.TerrainLiquidElevation, count, 2, nameof(trace.TerrainLiquidElevation));
        if (trace.ContentSha256.Length != 64 || !trace.ContentSha256.All(static c => char.IsAsciiHexDigit(c)))
            throw new InvalidDataException("Strict object fragment trace must carry a SHA-256 content hash.");
        ValidateStrictFragmentAssetTable(pack.ObjectGeometryTargetAssets);
        if (!trace.HasConsistentContentHash(pack.ObjectGeometryTargetAssets))
            throw new InvalidDataException("Strict object fragment trace content does not match its SHA-256 hash.");
        if (provenance.IsMaterialized
            && provenance.Status == ObjectGeometryTargetStatus.CompleteVisible
            && count == 0)
            throw new InvalidDataException("A visible strict union target cannot have an empty fragment trace.");

        for (int row = 0; row < count; row++)
        {
            int assetIndex = trace.SourceIds[row, 1];
            if (assetIndex < 0 || assetIndex >= pack.ObjectGeometryTargetAssets.Count)
                throw new InvalidDataException($"Strict fragment row {row} references missing asset index {assetIndex}.");
            ObjectGeometryPixelSource source = (ObjectGeometryPixelSource)trace.SourceClassification[row, 0];
            if (source == ObjectGeometryPixelSource.None || !Enum.IsDefined(source))
                throw new InvalidDataException($"Strict fragment row {row} has unknown source {trace.SourceClassification[row, 0]}.");
            if (pack.ObjectGeometryTargetAssets[assetIndex].Source != source)
            {
                throw new InvalidDataException(
                    $"Strict fragment row {row} source does not match asset index {assetIndex}.");
            }
            byte classification = trace.SourceClassification[row, 1];
            if (!Enum.IsDefined((ObjectGeometryFragmentClassification)classification))
                throw new InvalidDataException($"Strict fragment row {row} has unknown classification {classification}.");
        }
    }

    private static void ValidateStrictFragmentAssetTable(IReadOnlyList<ObjectGeometryTargetAsset> assets)
    {
        for (int index = 0; index < assets.Count; index++)
        {
            ObjectGeometryTargetAsset asset = assets[index];
            if (asset.AssetIndex != index)
                throw new InvalidDataException($"Strict fragment asset table index {index} has declared id {asset.AssetIndex}.");
            if (asset.Source == ObjectGeometryPixelSource.None || !Enum.IsDefined(asset.Source))
                throw new InvalidDataException($"Strict fragment asset index {index} has an unknown source.");
            if (string.IsNullOrWhiteSpace(asset.NormalizedAssetPath))
                throw new InvalidDataException($"Strict fragment asset index {index} has no normalized asset path.");
        }
    }

    private static void ValidateTraceShape(Array array, int rows, int columns, string name)
    {
        if (array.Rank != 2 || array.GetLength(0) != rows || array.GetLength(1) != columns)
            throw new InvalidDataException($"Strict fragment trace field {name} must have shape [{rows},{columns}].");
    }

    private static void WriteStrictObjectGeometryFragmentTrace(
        TerrainTileTensorPack pack,
        Stream outputStream)
    {
        ObjectGeometryFragmentTrace? trace = pack.ObjectGeometryFragmentTrace;
        if (trace is null)
            return;

        WriteArray(outputStream, "object_geometry_fragment_raster_xy", trace.RasterXy);
        WriteArray(outputStream, "object_geometry_fragment_world_xyze", trace.ObjectWorldXyzElevation);
        WriteArray(outputStream, "object_geometry_fragment_source_ids", trace.SourceIds);
        WriteArray(outputStream, "object_geometry_fragment_source_classification", trace.SourceClassification);
        WriteArray(outputStream, "object_geometry_fragment_terrain_vertex_dense_xy", trace.TerrainVertexDenseXy);
        WriteArray(outputStream, "object_geometry_fragment_terrain_vertex_z", trace.TerrainVertexZ);
        WriteArray(outputStream, "object_geometry_fragment_terrain_vertex_present", trace.TerrainVertexPresent);
        WriteArray(outputStream, "object_geometry_fragment_terrain_weights", trace.TerrainWeights);
        WriteArray(outputStream, "object_geometry_fragment_terrain_liquid_elevation", trace.TerrainLiquidElevation);
    }

    private static void WriteV16Arrays(TerrainTileTensorPack pack, Stream outputStream)
    {
        WriteArray(outputStream, "height_257", pack.Height257);
        WriteTerrainVertexArrays(pack, outputStream);
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
        WriteArray(outputStream, "object_geometry_visible_mask_257", pack.ObjectGeometryVisibleMask257);
        WriteArray(outputStream, "object_geometry_visible_top_elevation_257", pack.ObjectGeometryVisibleTopElevation257);
        WriteArray(outputStream, "object_geometry_visible_terrain_elevation_257", pack.ObjectGeometryVisibleTerrainElevation257);
        WriteArray(outputStream, "object_geometry_visible_source_257", pack.ObjectGeometryVisibleSource257);
        WriteArray(outputStream, "object_geometry_visible_instance_257", pack.ObjectGeometryVisibleInstance257);
        WriteStrictObjectGeometryFragmentTrace(pack, outputStream);
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
        WriteTerrainVertexArrays(pack, outputStream);
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
        WriteArray(outputStream, "object_geometry_visible_mask_257", pack.ObjectGeometryVisibleMask257);
        WriteArray(outputStream, "object_geometry_visible_top_elevation_257", pack.ObjectGeometryVisibleTopElevation257);
        WriteArray(outputStream, "object_geometry_visible_terrain_elevation_257", pack.ObjectGeometryVisibleTerrainElevation257);
        WriteArray(outputStream, "object_geometry_visible_source_257", pack.ObjectGeometryVisibleSource257);
        WriteArray(outputStream, "object_geometry_visible_instance_257", pack.ObjectGeometryVisibleInstance257);
        WriteStrictObjectGeometryFragmentTrace(pack, outputStream);
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
        WriteTerrainVertexArrays(pack, outputStream);
        WriteArray(outputStream, "normal_xyz", pack.McnrNormalXyz);
        WriteArray(outputStream, "normal_mask", pack.McnrMask257 ?? BuildNormalMask(pack.McnrNormalXyz));
        WriteArray(outputStream, "alpha_256", pack.McalAlphaPack256);
        WriteArray(outputStream, "holes_16", pack.HoleMask16);
        WriteArray(outputStream, "liquid_mask", Crop257To256(pack.UnifiedLiquidMask));
        WriteArray(outputStream, "liquid_height", Crop257To256(pack.UnifiedLiquidHeight));
        WriteArray(outputStream, "object_mask", pack.ObjectMask257);
        WriteArray(outputStream, "object_precise_mask", pack.ObjectPreciseMask257);
        WriteArray(outputStream, "object_instance_mask", pack.ObjectInstanceMask257);
        // Strict occlusion-aware object-geometry union targets (Spec 118), under the SAME canonical
        // _257 keys the v50 catalog expects -- identical to WriteV16/WriteFullArrays. V22 is the
        // proven v50 dataset profile (its normal_xyz/minimap_rgb names match the catalog), so these
        // arrays are added here to make V22 a true superset instead of forcing a switch to Full,
        // whose renamed core signals silently zero-filled normal_xyz.
        WriteArray(outputStream, "object_geometry_visible_mask_257", pack.ObjectGeometryVisibleMask257);
        WriteArray(outputStream, "object_geometry_visible_top_elevation_257", pack.ObjectGeometryVisibleTopElevation257);
        WriteArray(outputStream, "object_geometry_visible_terrain_elevation_257", pack.ObjectGeometryVisibleTerrainElevation257);
        WriteArray(outputStream, "object_geometry_visible_source_257", pack.ObjectGeometryVisibleSource257);
        WriteArray(outputStream, "object_geometry_visible_instance_257", pack.ObjectGeometryVisibleInstance257);
        WriteStrictObjectGeometryFragmentTrace(pack, outputStream);
        WriteArray(outputStream, "mcnk_flags_16", pack.McnkFlags16);
        WriteArray(outputStream, "mddf_mask", pack.MddfMask257);
        WriteArray(outputStream, "modf_mask", pack.ModfMask257);
        WriteArray(outputStream, "object_filtered_mask", pack.ObjectFilteredMask257);
        WriteArray(outputStream, "model_focus_mask", pack.ObjectFilteredMask257);
        WriteArray(outputStream, "object_roof_mask", pack.ObjectRoofMask256);
        WriteArray(outputStream, "object_roof_confidence", pack.ObjectRoofConfidence256);
        WriteArray(outputStream, "minimap_rgb", pack.MinimapRgb256);
        WriteArray(outputStream, "terrain_shadow_256", pack.TerrainShadow256);
        WriteArray(outputStream, "shadow_mask", pack.McshShadowMask256);
        WriteArray(outputStream, "mcly_texture_ids", pack.MclyTextureIds);
        WriteArray(outputStream, "mcly_layer_mask", BoolMaskToFloat(pack.MclyLayerMask));
        if (HasNameAlignedTexturePixels(pack))
        {
            IReadOnlyList<byte[,,]> texturePixels = pack.MclyTexturePixels!;
            for (int index = 0; index < texturePixels.Count; index++)
                WriteArray(outputStream, $"tileset_texture_rgb_{index}", texturePixels[index]);
        }
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
        WriteArray(outputStream, "model_above_terrain_mask",
            BuildModelAboveTerrainMask(pack.PlacementMddfData, pack.PlacementModfData, pack.Height257,
                pack.TileX ?? 0, pack.TileY ?? 0));
        WriteV22ModelPayloads(pack, outputStream);
    }

    /// <summary>
    /// Writes the model library sidecars promised by the V22 stream contract.  The stable hash
    /// keeps the raw-array namespace compact while metadata retains the canonical path and kind
    /// needed to resolve it.  This is dataset evidence only; it does not imply a renderer route.
    /// </summary>
    private static void WriteV22ModelPayloads(TerrainTileTensorPack pack, Stream outputStream)
    {
        foreach (V22ModelStreamRecord record in BuildV22ModelStreamRecords(pack))
        {
            string prefix = $"{record.Kind}_model_{record.StableKey}";
            WriteArray(outputStream, $"{prefix}_load_error", new[] { record.Payload.LoadError });

            foreach ((string arrayName, Array array) in record.Payload.RawArrays
                         .OrderBy(static entry => entry.Key, StringComparer.Ordinal))
            {
                WriteArray(outputStream, $"{prefix}_{arrayName}", array);
            }
        }
    }

    private static void WriteTerrainVertexArrays(TerrainTileTensorPack pack, Stream outputStream)
    {
        WriteArray(outputStream, "mcvt_vertex_z", pack.TerrainVertices?.VertexZ);
        WriteArray(outputStream, "mcvt_vertex_world_x", pack.TerrainVertices?.WorldX);
        WriteArray(outputStream, "mcvt_vertex_world_y", pack.TerrainVertices?.WorldY);
        WriteArray(outputStream, "mcvt_triangle_indices", pack.TerrainVertices is null ? null : TerrainVertexLattice.ChunkTriangleIndices);
        WriteArray(outputStream, "mcvt_vertex_present", pack.TerrainVertices?.Present);
        WriteArray(outputStream, "mcvt_vertex_mask_257", pack.TerrainVertices?.DenseValidMask);
        WriteArray(outputStream, "wdl_outer_17", pack.WdlLattice?.Outer17);
        WriteArray(outputStream, "wdl_inner_16", pack.WdlLattice?.Inner16);
        WriteArray(outputStream, "wdl_outer_present", pack.WdlLattice?.OuterPresent);
        WriteArray(outputStream, "wdl_inner_present", pack.WdlLattice?.InnerPresent);
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
        if (t == typeof(long)) return "<i8";
        if (t == typeof(ulong)) return "<u8";
        if (t == typeof(short)) return "<i2";
        if (t == typeof(ushort)) return "<u2";
        if (t == typeof(byte)) return "|u1";
        if (t == typeof(sbyte)) return "|i1";
        if (t == typeof(bool)) return "|b1";
        if (t == typeof(char)) return "<u2";
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
        if (et == typeof(uint))
        {
            var result = new byte[total * 4];
            int idx = 0;
            foreach (uint v in array)
                BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(idx++ * 4), v);
            return result;
        }
        if (et == typeof(long))
        {
            var result = new byte[total * 8];
            int idx = 0;
            foreach (long v in array)
                BinaryPrimitives.WriteInt64LittleEndian(result.AsSpan(idx++ * 8), v);
            return result;
        }
        if (et == typeof(ulong))
        {
            var result = new byte[total * 8];
            int idx = 0;
            foreach (ulong v in array)
                BinaryPrimitives.WriteUInt64LittleEndian(result.AsSpan(idx++ * 8), v);
            return result;
        }
        if (et == typeof(short))
        {
            var result = new byte[total * 2];
            int idx = 0;
            foreach (short v in array)
                BinaryPrimitives.WriteInt16LittleEndian(result.AsSpan(idx++ * 2), v);
            return result;
        }
        if (et == typeof(ushort) || et == typeof(char))
        {
            var result = new byte[total * 2];
            int idx = 0;
            foreach (object value in array)
            {
                ushort v = value is char c ? c : (ushort)value;
                BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(idx++ * 2), v);
            }
            return result;
        }
        if (et == typeof(double))
        {
            var result = new byte[total * 8];
            int idx = 0;
            foreach (double v in array)
                BinaryPrimitives.WriteInt64LittleEndian(result.AsSpan(idx++ * 8), BitConverter.DoubleToInt64Bits(v));
            return result;
        }
        if (et == typeof(sbyte))
        {
            var result = new byte[total];
            int idx = 0;
            foreach (sbyte v in array)
                result[idx++] = unchecked((byte)v);
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

        throw new NotSupportedException($"Raw array serialization does not support element type {et.FullName}.");
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

    private static float[,]? BuildModelAboveTerrainMask(
        float[,]? mddfData, float[,]? modfData, float[,]? height257, int tileX, int tileY)
    {
        if (height257 is null) return null;
        if (mddfData is null && modfData is null) return null;

        const int tileSize = 257;
        const float mapOrigin = 17066.666f;
        const float tileWorldSize = 533.33333f;
        const float epsilon = 1.0f;

        int mddfCount = mddfData?.GetLength(0) ?? 0;
        int modfCount = modfData?.GetLength(0) ?? 0;

        float[,] mask = new float[tileSize, tileSize];

        bool TryProjectToPixel(float posX, float posY, float posZ, out int px, out int py)
        {
            px = 0; py = 0;
            (float U, float V)[] candidates =
            [
                ((posX / tileWorldSize) - tileX, (posZ / tileWorldSize) - tileY),
                (((mapOrigin - posZ) / tileWorldSize) - tileX, ((mapOrigin - posX) / tileWorldSize) - tileY),
                ((posX / tileWorldSize) - tileX, (posY / tileWorldSize) - tileY),
                (((mapOrigin - posY) / tileWorldSize) - tileX, ((mapOrigin - posX) / tileWorldSize) - tileY),
            ];

            float bestOverflow = float.PositiveInfinity;
            (float U, float V) best = default;
            bool found = false;

            foreach ((float U, float V) candidate in candidates)
            {
                float overflow =
                    MathF.Max(0f, -candidate.U) + MathF.Max(0f, candidate.U - 1f) +
                    MathF.Max(0f, -candidate.V) + MathF.Max(0f, candidate.V - 1f);
                if (overflow < bestOverflow)
                {
                    bestOverflow = overflow;
                    best = candidate;
                    found = true;
                    if (overflow <= 0.000001f)
                        break;
                }
            }

            if (!found) return false;

            px = (int)(best.U * (tileSize - 1));
            py = (int)(best.V * (tileSize - 1));
            return px >= 0 && px < tileSize && py >= 0 && py < tileSize;
        }

        void ProcessPlacement(float[,] data, int row)
        {
            int cols = data.GetLength(1);
            float posX = data[row, 2];
            float posY = data[row, 3];
            float posZ = data[row, 4];

            if (!TryProjectToPixel(posX, posY, posZ, out int px, out int py))
                return;

            float terrainZ = height257[py, px];
            if (posZ >= terrainZ - epsilon)
                mask[py, px] = 1.0f;
        }

        for (int i = 0; i < mddfCount; i++)
            ProcessPlacement(mddfData!, i);
        for (int i = 0; i < modfCount; i++)
            ProcessPlacement(modfData!, i);

        return mask;
    }

    private static string BuildMetadataJson(TerrainTileTensorPack pack)
    {
        IReadOnlyList<V22ModelStreamRecord> modelRecords = BuildV22ModelStreamRecords(pack);
        return JsonSerializer.Serialize(new
        {
            tile_name = pack.TileName,
            map_name = pack.MapName,
            tile_x = pack.TileX,
            tile_y = pack.TileY,
            build_key = pack.BuildKey,
            source_adt_path = pack.SourceAdtPath,
            available_signals = pack.AvailableSignals.OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).ToArray(),
            mcly_texture_names = pack.MclyTextureNames,
            mtex_texture_paths = pack.MclyTextureNames,
            mtex_texture_payload_state = GetTexturePayloadState(pack),
            mtex_texture_fallbacks = pack.MinimapTextureFallbacks
                .OrderBy(static entry => entry.Key)
                .Select(static entry => new
                {
                    texture_id = entry.Value.TextureId,
                    requested_path = entry.Value.RequestedPath,
                    resolved_path = entry.Value.ResolvedPath,
                    resolution_kind = entry.Value.ResolutionKind,
                })
                .ToArray(),
            placement_mddf_names = pack.PlacementMddfNames,
            placement_modf_names = pack.PlacementModfNames,
            placement_mddf_asset_paths = BuildPlacementAssetPaths(pack.PlacementMddfData, pack.PlacementMddfNames),
            placement_modf_asset_paths = BuildPlacementAssetPaths(pack.PlacementModfData, pack.PlacementModfNames),
            object_roof_mask_source = pack.ObjectRoofMaskSource,
            object_geometry_target_version = ObjectGeometryTargetProvenance.ContractVersion,
            object_geometry_target_status = pack.ObjectGeometryTargetProvenance?.Status.ToString(),
            object_geometry_target_materialized = pack.ObjectGeometryTargetProvenance?.IsMaterialized ?? false,
            object_geometry_target_placement_count = pack.ObjectGeometryTargetProvenance?.PlacementCount,
            object_geometry_target_geometry_resolved_placement_count = pack.ObjectGeometryTargetProvenance?.GeometryResolvedPlacementCount,
            object_geometry_target_geometry_unresolved_placement_count = pack.ObjectGeometryTargetProvenance?.GeometryUnresolvedPlacementCount,
            object_geometry_target_fallback_required_placement_count = pack.ObjectGeometryTargetProvenance?.FallbackRequiredPlacementCount,
            object_geometry_target_triangle_count = pack.ObjectGeometryTargetProvenance?.TriangleCount,
            object_geometry_target_visible_pixel_count = pack.ObjectGeometryTargetProvenance?.VisiblePixelCount,
            object_geometry_target_occluded_pixel_count = pack.ObjectGeometryTargetProvenance?.OccludedPixelCount,
            object_geometry_target_terrain_unknown_pixel_count = pack.ObjectGeometryTargetProvenance?.TerrainUnknownPixelCount,
            object_geometry_target_liquid_evidence_status = pack.ObjectGeometryTargetProvenance?.LiquidEvidenceStatus.ToString(),
            object_geometry_target_liquid_covered_pixel_count = pack.ObjectGeometryTargetProvenance?.LiquidCoveredPixelCount,
            object_geometry_target_liquid_surface_unknown_pixel_count = pack.ObjectGeometryTargetProvenance?.LiquidSurfaceUnknownPixelCount,
            object_geometry_target_liquid_covered_fragment_count = pack.ObjectGeometryTargetProvenance?.LiquidCoveredFragmentCount,
            object_geometry_target_liquid_hidden_fragment_count = pack.ObjectGeometryTargetProvenance?.LiquidHiddenFragmentCount,
            object_geometry_target_liquid_above_surface_fragment_count = pack.ObjectGeometryTargetProvenance?.LiquidAboveSurfaceFragmentCount,
            object_geometry_target_liquid_unknown_fragment_count = pack.ObjectGeometryTargetProvenance?.LiquidUnknownFragmentCount,
            object_geometry_fragment_trace_schema = pack.ObjectGeometryFragmentTrace is null
                ? null
                : ObjectGeometryTargetProvenance.ContractVersion,
            object_geometry_fragment_count = pack.ObjectGeometryFragmentTrace?.Count,
            object_geometry_fragment_sha256 = pack.ObjectGeometryFragmentTrace?.ContentSha256,
            object_geometry_target_assets = pack.ObjectGeometryTargetAssets.Select(static asset => new
            {
                asset_index = asset.AssetIndex,
                source = asset.Source.ToString(),
                normalized_asset_path = asset.NormalizedAssetPath,
            }),
            object_geometry_target_unresolved_placements = pack.ObjectGeometryTargetUnresolvedPlacements.Select(static placement => new
            {
                placement_unique_id = placement.PlacementUniqueId,
                source = placement.Source.ToString(),
                normalized_asset_path = placement.NormalizedAssetPath,
                reason = placement.Reason,
            }),
            object_geometry_visible_instances = pack.ObjectGeometryVisibleInstances.Select(static instance => new
            {
                instance_id = instance.InstanceId,
                placement_unique_id = instance.PlacementUniqueId,
                asset_index = instance.AssetIndex,
                source = instance.Source.ToString(),
                visible_pixel_count = instance.VisiblePixelCount,
            }),
            placement_mddf_count = pack.PlacementMddfCount,
            placement_modf_count = pack.PlacementModfCount,
            tile_model_paths = modelRecords.Select(static record => record.CanonicalPath).ToArray(),
            tile_model_kinds = modelRecords.Select(static record => record.Kind).ToArray(),
            tile_model_payloads = modelRecords.Select(static record => new
            {
                canonical_path = record.CanonicalPath,
                kind = record.Kind,
                stable_key = record.StableKey,
                load_error = record.Payload.LoadError,
            }).ToArray(),
            minimap_source_tag = pack.MinimapSourceTag,
            minimap_lighting = pack.MinimapLightingProvenance?.ToMetadata(),
        });
    }

    private static IReadOnlyList<V22ModelStreamRecord> BuildV22ModelStreamRecords(TerrainTileTensorPack pack)
    {
        if (pack.PerTileModelPayloads is not { Count: > 0 } payloads)
            return [];

        return payloads
            .Select(static entry =>
            {
                string canonicalPath = string.IsNullOrWhiteSpace(entry.Value.CanonicalPath)
                    ? entry.Key
                    : entry.Value.CanonicalPath;
                string kind = entry.Value.Kind switch
                {
                    V22ModelPayload.ModelKind.M2 => "m2",
                    V22ModelPayload.ModelKind.Wmo => "wmo",
                    _ => "unknown",
                };
                return new V22ModelStreamRecord(
                    canonicalPath,
                    kind,
                    BuildStableModelKey(canonicalPath),
                    entry.Value);
            })
            .OrderBy(static record => record.CanonicalPath, StringComparer.OrdinalIgnoreCase)
            .ThenBy(static record => record.CanonicalPath, StringComparer.Ordinal)
            .ToArray();
    }

    private static string BuildStableModelKey(string canonicalPath)
    {
        string normalizedPath = canonicalPath.Replace('\\', '/').Trim().ToLowerInvariant();
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(normalizedPath));
        return Convert.ToHexString(digest.AsSpan(0, 4)).ToLowerInvariant();
    }

    private sealed record V22ModelStreamRecord(
        string CanonicalPath,
        string Kind,
        string StableKey,
        V22ModelPayload Payload);

    private static bool HasNameAlignedTexturePixels(TerrainTileTensorPack pack)
        => pack.MclyTexturePixels is { Count: > 0 } pixels
           && pixels.Count == pack.MclyTextureNames.Count;

    private static string GetTexturePayloadState(TerrainTileTensorPack pack)
    {
        if (pack.MclyTextureNames.Count == 0)
            return "no_mtex_textures";
        if (pack.MclyTexturePixels is not { Count: > 0 })
            return "not_available";
        return HasNameAlignedTexturePixels(pack)
            ? pack.MinimapTextureFallbacks.Count > 0
                ? "complete_name_aligned_with_rgb_proxy"
                : "complete_name_aligned"
            : "incomplete_not_serialized";
    }

    private static string[] BuildPlacementAssetPaths(float[,]? placementData, IReadOnlyList<string> names)
    {
        if (placementData is null)
            return [];

        string[] paths = new string[placementData.GetLength(0)];
        for (int i = 0; i < paths.Length; i++)
        {
            int nameId = placementData.GetLength(1) > 0 ? (int)placementData[i, 0] : -1;
            paths[i] = nameId >= 0 && nameId < names.Count ? names[nameId] : string.Empty;
        }
        return paths;
    }
}
