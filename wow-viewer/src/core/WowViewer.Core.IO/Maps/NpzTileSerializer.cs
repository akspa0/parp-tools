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
        ValidateStrictUnionTargetSerialization(pack);

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using FileStream fs = File.Create(outputPath);
        SerializeValidated(pack, fs);
    }

    public static void Serialize(TerrainTileTensorPack pack, Stream outputStream)
    {
        ArgumentNullException.ThrowIfNull(pack);
        ArgumentNullException.ThrowIfNull(outputStream);
        ValidateStrictUnionTargetSerialization(pack);
        SerializeValidated(pack, outputStream);
    }

    private static void SerializeValidated(TerrainTileTensorPack pack, Stream outputStream)
    {

        using ZipOutputStream zip = new(outputStream);
        zip.SetLevel(3); // balance speed vs compression

        // Write each present signal as a .npy entry
        WriteArray(zip, "height_257", pack.Height257, "<f4");
        WriteArray(zip, "mcvt_vertex_z", pack.TerrainVertices?.VertexZ, "<f4");
        WriteArray(zip, "mcvt_vertex_world_x", pack.TerrainVertices?.WorldX, "<f4");
        WriteArray(zip, "mcvt_vertex_world_y", pack.TerrainVertices?.WorldY, "<f4");
        WriteArray(zip, "mcvt_triangle_indices", pack.TerrainVertices is null ? null : TerrainVertexLattice.ChunkTriangleIndices, "<i4");
        WriteArray(zip, "mcvt_vertex_present", pack.TerrainVertices?.Present, "|b1");
        WriteArray(zip, "mcvt_vertex_mask_257", pack.TerrainVertices?.DenseValidMask, "|b1");
        WriteArray(zip, "wdl_outer_17", pack.WdlLattice?.Outer17, "<f4");
        WriteArray(zip, "wdl_inner_16", pack.WdlLattice?.Inner16, "<f4");
        WriteArray(zip, "wdl_outer_present", pack.WdlLattice?.OuterPresent, "|b1");
        WriteArray(zip, "wdl_inner_present", pack.WdlLattice?.InnerPresent, "|b1");
        WriteArray(zip, "height_65", pack.Height65, "<f4");
        WriteArray(zip, "height_17", pack.Height17, "<f4");
        WriteArray(zip, "mcly_texture_ids", pack.MclyTextureIds, "<i4");
        WriteArray(zip, "mcly_layer_mask", pack.MclyLayerMask, "|b1");
        WriteArray(zip, "mcmt_material_ids", pack.McmtMaterialIds, "|u1");
        WriteArray(zip, "mamp_value", pack.MampValue, "|u1");
        WriteArray(zip, "mcal_alpha_pack", pack.McalAlphaPack, "<f4");
        WriteArray(zip, "mcal_alpha_pack_256", pack.McalAlphaPack256, "<f4");
        WriteArray(zip, "mccv_rgb", pack.MccvRgb, "<f4");
        WriteArray(zip, "mclv_lighting_bytes", pack.MclvLightingBytes, "|u1");
        WriteArray(zip, "mcnr_normal_xyz", pack.McnrNormalXyz, "<f4");
        WriteArray(zip, "mcnr_mask_257", pack.McnrMask257, "|b1");
        WriteArray(zip, "mfbo_flight_bounds", pack.MfboFlightBounds, "<i4");
        WriteArray(zip, "mh2o_surface_height", pack.Mh2oSurfaceHeight, "<f4");
        WriteArray(zip, "mh2o_depth", pack.Mh2oDepth, "<f4");
        WriteArray(zip, "mh2o_type_mask", pack.Mh2oTypeMask, "<i4");
        WriteArray(zip, "mh2o_presence_mask", pack.Mh2oPresenceMask, "|b1");
        WriteArray(zip, "mclq_surface_height", pack.MclqSurfaceHeight, "<f4");
        WriteArray(zip, "mclq_type_mask", pack.MclqTypeMask, "<i4");
        WriteArray(zip, "mclq_presence_mask", pack.MclqPresenceMask, "|b1");
        WriteArray(zip, "wl_liquid_mask", pack.WlLiquidMask, "<f4");
        WriteArray(zip, "wl_liquid_height", pack.WlLiquidHeight, "<f4");
        WriteArray(zip, "unified_liquid_mask", pack.UnifiedLiquidMask, "<f4");
        WriteArray(zip, "unified_liquid_height", pack.UnifiedLiquidHeight, "<f4");
        WriteArray(zip, "liquid_basic_type_257", pack.LiquidBasicType257, "|u1");
        WriteArray(zip, "object_mask_257", pack.ObjectMask257, "<f4");
        WriteArray(zip, "object_precise_mask_257", pack.ObjectPreciseMask257, "<f4");
        WriteArray(zip, "object_geometry_visible_mask_257", pack.ObjectGeometryVisibleMask257, "<f4");
        WriteArray(zip, "object_geometry_visible_top_elevation_257", pack.ObjectGeometryVisibleTopElevation257, "<f4");
        WriteArray(zip, "object_geometry_visible_terrain_elevation_257", pack.ObjectGeometryVisibleTerrainElevation257, "<f4");
        WriteArray(zip, "object_geometry_visible_source_257", pack.ObjectGeometryVisibleSource257, "|u1");
        WriteStrictObjectGeometryFragmentTrace(zip, pack);
        WriteArray(zip, "object_instance_mask_257", pack.ObjectInstanceMask257, "<i4");
        WriteArray(zip, "mcnk_flags_16", pack.McnkFlags16, "<i4");
        WriteArray(zip, "mddf_mask_257", pack.MddfMask257, "<f4");
        WriteArray(zip, "modf_mask_257", pack.ModfMask257, "<f4");
        WriteArray(zip, "object_filtered_mask_257", pack.ObjectFilteredMask257, "<f4");
        WriteArray(zip, "pm4_path_mask", pack.Pm4PathMask, "<f4");
        WriteArray(zip, "pm4_building_footprint_mask", pack.Pm4BuildingFootprintMask, "<f4");
        WriteArray(zip, "pm4_mprl_mask", pack.Pm4MprlMask, "<f4");
        WriteArray(zip, "mcsh_shadow_mask_256", pack.McshShadowMask256, "<f4");
        WriteArray(zip, "shadow_residual_mask_256", pack.ShadowResidualMask256, "<f4");
        WriteArray(zip, "object_roof_mask_256", pack.ObjectRoofMask256, "<f4");
        WriteArray(zip, "object_roof_confidence_256", pack.ObjectRoofConfidence256, "<f4");
        WriteArray(zip, "minimap_rgb_256", pack.MinimapRgb256, "|u1");
        WriteArray(zip, "hole_mask_16", pack.HoleMask16, "|b1");
        WriteArray(zip, "mtxf_animated_mask", pack.MtxfAnimatedMask, "<i4");
        WriteArray(zip, "mtxf_transform_id", pack.MtxfTransformId, "<i4");
        WriteArray(zip, "mcse_emitter_counts_16", pack.McseEmitterCounts16, "<i4");
        WriteArray(zip, "mcse_entry_ids", pack.McseEntryIds, "<i4");
        WriteArray(zip, "mcse_position_xyz", pack.McsePositionXyz, "<f4");
        WriteArray(zip, "mcse_entry_bytes", pack.McseEntryBytes, "|u1");
        WriteArray(zip, "mcrf_doodad_ref_counts_16", pack.McrfDoodadRefCounts16, "<i4");
        WriteArray(zip, "mcrf_doodad_ref_indices", pack.McrfDoodadRefIndices, "<i4");
        WriteArray(zip, "mcrf_wmo_ref_counts_16", pack.McrfWmoRefCounts16, "<i4");
        WriteArray(zip, "mcrf_wmo_ref_indices", pack.McrfWmoRefIndices, "<i4");
        WriteArray(zip, "mcrd_ref_counts_16", pack.McrdRefCounts16, "<i4");
        WriteArray(zip, "mcrd_ref_indices", pack.McrdRefIndices, "<i4");
        WriteArray(zip, "mcrw_ref_counts_16", pack.McrwRefCounts16, "<i4");
        WriteArray(zip, "mcrw_ref_indices", pack.McrwRefIndices, "<i4");
        WriteArray(zip, "placement_mddf_data", pack.PlacementMddfData, "<f4");
        WriteArray(zip, "placement_modf_data", pack.PlacementModfData, "<f4");

        // Texture swatches for tileset identification
        if (pack.MclyTexturePixels is { Count: > 0 } pixels)
        {
            for (int i = 0; i < pixels.Count; i++)
                WriteArray(zip, $"mcly_texture_pixels_{i}", pixels[i], "|u1");
        }

        WriteRawChunks(zip, pack.RawChunks);

        // Write metadata JSON
        WriteMetadata(zip, pack);

        zip.Finish();
    }

    /// <summary>
    /// NPZ always writes the four strict union arrays when present, so reject
    /// any pack whose provenance says the target is incomplete. Fragment-trace
    /// sidecars are intentionally outside this gate: they remain valid audit
    /// evidence for nonmaterialized targets.
    /// </summary>
    private static void ValidateStrictUnionTargetSerialization(TerrainTileTensorPack pack)
    {
        // Incomplete targets may preserve a fragment trace even though they
        // deliberately omit all four union-label arrays. Validate that sidecar
        // without making its incompleteness a reason to reject it.
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
        ZipOutputStream zip,
        TerrainTileTensorPack pack)
    {
        ObjectGeometryFragmentTrace? trace = pack.ObjectGeometryFragmentTrace;
        if (trace is null)
            return;

        WriteArray(zip, "object_geometry_fragment_raster_xy", trace.RasterXy, "<i4");
        WriteArray(zip, "object_geometry_fragment_world_xyze", trace.ObjectWorldXyzElevation, "<f4");
        WriteArray(zip, "object_geometry_fragment_source_ids", trace.SourceIds, "<i4");
        WriteArray(zip, "object_geometry_fragment_source_classification", trace.SourceClassification, "|u1");
        WriteArray(zip, "object_geometry_fragment_terrain_vertex_dense_xy", trace.TerrainVertexDenseXy, "<i4");
        WriteArray(zip, "object_geometry_fragment_terrain_vertex_z", trace.TerrainVertexZ, "<f4");
        WriteArray(zip, "object_geometry_fragment_terrain_vertex_present", trace.TerrainVertexPresent, "|u1");
        WriteArray(zip, "object_geometry_fragment_terrain_weights", trace.TerrainWeights, "<f4");
        WriteArray(zip, "object_geometry_fragment_terrain_liquid_elevation", trace.TerrainLiquidElevation, "<f4");
    }

    private static void WriteRawChunks(ZipOutputStream zip, IReadOnlyList<TerrainRawChunkBlob> rawChunks)
    {
        foreach (TerrainRawChunkBlob rawChunk in rawChunks)
        {
            if (rawChunk.Data.Length == 0 || string.IsNullOrWhiteSpace(rawChunk.EntryName))
                continue;

            WriteArray(zip, rawChunk.EntryName, rawChunk.Data, "|u1");
        }
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
            tile_x = pack.TileX,
            tile_y = pack.TileY,
            build_key = pack.BuildKey,
            source_adt_path = pack.SourceAdtPath,
            available_signals = pack.AvailableSignals.OrderBy(static signal => signal, StringComparer.OrdinalIgnoreCase),
            mcly_texture_names = pack.MclyTextureNames,
            mcly_texture_name_table = BuildNameTable(pack.MclyTextureNames),
            placement_mddf_names = pack.PlacementMddfNames,
            placement_mddf_name_table = BuildNameTable(pack.PlacementMddfNames),
            placement_modf_names = pack.PlacementModfNames,
            placement_modf_name_table = BuildNameTable(pack.PlacementModfNames),
            placement_mddf_count = pack.PlacementMddfCount,
            placement_modf_count = pack.PlacementModfCount,
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
            minimap_source_tag = pack.MinimapSourceTag,
            raw_chunks = pack.RawChunks.Select(static rawChunk => new
            {
                entry_name = rawChunk.EntryName,
                source_kind = rawChunk.SourceKind,
                source_path = rawChunk.SourcePath,
                scope = rawChunk.Scope,
                chunk_id = rawChunk.ChunkId,
                chunk_index = rawChunk.ChunkIndex,
                chunk_x = rawChunk.ChunkX,
                chunk_y = rawChunk.ChunkY,
                byte_length = rawChunk.Data.Length,
            }),
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

    private static object[] BuildNameTable(IReadOnlyList<string> names)
    {
        return names
            .Select(static (path, index) => new { index, path })
            .Where(static entry => !string.IsNullOrEmpty(entry.path))
            .Cast<object>()
            .ToArray();
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
            case 1:
                var i1 = (int[])array;
                for (int valueIndex = 0; valueIndex < i1.Length; valueIndex++)
                    BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(valueIndex * sizeof(int), sizeof(int)), i1[valueIndex]);
                break;
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
            case 1:
                var b1 = (byte[])array;
                Buffer.BlockCopy(b1, 0, result, 0, b1.Length);
                break;
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
