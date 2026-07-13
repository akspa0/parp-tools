using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class RawArraySerializerTests
{
    [Fact]
    public void Serialize_V22_WritesFinalDatasetKeysAndDerivedArrays()
    {
        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            MapName = "Azeroth",
            BuildKey = "3_3_5_12340",
            SourceAdtPath = "World/Maps/Azeroth/Azeroth_1_2.adt",
            Height257 = new float[257, 257],
            McnrNormalXyz = BuildNormals(),
            McnrMask257 = BuildCheckerMask(),
            McalAlphaPack256 = new float[256, 256, 4],
            HoleMask16 = new bool[16, 16],
            UnifiedLiquidMask = new float[257, 257],
            UnifiedLiquidHeight = new float[257, 257],
            LiquidBasicType257 = BuildLiquidTypes(),
            ObjectMask257 = new float[257, 257],
            ObjectPreciseMask257 = BuildObjectMask(),
            ObjectInstanceMask257 = new int[257, 257],
            McnkFlags16 = new int[16, 16],
            MddfMask257 = new float[257, 257],
            ModfMask257 = new float[257, 257],
            ObjectFilteredMask257 = new float[257, 257],
            ObjectRoofMask256 = new float[256, 256],
            ObjectRoofConfidence256 = new float[256, 256],
            MinimapRgb256 = new byte[256, 256, 3],
            McshShadowMask256 = new float[256, 256],
            MclyTextureIds = new int[16, 16, 4],
            MclyLayerMask = new bool[16, 16, 4],
            MclyTextureNames = ["Tileset/Test/Test.blp"],
            MclyTexturePixels = [BuildTexturePixels()],
            PlacementMddfCount = 1,
            PlacementModfCount = 1,
            PlacementMddfData = new float[,] { { 7, 101, 1, 2, 3, 4, 5, 6, 1.5f } },
            PlacementModfData = new float[,] { { 9, 202, 10, 11, 12, 13, 14, 15, -1, -2, -3, 4, 5, 6 } },
            PlacementMddfNames = BuildNameTable(7, "World/Generic/PassiveDoodads/Test/Test.m2"),
            PlacementModfNames = BuildNameTable(9, "World/Wmo/Test/Test.wmo"),
            PerTileModelPayloads = new Dictionary<string, V22ModelPayload>
            {
                ["World/M2/Test.m2"] = new V22ModelPayload
                {
                    Kind = V22ModelPayload.ModelKind.M2,
                    LoadError = 0,
                    CanonicalPath = "World/M2/Test.m2",
                    RawArrays = new Dictionary<string, Array>
                    {
                        ["vertices"] = new float[4, 3],
                        ["triangles"] = new int[2, 3],
                        ["normals"] = new float[4, 3],
                        ["bounds"] = new float[2, 3],
                        ["render_flags"] = new uint[1],
                        ["blend_modes"] = new byte[1],
                    }
                }
            },
        };

        using MemoryStream stream = new();
        RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V22);

        byte[] bytes = stream.ToArray();
        string metadata = ReadMetadataJson(bytes);
        Dictionary<string, RawArrayInfo> arrays = ReadArrayIndex(bytes);

        Assert.Contains("normal_xyz", arrays.Keys);
        Assert.Contains("alpha_256", arrays.Keys);
        Assert.Contains("holes_16", arrays.Keys);
        Assert.Contains("liquid_type_256", arrays.Keys);
        Assert.Contains("ground_intent_height_257", arrays.Keys);
        Assert.Contains("model_focus_mask", arrays.Keys);
        Assert.Contains("model_above_terrain_mask", arrays.Keys);
        Assert.Contains("mddf_placement_data", arrays.Keys);
        Assert.Contains("modf_placement_data", arrays.Keys);
        Assert.Contains("mddf_model_ids", arrays.Keys);
        Assert.Contains("modf_model_ids", arrays.Keys);
        Assert.Contains("tileset_texture_rgb_0", arrays.Keys);
        Assert.True(arrays.Keys.Any(k => k.StartsWith("m2_model_") && k.EndsWith("_vertices")), "Missing m2_model_*_vertices");
        Assert.DoesNotContain("mcnr_normal_xyz", arrays.Keys);
        Assert.DoesNotContain("mcal_alpha_pack_256", arrays.Keys);
        Assert.Equal([1, 17], arrays["modf_placement_data"].Shape);
        Assert.Equal([256, 256], arrays["liquid_type_256"].Shape);
        Assert.Equal("|u1", arrays["liquid_type_256"].Dtype);
        Assert.Equal("<f4", arrays["mcly_layer_mask"].Dtype);
        Assert.Equal([257, 257], arrays["model_focus_mask"].Shape);
        Assert.Equal([257, 257], arrays["model_above_terrain_mask"].Shape);
        Assert.Equal([2, 2, 3], arrays["tileset_texture_rgb_0"].Shape);
        Assert.Contains("\"mtex_texture_paths\":[\"Tileset/Test/Test.blp\"]", metadata, StringComparison.Ordinal);
        Assert.Contains("\"placement_mddf_asset_paths\":[\"World/Generic/PassiveDoodads/Test/Test.m2\"]", metadata, StringComparison.Ordinal);
        Assert.Contains("\"placement_modf_asset_paths\":[\"World/Wmo/Test/Test.wmo\"]", metadata, StringComparison.Ordinal);
    }

    [Fact]
    public void Serialize_V16_WritesStrictValidMetadataJson()
    {
        TerrainTileTensorPack pack = new()
        {
            TileName = "Quoted \\\"tile\\\"",
            MapName = "Azeroth",
            SourceAdtPath = @"World\\Maps\\Azeroth\\Azeroth_1_2.adt",
        };

        using MemoryStream stream = new();
        RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V16);

        string metadata = ReadMetadataJson(stream.ToArray());
        using JsonDocument parsed = JsonDocument.Parse(metadata);
        Assert.Equal("Quoted \\\"tile\\\"", parsed.RootElement.GetProperty("tile_name").GetString());
        Assert.Equal(JsonValueKind.Null, parsed.RootElement.GetProperty("tile_x").ValueKind);
        Assert.Equal(0, parsed.RootElement.GetProperty("placement_modf_count").GetInt32());
    }

    [Fact]
    public void Serialize_V16_WritesStrictGeometryTargetSeparatelyFromLegacyMask()
    {
        float[,] strictMask = new float[257, 257];
        float[,] strictTopElevation = new float[257, 257];
        float[,] strictTerrainElevation = new float[257, 257];
        byte[,] strictSource = new byte[257, 257];
        strictMask[12, 34] = 1f;
        strictTopElevation[12, 34] = 42f;
        strictTerrainElevation[12, 34] = 12f;
        strictSource[12, 34] = (byte)ObjectGeometryPixelSource.WmoTriangle;
        ObjectGeometryTargetAsset[] strictAssets =
        [
            new ObjectGeometryTargetAsset(
                AssetIndex: 0,
                Source: ObjectGeometryPixelSource.WmoTriangle,
                NormalizedAssetPath: "world/wmo/test.wmo"),
        ];
        ObjectGeometryFragmentTrace strictTrace = ObjectGeometryFragmentTrace.Create(
        [
            new ObjectGeometryFragmentRecord(
                RasterX: 34,
                RasterY: 12,
                ObjectWorldX: 100f,
                ObjectWorldY: 200f,
                ObjectWorldZ: 42f,
                ObjectElevation: 42f,
                PlacementUniqueId: 77,
                AssetIndex: 0,
                SourceTriangleIndex: 3,
                Source: ObjectGeometryPixelSource.WmoTriangle,
                Classification: ObjectGeometryFragmentClassification.TerrainVisible,
                TerrainVertex0X: 33,
                TerrainVertex0Y: 12,
                TerrainVertex1X: 34,
                TerrainVertex1Y: 11,
                TerrainVertex2X: 35,
                TerrainVertex2Y: 12,
                TerrainVertex0Z: 12f,
                TerrainVertex1Z: 12f,
                TerrainVertex2Z: 12f,
                TerrainVertex0Present: true,
                TerrainVertex1Present: true,
                TerrainVertex2Present: true,
                TerrainWeight0: 0.5f,
                TerrainWeight1: 0.25f,
                TerrainWeight2: 0.25f,
                TerrainElevation: 12f,
                LiquidSurfaceElevation: float.NaN),
        ],
            strictAssets);

        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            ObjectGeometryVisibleMask257 = strictMask,
            ObjectGeometryVisibleTopElevation257 = strictTopElevation,
            ObjectGeometryVisibleTerrainElevation257 = strictTerrainElevation,
            ObjectGeometryVisibleSource257 = strictSource,
            ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                ObjectGeometryTargetStatus.CompleteVisible,
                PlacementCount: 2,
                GeometryResolvedPlacementCount: 2,
                GeometryUnresolvedPlacementCount: 0,
                FallbackRequiredPlacementCount: 0,
                TriangleCount: 64,
                VisiblePixelCount: 1,
                OccludedPixelCount: 3,
                TerrainUnknownPixelCount: 0,
                LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
            ObjectGeometryTargetAssets = strictAssets,
            ObjectGeometryFragmentTrace = strictTrace,
        };

        using MemoryStream stream = new();
        RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V16);

        Dictionary<string, RawArrayInfo> arrays = ReadArrayIndex(stream.ToArray());
        using JsonDocument metadata = JsonDocument.Parse(ReadMetadataJson(stream.ToArray()));

        Assert.Contains("object_geometry_visible_mask_257", arrays.Keys);
        Assert.Contains("object_geometry_visible_top_elevation_257", arrays.Keys);
        Assert.Contains("object_geometry_visible_terrain_elevation_257", arrays.Keys);
        Assert.Contains("object_geometry_visible_source_257", arrays.Keys);
        Assert.Contains("object_geometry_fragment_raster_xy", arrays.Keys);
        Assert.Contains("object_geometry_fragment_world_xyze", arrays.Keys);
        Assert.Contains("object_geometry_fragment_source_ids", arrays.Keys);
        Assert.Contains("object_geometry_fragment_source_classification", arrays.Keys);
        Assert.Contains("object_geometry_fragment_terrain_vertex_dense_xy", arrays.Keys);
        Assert.Contains("object_geometry_fragment_terrain_vertex_z", arrays.Keys);
        Assert.Contains("object_geometry_fragment_terrain_vertex_present", arrays.Keys);
        Assert.Contains("object_geometry_fragment_terrain_weights", arrays.Keys);
        Assert.Contains("object_geometry_fragment_terrain_liquid_elevation", arrays.Keys);
        Assert.DoesNotContain("object_precise_mask_257", arrays.Keys);
        Assert.Equal([257, 257], arrays["object_geometry_visible_mask_257"].Shape);
        Assert.Equal([257, 257], arrays["object_geometry_visible_source_257"].Shape);
        Assert.Equal("CompleteVisible", metadata.RootElement.GetProperty("object_geometry_target_status").GetString());
        Assert.Equal(ObjectGeometryTargetProvenance.ContractVersion, metadata.RootElement.GetProperty("object_geometry_target_version").GetString());
        Assert.True(metadata.RootElement.GetProperty("object_geometry_target_materialized").GetBoolean());
        Assert.Equal(0, metadata.RootElement.GetProperty("object_geometry_target_fallback_required_placement_count").GetInt32());
        Assert.Equal(0, metadata.RootElement.GetProperty("object_geometry_target_terrain_unknown_pixel_count").GetInt32());
        Assert.Equal(1, metadata.RootElement.GetProperty("object_geometry_fragment_count").GetInt32());
        Assert.Equal(strictTrace.ContentSha256, metadata.RootElement.GetProperty("object_geometry_fragment_sha256").GetString());
    }

    [Fact]
    public void Serialize_V16_IncompleteStrictGeometryWritesProvenanceButNeverAnEmptyTarget()
    {
        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                ObjectGeometryTargetStatus.IncompleteGeometry,
                PlacementCount: 3,
                GeometryResolvedPlacementCount: 2,
                GeometryUnresolvedPlacementCount: 1,
                FallbackRequiredPlacementCount: 1,
                TriangleCount: 64,
                VisiblePixelCount: 0,
                OccludedPixelCount: 0,
                TerrainUnknownPixelCount: 0,
                LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
        };

        using MemoryStream stream = new();
        RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V16);

        Dictionary<string, RawArrayInfo> arrays = ReadArrayIndex(stream.ToArray());
        using JsonDocument metadata = JsonDocument.Parse(ReadMetadataJson(stream.ToArray()));

        Assert.DoesNotContain("object_geometry_visible_mask_257", arrays.Keys);
        Assert.DoesNotContain("object_geometry_visible_top_elevation_257", arrays.Keys);
        Assert.DoesNotContain("object_geometry_visible_terrain_elevation_257", arrays.Keys);
        Assert.DoesNotContain("object_geometry_visible_source_257", arrays.Keys);
        Assert.Equal("IncompleteGeometry", metadata.RootElement.GetProperty("object_geometry_target_status").GetString());
        Assert.False(metadata.RootElement.GetProperty("object_geometry_target_materialized").GetBoolean());
        Assert.Equal(1, metadata.RootElement.GetProperty("object_geometry_target_fallback_required_placement_count").GetInt32());
    }

    [Fact]
    public void Serialize_V16_IncompleteStrictGeometryRetainsValidFragmentTraceWithoutUnionArrays()
    {
        (ObjectGeometryTargetAsset[] assets, ObjectGeometryFragmentTrace trace) = CreateFragmentTraceFixture();
        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            ObjectGeometryTargetProvenance = CreateIncompleteGeometryProvenance(),
            ObjectGeometryTargetAssets = assets,
            ObjectGeometryFragmentTrace = trace,
        };

        using MemoryStream stream = new();
        RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V16);

        Dictionary<string, RawArrayInfo> arrays = ReadArrayIndex(stream.ToArray());
        using JsonDocument metadata = JsonDocument.Parse(ReadMetadataJson(stream.ToArray()));

        Assert.DoesNotContain("object_geometry_visible_mask_257", arrays.Keys);
        Assert.Contains("object_geometry_fragment_raster_xy", arrays.Keys);
        Assert.Contains("object_geometry_fragment_world_xyze", arrays.Keys);
        Assert.Equal([1, 2], arrays["object_geometry_fragment_raster_xy"].Shape);
        Assert.Equal([1, 4], arrays["object_geometry_fragment_world_xyze"].Shape);
        Assert.Equal(ObjectGeometryTargetProvenance.ContractVersion,
            metadata.RootElement.GetProperty("object_geometry_fragment_trace_schema").GetString());
        Assert.Equal(trace.ContentSha256,
            metadata.RootElement.GetProperty("object_geometry_fragment_sha256").GetString());
    }

    [Fact]
    public void Serialize_V16_RejectsMutatedIncompleteFragmentTraceBeforeWriting()
    {
        (ObjectGeometryTargetAsset[] assets, ObjectGeometryFragmentTrace trace) = CreateFragmentTraceFixture();
        trace.ObjectWorldXyzElevation[0, 0] += 1f;
        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            ObjectGeometryTargetProvenance = CreateIncompleteGeometryProvenance(),
            ObjectGeometryTargetAssets = assets,
            ObjectGeometryFragmentTrace = trace,
        };

        using MemoryStream stream = new();
        InvalidDataException failure = Assert.Throws<InvalidDataException>(
            () => RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V16));

        Assert.Contains("does not match", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, stream.Length);
    }

    [Fact]
    public void Serialize_V16_RejectsMaterializedStrictUnionWithoutFragmentTrace()
    {
        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            ObjectGeometryVisibleMask257 = new float[257, 257],
            ObjectGeometryVisibleTopElevation257 = new float[257, 257],
            ObjectGeometryVisibleTerrainElevation257 = new float[257, 257],
            ObjectGeometryVisibleSource257 = new byte[257, 257],
            ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                ObjectGeometryTargetStatus.CompleteEmpty,
                PlacementCount: 0,
                GeometryResolvedPlacementCount: 0,
                GeometryUnresolvedPlacementCount: 0,
                FallbackRequiredPlacementCount: 0,
                TriangleCount: 0,
                VisiblePixelCount: 0,
                OccludedPixelCount: 0,
                TerrainUnknownPixelCount: 0,
                LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
        };

        using MemoryStream stream = new();
        InvalidDataException failure = Assert.Throws<InvalidDataException>(
            () => RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V16));

        Assert.Contains("fragment trace", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, stream.Length);
    }

    [Fact]
    public void Serialize_V16_RejectsIncompleteStrictProvenanceWithUnionArrays()
    {
        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            ObjectGeometryVisibleMask257 = new float[257, 257],
            ObjectGeometryVisibleTopElevation257 = new float[257, 257],
            ObjectGeometryVisibleTerrainElevation257 = new float[257, 257],
            ObjectGeometryVisibleSource257 = new byte[257, 257],
            ObjectGeometryTargetProvenance = new ObjectGeometryTargetProvenance(
                ObjectGeometryTargetStatus.IncompleteGeometry,
                PlacementCount: 1,
                GeometryResolvedPlacementCount: 0,
                GeometryUnresolvedPlacementCount: 1,
                FallbackRequiredPlacementCount: 1,
                TriangleCount: 0,
                VisiblePixelCount: 0,
                OccludedPixelCount: 0,
                TerrainUnknownPixelCount: 0,
                LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry),
        };

        using MemoryStream stream = new();
        InvalidDataException failure = Assert.Throws<InvalidDataException>(
            () => RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V16));

        Assert.Contains("nonmaterialized", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, stream.Length);
    }

    [Fact]
    public void Serialize_V22_WritesCanonicalMcvtArrays()
    {
        TerrainVertexLattice terrainVertices = BuildTerrainVertices();
        TerrainTileTensorPack pack = new()
        {
            TileName = "Azeroth_1_2",
            TerrainVertices = terrainVertices,
            WdlLattice = TerrainWdlLattice.FromTerrainVertices(terrainVertices),
        };

        using MemoryStream stream = new();
        RawArraySerializer.Serialize(pack, stream, RawArraySerializer.StreamProfile.V22);
        Dictionary<string, RawArrayInfo> arrays = ReadArrayIndex(stream.ToArray());

        Assert.Equal([16, 16, 145], arrays["mcvt_vertex_z"].Shape);
        Assert.Equal([16, 16, 145], arrays["mcvt_vertex_present"].Shape);
        Assert.Equal([16, 16, 145], arrays["mcvt_vertex_world_x"].Shape);
        Assert.Equal([16, 16, 145], arrays["mcvt_vertex_world_y"].Shape);
        Assert.Equal([256, 3], arrays["mcvt_triangle_indices"].Shape);
        Assert.Equal([257, 257], arrays["mcvt_vertex_mask_257"].Shape);
        Assert.Equal("<f4", arrays["mcvt_vertex_z"].Dtype);
        Assert.Equal("|b1", arrays["mcvt_vertex_present"].Dtype);
        Assert.Equal("|b1", arrays["mcvt_vertex_mask_257"].Dtype);
        Assert.Equal([17, 17], arrays["wdl_outer_17"].Shape);
        Assert.Equal([16, 16], arrays["wdl_inner_16"].Shape);
        Assert.Equal([17, 17], arrays["wdl_outer_present"].Shape);
        Assert.Equal([16, 16], arrays["wdl_inner_present"].Shape);
    }

    private static float[,,] BuildNormals()
    {
        float[,,] normals = new float[257, 257, 3];
        normals[0, 0, 2] = 1f;
        return normals;
    }

    private static TerrainVertexLattice BuildTerrainVertices()
    {
        return new TerrainVertexLattice(
            new float[16, 16, 145],
            new float[16, 16, 145],
            new float[16, 16, 145],
            new bool[16, 16, 145],
            new bool[257, 257]);
    }

    private static bool[,] BuildCheckerMask()
    {
        bool[,] mask = new bool[257, 257];
        mask[0, 0] = true;
        return mask;
    }

    private static byte[,] BuildLiquidTypes()
    {
        byte[,] types = new byte[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                types[y, x] = 0xFF;
        types[0, 0] = 0;
        types[0, 1] = 2;
        return types;
    }

    private static float[,] BuildObjectMask()
    {
        float[,] mask = new float[257, 257];
        mask[10, 10] = 1f;
        return mask;
    }

    private static string[] BuildNameTable(int index, string value)
    {
        string[] names = new string[index + 1];
        Array.Fill(names, string.Empty);
        names[index] = value;
        return names;
    }

    private static byte[,,] BuildTexturePixels()
    {
        byte[,,] pixels = new byte[2, 2, 3];
        pixels[0, 0, 0] = 255;
        return pixels;
    }

    private static ObjectGeometryTargetProvenance CreateIncompleteGeometryProvenance()
        => new(
            ObjectGeometryTargetStatus.IncompleteGeometry,
            PlacementCount: 1,
            GeometryResolvedPlacementCount: 0,
            GeometryUnresolvedPlacementCount: 1,
            FallbackRequiredPlacementCount: 1,
            TriangleCount: 1,
            VisiblePixelCount: 0,
            OccludedPixelCount: 1,
            TerrainUnknownPixelCount: 0,
            LiquidEvidenceStatus: ObjectGeometryLiquidEvidenceStatus.Dry);

    private static (ObjectGeometryTargetAsset[] Assets, ObjectGeometryFragmentTrace Trace) CreateFragmentTraceFixture()
    {
        ObjectGeometryTargetAsset[] assets =
        [
            new ObjectGeometryTargetAsset(
                AssetIndex: 0,
                Source: ObjectGeometryPixelSource.M2Triangle,
                NormalizedAssetPath: "world/m2/test.m2"),
        ];
        ObjectGeometryFragmentTrace trace = ObjectGeometryFragmentTrace.Create(
        [
            new ObjectGeometryFragmentRecord(
                RasterX: 3,
                RasterY: 4,
                ObjectWorldX: 100f,
                ObjectWorldY: 200f,
                ObjectWorldZ: 30f,
                ObjectElevation: 30f,
                PlacementUniqueId: 42,
                AssetIndex: 0,
                SourceTriangleIndex: 1,
                Source: ObjectGeometryPixelSource.M2Triangle,
                Classification: ObjectGeometryFragmentClassification.TerrainHidden,
                TerrainVertex0X: 2,
                TerrainVertex0Y: 4,
                TerrainVertex1X: 4,
                TerrainVertex1Y: 4,
                TerrainVertex2X: 3,
                TerrainVertex2Y: 5,
                TerrainVertex0Z: 31f,
                TerrainVertex1Z: 31f,
                TerrainVertex2Z: 31f,
                TerrainVertex0Present: true,
                TerrainVertex1Present: true,
                TerrainVertex2Present: true,
                TerrainWeight0: 0.25f,
                TerrainWeight1: 0.25f,
                TerrainWeight2: 0.5f,
                TerrainElevation: 31f,
                LiquidSurfaceElevation: float.NaN),
        ],
            assets);
        return (assets, trace);
    }

    private static string ReadMetadataJson(byte[] bytes)
    {
        int offset = 0;
        Assert.Equal("ARRY", Encoding.ASCII.GetString(bytes, offset, 4));
        offset += 4;
        int metadataLength = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4));
        offset += 4;
        return Encoding.UTF8.GetString(bytes, offset, metadataLength);
    }

    private static Dictionary<string, RawArrayInfo> ReadArrayIndex(byte[] bytes)
    {
        int offset = 0;
        Assert.Equal("ARRY", Encoding.ASCII.GetString(bytes, offset, 4));
        offset += 4;

        int metadataLength = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4));
        offset += 4 + metadataLength;

        Dictionary<string, RawArrayInfo> arrays = new(StringComparer.Ordinal);
        while (offset + 4 <= bytes.Length)
        {
            string magicOrLength = Encoding.ASCII.GetString(bytes, offset, 4);
            if (magicOrLength == "ENDS")
                break;

            int nameLength = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4));
            offset += 4;
            string name = Encoding.UTF8.GetString(bytes, offset, nameLength);
            offset += nameLength;

            int rank = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4));
            offset += 4;
            int[] shape = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4));
                offset += 4;
            }

            string dtype = Encoding.ASCII.GetString(bytes, offset, 8).TrimEnd('\0');
            offset += 8;
            long dataLength = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(offset, 8));
            offset += 8 + checked((int)dataLength);
            arrays.Add(name, new RawArrayInfo(shape, dtype));
        }

        return arrays;
    }

    private sealed record RawArrayInfo(int[] Shape, string Dtype);
}
