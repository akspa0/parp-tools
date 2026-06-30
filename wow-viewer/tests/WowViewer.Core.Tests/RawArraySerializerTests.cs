using System.Buffers.Binary;
using System.Text;
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
        Assert.Contains("mddf_placement_data", arrays.Keys);
        Assert.Contains("modf_placement_data", arrays.Keys);
        Assert.Contains("mddf_model_ids", arrays.Keys);
        Assert.Contains("modf_model_ids", arrays.Keys);
        Assert.Contains("tileset_texture_rgb_0", arrays.Keys);
        Assert.DoesNotContain("mcnr_normal_xyz", arrays.Keys);
        Assert.DoesNotContain("mcal_alpha_pack_256", arrays.Keys);
        Assert.Equal([1, 17], arrays["modf_placement_data"].Shape);
        Assert.Equal([256, 256], arrays["liquid_type_256"].Shape);
        Assert.Equal("|u1", arrays["liquid_type_256"].Dtype);
        Assert.Equal("<f4", arrays["mcly_layer_mask"].Dtype);
        Assert.Equal([2, 2, 3], arrays["tileset_texture_rgb_0"].Shape);
        Assert.Contains("\"mtex_texture_paths\":[\"Tileset/Test/Test.blp\"]", metadata, StringComparison.Ordinal);
        Assert.Contains("\"placement_mddf_asset_paths\":[\"World/Generic/PassiveDoodads/Test/Test.m2\"]", metadata, StringComparison.Ordinal);
        Assert.Contains("\"placement_modf_asset_paths\":[\"World/Wmo/Test/Test.wmo\"]", metadata, StringComparison.Ordinal);
    }

    private static float[,,] BuildNormals()
    {
        float[,,] normals = new float[257, 257, 3];
        normals[0, 0, 2] = 1f;
        return normals;
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
