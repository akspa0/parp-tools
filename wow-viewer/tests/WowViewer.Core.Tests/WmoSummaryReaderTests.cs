using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoSummaryReaderTests
{
    [Fact]
    public void Read_WmoBuffer_ProducesSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(17)),
            .. CreateChunk("MOHD", CreateMohdPayload(
                materialCount: 2,
                groupCount: 4,
                portalCount: 1,
                lightCount: 3,
                doodadNameCount: 5,
                doodadDefCount: 6,
                doodadSetCount: 2,
                flags: 0x1234,
                boundsMin: new Vector3(-1f, -2f, -3f),
                boundsMax: new Vector3(4f, 5f, 6f))),
            .. CreateChunk("MOTX", CreateStringBlock("a.blp", "b.blp")),
            .. CreateChunk("MOMT", new byte[2 * 64]),
            .. CreateChunk("MOGI", new byte[4 * 32]),
            .. CreateChunk("MOSB", CreateStringBlock("world\\sky\\test.sky")),
            .. CreateChunk("MODS", new byte[2 * 32]),
            .. CreateChunk("MODN", CreateStringBlock("a.mdx", "b.mdx", "c.mdx", "d.mdx", "e.mdx")),
            .. CreateChunk("MODD", new byte[6 * 40]),
        ];

        using MemoryStream stream = new(bytes);
        var summary = WmoSummaryReader.Read(stream, "synthetic.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(2, summary.ReportedMaterialCount);
        Assert.Equal(2, summary.MaterialEntryCount);
        Assert.Equal(4, summary.ReportedGroupCount);
        Assert.Equal(4, summary.GroupInfoCount);
        Assert.Equal(1, summary.ReportedPortalCount);
        Assert.Equal(3, summary.ReportedLightCount);
        Assert.Equal(2, summary.TextureNameCount);
        Assert.Equal(5, summary.ReportedDoodadNameCount);
        Assert.Equal(5, summary.DoodadNameTableCount);
        Assert.Equal(6, summary.ReportedDoodadPlacementCount);
        Assert.Equal(6, summary.DoodadPlacementEntryCount);
        Assert.Equal(2, summary.ReportedDoodadSetCount);
        Assert.Equal(2, summary.DoodadSetEntryCount);
        Assert.True(summary.HasSkybox);
        Assert.Equal(0x1234u, summary.Flags);
        Assert.Equal(new Vector3(-1f, -2f, -3f), summary.BoundsMin);
        Assert.Equal(new Vector3(4f, 5f, 6f), summary.BoundsMax);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }

    private static byte[] CreateMohdPayload(
        uint materialCount,
        uint groupCount,
        uint portalCount,
        uint lightCount,
        uint doodadNameCount,
        uint doodadDefCount,
        uint doodadSetCount,
        uint flags,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), materialCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), portalCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), lightCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16, 4), doodadNameCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(20, 4), doodadDefCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24, 4), doodadSetCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28, 4), 0xFF000000);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(32, 4), 1);
        WriteSingle(bytes, 36, boundsMin.X);
        WriteSingle(bytes, 40, boundsMin.Y);
        WriteSingle(bytes, 44, boundsMin.Z);
        WriteSingle(bytes, 48, boundsMax.X);
        WriteSingle(bytes, 52, boundsMax.Y);
        WriteSingle(bytes, 56, boundsMax.Z);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(60, 4), flags);
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
