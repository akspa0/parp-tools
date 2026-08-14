using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public class LkAdtReaderTests
{
    [Fact]
    public void Read_UnpaddedOddChunkBeforeMcnk_KeepsChunkWalkAligned()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(writer, "REVM", BitConverter.GetBytes(18));
        WriteChunk(writer, "XDMM", [1, 2, 3]);
        WriteMcnkWithThreeLayerPayloadButTwoDeclaredLayers(writer);

        var adt = LkAdtReader.Read(ms.ToArray(), null, null, 0, 0);

        Assert.Single(adt.Chunks);
        Assert.Equal(2, adt.Chunks[0].Layers.Count);
    }

    [Fact]
    public void Read_McnkTrailingSubchunkWithOverflowingSize_DoesNotThrow()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(writer, "REVM", BitConverter.GetBytes(18));
        WriteMcnkWithOverflowingTrailingSubchunk(writer);

        var adt = LkAdtReader.Read(ms.ToArray(), null, null, 0, 0);

        Assert.Single(adt.Chunks);
        Assert.Equal(0, adt.Chunks[0].IndexX);
        Assert.Equal(0, adt.Chunks[0].IndexY);
    }

    [Fact]
    public void Read_SplitTextureMcnkPreservesMccvWithoutRootLayersOrAlpha()
    {
        byte[] expectedMccv = new byte[145 * 4];
        for (int index = 0; index < expectedMccv.Length; index++)
            expectedMccv[index] = (byte)(index % 251);

        byte[] root = CreateMcnkStream([]);
        byte[] tex0 = CreateMcnkStream([("MCCV", expectedMccv)]);

        LkAdtData adt = LkAdtReader.Read(root, tex0, null, 0, 0);

        Assert.Single(adt.Chunks);
        Assert.Empty(adt.Chunks[0].Layers);
        Assert.Null(adt.Chunks[0].AlphaMapData);
        Assert.Equal(expectedMccv, adt.Chunks[0].MccvColors);
    }

    [Fact]
    public void Read_SparseMcnkRecoversMccvAfterShortMcnr()
    {
        byte[] expectedMccv = new byte[145 * 4];
        for (int index = 0; index < expectedMccv.Length; index++)
            expectedMccv[index] = (byte)(index % 251);

        LkAdtData adt = LkAdtReader.Read(
            CreateMcnkStream([
                ("MCNR", new byte[145 * 3]),
                ("MCCV", expectedMccv)
            ]),
            null,
            null,
            0,
            0);

        Assert.Single(adt.Chunks);
        Assert.Equal(expectedMccv, adt.Chunks[0].MccvColors);
    }

    private static void WriteMcnkWithThreeLayerPayloadButTwoDeclaredLayers(BinaryWriter writer)
    {
        byte[] header = new byte[128];
        BitConverter.GetBytes(2).CopyTo(header, 0x0C);
        BitConverter.GetBytes(136).CopyTo(header, 0x1C);

        byte[] mcly = new byte[3 * 16];
        for (int layer = 0; layer < 3; layer++)
            BitConverter.GetBytes(layer + 10).CopyTo(mcly, layer * 16);

        writer.Write(Encoding.ASCII.GetBytes("KNCM"));
        writer.Write(header.Length + 8 + mcly.Length);
        writer.Write(header);
        writer.Write(Encoding.ASCII.GetBytes("YLCM"));
        writer.Write(mcly.Length);
        writer.Write(mcly);
    }

    private static void WriteMcnkWithOverflowingTrailingSubchunk(BinaryWriter writer)
    {
        byte[] header = new byte[128];
        BitConverter.GetBytes(0).CopyTo(header, 0x04);
        BitConverter.GetBytes(0).CopyTo(header, 0x08);

        writer.Write(Encoding.ASCII.GetBytes("KNCM"));
        writer.Write(header.Length + 8);
        writer.Write(header);
        writer.Write(Encoding.ASCII.GetBytes("ZZZZ"));
        writer.Write(int.MaxValue);
    }

    private static byte[] CreateMcnkStream((string Id, byte[] Payload)[] subchunks)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        byte[] header = new byte[128];
        WriteChunk(writer, "KNCM", [.. header, .. CreateSubchunkBytes(subchunks)]);
        return ms.ToArray();
    }

    private static byte[] CreateSubchunkBytes((string Id, byte[] Payload)[] subchunks)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);
        foreach ((string id, byte[] payload) in subchunks)
            WriteChunk(writer, ReverseTag(id), payload);
        return ms.ToArray();
    }

    private static string ReverseTag(string id)
    {
        char[] chars = id.ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    private static void WriteChunk(BinaryWriter writer, string tag, byte[] payload)
    {
        writer.Write(Encoding.ASCII.GetBytes(tag));
        writer.Write(payload.Length);
        writer.Write(payload);
    }
}
