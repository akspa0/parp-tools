using System.Text;
using WowViewer.Core.IO.Maps;

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
    public void Read_OriginalDevelopmentSplitFamily_MergesTexAndObjWithoutThrowing()
    {
        string rootPath = Path.Combine(MapTestPaths.OriginalDevelopmentDirectoryPath, "development_0_0.adt");
        string texPath = Path.Combine(MapTestPaths.OriginalDevelopmentDirectoryPath, "development_0_0_tex0.adt");
        string objPath = Path.Combine(MapTestPaths.OriginalDevelopmentDirectoryPath, "development_0_0_obj0.adt");

        var adt = LkAdtReader.Read(
            File.ReadAllBytes(rootPath),
            File.ReadAllBytes(texPath),
            File.ReadAllBytes(objPath),
            0,
            0);

        Assert.NotEmpty(adt.Chunks);
        Assert.Contains(adt.Chunks, static chunk => chunk.Layers.Count > 0);
        Assert.True(adt.ModelNames.Count > 0 || adt.WorldModelNames.Count > 0);
        Assert.True(adt.ModelPlacements.Count > 0 || adt.WorldModelPlacements.Count > 0);
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

    private static void WriteChunk(BinaryWriter writer, string tag, byte[] payload)
    {
        writer.Write(Encoding.ASCII.GetBytes(tag));
        writer.Write(payload.Length);
        writer.Write(payload);
    }
}
