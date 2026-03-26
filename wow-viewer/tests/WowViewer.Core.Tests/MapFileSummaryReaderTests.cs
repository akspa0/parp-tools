using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class MapFileSummaryReaderTests
{
    [Fact]
    public void Read_WdtBuffer_ProducesExpectedSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MPHD", new byte[32]),
            .. CreateChunk("MAIN", new byte[64]),
            .. CreateChunk("MWMO", new byte[0]),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic.wdt");

        Assert.Equal(MapFileKind.Wdt, summary.Kind);
        Assert.Equal(18u, summary.Version);
        Assert.Equal(4, summary.ChunkCount);
        Assert.True(summary.HasChunk(MapChunkIds.Main));
        Assert.Equal(1, summary.CountChunks(MapChunkIds.Mphd));
    }

    [Fact]
    public void Read_AdtBuffer_ProducesExpectedSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MHDR", new byte[64]),
            .. CreateChunk("MCIN", new byte[4096]),
            .. CreateChunk("MCNK", new byte[128]),
            .. CreateChunk("MCNK", new byte[128]),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.Equal(18u, summary.Version);
        Assert.Equal(5, summary.ChunkCount);
        Assert.True(summary.HasChunk(MapChunkIds.Mhdr));
        Assert.Equal(2, summary.CountChunks(MapChunkIds.Mcnk));
    }

    [Fact]
    public void Read_DevelopmentWdt_ProducesExpectedTopLevelSignals()
    {
        MapFileSummary summary = MapFileSummaryReader.Read(MapTestPaths.DevelopmentWdtPath);

        Assert.Equal(MapFileKind.Wdt, summary.Kind);
        Assert.Equal(18u, summary.Version);
        Assert.True(summary.HasChunk(MapChunkIds.Mphd));
        Assert.True(summary.HasChunk(MapChunkIds.Main));
        Assert.True(summary.ChunkCount >= 3);
    }

    [Fact]
    public void Read_DevelopmentRootAdt_ProducesExpectedTopLevelSignals()
    {
        MapFileSummary summary = MapFileSummaryReader.Read(MapTestPaths.DevelopmentRootAdtPath);

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.Equal(18u, summary.Version);
        Assert.True(summary.HasChunk(MapChunkIds.Mhdr));
        Assert.True(summary.HasChunk(MapChunkIds.Mcnk));
        Assert.True(summary.HasChunk(MapChunkIds.Mfbo));
        Assert.Equal(256, summary.CountChunks(MapChunkIds.Mcnk));
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
}

internal static class MapTestPaths
{
    public static string DevelopmentDirectoryPath => Path.Combine(GetWowViewerRoot(), "..", "gillijimproject_refactor", "test_data", "development", "World", "Maps", "development");

    public static string DevelopmentWdtPath => Path.Combine(DevelopmentDirectoryPath, "development.wdt");

    public static string DevelopmentRootAdtPath => Path.Combine(DevelopmentDirectoryPath, "development_0_0.adt");

    private static string GetWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the test output directory.");
    }
}