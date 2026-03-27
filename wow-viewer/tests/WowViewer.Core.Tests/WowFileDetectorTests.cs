using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class WowFileDetectorTests
{
    [Fact]
    public void Detect_SyntheticWdtBuffer_ReturnsWdtAndVersion()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MPHD", new byte[32]),
            .. CreateChunk("MAIN", new byte[64]),
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "synthetic.wdt");

        Assert.Equal(WowFileKind.Wdt, detection.Kind);
        Assert.Equal(18u, detection.Version);
    }

    [Fact]
    public void Detect_SyntheticWmoGroupBufferWithoutMver_ReturnsWmoGroup()
    {
        byte[] bytes =
        [
            .. CreateChunk("MOGP", new byte[0x80 + 8 + 12]),
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "synthetic_000.wmo");

        Assert.Equal(WowFileKind.WmoGroup, detection.Kind);
        Assert.Null(detection.Version);
    }

    [Theory]
    [InlineData("development.wdt", WowFileKind.Wdt, 18u)]
    [InlineData("development_0_0.adt", WowFileKind.Adt, 18u)]
    [InlineData("development_0_0_tex0.adt", WowFileKind.AdtTex, 18u)]
    [InlineData("development_0_0_obj0.adt", WowFileKind.AdtObj, 18u)]
    [InlineData("development_00_00.pm4", WowFileKind.Pm4, 12304u)]
    public void Detect_DevelopmentFiles_ReturnExpectedKinds(string fileName, WowFileKind expectedKind, uint expectedVersion)
    {
        WowFileDetection detection = WowFileDetector.Detect(Path.Combine(MapTestPaths.DevelopmentDirectoryPath, fileName));

        Assert.Equal(expectedKind, detection.Kind);
        Assert.Equal(expectedVersion, detection.Version);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }
}

internal static class MapFileSummaryReaderTestsAccessor
{
    public static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    public static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }
}
