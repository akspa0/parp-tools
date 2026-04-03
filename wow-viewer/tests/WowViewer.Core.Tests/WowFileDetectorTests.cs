using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class WowFileDetectorTests
{
    [Fact]
    public void Detect_SyntheticMdxBuffer_ReturnsMdx()
    {
        byte[] bytes =
        [
            (byte)'M', (byte)'D', (byte)'L', (byte)'X',
            .. CreateChunk("VERS", CreateUInt32Payload(1300)),
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "synthetic.mdx");

        Assert.Equal(WowFileKind.Mdx, detection.Kind);
        Assert.Null(detection.Version);
    }

    [Fact]
    public void Detect_SyntheticBlp2Buffer_ReturnsBlp()
    {
        byte[] bytes =
        [
            (byte)'B', (byte)'L', (byte)'P', (byte)'2',
            1, 0, 0, 0,
            2, 8, 7, 1,
            64, 0, 0, 0,
            64, 0, 0, 0,
            .. new byte[128],
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "synthetic.blp");

        Assert.Equal(WowFileKind.Blp, detection.Kind);
        Assert.Null(detection.Version);
    }

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
    public void Detect_SyntheticAdtV23Buffer_ReturnsAdtV23AndVersion()
    {
        byte[] bytes =
        [
            .. CreateChunk("AHDR", CreateAhdrPayload(version: 23, verticesX: 129, verticesY: 129, chunksX: 16, chunksY: 16)),
            .. CreateChunk("AVTX", new byte[16]),
            .. CreateChunk("ACNK", new byte[64]),
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "synthetic_0_0.adt");

        Assert.Equal(WowFileKind.AdtV23, detection.Kind);
        Assert.Equal(23u, detection.Version);
    }

    [Fact]
    public void Detect_SyntheticAdtV23ErrorBuffer_ReturnsAdtV23Error()
    {
        byte[] bytes =
        [
            .. CreateChunk("AHDR", CreateAhdrPayload(version: 23, verticesX: 129, verticesY: 129, chunksX: 16, chunksY: 16)),
            .. CreateChunk("ACNK", new byte[64]),
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "project_tile.error");

        Assert.Equal(WowFileKind.AdtV23Error, detection.Kind);
        Assert.Equal(23u, detection.Version);
    }

    [Fact]
    public void Detect_SyntheticLitBuffer_ReturnsLit()
    {
        byte[] bytes =
        [
            0x04, 0x00, 0x00, 0x80,
            0x00, 0x00, 0x00, 0x00,
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "lights.lit");

        Assert.Equal(WowFileKind.Lit, detection.Kind);
        Assert.Null(detection.Version);
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

    [Fact]
    public void Detect_SyntheticAdtLodBuffer_ReturnsAdtLod()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MLHD", new byte[32]),
        ];

        using MemoryStream stream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(stream, "synthetic_0_0_lod.adt");

        Assert.Equal(WowFileKind.AdtLod, detection.Kind);
        Assert.Equal(18u, detection.Version);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }

    private static byte[] CreateAhdrPayload(uint version, uint verticesX, uint verticesY, uint chunksX, uint chunksY)
    {
        byte[] bytes = new byte[0x40];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x00, 4), version);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x04, 4), verticesX);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x08, 4), verticesY);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x0C, 4), chunksX);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x10, 4), chunksY);
        return bytes;
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
