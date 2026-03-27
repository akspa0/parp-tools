using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupLiquidSummaryReaderTests
{
    [Fact]
    public void Read_WmoGroupLiquidBuffer_ProducesLiquidSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(17)),
            .. CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x80,
                flags: 0x80000,
                boundsMin: Vector3.Zero,
                boundsMax: Vector3.One,
                portalStart: 0,
                portalCount: 0,
                transBatchCount: 0,
                intBatchCount: 0,
                extBatchCount: 0,
                groupLiquid: 4,
                subchunks:
                [
                    ("MLIQ", CreateMliqPayload(2, 2, 1, 1, new Vector3(10f, 20f, 30f), 9, [1f, 2f, 3f, 4f], [0x00])),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupLiquidSummary summary = WmoGroupLiquidSummaryReader.Read(stream, "synthetic_liquid_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(63, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.XVertexCount);
        Assert.Equal(2, summary.YVertexCount);
        Assert.Equal(1, summary.XTileCount);
        Assert.Equal(1, summary.YTileCount);
        Assert.Equal(1, summary.TileCount);
        Assert.Equal(new Vector3(10f, 20f, 30f), summary.Corner);
        Assert.Equal(9, summary.MaterialId);
        Assert.Equal(4, summary.HeightCount);
        Assert.Equal(1f, summary.MinHeight);
        Assert.Equal(4f, summary.MaxHeight);
        Assert.Equal(1, summary.TileFlagByteCount);
        Assert.True(summary.HasCompleteTileFlags);
        Assert.Equal(1, summary.VisibleTileCount);
        Assert.Equal(WmoLiquidBasicType.Ocean, summary.LiquidType);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }

    private static byte[] CreateMogpPayload(
        int headerSize,
        uint flags,
        Vector3 boundsMin,
        Vector3 boundsMax,
        ushort portalStart,
        ushort portalCount,
        ushort transBatchCount,
        ushort intBatchCount,
        ushort extBatchCount,
        uint groupLiquid,
        params (string Id, byte[] Payload)[] subchunks)
    {
        byte[] header = new byte[headerSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), flags);
        WriteSingle(header, 0x0C, boundsMin.X);
        WriteSingle(header, 0x10, boundsMin.Y);
        WriteSingle(header, 0x14, boundsMin.Z);
        WriteSingle(header, 0x18, boundsMax.X);
        WriteSingle(header, 0x1C, boundsMax.Y);
        WriteSingle(header, 0x20, boundsMax.Z);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x24, 2), portalStart);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x26, 2), portalCount);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x28, 2), transBatchCount);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x2A, 2), intBatchCount);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x2C, 2), extBatchCount);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), groupLiquid);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(CreateChunk(id, payload));

        return stream.ToArray();
    }

    private static byte[] CreateMliqPayload(int xVerts, int yVerts, int xTiles, int yTiles, Vector3 corner, ushort materialId, float[] heights, byte[] tileFlags)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream);
        writer.Write(xVerts);
        writer.Write(yVerts);
        writer.Write(xTiles);
        writer.Write(yTiles);
        writer.Write(corner.X);
        writer.Write(corner.Y);
        writer.Write(corner.Z);
        writer.Write(materialId);
        foreach (float height in heights)
        {
            writer.Write(0);
            writer.Write(height);
        }

        writer.Write(tileFlags);
        return stream.ToArray();
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
